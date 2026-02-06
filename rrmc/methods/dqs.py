"""
DQS – Deliberative Question Selection for RRMC.
================================================

At each turn, instead of letting the LLM fire off a single question,
DQS has the LLM *deliberate* by exploring M parallel paths and then
selecting the most promising one.

Architecture (per turn)
-----------------------
1. **M parallel LLM calls** – each independently proposes one candidate
   question/guess.  Using temperature > 0 gives natural diversity.
2. **M parallel LLM calls** – for each candidate, a separate LLM call
   imagines what feedback / response the candidate would receive.
   (Parallel across candidates, but sequential to step 1 per candidate.)
3. **1 orchestrator LLM call** – sees ALL M (candidate, imagined-feedback)
   pairs and decides which single candidate to actually play.

The LLM is in control at *every* step, including the final selection.
No mathematical scoring formulas — the orchestrator LLM picks directly.

Inference cost: ``2·M + 1`` LLM calls per turn.

Supports **DC** (Detective Cases) and **GN** (Guessing Numbers).

This module is **fully portable**: it does NOT modify any existing RRMC
core logic, prompts, stopping rules, or evaluation harness.
"""

import re
import json
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.llm import LLMWrapper
from ..utils.gn_utils import extract_guess
from .question_selector import CandidateQuestion, QuestionSelectionResult


# =====================================================================
# Prompts — DC
# =====================================================================

_PROPOSE_ONE_DC = """\
You are a detective investigating a murder case.

Case Background:
{background}

Previous Interrogation:
{history}

Suspects: {suspect_names}

Propose ONE strategic question to ask ONE of the suspects. Choose the \
suspect and question that you believe would be most informative for \
identifying the true murderer.

Output as JSON (no other text):
{{"suspect": "<full name>", "question": "<your question>"}}"""

_SIMULATE_ONE_DC = """\
You are role-playing as a suspect in a murder investigation.

Your Character: {suspect_name}
Case Background:
{background}

Previous Q&A in this investigation:
{history}

A detective now asks you: "{question}"

Respond in-character with a brief, plausible one-sentence answer. Stay \
consistent with the case details and your character.
Response:"""

_ORCHESTRATE_DC = """\
You are a detective investigating a murder case.

Case Background:
{background}

Previous Interrogation:
{history}

Suspects: {suspect_names}

Your team has proposed {m} possible lines of questioning. For each one, \
they have also imagined how the suspect might respond:

{branches}

Now YOU must decide: which of these {m} options would give you the most \
useful information to identify the true murderer? Consider:
- Which question targets the most suspicious behavior?
- Which imagined response reveals the most new information?
- Which angle hasn't been explored yet?

Pick the single best option. Output ONLY the option number (1-{m}), \
nothing else.
Best option:"""

# =====================================================================
# Prompts — GN
# =====================================================================

_PROPOSE_ONE_GN = """\
You are playing a number guessing game (Bulls & Cows).

Rules: Guess a 4-digit number where all digits are unique (0-9). \
After each guess you get feedback:
- "Bulls": digits in the correct position
- "Cows": correct digits but in the wrong position

Previous guesses and feedback:
{history}

Propose ONE strategic 4-digit guess (all digits unique) that would \
help narrow down the secret number. Think about what digits and \
positions have been ruled in or out by the feedback so far.

Output ONLY the 4-digit number, nothing else.
Guess:"""

_SIMULATE_ONE_GN = """\
You are playing a number guessing game (Bulls & Cows).

Rules:
- The secret is a 4-digit number with all unique digits.
- "Bulls" = digits in the correct position.
- "Cows" = correct digits but in the wrong position.

Previous guesses and feedback:
{history}

If you were to guess **{guess}**, what feedback do you think you would \
most likely receive? Reason briefly about which digits are likely \
correct based on the patterns so far, then give your prediction.

Respond with ONLY the feedback in this format:
X digits in correct position, Y digits in wrong position"""

_ORCHESTRATE_GN = """\
You are playing a number guessing game (Bulls & Cows).

Rules:
- The secret is a 4-digit number with all unique digits (0-9).
- "Bulls" = digits in the correct position.
- "Cows" = correct digits but in the wrong position.

Previous guesses and feedback:
{history}

You are considering {m} possible guesses. For each, you've imagined \
what feedback it might produce:

{branches}

Now decide: which guess would give you the most useful information \
to narrow down the secret number? Consider:
- Which guess tests the most uncertain digits/positions?
- Which imagined feedback would eliminate the most possibilities?
- Which guess is most consistent with all previous feedback?

Pick the single best option. Output ONLY the option number (1-{m}), \
nothing else.
Best option:"""


# =====================================================================
# Data classes
# =====================================================================

@dataclass
class DQSBranch:
    """One candidate branch: a proposed action + its imagined outcome."""
    candidate: CandidateQuestion
    imagined_feedback: str


@dataclass
class DQSMetrics:
    """Per-turn metrics produced by a DQS selection step."""
    n_candidates: int
    branches: List[DQSBranch]
    selected_idx: int
    selected_suspect: str
    selected_question: str
    orchestrator_raw: str


# =====================================================================
# Main class
# =====================================================================

class DQSQuestionSelector:
    """
    Deliberative Question Selection (DQS).

    Per-turn flow:
      1. M parallel LLM calls → M independent candidate proposals
      2. M parallel LLM calls → M imagined feedbacks (one per candidate)
      3. 1 orchestrator LLM call → sees all branches, picks the best

    Total: 2M + 1 LLM calls per turn.  The LLM decides everything.

    Parameters
    ----------
    llm : LLMWrapper
        Shared LLM wrapper (thread-safe).
    m_candidates : int
        Number of parallel candidate proposals per turn (*M*).
    temperature : float
        Sampling temperature for proposal and simulation calls.
    orchestrator_temperature : float
        Temperature for the final orchestrator selection call.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        m_candidates: int = 5,
        r_simulations: int = 1,          # kept for config compat, not used
        temperature: float = 0.7,
        eval_temperature: float = 0.3,   # used as orchestrator temp
    ):
        self.llm = llm
        self.m_candidates = m_candidates
        self.temperature = temperature
        self.orchestrator_temperature = eval_temperature
        self._last_metrics: Optional[DQSMetrics] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_question(
        self,
        task_type: str,
        state: Dict[str, Any],
        current_mi: float = 0.0,
    ) -> QuestionSelectionResult:
        """Select the best question/guess via deliberation.

        Parameters
        ----------
        task_type : str
            ``"DC"`` or ``"GN"``.
        state : dict
            Current environment state.
        current_mi : float
            Unused (interface compatibility).

        Returns
        -------
        QuestionSelectionResult
        """
        if task_type == "DC":
            return self._select_dc(state, current_mi)
        if task_type == "GN":
            return self._select_gn(state, current_mi)
        raise NotImplementedError(f"DQS supports DC and GN, got {task_type}")

    # ==================================================================
    # DC
    # ==================================================================

    def _select_dc(
        self, state: Dict[str, Any], current_mi: float
    ) -> QuestionSelectionResult:
        background = state.get("initial_info", "")
        history_str = state.get("history_string", "") or "(No interrogation yet)"
        suspect_names = state.get("suspect_names", [])
        suspect_str = ", ".join(suspect_names)

        # --- Step 1: M parallel proposals ---
        candidates = self._parallel_propose_dc(
            background, history_str, suspect_names
        )

        # --- Step 2: M parallel simulations ---
        branches = self._parallel_simulate_dc(
            candidates, background, history_str
        )

        # --- Step 3: 1 orchestrator selection ---
        selected, sel_idx, orch_raw = self._orchestrate_dc(
            branches, background, history_str, suspect_str
        )

        self._last_metrics = DQSMetrics(
            n_candidates=len(branches),
            branches=branches,
            selected_idx=sel_idx,
            selected_suspect=selected.suspect,
            selected_question=selected.question,
            orchestrator_raw=orch_raw,
        )

        return QuestionSelectionResult(
            selected=selected,
            candidates=[b.candidate for b in branches],
            current_mi=current_mi,
        )

    def _parallel_propose_dc(
        self, background: str, history_str: str, suspect_names: List[str]
    ) -> List[CandidateQuestion]:
        """M independent parallel calls, each proposing one question."""
        prompt = _PROPOSE_ONE_DC.format(
            background=background,
            history=history_str,
            suspect_names=", ".join(suspect_names),
        )

        # Fire M parallel calls with temperature for diversity
        responses = self.llm.sample_n(
            messages=[{"role": "user", "content": prompt}],
            n=self.m_candidates,
            temperature=self.temperature,
            max_tokens=256,
            parallel=True,
        )

        candidates: List[CandidateQuestion] = []
        seen: set = set()
        for resp in responses:
            c = self._parse_one_dc(resp.content, suspect_names)
            if c:
                key = (c.suspect.lower(), c.question.strip().lower())
                if key not in seen:
                    seen.add(key)
                    candidates.append(c)

        # Fallback if parsing failed
        if not candidates:
            candidates.append(CandidateQuestion(
                suspect=suspect_names[0] if suspect_names else "",
                question="Where were you at the time of the crime?",
            ))

        return candidates

    def _parse_one_dc(
        self, text: str, suspect_names: List[str]
    ) -> Optional[CandidateQuestion]:
        """Parse a single {suspect, question} JSON from one proposal."""
        try:
            # Try JSON object
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                suspect = obj.get("suspect", "")
                question = obj.get("question", "")
                if question:
                    suspect = self._match_suspect(suspect, suspect_names)
                    return CandidateQuestion(suspect=suspect, question=question)
        except (json.JSONDecodeError, AttributeError):
            pass
        return None

    def _parallel_simulate_dc(
        self,
        candidates: List[CandidateQuestion],
        background: str,
        history_str: str,
    ) -> List[DQSBranch]:
        """M parallel simulation calls (one per candidate)."""
        branches: List[DQSBranch] = []

        def _sim_one(c: CandidateQuestion) -> DQSBranch:
            prompt = _SIMULATE_ONE_DC.format(
                suspect_name=c.suspect,
                background=background,
                history=history_str,
                question=c.question,
            )
            resp = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=128,
            )
            return DQSBranch(candidate=c, imagined_feedback=resp.content.strip())

        max_w = min(len(candidates), getattr(self.llm, "max_workers", 4))
        if max_w <= 1 or len(candidates) <= 1:
            branches = [_sim_one(c) for c in candidates]
        else:
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futs = {pool.submit(_sim_one, c): i for i, c in enumerate(candidates)}
                result_map: Dict[int, DQSBranch] = {}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        result_map[idx] = f.result()
                    except Exception:
                        result_map[idx] = DQSBranch(
                            candidate=candidates[idx],
                            imagined_feedback="(no response)",
                        )
                branches = [result_map[i] for i in range(len(candidates))]

        return branches

    def _orchestrate_dc(
        self,
        branches: List[DQSBranch],
        background: str,
        history_str: str,
        suspect_str: str,
    ) -> Tuple[CandidateQuestion, int, str]:
        """One orchestrator call — LLM picks the best branch."""
        branch_strs = []
        for i, b in enumerate(branches):
            branch_strs.append(
                f"Option {i+1}: Ask {b.candidate.suspect}: "
                f"\"{b.candidate.question}\"\n"
                f"  Imagined response: \"{b.imagined_feedback}\""
            )

        prompt = _ORCHESTRATE_DC.format(
            background=background,
            history=history_str,
            suspect_names=suspect_str,
            m=len(branches),
            branches="\n\n".join(branch_strs),
        )

        resp = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.orchestrator_temperature,
            max_tokens=32,
        )

        raw = resp.content.strip()
        sel_idx = self._parse_option_number(raw, len(branches))
        return branches[sel_idx].candidate, sel_idx, raw

    # ==================================================================
    # GN
    # ==================================================================

    def _select_gn(
        self, state: Dict[str, Any], current_mi: float
    ) -> QuestionSelectionResult:
        history_str = state.get("history_string", "") or "(No guesses yet)"

        # --- Step 1: M parallel proposals ---
        candidates = self._parallel_propose_gn(history_str)

        # --- Step 2: M parallel simulations ---
        branches = self._parallel_simulate_gn(candidates, history_str)

        # --- Step 3: 1 orchestrator selection ---
        selected, sel_idx, orch_raw = self._orchestrate_gn(
            branches, history_str
        )

        self._last_metrics = DQSMetrics(
            n_candidates=len(branches),
            branches=branches,
            selected_idx=sel_idx,
            selected_suspect="",
            selected_question=selected.question,
            orchestrator_raw=orch_raw,
        )

        return QuestionSelectionResult(
            selected=selected,
            candidates=[b.candidate for b in branches],
            current_mi=current_mi,
        )

    def _parallel_propose_gn(
        self, history_str: str
    ) -> List[CandidateQuestion]:
        """M independent parallel calls, each proposing one guess."""
        prompt = _PROPOSE_ONE_GN.format(history=history_str)

        responses = self.llm.sample_n(
            messages=[{"role": "user", "content": prompt}],
            n=self.m_candidates,
            temperature=self.temperature,
            max_tokens=64,
            parallel=True,
        )

        candidates: List[CandidateQuestion] = []
        seen: set = set()
        for resp in responses:
            g = extract_guess(resp.content)
            if g and len(set(g)) == 4 and g not in seen:
                seen.add(g)
                candidates.append(CandidateQuestion(suspect="", question=g))

        # Fallback
        if not candidates:
            candidates.append(CandidateQuestion(suspect="", question="1234"))

        return candidates

    def _parallel_simulate_gn(
        self,
        candidates: List[CandidateQuestion],
        history_str: str,
    ) -> List[DQSBranch]:
        """M parallel simulation calls (one per candidate)."""
        branches: List[DQSBranch] = []

        def _sim_one(c: CandidateQuestion) -> DQSBranch:
            prompt = _SIMULATE_ONE_GN.format(
                history=history_str,
                guess=c.question,
            )
            resp = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=128,
            )
            return DQSBranch(candidate=c, imagined_feedback=resp.content.strip())

        max_w = min(len(candidates), getattr(self.llm, "max_workers", 4))
        if max_w <= 1 or len(candidates) <= 1:
            branches = [_sim_one(c) for c in candidates]
        else:
            with ThreadPoolExecutor(max_workers=max_w) as pool:
                futs = {pool.submit(_sim_one, c): i for i, c in enumerate(candidates)}
                result_map: Dict[int, DQSBranch] = {}
                for f in as_completed(futs):
                    idx = futs[f]
                    try:
                        result_map[idx] = f.result()
                    except Exception:
                        result_map[idx] = DQSBranch(
                            candidate=candidates[idx],
                            imagined_feedback="(no response)",
                        )
                branches = [result_map[i] for i in range(len(candidates))]

        return branches

    def _orchestrate_gn(
        self,
        branches: List[DQSBranch],
        history_str: str,
    ) -> Tuple[CandidateQuestion, int, str]:
        """One orchestrator call — LLM picks the best guess."""
        branch_strs = []
        for i, b in enumerate(branches):
            branch_strs.append(
                f"Option {i+1}: Guess {b.candidate.question}\n"
                f"  Imagined feedback: {b.imagined_feedback}"
            )

        prompt = _ORCHESTRATE_GN.format(
            history=history_str,
            m=len(branches),
            branches="\n\n".join(branch_strs),
        )

        resp = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.orchestrator_temperature,
            max_tokens=32,
        )

        raw = resp.content.strip()
        sel_idx = self._parse_option_number(raw, len(branches))
        return branches[sel_idx].candidate, sel_idx, raw

    # ==================================================================
    # Shared helpers
    # ==================================================================

    @staticmethod
    def _match_suspect(name: str, suspect_names: List[str]) -> str:
        name_lower = name.strip().lower()
        for sn in suspect_names:
            if sn.lower() == name_lower:
                return sn
        for sn in suspect_names:
            if sn.lower() in name_lower or name_lower in sn.lower():
                return sn
        return suspect_names[0] if suspect_names else name

    @staticmethod
    def _parse_option_number(text: str, n_options: int) -> int:
        """Parse the orchestrator's choice (1-based) → 0-based index."""
        # Look for a bare number
        m = re.search(r"\b(\d+)\b", text)
        if m:
            num = int(m.group(1))
            if 1 <= num <= n_options:
                return num - 1
        # Fallback: first option
        return 0
