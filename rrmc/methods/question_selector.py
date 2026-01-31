"""
VoI (Value of Information) Question Selector for RRMC.

Selects the next question by generating M candidate questions,
simulating R outcomes for each, and picking the candidate that
maximizes expected robust MI reduction.

Includes DPP (Determinantal Point Process) slate selection for
choosing a diverse set of high-quality candidate questions.

Currently supports DC task only.
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from ..core.llm import LLMWrapper
from ..core.clustering import SemanticClusterer
from ..core.mi_estimator import RobustMI


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

CANDIDATE_GENERATION_DC = """You are a detective investigating a murder case.

Case Background:
{initial_info}

Interrogation History:
{history}

Suspects: {suspect_names}

Generate {m} distinct questions you could ask the suspects to help identify the murderer.
For each question, specify which suspect to ask and why this question is informative.

Output as a JSON list:
[
  {{"suspect": "<name>", "question": "<question text>"}},
  ...
]

Output ONLY the JSON list, nothing else."""

SIMULATE_RESPONSE_DC = """You will play the role of a suspect being interrogated.

Your Name: {suspect_name}
Case Background:
{initial_info}

Interrogation History:
{history}

A detective asks you: "{question}"

Provide a plausible one-sentence response as this suspect would answer.
Response:"""


@dataclass
class CandidateQuestion:
    """A candidate question with its VoI score."""
    suspect: str
    question: str
    expected_mi_reduction: float = 0.0
    simulated_mis: List[float] = field(default_factory=list)


@dataclass
class SlateInfo:
    """DPP slate selection metadata."""
    slate_indices: List[int] = field(default_factory=list)
    quality_scores: List[float] = field(default_factory=list)
    pairwise_similarities: Optional[List[List[float]]] = None


@dataclass
class QuestionSelectionResult:
    """Result of question selection."""
    selected: CandidateQuestion
    candidates: List[CandidateQuestion]
    current_mi: float
    slate: Optional[List[CandidateQuestion]] = None
    slate_info: Optional[SlateInfo] = None


class VoIQuestionSelector:
    """
    Value-of-Information question selector.

    1) Generate M candidate questions (via LLM).
    2) For each candidate, simulate R outcomes by role-playing suspect responses.
    3) Compute robust MI on each resulting next-state.
    4) Score = E[ current_mi - next_mi ].
    5) Pick candidate with highest expected MI reduction.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        robust_mi: RobustMI,
        m_candidates: int = 5,
        r_simulations: int = 2,
        k_samples: int = 4,
        variants: Optional[List[str]] = None,
        use_dpp: bool = False,
        dpp_slate_size: int = 3,
    ):
        """
        Args:
            llm: LLM wrapper for generation.
            clusterer: Semantic clusterer.
            robust_mi: RobustMI estimator (reused for scoring).
            m_candidates: Number of candidate questions to generate.
            r_simulations: Simulated outcomes per candidate.
            k_samples: MI samples for scoring next states.
            variants: Prompt variants for robust MI scoring.
            use_dpp: Whether to use DPP slate selection.
            dpp_slate_size: Number of diverse questions to select (m in DPP).
        """
        self.llm = llm
        self.clusterer = clusterer
        self.robust_mi = robust_mi
        self.m_candidates = m_candidates
        self.r_simulations = r_simulations
        self.k_samples = k_samples
        self.variants = variants or ["base", "skeptical"]
        self.use_dpp = use_dpp
        self.dpp_slate_size = dpp_slate_size
        self.dpp_selector = DPPSlateSelector(clusterer=clusterer) if use_dpp else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_question(
        self,
        task_type: str,
        state: Dict[str, Any],
        current_mi: float,
    ) -> QuestionSelectionResult:
        """
        Select the best question to ask.

        Args:
            task_type: Task type string (currently only "DC" supported).
            state: Current environment state.
            current_mi: Robust MI at the current state.

        Returns:
            QuestionSelectionResult with best question and all candidates.
        """
        if task_type != "DC":
            raise NotImplementedError(
                f"VoI question selection is only implemented for DC, got {task_type}"
            )

        # Step 1: generate candidates
        candidates = self._generate_candidates(state)

        # Step 2: score each candidate via simulated outcomes
        self._score_candidates(task_type, state, candidates, current_mi)

        # Step 3: pick best (optionally via DPP slate)
        candidates.sort(key=lambda c: c.expected_mi_reduction, reverse=True)

        slate = None
        slate_info = None
        if self.dpp_selector is not None and len(candidates) >= 2:
            questions = [c.question for c in candidates]
            quality = [c.expected_mi_reduction for c in candidates]
            slate_indices, sim_matrix = self.dpp_selector.select(
                questions, quality, m=self.dpp_slate_size
            )
            slate = [candidates[i] for i in slate_indices]
            slate_info = SlateInfo(
                slate_indices=slate_indices,
                quality_scores=[candidates[i].expected_mi_reduction for i in slate_indices],
                pairwise_similarities=sim_matrix.tolist() if sim_matrix is not None else None,
            )
            # Top-1 from the slate is the one asked
            selected = slate[0] if slate else candidates[0]
        else:
            selected = candidates[0] if candidates else CandidateQuestion(
                suspect=state.get("suspect_names", [""])[0],
                question="What can you tell me about the night of the crime?",
            )

        return QuestionSelectionResult(
            selected=selected,
            candidates=candidates,
            current_mi=current_mi,
            slate=slate,
            slate_info=slate_info,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _generate_candidates(
        self, state: Dict[str, Any]
    ) -> List[CandidateQuestion]:
        """Generate M candidate questions using the LLM."""
        suspect_names = state.get("suspect_names", [])
        prompt = CANDIDATE_GENERATION_DC.format(
            initial_info=state.get("initial_info", ""),
            history=state.get("history_string", ""),
            suspect_names=", ".join(suspect_names),
            m=self.m_candidates,
        )

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
        )

        candidates = self._parse_candidates(response.content, suspect_names)

        # Deduplicate by normalised text
        seen = set()
        unique = []
        for c in candidates:
            key = c.question.strip().lower()
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique[:self.m_candidates]

    def _parse_candidates(
        self, text: str, suspect_names: List[str]
    ) -> List[CandidateQuestion]:
        """Parse candidate JSON from LLM output."""
        candidates: List[CandidateQuestion] = []
        try:
            # Find JSON array in response
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                items = json.loads(match.group())
                for item in items:
                    suspect = item.get("suspect", "")
                    question = item.get("question", "")
                    # Normalise suspect name
                    suspect = self._match_suspect(suspect, suspect_names)
                    if question:
                        candidates.append(
                            CandidateQuestion(suspect=suspect, question=question)
                        )
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: if parsing failed, generate a simple question per suspect
        if not candidates and suspect_names:
            for name in suspect_names[:self.m_candidates]:
                candidates.append(CandidateQuestion(
                    suspect=name,
                    question=f"Where were you at the time of the crime, {name}?",
                ))

        return candidates

    def _match_suspect(self, name: str, suspect_names: List[str]) -> str:
        """Fuzzy-match a suspect name."""
        name_lower = name.strip().lower()
        for sn in suspect_names:
            if sn.lower() == name_lower or sn.lower() in name_lower or name_lower in sn.lower():
                return sn
        return suspect_names[0] if suspect_names else name

    def _score_candidates(
        self,
        task_type: str,
        state: Dict[str, Any],
        candidates: List[CandidateQuestion],
        current_mi: float,
    ) -> None:
        """Score each candidate by expected MI reduction."""

        def score_one(candidate: CandidateQuestion) -> None:
            sim_mis = []
            for _ in range(self.r_simulations):
                # Simulate a suspect response
                simulated_response = self._simulate_response(state, candidate)

                # Build next-state by appending to history
                next_state = self._build_next_state(state, candidate, simulated_response)

                # Compute robust MI on next state
                try:
                    next_mi, _ = self.robust_mi.estimate(
                        task_type=task_type,
                        state=next_state,
                        variants=self.variants,
                    )
                except Exception:
                    next_mi = current_mi  # If estimation fails, assume no reduction
                sim_mis.append(next_mi)

            candidate.simulated_mis = sim_mis
            candidate.expected_mi_reduction = current_mi - float(np.mean(sim_mis))

        # Run scoring in parallel across candidates
        max_workers = min(len(candidates), getattr(self.llm, 'max_workers', 4))
        if max_workers <= 1 or len(candidates) <= 1:
            for c in candidates:
                score_one(c)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(score_one, c): c for c in candidates}
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        c = futures[future]
                        c.expected_mi_reduction = 0.0

    def _simulate_response(
        self, state: Dict[str, Any], candidate: CandidateQuestion
    ) -> str:
        """Simulate a suspect's response to a candidate question."""
        prompt = SIMULATE_RESPONSE_DC.format(
            suspect_name=candidate.suspect,
            initial_info=state.get("initial_info", ""),
            history=state.get("history_string", ""),
            question=candidate.question,
        )

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=128,
        )
        return response.content.strip()

    def _build_next_state(
        self,
        state: Dict[str, Any],
        candidate: CandidateQuestion,
        simulated_response: str,
    ) -> Dict[str, Any]:
        """Build a next-state dict by appending the simulated Q&A to history."""
        next_state = dict(state)
        history = list(state.get("history", []))
        turn = state.get("turn", 0) + 1

        history.append({
            "turn": turn,
            "suspect": candidate.suspect,
            "question": candidate.question,
            "feedback": simulated_response,
        })

        # Rebuild history string
        lines = []
        for h in history:
            if "suspect" in h:
                lines.append(
                    f"Turn {h['turn']}: Question for {h['suspect']}: {h['question']} "
                    f"Feedback: {h['feedback']}"
                )
            else:
                lines.append(f"Q: {h.get('question', '')} A: {h.get('feedback', '')}")

        next_state["history"] = history
        next_state["history_string"] = "\n".join(lines)
        next_state["turn"] = turn
        return next_state


# ---------------------------------------------------------------------------
# DPP Slate Selector
# ---------------------------------------------------------------------------

class DPPSlateSelector:
    """
    Greedy DPP (Determinantal Point Process) slate selection.

    Selects a diverse subset of m questions from M candidates by
    approximately maximising det(L_S), where:
        L = diag(q) @ K @ diag(q)
        q_i = quality score of candidate i
        K_ij = cosine similarity between question embeddings i and j

    Uses greedy marginal-gain selection (O(m*M) per call).
    """

    def __init__(
        self,
        clusterer: Optional[SemanticClusterer] = None,
    ):
        """
        Args:
            clusterer: Semantic clusterer (used for embeddings). If the
                       clusterer has a loaded sentence-transformer model,
                       cosine similarity is embedding-based; otherwise
                       falls back to normalised n-gram overlap.
        """
        self.clusterer = clusterer

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def select(
        self,
        questions: List[str],
        quality_scores: List[float],
        m: int = 3,
    ) -> Tuple[List[int], Optional[np.ndarray]]:
        """
        Select a diverse slate of m questions from M candidates.

        Args:
            questions: List of M candidate question strings.
            quality_scores: Quality score per candidate (higher = better).
            m: Slate size.

        Returns:
            Tuple of (selected_indices, pairwise_similarity_matrix).
        """
        M = len(questions)
        if M <= m:
            return list(range(M)), None

        # Build similarity kernel K (M x M)
        K = self._similarity_kernel(questions)

        # Build L-ensemble kernel: L = diag(q) @ K @ diag(q)
        q = np.array(quality_scores, dtype=float)
        # Shift quality scores to be non-negative (DPP requires q >= 0)
        q = q - q.min() + 1e-6
        Q = np.diag(q)
        L = Q @ K @ Q

        # Greedy selection
        selected = self._greedy_select(L, m)

        return selected, K

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _similarity_kernel(self, questions: List[str]) -> np.ndarray:
        """Compute cosine similarity kernel between questions."""
        embeddings = self._embed(questions)
        # Normalise rows
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normed = embeddings / norms
        K = normed @ normed.T
        # Clamp to [0, 1] for numerical safety
        np.clip(K, 0.0, 1.0, out=K)
        return K

    def _embed(self, texts: List[str]) -> np.ndarray:
        """
        Embed texts. Uses sentence-transformers if the clusterer has a
        model loaded; otherwise falls back to TF-IDF-style n-gram vectors.
        """
        # Try sentence-transformer embeddings via the clusterer
        if self.clusterer is not None:
            model = getattr(self.clusterer, "model", None)
            if model is not None:
                try:
                    return model.encode(texts, show_progress_bar=False)
                except Exception:
                    pass

        # Fallback: character 3-gram TF vectors
        return self._ngram_vectors(texts, n=3)

    @staticmethod
    def _ngram_vectors(texts: List[str], n: int = 3) -> np.ndarray:
        """Simple character n-gram frequency vectors."""
        from collections import Counter as _Counter

        # Build vocabulary
        vocab: Dict[str, int] = {}
        counters = []
        for text in texts:
            text_lower = text.lower()
            grams = [text_lower[i:i + n] for i in range(len(text_lower) - n + 1)]
            c = _Counter(grams)
            counters.append(c)
            for g in c:
                if g not in vocab:
                    vocab[g] = len(vocab)

        # Build matrix
        mat = np.zeros((len(texts), len(vocab)), dtype=float)
        for i, c in enumerate(counters):
            for g, cnt in c.items():
                mat[i, vocab[g]] = cnt

        return mat

    @staticmethod
    def _greedy_select(L: np.ndarray, m: int) -> List[int]:
        """
        Greedy DPP selection: iteratively pick the item that
        maximises the marginal gain in log det(L_S).
        """
        M = L.shape[0]
        selected: List[int] = []
        remaining = set(range(M))

        for _ in range(min(m, M)):
            best_idx = -1
            best_gain = -float("inf")

            for j in remaining:
                trial = selected + [j]
                L_sub = L[np.ix_(trial, trial)]
                sign, logdet = np.linalg.slogdet(L_sub)
                gain = logdet if sign > 0 else -float("inf")
                if gain > best_gain:
                    best_gain = gain
                    best_idx = j

            if best_idx < 0:
                break
            selected.append(best_idx)
            remaining.discard(best_idx)

        return selected
