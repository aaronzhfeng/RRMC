"""
RRMC Evaluation Harness.

Runs experiments comparing stopping rules on AR-Bench tasks.
"""

import json
import os
import time
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Any, Type
from dataclasses import dataclass, field, asdict
from datetime import datetime
import numpy as np
from scipy import stats as scipy_stats
from tqdm import tqdm

from ..core.llm import LLMWrapper
from ..core.environment import ARBenchEnv, TaskType, ActionASK, ActionANSWER
from ..core.clustering import SemanticClusterer
from ..core.mi_estimator import RobustMI
from ..core.calibration import RiskControlledCalibrator, CalibrationResult
from ..core.mi_estimator import compute_homogeneity_score
from .metrics import (
    generate_metrics_report,
    bootstrap_accuracy_ci,
    bootstrap_turns_ci,
    MetricsReport,
)
from ..methods import get_method
from ..methods.stopping_rules import (
    BaseStoppingRule,
    FixedTurnsStopping,
    SelfConsistencyStopping,
    VerbalizedConfidenceStopping,
    SemanticEntropyStopping,
    MIOnlyStopping,
    RobustMIStopping,
    KnowNoStopping,
    CIPLiteStopping,
    UoTLiteStopping,
    AllSuspectsWrapper,
)
from ..methods.question_selector import VoIQuestionSelector, QuestionSelectionResult
from ..methods.trajectory_ensemble import TrajectoryEnsemble, TrajectoryResult, EnsembleResult


# =============================================================================
# AR-Bench Prompts (ported from AR-Bench/arbench/reasoner/dc/prompt.py)
# =============================================================================

DC_PROPOSE_TEMPLATE = """You will take on the role of a detective tasked with finding the real murderer in this case. Your goal is to solve the mystery by questioning the suspects. You will take turns asking these questions and using the answers to gather evidence and piece together the truth. The game will conduct in {turn} turns, in each turn you can only propose one question
The case background is:
{background}
"""

DC_SELECT_SUSPECT_TEMPLATE = "Turn {turn}: Now choose a suspect to interrogate. {suspect_names}. Output ONLY the suspect's full name, nothing else. No explanations, no reasoning, just the name."

DC_REFINE_SELECT_SUSPECT = "Invalid response. Output ONLY the suspect's full name. No other text."

DC_QUESTION_PROPOSE_PROMPT = """Turn {turn}: Now give your question to the suspect. Output ONLY your question, nothing else. No reasoning, no explanation, just the question.
Example format: "Where were you at 8pm last night?"
Your question:"""

DC_SELECT_MURDERER_TEMPLATE = """Now, based on your obtained information, you should tell me who is more likely to be the true murderer, {choice}
You should only output the index of the candidate suspect like: A , B, C, D or E.
you should strictly follow this answer format:
Reason: [Your inference step by step]
Answer: [A, B, C, D or E]
"""

# SP Prompts (from AR-Bench/arbench/reasoner/sp/prompt.py)
SP_PROPOSE_TEMPLATE = """The scenario is: {scenario}

Your task is to find out what truly happened by asking yes-or-no questions. You will take turns asking questions.

The game will conduct in {turn} turns. In each turn, output ONLY your question - no reasoning or explanation.
"""

SP_QUESTION_PROMPT = """Turn {turn}: Ask a yes/no question. Output ONLY the question itself, nothing else.
Your question:"""

# GN Prompts - Aligned with AR-Bench/arbench/reasoner/gn/prompt.py
# Using propose_template_with_1_shot for better format compliance
GN_PROPOSE_TEMPLATE = """Let's play a game of guessing number 
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once. For example: 0123 or 9468. You will take turns guessing the number and using feedbacks to progressively reveal the true number. The game will conduct {turn} turns:
In the game when a guessing number is proposed, I will return the feedback of two information: 
1. How many digits are present in the answer and in the correct position 
2. How many digits are present in the answer but in the different position from the guessing number 
For example: 0 digits are present in the answer and in the correct positions, 2 digits are present in the answer but in the different positions 

Here is an example trajectory:
Turn 1: Guess: 1234, Feedback: 0 digits are present in the answer and in the correct positions, 0 digits are present in the answer but in the different positions 
Turn 2: Guess: 5678, Feedback: 0 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions 
Turn 3: Guess: 5690, Feedback: 1 digits are present in the answer and in the correct positions, 3 digits are present in the answer but in the different positions
Turn 4: Guess: 6850, Feedback: 4 digits are present in the answer and in the correct positions, 0 digits are present in the answer but in the different positions

Final Answer: 6850

Game start:
"""

# guess_prompt from AR-Bench
GN_GUESS_PROMPT = """Turn {turn}: Now give me the number you guess in this format, and do not give any other statement:
Guess: [number]
"""

# eval_prompt from AR-Bench (for feedback)
GN_EVAL_PROMPT = """{same_pos} digits are present in the answer and in the correct positions 
{diff_pos} digits are present in the answer but in the different positions 
"""

# final_guess_prompt from AR-Bench
GN_FINAL_GUESS_PROMPT = """
You have finished all the rounds of interaction, please give your final answer based on the guesses and feedback above:
Guess: [number]
"""

# refine_prompt from AR-Bench
GN_REFINE_PROMPT = """
Your output does not follow this format: Guess: [number]
please propose a guess number, for example: Guess: 1234
"""


class DCConversationManager:
    """
    Manages the multi-turn conversation structure for DC task.
    
    Uses AR-Bench's prompt structure with separate suspect selection
    and question generation steps.
    """
    
    def __init__(self, llm: LLMWrapper, max_turns: int = 25):
        self.llm = llm
        self.max_turns = max_turns
        self.conversation: List[Dict[str, str]] = []
        self.initialized = False
        
    def initialize(self, background: str, suspect_names: List[str]) -> None:
        """Initialize the conversation with system prompt."""
        self.suspect_names = suspect_names
        self.suspect_names_str = ", ".join(suspect_names)
        
        # System message with case background
        system_content = DC_PROPOSE_TEMPLATE.format(
            turn=self.max_turns,
            background=background
        )
        
        self.conversation = [{
            "role": "system",
            "content": system_content
        }]
        self.initialized = True
        
    def generate_question(self, turn: int) -> tuple:
        """
        Generate a question using AR-Bench's two-step process:
        1. Select suspect
        2. Generate question
        
        Returns:
            (question, suspect) tuple
        """
        if not self.initialized:
            raise ValueError("Conversation not initialized. Call initialize() first.")
        
        # Step 1: Select suspect
        select_prompt = DC_SELECT_SUSPECT_TEMPLATE.format(
            turn=turn,
            suspect_names=self.suspect_names_str
        )
        
        self.conversation.append({"role": "user", "content": select_prompt})
        
        response = self.llm.generate(
            messages=self.conversation.copy(),
            temperature=0.7,
            max_tokens=64,
        )
        
        suspect = self._parse_suspect(response.content)
        self.conversation.append({"role": "assistant", "content": suspect})
        
        # Retry if suspect parsing failed
        if suspect not in self.suspect_names:
            # Add refine prompt
            self.conversation.append({"role": "user", "content": DC_REFINE_SELECT_SUSPECT})
            response = self.llm.generate(
                messages=self.conversation.copy(),
                temperature=0.7,
                max_tokens=64,
            )
            suspect = self._parse_suspect(response.content)
            self.conversation.append({"role": "assistant", "content": suspect})
        
        # Step 2: Generate question
        question_prompt = DC_QUESTION_PROPOSE_PROMPT.format(turn=turn)
        self.conversation.append({"role": "user", "content": question_prompt})
        
        response = self.llm.generate(
            messages=self.conversation.copy(),
            temperature=0.7,
            max_tokens=256,
        )
        
        question = response.content.strip()
        self.conversation.append({"role": "assistant", "content": question})
        
        return question, suspect
    
    def add_feedback(self, feedback: str) -> None:
        """Add NPC feedback to conversation."""
        self.conversation.append({"role": "user", "content": feedback})
    
    def _parse_suspect(self, text: str) -> str:
        """Parse suspect name from model response."""
        text = text.strip()
        
        # Try exact match first
        for name in self.suspect_names:
            if name.lower() == text.lower():
                return name
        
        # Try partial match
        for name in self.suspect_names:
            if name.lower() in text.lower():
                return name
        
        # Fallback to first suspect
        return self.suspect_names[0] if self.suspect_names else ""
    
    def get_final_answer_prompt(self, choices: Dict[str, str]) -> List[Dict[str, str]]:
        """Get conversation with final answer prompt."""
        choice_str = ", ".join([f"{k}. {v}" for k, v in choices.items()])
        
        final_prompt = DC_SELECT_MURDERER_TEMPLATE.format(choice=choice_str)
        
        messages = self.conversation.copy()
        messages.append({"role": "user", "content": final_prompt})
        
        return messages


class SPConversationManager:
    """Manages multi-turn conversation for Situation Puzzles."""
    
    def __init__(self, llm: LLMWrapper, max_turns: int = 25):
        self.llm = llm
        self.max_turns = max_turns
        self.conversation: List[Dict[str, str]] = []
        self.initialized = False
        
    def initialize(self, scenario: str) -> None:
        """Initialize with the puzzle scenario."""
        system_content = SP_PROPOSE_TEMPLATE.format(
            scenario=scenario,
            turn=self.max_turns
        )
        
        self.conversation = [{
            "role": "system",
            "content": system_content
        }]
        self.initialized = True
        
    def generate_question(self, turn: int) -> str:
        """Generate a yes/no question."""
        if not self.initialized:
            raise ValueError("Conversation not initialized")
            
        question_prompt = SP_QUESTION_PROMPT.format(turn=turn)
        self.conversation.append({"role": "user", "content": question_prompt})
        
        response = self.llm.generate(
            messages=self.conversation.copy(),
            temperature=0.7,
            max_tokens=128,
        )
        
        question = response.content.strip()
        self.conversation.append({"role": "assistant", "content": question})
        
        return question
    
    def add_feedback(self, feedback: str) -> None:
        """Add NPC feedback."""
        self.conversation.append({"role": "user", "content": feedback})


class GNConversationManager:
    """Manages multi-turn conversation for Guessing Numbers.
    
    Aligned with AR-Bench's gn_evaluator.py _run_traditional_evaluation.
    """
    
    MAX_RETRIES = 3
    
    def __init__(self, llm: LLMWrapper, max_turns: int = 25):
        self.llm = llm
        self.max_turns = max_turns
        self.conversation: List[Dict[str, str]] = []
        self.initialized = False
        
    def initialize(self) -> None:
        """Initialize the game."""
        system_content = GN_PROPOSE_TEMPLATE.format(turn=self.max_turns)
        
        self.conversation = [{
            "role": "system",
            "content": system_content
        }]
        self.initialized = True
    
    def _extract_guess(self, response_text: str) -> Optional[str]:
        """Extract 4-digit guess from response, handling Qwen3's reasoning output.
        
        Looks for:
        1. "Guess: [number]" format (AR-Bench standard)
        2. Intent phrases like "I'll guess", "my guess is", "let's try"
        3. Last 4-digit number with unique digits as fallback
        """
        # Remove brackets
        text = response_text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
        text = text.replace("{", "").replace("}", "").replace("*", "")
        
        # 1. Look for "Guess: XXXX" pattern (case-insensitive)
        guess_pattern = r"guess:\s*(\d{4})"
        match = re.search(guess_pattern, text.lower())
        if match:
            candidate = match.group(1)
            if len(set(candidate)) == 4:
                return candidate
        
        # 2. Look for intent phrases indicating what the model wants to guess
        intent_patterns = [
            r"(?:i'?ll guess|my guess is|let'?s try|i'?ll try|going to guess|start with)\s*(\d{4})",
            r"(\d{4})\s*(?:as my guess|would be|seems)",
        ]
        for pattern in intent_patterns:
            match = re.search(pattern, text.lower())
            if match:
                candidate = match.group(1)
                if len(set(candidate)) == 4:
                    return candidate
        
        # 3. Fallback: find all 4-digit numbers with unique digits, prefer last one
        four_digit = re.findall(r"\d{4}", text)
        # Filter to those with unique digits
        valid_guesses = [n for n in four_digit if len(set(n)) == 4]
        if valid_guesses:
            return valid_guesses[-1]  # Take the last valid one
        
        # 4. If no valid guess found, take any 4-digit number
        if four_digit:
            return four_digit[-1]
        
        return None
        
    def generate_guess(self, turn: int) -> str:
        """Generate a 4-digit guess using AR-Bench's format."""
        if not self.initialized:
            raise ValueError("Conversation not initialized")
            
        guess_prompt = GN_GUESS_PROMPT.format(turn=turn)
        self.conversation.append({"role": "user", "content": guess_prompt})
        
        response = self.llm.generate(
            messages=self.conversation.copy(),
            temperature=0.7,
            max_tokens=64,  # Increased for "Guess: XXXX" format
        )
        
        # Store the full response
        self.conversation.append({"role": "assistant", "content": response.content})
        
        # Extract guess
        guess = self._extract_guess(response.content)
        
        # Retry if extraction failed
        retries = 0
        while guess is None and retries < self.MAX_RETRIES:
            retries += 1
            # Add refine prompt
            self.conversation.append({"role": "user", "content": GN_REFINE_PROMPT})
            
            response = self.llm.generate(
                messages=self.conversation.copy(),
                temperature=0.7,
                max_tokens=64,
            )
            self.conversation.append({"role": "assistant", "content": response.content})
            guess = self._extract_guess(response.content)
        
        # Validate uniqueness
        if guess and len(set(guess)) != 4:
            # Invalid guess - generate fallback
            guess = None
        
        # Fallback: generate valid random guess
        if not guess:
            import random
            digits = list("0123456789")
            random.shuffle(digits)
            guess = "".join(digits[:4])
        
        return guess
    
    def add_feedback(self, feedback: str) -> None:
        """Add game feedback (AR-Bench's eval_prompt format)."""
        self.conversation.append({"role": "user", "content": feedback})
    
    def generate_final_answer(self) -> str:
        """Generate final answer using AR-Bench's final_guess_prompt.
        
        This continues the existing conversation, matching AR-Bench's flow.
        """
        if not self.initialized:
            raise ValueError("Conversation not initialized")
        
        # Add final guess prompt (AR-Bench's format)
        self.conversation.append({"role": "user", "content": GN_FINAL_GUESS_PROMPT})
        
        response = self.llm.generate(
            messages=self.conversation.copy(),
            temperature=0.3,  # Lower temp for final answer
            max_tokens=64,
        )
        
        self.conversation.append({"role": "assistant", "content": response.content})
        
        # Extract guess
        guess = self._extract_guess(response.content)
        
        # Validate and fallback
        if guess and len(set(guess)) == 4:
            return guess
        
        # Fallback: generate valid random guess
        import random
        digits = list("0123456789")
        random.shuffle(digits)
        return "".join(digits[:4])


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    puzzle_idx: int
    method: str
    correct: bool
    prediction: Any
    ground_truth: Any
    turns_used: int
    mi_scores: List[float] = field(default_factory=list)
    decisions: List[Dict] = field(default_factory=list)
    total_time: float = 0.0
    raw_answer: Optional[str] = None
    raw_samples: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Aggregated evaluation results."""
    method: str
    task_type: str
    n_episodes: int
    accuracy: float
    avg_turns: float
    std_turns: float
    avg_mi: float
    episode_results: List[EpisodeResult] = field(default_factory=list)


@dataclass
class CalibrationData:
    """Data collected during calibration phase."""
    task_type: str
    n_states: int
    n_puzzles: int
    calibration_result: Optional[CalibrationResult] = None
    threshold: float = 0.3  # Default fallback


class RRMCEvaluator:
    """
    Evaluation harness for RRMC experiments.

    Runs episodes with different stopping rules and compares performance.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        data_path: str,
        task_type: TaskType = TaskType.DC,
        max_turns: int = 25,
        output_dir: str = "./results",
        regime: str = "normal",
        max_workers: int = 5,
        question_selector: Optional[VoIQuestionSelector] = None,
    ):
        """
        Initialize evaluator.

        Args:
            llm: LLM wrapper
            data_path: Path to AR-Bench dataset
            task_type: Task type
            max_turns: Maximum turns per episode
            output_dir: Directory for saving results
            regime: Decoding regime ("normal" or "homogeneous")
            max_workers: Max concurrent puzzle evaluations (default: 5)
            question_selector: Optional VoI question selector for DC
        """
        self.llm = llm
        self.data_path = data_path
        self.task_type = task_type
        self.max_turns = max_turns
        self.output_dir = output_dir
        self.regime = regime
        self.max_workers = max_workers
        self.question_selector = question_selector

        # Initialize environment (for sequential runs and metadata)
        self.env = ARBenchEnv(
            task_type=task_type,
            data_path=data_path,
            llm=llm,
            max_turns=max_turns,
        )

        # Initialize clusterer (without sentence-transformers dependency for now)
        self.clusterer = SemanticClusterer(use_entailment=False)

        os.makedirs(output_dir, exist_ok=True)

    def _clone_stopping_rule(self, stopping_rule: BaseStoppingRule) -> BaseStoppingRule:
        """Create a thread-local stopping rule instance."""
        if hasattr(stopping_rule, "clone") and callable(getattr(stopping_rule, "clone")):
            return stopping_rule.clone()

        # Handle AllSuspectsWrapper by cloning inner rule and re-wrapping
        if isinstance(stopping_rule, AllSuspectsWrapper):
            cloned_inner = self._clone_stopping_rule(stopping_rule.inner_rule)
            return AllSuspectsWrapper(
                inner_rule=cloned_inner,
                enabled=stopping_rule.enabled,
            )

        if isinstance(stopping_rule, FixedTurnsStopping):
            return FixedTurnsStopping(
                llm=stopping_rule.llm,
                fixed_turns=stopping_rule.fixed_turns,
                max_turns=stopping_rule.max_turns,
            )
        if isinstance(stopping_rule, SelfConsistencyStopping):
            return SelfConsistencyStopping(
                llm=stopping_rule.llm,
                clusterer=stopping_rule.clusterer,
                k_samples=stopping_rule.k_samples,
                consistency_threshold=stopping_rule.consistency_threshold,
                max_turns=stopping_rule.max_turns,
                temperature=stopping_rule.temperature,
            )
        if isinstance(stopping_rule, SemanticEntropyStopping):
            return SemanticEntropyStopping(
                llm=stopping_rule.llm,
                clusterer=stopping_rule.clusterer,
                k_samples=stopping_rule.k_samples,
                entropy_threshold=stopping_rule.entropy_threshold,
                max_turns=stopping_rule.max_turns,
                temperature=stopping_rule.temperature,
            )
        if isinstance(stopping_rule, VerbalizedConfidenceStopping):
            return VerbalizedConfidenceStopping(
                llm=stopping_rule.llm,
                confidence_threshold=stopping_rule.confidence_threshold,
                max_turns=stopping_rule.max_turns,
            )
        if isinstance(stopping_rule, MIOnlyStopping):
            mi_estimator = stopping_rule.mi_estimator
            return MIOnlyStopping(
                llm=mi_estimator.llm,
                clusterer=mi_estimator.clusterer,
                mi_threshold=stopping_rule.mi_threshold,
                k_samples=mi_estimator.k_samples,
                max_turns=stopping_rule.max_turns,
                temperature=mi_estimator.temperature,
            )
        if isinstance(stopping_rule, RobustMIStopping):
            mi_estimator = stopping_rule.robust_mi.mi_estimator
            return RobustMIStopping(
                llm=mi_estimator.llm,
                clusterer=mi_estimator.clusterer,
                threshold=stopping_rule.threshold,
                k_samples=mi_estimator.k_samples,
                max_turns=stopping_rule.max_turns,
                temperature=mi_estimator.temperature,
                variants=stopping_rule.variants,
                use_diversity_sampling=stopping_rule.robust_mi.use_diversity_sampling,
                regime=stopping_rule.regime,
            )

        return stopping_rule

    def run_episode(
        self,
        puzzle_idx: int,
        stopping_rule: BaseStoppingRule,
        question_generator: Optional[Any] = None,
    ) -> EpisodeResult:
        """
        Run a single episode.

        Args:
            puzzle_idx: Index of puzzle to run
            stopping_rule: Stopping rule to use
            question_generator: Question generator (uses AR-Bench prompts if None)

        Returns:
            EpisodeResult
        """
        start_time = time.time()

        # Reset environment
        obs = self.env.reset(puzzle_idx)

        # Map task type to correct string
        if self.task_type == TaskType.DC:
            task_type_str = "DC"
        elif self.task_type == TaskType.SP:
            task_type_str = "SP"
        elif self.task_type == TaskType.GN:
            task_type_str = "GN"
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # Create conversation manager for this episode
        conv_manager = None
        if question_generator is None:
            if self.task_type == TaskType.DC:
                conv_manager = DCConversationManager(self.llm, self.max_turns)
                conv_manager.initialize(
                    background=obs.get("initial_info", ""),
                    suspect_names=obs.get("suspect_names", [])
                )
            elif self.task_type == TaskType.SP:
                conv_manager = SPConversationManager(self.llm, self.max_turns)
                conv_manager.initialize(scenario=obs.get("surface", ""))
            elif self.task_type == TaskType.GN:
                conv_manager = GNConversationManager(self.llm, self.max_turns)
                conv_manager.initialize()

        mi_scores = []
        decisions = []
        turn = 0
        final_raw_answer = None
        final_raw_samples: List[str] = []

        while not self.env.done and turn < self.max_turns:
            turn += 1

            # Get current state
            state = self.env.get_state()

            # Make stopping decision
            decision = stopping_rule.should_stop(task_type_str, state, turn)
            mi_scores.append(float(decision.score))
            decisions.append({
                "turn": turn,
                "score": float(decision.score),
                "should_stop": bool(decision.should_stop),
                "reason": decision.reason,
            })

            if decision.should_stop:
                # Submit answer
                answer = decision.prediction or stopping_rule.get_best_answer(task_type_str, state)
                final_raw_answer = getattr(stopping_rule, "_last_raw_answer", None)
                final_raw_samples = getattr(stopping_rule, "_last_raw_samples", [])
                result = self.env.step(ActionANSWER(answer=answer))
                break
            else:
                # Generate and ask question
                if question_generator:
                    generated = question_generator(state)
                    if isinstance(generated, tuple):
                        question = generated[0] if len(generated) > 0 else ""
                        suspect = generated[1] if len(generated) > 1 else None
                    else:
                        question, suspect = generated, None
                    if not question:
                        question, suspect = self._generate_question(state, conv_manager, turn)
                    if self.task_type == TaskType.DC:
                        result = self.env.step(ActionASK(question=question, suspect=suspect))
                    else:
                        result = self.env.step(ActionASK(question=question))
                elif self.question_selector is not None and self.task_type in (
                    TaskType.DC, TaskType.GN
                ):
                    # DQS / VoI question selector
                    selection = self.question_selector.select_question(
                        task_type=task_type_str,
                        state=state,
                        current_mi=decision.score,
                    )
                    question = selection.selected.question
                    suspect = selection.selected.suspect or None
                    if self.task_type == TaskType.DC:
                        result = self.env.step(ActionASK(question=question, suspect=suspect))
                    else:
                        result = self.env.step(ActionASK(question=question))
                else:
                    # Use conversation manager for proper multi-turn conversation
                    if self.task_type == TaskType.DC:
                        question, suspect = conv_manager.generate_question(turn)
                        result = self.env.step(ActionASK(question=question, suspect=suspect))
                    elif self.task_type == TaskType.SP:
                        question = conv_manager.generate_question(turn)
                        result = self.env.step(ActionASK(question=question))
                    else:  # GN
                        question = conv_manager.generate_guess(turn)
                        result = self.env.step(ActionASK(question=question))
                
                # Add NPC feedback to conversation manager
                if conv_manager is not None:
                    conv_manager.add_feedback(result.observation)

        # If we exhausted turns without stopping, force answer
        if not self.env.done:
            state = self.env.get_state()
            answer = stopping_rule.get_best_answer(task_type_str, state)
            final_raw_answer = getattr(stopping_rule, "_last_raw_answer", None)
            final_raw_samples = getattr(stopping_rule, "_last_raw_samples", [])
            result = self.env.step(ActionANSWER(answer=answer))

        total_time = time.time() - start_time

        # Extract result info
        info = result.info if hasattr(result, 'info') else {}

        return EpisodeResult(
            puzzle_idx=puzzle_idx,
            method=type(stopping_rule).__name__,
            correct=info.get("correct", False),
            prediction=info.get("prediction", ""),
            ground_truth=info.get("ground_truth", ""),
            turns_used=turn,
            mi_scores=mi_scores,
            decisions=decisions,
            total_time=total_time,
            raw_answer=final_raw_answer,
            raw_samples=final_raw_samples,
            history=self.env.get_history(),
        )

    def _generate_question(self, state: Dict[str, Any], conv_manager=None, turn: int = 1) -> tuple:
        """
        Generate next question using AR-Bench's multi-turn conversation structure.
        
        Args:
            state: Current environment state
            conv_manager: Conversation manager (created if None)
            turn: Current turn number
            
        Returns:
            (question, suspect) tuple
        """
        if self.task_type == TaskType.DC:
            # Use conversation manager for proper multi-turn structure
            if conv_manager is None:
                conv_manager = DCConversationManager(self.llm, self.max_turns)
                conv_manager.initialize(
                    background=state.get("initial_info", ""),
                    suspect_names=state.get("suspect_names", [])
                )
                # Replay history into conversation
                for h in state.get("history", []):
                    # Each history item was a turn - add the feedback
                    conv_manager.add_feedback(h.get("feedback", ""))
            
            question, suspect = conv_manager.generate_question(turn)
            return question, suspect

        elif self.task_type == TaskType.SP:
            if conv_manager is None:
                conv_manager = SPConversationManager(self.llm, self.max_turns)
                conv_manager.initialize(scenario=state.get("surface", ""))
                # Replay history
                for h in state.get("history", []):
                    conv_manager.add_feedback(h.get("feedback", ""))
            
            question = conv_manager.generate_question(turn)
            return question, None

        else:  # GN
            if conv_manager is None:
                conv_manager = GNConversationManager(self.llm, self.max_turns)
                conv_manager.initialize()
                # Replay history using AR-Bench's eval_prompt format
                for h in state.get("history", []):
                    same_pos = h.get('bulls', 0)
                    diff_pos = h.get('cows', 0)
                    feedback = GN_EVAL_PROMPT.format(same_pos=same_pos, diff_pos=diff_pos)
                    conv_manager.add_feedback(feedback)
            
            guess = conv_manager.generate_guess(turn)
            return guess, None

    def evaluate_method(
        self,
        stopping_rule: BaseStoppingRule,
        puzzle_indices: Optional[List[int]] = None,
        verbose: bool = True,
        parallel: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a stopping rule on multiple puzzles.

        Args:
            stopping_rule: Stopping rule to evaluate
            puzzle_indices: Which puzzles to run (default: all)
            verbose: Whether to print progress
            parallel: Whether to run puzzles in parallel (default: True)

        Returns:
            EvaluationResult with aggregated metrics
        """
        if puzzle_indices is None:
            puzzle_indices = list(range(len(self.env)))

        method_name = type(stopping_rule).__name__
        if verbose:
            print(f"\nEvaluating {method_name} on {len(puzzle_indices)} puzzles...")

        if not parallel or len(puzzle_indices) <= 1 or self.max_workers <= 1:
            # Sequential execution
            return self._evaluate_method_sequential(stopping_rule, puzzle_indices, verbose)
        else:
            # Parallel execution
            return self._evaluate_method_parallel(stopping_rule, puzzle_indices, verbose)

    def _evaluate_method_sequential(
        self,
        stopping_rule: BaseStoppingRule,
        puzzle_indices: List[int],
        verbose: bool,
    ) -> EvaluationResult:
        """Sequential puzzle evaluation (original behavior)."""
        method_name = type(stopping_rule).__name__
        results = []
        correct_count = 0
        total_turns = 0

        puzzle_iter = tqdm(
            puzzle_indices,
            desc=f"  {method_name[:20]}",
            leave=False,
            disable=not verbose,
        )
        for idx in puzzle_iter:
            result = self.run_episode(idx, stopping_rule)
            results.append(result)

            if result.correct:
                correct_count += 1
            total_turns += result.turns_used
            
            # Update progress bar postfix with running stats
            acc = correct_count / len(results)
            puzzle_iter.set_postfix(acc=f"{acc:.0%}", turns=f"{total_turns/len(results):.1f}")

        return self._build_evaluation_result(results, puzzle_indices, verbose)

    def _evaluate_method_parallel(
        self,
        stopping_rule: BaseStoppingRule,
        puzzle_indices: List[int],
        verbose: bool,
    ) -> EvaluationResult:
        """Parallel puzzle evaluation using ThreadPoolExecutor."""
        method_name = type(stopping_rule).__name__
        
        # Thread-safe counters
        results_lock = threading.Lock()
        results_dict = {}
        correct_count = [0]
        total_turns = [0]
        completed = [0]
        
        # Progress bar for puzzles
        pbar = tqdm(
            total=len(puzzle_indices),
            desc=f"  {method_name[:20]}",
            leave=False,
            disable=not verbose,
        )

        def run_single_puzzle(idx: int) -> EpisodeResult:
            """Run a single puzzle with its own environment (thread-safe)."""
            # Create fresh environment for this thread
            thread_env = ARBenchEnv(
                task_type=self.task_type,
                data_path=self.data_path,
                llm=self.llm,
                max_turns=self.max_turns,
            )

            # Clone stopping rule to avoid shared mutable state across threads
            local_rule = self._clone_stopping_rule(stopping_rule)

            # Run episode with the thread-local environment
            result = self._run_episode_with_env(idx, local_rule, thread_env)
            
            # Update counters (thread-safe)
            with results_lock:
                results_dict[idx] = result
                completed[0] += 1
                if result.correct:
                    correct_count[0] += 1
                total_turns[0] += result.turns_used
                
                # Update progress bar
                pbar.update(1)
                acc = correct_count[0] / completed[0] if completed[0] > 0 else 0
                avg_t = total_turns[0] / completed[0] if completed[0] > 0 else 0
                pbar.set_postfix(acc=f"{acc:.0%}", turns=f"{avg_t:.1f}")
            
            return result

        # Run puzzles in parallel
        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(run_single_puzzle, idx): idx for idx in puzzle_indices}
                
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        future.result()  # Raises exception if any
                    except Exception as e:
                        print(f"\nError in puzzle {idx}: {e}")
                        with results_lock:
                            results_dict[idx] = EpisodeResult(
                                puzzle_idx=idx,
                                method=method_name,
                                correct=False,
                                prediction="",
                                ground_truth="",
                                turns_used=0,
                                mi_scores=[],
                                decisions=[],
                                total_time=0.0,
                            )
        finally:
            pbar.close()

        # Sort results by puzzle index
        results = [results_dict[idx] for idx in sorted(results_dict.keys())]
        return self._build_evaluation_result(results, puzzle_indices, verbose)

    def _run_episode_with_env(
        self,
        puzzle_idx: int,
        stopping_rule: BaseStoppingRule,
        env: ARBenchEnv,
    ) -> EpisodeResult:
        """Run a single episode with a specific environment (thread-safe)."""
        start_time = time.time()

        # Reset environment
        obs = env.reset(puzzle_idx)

        # Map task type to correct string
        if self.task_type == TaskType.DC:
            task_type_str = "DC"
        elif self.task_type == TaskType.SP:
            task_type_str = "SP"
        elif self.task_type == TaskType.GN:
            task_type_str = "GN"
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        # Create conversation manager for this episode (maintains multi-turn context)
        conv_manager = None
        if self.task_type == TaskType.DC:
            conv_manager = DCConversationManager(self.llm, self.max_turns)
            conv_manager.initialize(
                background=obs.get("initial_info", ""),
                suspect_names=obs.get("suspect_names", [])
            )
        elif self.task_type == TaskType.SP:
            conv_manager = SPConversationManager(self.llm, self.max_turns)
            conv_manager.initialize(scenario=obs.get("surface", ""))
        elif self.task_type == TaskType.GN:
            conv_manager = GNConversationManager(self.llm, self.max_turns)
            conv_manager.initialize()

        mi_scores = []
        decisions = []
        turn = 0
        final_raw_answer = None
        final_raw_samples: List[str] = []

        while not env.done and turn < self.max_turns:
            turn += 1

            # Get current state
            state = env.get_state()

            # Make stopping decision
            decision = stopping_rule.should_stop(task_type_str, state, turn)
            mi_scores.append(float(decision.score))
            decisions.append({
                "turn": turn,
                "score": float(decision.score),
                "should_stop": bool(decision.should_stop),
                "reason": decision.reason,
            })

            if decision.should_stop:
                # Submit answer
                answer = decision.prediction or stopping_rule.get_best_answer(task_type_str, state)
                final_raw_answer = getattr(stopping_rule, "_last_raw_answer", None)
                final_raw_samples = getattr(stopping_rule, "_last_raw_samples", [])
                result = env.step(ActionANSWER(answer=answer))
                break
            else:
                # Generate and ask question
                if self.task_type == TaskType.DC and self.question_selector is not None:
                    # Use DQS / VoI question selector (DC)
                    current_mi = decision.score
                    selection = self.question_selector.select_question(
                        task_type=task_type_str,
                        state=state,
                        current_mi=current_mi,
                    )
                    question = selection.selected.question
                    suspect = selection.selected.suspect
                    result = env.step(ActionASK(question=question, suspect=suspect))
                    conv_manager.add_feedback(result.observation)
                elif self.task_type == TaskType.GN and self.question_selector is not None:
                    # Use DQS question selector (GN)
                    selection = self.question_selector.select_question(
                        task_type=task_type_str,
                        state=state,
                        current_mi=decision.score,
                    )
                    guess = selection.selected.question
                    result = env.step(ActionASK(question=guess))
                    conv_manager.add_feedback(result.observation)
                elif self.task_type == TaskType.DC:
                    question, suspect = conv_manager.generate_question(turn)
                    result = env.step(ActionASK(question=question, suspect=suspect))
                    conv_manager.add_feedback(result.observation)
                elif self.task_type == TaskType.SP:
                    question = conv_manager.generate_question(turn)
                    result = env.step(ActionASK(question=question))
                    conv_manager.add_feedback(result.observation)
                else:  # GN
                    guess = conv_manager.generate_guess(turn)
                    result = env.step(ActionASK(question=guess))
                    conv_manager.add_feedback(result.observation)

        # If we exhausted turns without stopping, force answer
        if not env.done:
            state = env.get_state()
            answer = stopping_rule.get_best_answer(task_type_str, state)
            final_raw_answer = getattr(stopping_rule, "_last_raw_answer", None)
            final_raw_samples = getattr(stopping_rule, "_last_raw_samples", [])
            result = env.step(ActionANSWER(answer=answer))

        total_time = time.time() - start_time

        # Extract result info
        info = result.info if hasattr(result, 'info') else {}

        return EpisodeResult(
            puzzle_idx=puzzle_idx,
            method=type(stopping_rule).__name__,
            correct=info.get("correct", False),
            prediction=info.get("prediction", ""),
            ground_truth=info.get("ground_truth", ""),
            turns_used=turn,
            mi_scores=mi_scores,
            decisions=decisions,
            total_time=total_time,
            raw_answer=final_raw_answer,
            raw_samples=final_raw_samples,
            history=env.get_history(),
        )

    def _generate_question_with_env(self, state: Dict[str, Any], env: ARBenchEnv, conv_manager=None, turn: int = 1) -> tuple:
        """
        Generate next question using AR-Bench's multi-turn conversation structure.
        Thread-safe version that works with parallel evaluation.
        
        Args:
            state: Current environment state
            env: The environment instance (for thread-safety)
            conv_manager: Conversation manager (created if None)
            turn: Current turn number
            
        Returns:
            (question, suspect) tuple
        """
        # Delegate to _generate_question which now uses conversation managers
        return self._generate_question(state, conv_manager, turn)

    def _build_evaluation_result(
        self,
        results: List[EpisodeResult],
        puzzle_indices: List[int],
        verbose: bool,
    ) -> EvaluationResult:
        """Build EvaluationResult from episode results."""
        if not results:
            return EvaluationResult(
                method="Unknown",
                task_type=self.task_type.value,
                n_episodes=0,
                accuracy=0.0,
                avg_turns=0.0,
                std_turns=0.0,
                avg_mi=0.0,
                episode_results=[],
            )

        method_name = results[0].method
        correct_count = sum(1 for r in results if r.correct)
        total_turns = sum(r.turns_used for r in results)
        
        accuracy = correct_count / len(puzzle_indices) if puzzle_indices else 0.0
        avg_turns = total_turns / len(puzzle_indices) if puzzle_indices else 0.0
        std_turns = np.std([r.turns_used for r in results]) if results else 0.0
        avg_mi = np.mean([np.mean(r.mi_scores) if r.mi_scores else 0.0 for r in results])

        # Compute bootstrap CIs
        correct_flags = [r.correct for r in results]
        turns_list = [r.turns_used for r in results]
        acc_ci = bootstrap_accuracy_ci(correct_flags)
        turns_ci_result = bootstrap_turns_ci(turns_list)

        if verbose:
            print(f"  Results: Accuracy={accuracy:.2%} [{acc_ci.ci_lower:.2%}, {acc_ci.ci_upper:.2%}], "
                  f"Avg Turns={avg_turns:.1f} [{turns_ci_result.ci_lower:.1f}, {turns_ci_result.ci_upper:.1f}]")

        eval_result = EvaluationResult(
            method=method_name,
            task_type=self.task_type.value,
            n_episodes=len(puzzle_indices),
            accuracy=accuracy,
            avg_turns=avg_turns,
            std_turns=std_turns,
            avg_mi=avg_mi,
            episode_results=results,
        )
        # Attach CIs as extra attributes
        eval_result.accuracy_ci = acc_ci
        eval_result.turns_ci = turns_ci_result
        return eval_result

    def evaluate_method_ensemble(
        self,
        stopping_rule: BaseStoppingRule,
        ensemble: TrajectoryEnsemble,
        puzzle_indices: Optional[List[int]] = None,
        verbose: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate a stopping rule using trajectory ensemble on multiple puzzles.

        For each puzzle, runs N independent trajectories with varying temperatures,
        then aggregates via majority vote.

        Args:
            stopping_rule: Base stopping rule (will be cloned per trajectory)
            ensemble: TrajectoryEnsemble configuration
            puzzle_indices: Which puzzles to run
            verbose: Whether to print progress

        Returns:
            EvaluationResult with aggregated metrics
        """
        if puzzle_indices is None:
            puzzle_indices = list(range(len(self.env)))

        method_name = f"Ensemble({type(stopping_rule).__name__})"
        if verbose:
            print(f"\nEvaluating {method_name} on {len(puzzle_indices)} puzzles "
                  f"({ensemble.n_trajectories} trajectories each)...")

        temperatures = ensemble.generate_temperatures()
        results = []
        correct_count = 0
        total_turns = 0

        puzzle_iter = tqdm(
            puzzle_indices,
            desc=f"  {method_name[:20]}",
            leave=False,
            disable=not verbose,
        )
        for puzzle_idx in puzzle_iter:
            traj_results: List[TrajectoryResult] = []

            for traj_id in range(ensemble.n_trajectories):
                temp = temperatures[traj_id]

                # Create a fresh env and cloned stopping rule for this trajectory
                thread_env = ARBenchEnv(
                    task_type=self.task_type,
                    data_path=self.data_path,
                    llm=self.llm,
                    max_turns=self.max_turns,
                )
                local_rule = self._clone_stopping_rule(stopping_rule)

                # Run episode with modified temperature
                episode = self._run_episode_with_env(
                    puzzle_idx, local_rule, thread_env
                )

                traj_results.append(TrajectoryResult(
                    trajectory_id=traj_id,
                    final_answer=str(episode.prediction),
                    turns_used=episode.turns_used,
                    history=episode.history,
                    confidence=1.0 if episode.correct else 0.0,
                    stopping_reason=episode.decisions[-1]["reason"] if episode.decisions else "max_turns",
                ))

                # Early consensus check
                if ensemble.early_consensus and ensemble.check_early_consensus(traj_results):
                    break

            # Aggregate trajectory results
            ensemble_result = ensemble.aggregate_results(traj_results)

            # Determine correctness: compare ensemble answer to ground truth
            # We need ground truth — get it from a fresh env reset
            check_env = ARBenchEnv(
                task_type=self.task_type,
                data_path=self.data_path,
                llm=self.llm,
                max_turns=self.max_turns,
            )
            check_env.reset(puzzle_idx)
            ground_truth = check_env.get_ground_truth()

            # Determine correctness based on task type
            if self.task_type == TaskType.DC:
                # final_answer is already a 0-indexed string from trajectory predictions
                # Don't re-parse it (that would treat "2" as 1-based and return 1)
                try:
                    pred_idx = int(ensemble_result.final_answer)
                except (ValueError, TypeError):
                    # Fallback: if somehow it's a name, parse it
                    suspect_names = check_env.get_state().get("suspect_names", [])
                    pred_idx = check_env._parse_dc_answer(ensemble_result.final_answer, suspect_names)
                correct = pred_idx == ground_truth
                prediction = pred_idx
            elif self.task_type == TaskType.GN:
                import re as _re
                guess = _re.sub(r'[^0-9]', '', str(ensemble_result.final_answer))[:4]
                gt = _re.sub(r'[^0-9]', '', str(ground_truth))[:4]
                correct = guess == gt
                prediction = guess
            else:
                correct = ensemble_result.final_answer == ground_truth
                prediction = ensemble_result.final_answer

            episode_result = EpisodeResult(
                puzzle_idx=puzzle_idx,
                method=method_name,
                correct=correct,
                prediction=prediction,
                ground_truth=ground_truth,
                turns_used=int(ensemble_result.avg_turns),
                mi_scores=[],
                decisions=[{
                    "ensemble_consensus": ensemble_result.consensus_count,
                    "ensemble_total": ensemble_result.total_trajectories,
                    "ensemble_confidence": ensemble_result.confidence,
                }],
                total_time=0.0,
                history=[],
            )
            results.append(episode_result)

            if correct:
                correct_count += 1
            total_turns += int(ensemble_result.avg_turns)

            acc = correct_count / len(results)
            puzzle_iter.set_postfix(
                acc=f"{acc:.0%}",
                turns=f"{total_turns/len(results):.1f}",
                cons=f"{ensemble_result.confidence:.0%}",
            )

        return self._build_evaluation_result(results, puzzle_indices, verbose)

    def run_comparison(
        self,
        puzzle_indices: Optional[List[int]] = None,
        methods: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, EvaluationResult]:
        """
        Run comparison of multiple methods.

        Args:
            puzzle_indices: Which puzzles to run
            methods: Which methods to compare (default: all)
            verbose: Whether to print progress

        Returns:
            Dict mapping method name to EvaluationResult
        """
        if methods is None:
            methods = ["fixed_turns", "self_consistency", "semantic_entropy", "mi_only", "robust_mi"]

        results = {}

        for method in methods:
            stopping_rule = self._create_stopping_rule(method)
            result = self.evaluate_method(stopping_rule, puzzle_indices, verbose)
            results[method] = result

        # Save results
        self._save_results(results)

        return results

    def _create_stopping_rule(self, method: str) -> BaseStoppingRule:
        """Create stopping rule by name."""
        if method == "fixed_turns":
            return FixedTurnsStopping(
                llm=self.llm,
                fixed_turns=10,
                max_turns=self.max_turns,
            )
        elif method == "self_consistency":
            return SelfConsistencyStopping(
                llm=self.llm,
                clusterer=self.clusterer,
                k_samples=8,
                consistency_threshold=0.7,
                max_turns=self.max_turns,
            )
        elif method == "semantic_entropy":
            return SemanticEntropyStopping(
                llm=self.llm,
                clusterer=self.clusterer,
                k_samples=8,
                entropy_threshold=0.5,
                max_turns=self.max_turns,
            )
        elif method == "mi_only":
            return MIOnlyStopping(
                llm=self.llm,
                clusterer=self.clusterer,
                mi_threshold=0.3,
                k_samples=6,
                max_turns=self.max_turns,
            )
        elif method == "robust_mi":
            return RobustMIStopping(
                llm=self.llm,
                clusterer=self.clusterer,
                threshold=0.3,
                k_samples=4,  # Fewer samples for efficiency
                max_turns=self.max_turns,
                variants=["base", "skeptical"],  # Use 2 variants for MVE
                regime=self.regime,
            )
        else:
            raise ValueError(f"Unknown method: {method}")

    def _save_results(self, results: Dict[str, EvaluationResult]):
        """Save results to per-method files only (no comparison JSON)."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._save_arbench_results(results, timestamp)

    def _save_arbench_results(self, results: Dict[str, EvaluationResult], timestamp: str) -> None:
        """Save per-method logs in AR-Bench-style hierarchy."""
        task_map = {
            TaskType.DC: "dc",
            TaskType.SP: "sp",
            TaskType.GN: "gn",
        }
        task_slug = task_map.get(self.task_type, "dc")
        model_slug = getattr(self.llm, "model", "model").replace("/", "_").replace(".", "-")
        base_dir = os.path.join(self.output_dir, "baseline", task_slug)

        for method, result in results.items():
            method_dir = os.path.join(base_dir, method)
            os.makedirs(method_dir, exist_ok=True)
            output_path = os.path.join(method_dir, f"{model_slug}_{timestamp}.json")

            records = []
            for ep in result.episode_results:
                records.append({
                    "idx": ep.puzzle_idx,
                    "pred": ep.prediction,
                    "label": ep.ground_truth,
                    "record": ep.history,
                    "round": ep.turns_used,
                    "correctness": ep.correct,
                    "raw_pred": ep.raw_answer or "",
                    "raw_samples": ep.raw_samples,
                    "mi_scores": ep.mi_scores,
                    "decisions": ep.decisions,
                })

            with open(output_path, "w") as f:
                json.dump(records, f, indent=2)

    def collect_calibration_states(
        self,
        puzzle_indices: List[int],
        k_samples: int = 4,
        variants: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> RiskControlledCalibrator:
        """
        Collect calibration states by running a questioning policy on train puzzles.

        At each visited state, computes robust MI and records whether the
        model's prediction would be correct.

        Args:
            puzzle_indices: Which puzzles to run for calibration
            k_samples: Number of MI samples (fewer for speed)
            variants: Prompt variants for robust MI
            verbose: Whether to print progress

        Returns:
            RiskControlledCalibrator with collected states
        """
        if variants is None:
            variants = ["base", "skeptical"]

        if verbose:
            print(f"\n{'='*60}")
            print("CALIBRATION PHASE")
            print(f"{'='*60}")
            print(f"Collecting states from {len(puzzle_indices)} puzzles...")

        # Initialize calibrator
        calibrator = RiskControlledCalibrator(
            target_error=0.10,
            confidence_level=0.95,
        )

        # Create a robust MI stopping rule for state evaluation
        # Use a very high threshold so it always asks (we want to visit many states)
        robust_mi_rule = RobustMIStopping(
            llm=self.llm,
            clusterer=self.clusterer,
            threshold=float('inf'),  # Never stop on MI, only on max turns
            k_samples=k_samples,
            max_turns=self.max_turns,
            variants=variants,
            regime=self.regime,
        )

        # Map task type to string
        if self.task_type == TaskType.DC:
            task_type_str = "DC"
        elif self.task_type == TaskType.SP:
            task_type_str = "SP"
        elif self.task_type == TaskType.GN:
            task_type_str = "GN"
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        states_collected = 0

        puzzle_iter = tqdm(
            puzzle_indices,
            desc="  Calibrating",
            leave=False,
            disable=not verbose,
        )
        for puzzle_idx in puzzle_iter:

            # Reset environment
            obs = self.env.reset(puzzle_idx)
            ground_truth = self.env.get_ground_truth()
            if task_type_str == "GN" and ground_truth is not None:
                ground_truth = re.sub(r"[^0-9]", "", str(ground_truth))[:4]

            # Create conversation manager for this puzzle
            conv_manager = None
            if self.task_type == TaskType.DC:
                conv_manager = DCConversationManager(self.llm, self.max_turns)
                conv_manager.initialize(
                    background=obs.get("initial_info", ""),
                    suspect_names=obs.get("suspect_names", [])
                )
            elif self.task_type == TaskType.SP:
                conv_manager = SPConversationManager(self.llm, self.max_turns)
                conv_manager.initialize(scenario=obs.get("surface", ""))
            elif self.task_type == TaskType.GN:
                conv_manager = GNConversationManager(self.llm, self.max_turns)
                conv_manager.initialize()

            turn = 0
            while not self.env.done and turn < self.max_turns:
                turn += 1

                # Get current state
                state = self.env.get_state()

                # Compute robust MI for this state
                decision = robust_mi_rule.should_stop(task_type_str, state, turn)
                mi_score = decision.score

                # Get the model's current best prediction
                prediction = robust_mi_rule.get_best_answer(task_type_str, state)
                if task_type_str == "DC":
                    suspect_names = state.get("suspect_names", [])
                    prediction = self.env._parse_dc_answer(str(prediction), suspect_names)
                elif task_type_str == "GN":
                    prediction = re.sub(r"[^0-9]", "", str(prediction))[:4]

                # Compute homogeneity for diagnostics
                homogeneity = None
                if robust_mi_rule._last_estimates:
                    all_answers = []
                    for est in robust_mi_rule._last_estimates.values():
                        all_answers.extend(est.revised_answers)
                    if all_answers:
                        homogeneity = compute_homogeneity_score(
                            all_answers, self.clusterer, task_type_str
                        )

                # Add state to calibrator
                calibrator.add_state(
                    puzzle_idx=puzzle_idx,
                    turn=turn,
                    score=mi_score,
                    prediction=prediction,
                    ground_truth=ground_truth,
                    homogeneity=homogeneity,
                    task_type=task_type_str,
                )
                states_collected += 1
                puzzle_iter.set_postfix(states=states_collected)

                # Ask a question using conversation manager to continue the episode
                if self.task_type == TaskType.DC:
                    question, suspect = conv_manager.generate_question(turn)
                    result = self.env.step(ActionASK(question=question, suspect=suspect))
                    conv_manager.add_feedback(result.observation)
                elif self.task_type == TaskType.SP:
                    question = conv_manager.generate_question(turn)
                    result = self.env.step(ActionASK(question=question))
                    conv_manager.add_feedback(result.observation)
                else:  # GN
                    guess = conv_manager.generate_guess(turn)
                    result = self.env.step(ActionASK(question=guess))
                    conv_manager.add_feedback(result.observation)

        if verbose:
            print(f"\n  Collected {states_collected} calibration states from {len(puzzle_indices)} puzzles")

        return calibrator

    def calibrate_threshold(
        self,
        calibrator: RiskControlledCalibrator,
        target_error: float = 0.10,
        verbose: bool = True,
    ) -> CalibrationResult:
        """
        Compute calibrated threshold from collected states.

        Args:
            calibrator: Calibrator with collected states
            target_error: Target error rate (delta)
            verbose: Whether to print results

        Returns:
            CalibrationResult with optimal threshold
        """
        calibrator.target_error = target_error
        result = calibrator.calibrate()

        # Compute MI-error Spearman correlation
        mi_error_correlation = self._compute_mi_error_correlation(calibrator)

        if verbose:
            print(f"\n{'='*60}")
            print("CALIBRATION RESULT")
            print(f"{'='*60}")
            print(f"  Total states: {result.n_states}")
            print(f"  Target error rate: {result.target_error:.1%}")
            print(f"  Calibrated threshold (τ): {result.threshold:.4f}")
            print(f"  States covered at τ: {result.n_covered}")
            print(f"  Empirical error at τ: {result.empirical_error:.2%}")
            print(f"  UCB error at τ: {result.ucb_error:.2%}")

            if mi_error_correlation is not None:
                print(f"  MI-Error Spearman ρ: {mi_error_correlation['rho']:.4f} (p={mi_error_correlation['pvalue']:.4f})")

            if result.threshold == -np.inf:
                print("\n  WARNING: No valid threshold found!")
                print("  The system will never answer (always ask).")
                print("  Consider relaxing target_error or collecting more states.")

        # Store correlation in calibrator for later access
        calibrator.mi_error_correlation = mi_error_correlation

        return result

    def _compute_mi_error_correlation(
        self,
        calibrator: RiskControlledCalibrator,
    ) -> Optional[Dict[str, float]]:
        """
        Compute Spearman correlation between MI scores and errors.

        Args:
            calibrator: Calibrator with collected states

        Returns:
            Dict with 'rho' and 'pvalue', or None if insufficient data
        """
        if len(calibrator.states) < 3:
            return None

        scores = [s.score for s in calibrator.states]
        errors = [s.error for s in calibrator.states]

        # Need variance in both arrays for correlation
        if len(set(scores)) < 2 or len(set(errors)) < 2:
            return None

        try:
            rho, pvalue = scipy_stats.spearmanr(scores, errors)
            return {"rho": float(rho), "pvalue": float(pvalue)}
        except Exception:
            return None

    def run_calibrated_evaluation(
        self,
        train_indices: List[int],
        test_indices: List[int],
        target_error: float = 0.10,
        k_samples_calibration: int = 4,
        k_samples_eval: int = 4,
        variants: Optional[List[str]] = None,
        other_methods: Optional[List[str]] = None,
        verbose: bool = True,
        save_calibration: bool = True,
    ) -> Dict[str, Any]:
        """
        Full calibrated evaluation pipeline:
        1. Collect calibration states on train split
        2. Compute optimal threshold via Clopper-Pearson UCB
        3. Evaluate robust_mi with calibrated threshold on test split
        4. Compare against other methods

        Args:
            train_indices: Puzzle indices for calibration (train split)
            test_indices: Puzzle indices for evaluation (test split)
            target_error: Target error rate for calibration
            k_samples_calibration: MI samples during calibration
            k_samples_eval: MI samples during evaluation
            variants: Prompt variants for robust MI
            other_methods: Other methods to compare against
            verbose: Whether to print progress
            save_calibration: Whether to save calibration data

        Returns:
            Dict with calibration result and evaluation results
        """
        if variants is None:
            variants = ["base", "skeptical"]
        if other_methods is None:
            other_methods = ["fixed_turns", "self_consistency", "mi_only"]

        # Phase 1: Collect calibration states
        calibrator = self.collect_calibration_states(
            puzzle_indices=train_indices,
            k_samples=k_samples_calibration,
            variants=variants,
            verbose=verbose,
        )

        # Phase 2: Compute calibrated threshold
        calibration_result = self.calibrate_threshold(
            calibrator=calibrator,
            target_error=target_error,
            verbose=verbose,
        )

        # Save calibration data
        if save_calibration:
            calibration_path = os.path.join(
                self.output_dir,
                f"calibration_{self.task_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            calibrator.save(calibration_path)
            if verbose:
                print(f"  Calibration data saved to: {calibration_path}")

        # Phase 3: Create calibrated robust MI stopping rule
        calibrated_threshold = calibration_result.threshold
        if calibrated_threshold == -np.inf:
            # Fallback to a conservative fixed threshold
            calibrated_threshold = 0.1
            if verbose:
                print(f"  Using fallback threshold: {calibrated_threshold}")

        # Phase 4: Evaluate methods
        if verbose:
            print(f"\n{'='*60}")
            print("EVALUATION PHASE")
            print(f"{'='*60}")

        results = {}

        # Evaluate calibrated robust MI (RRMC)
        calibrated_rule = RobustMIStopping(
            llm=self.llm,
            clusterer=self.clusterer,
            threshold=calibrated_threshold,
            k_samples=k_samples_eval,
            max_turns=self.max_turns,
            variants=variants,
            regime=self.regime,
        )
        results["rrmc_calibrated"] = self.evaluate_method(
            stopping_rule=calibrated_rule,
            puzzle_indices=test_indices,
            verbose=verbose,
        )

        # Evaluate other methods for comparison
        for method in other_methods:
            stopping_rule = self._create_stopping_rule(method)
            results[method] = self.evaluate_method(
                stopping_rule=stopping_rule,
                puzzle_indices=test_indices,
                verbose=verbose,
            )

        # Save results
        self._save_results(results)

        # Get MI-error correlation from calibrator
        mi_error_correlation = getattr(calibrator, 'mi_error_correlation', None)

        return {
            "calibration": {
                "threshold": calibration_result.threshold,
                "n_states": calibration_result.n_states,
                "n_covered": calibration_result.n_covered,
                "empirical_error": calibration_result.empirical_error,
                "ucb_error": calibration_result.ucb_error,
                "target_error": calibration_result.target_error,
                "mi_error_correlation": mi_error_correlation,
            },
            "evaluation": results,
        }

    def load_calibration_and_evaluate(
        self,
        calibration_path: str,
        test_indices: List[int],
        target_error: float = 0.10,
        k_samples: int = 4,
        variants: Optional[List[str]] = None,
        other_methods: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Load existing calibration data and run evaluation.

        Args:
            calibration_path: Path to saved calibration data
            test_indices: Puzzle indices for evaluation
            target_error: Target error rate for threshold selection
            k_samples: MI samples during evaluation
            variants: Prompt variants
            other_methods: Other methods to compare
            verbose: Whether to print progress

        Returns:
            Dict with calibration result and evaluation results
        """
        if variants is None:
            variants = ["base", "skeptical"]
        if other_methods is None:
            other_methods = ["fixed_turns", "self_consistency", "mi_only"]

        # Load calibration data
        calibrator = RiskControlledCalibrator()
        calibrator.load(calibration_path)

        if verbose:
            print(f"Loaded calibration data from: {calibration_path}")
            print(f"  States: {len(calibrator.states)}")

        # Recompute threshold with potentially different target_error
        calibration_result = self.calibrate_threshold(
            calibrator=calibrator,
            target_error=target_error,
            verbose=verbose,
        )

        # Get threshold
        calibrated_threshold = calibration_result.threshold
        if calibrated_threshold == -np.inf:
            calibrated_threshold = 0.1
            if verbose:
                print(f"  Using fallback threshold: {calibrated_threshold}")

        # Evaluate
        if verbose:
            print(f"\n{'='*60}")
            print("EVALUATION PHASE")
            print(f"{'='*60}")

        results = {}

        # Calibrated RRMC
        calibrated_rule = RobustMIStopping(
            llm=self.llm,
            clusterer=self.clusterer,
            threshold=calibrated_threshold,
            k_samples=k_samples,
            max_turns=self.max_turns,
            variants=variants,
            regime=self.regime,
        )
        results["rrmc_calibrated"] = self.evaluate_method(
            stopping_rule=calibrated_rule,
            puzzle_indices=test_indices,
            verbose=verbose,
        )

        # Other methods
        for method in other_methods:
            stopping_rule = self._create_stopping_rule(method)
            results[method] = self.evaluate_method(
                stopping_rule=stopping_rule,
                puzzle_indices=test_indices,
                verbose=verbose,
            )

        self._save_results(results)

        # Get MI-error correlation from calibrator
        mi_error_correlation = getattr(calibrator, 'mi_error_correlation', None)

        return {
            "calibration": {
                "threshold": calibration_result.threshold,
                "n_states": calibration_result.n_states,
                "n_covered": calibration_result.n_covered,
                "empirical_error": calibration_result.empirical_error,
                "ucb_error": calibration_result.ucb_error,
                "target_error": calibration_result.target_error,
                "mi_error_correlation": mi_error_correlation,
            },
            "evaluation": results,
        }


def print_comparison_table(
    results: Dict[str, EvaluationResult],
    mi_error_correlation: Optional[Dict[str, float]] = None,
):
    """Print a comparison table of results with bootstrap 95% CIs."""
    print("\n" + "=" * 90)
    print("COMPARISON RESULTS")
    print("=" * 90)
    print(f"{'Method':<25} {'Accuracy':>10} {'95% CI':>16} {'Avg Turns':>10} {'95% CI':>16}")
    print("-" * 90)

    for method, result in sorted(results.items(), key=lambda x: -x[1].accuracy):
        acc_ci = getattr(result, "accuracy_ci", None)
        turns_ci = getattr(result, "turns_ci", None)
        if acc_ci:
            acc_ci_str = f"[{acc_ci.ci_lower:.2%},{acc_ci.ci_upper:.2%}]"
        else:
            acc_ci_str = ""
        if turns_ci:
            turns_ci_str = f"[{turns_ci.ci_lower:.1f},{turns_ci.ci_upper:.1f}]"
        else:
            turns_ci_str = ""
        print(f"{result.method:<25} {result.accuracy:>10.2%} {acc_ci_str:>16} {result.avg_turns:>10.1f} {turns_ci_str:>16}")

    print("=" * 90)

    # Print MI-error correlation if available
    if mi_error_correlation is not None:
        print(f"\nMI-Error Correlation (Spearman):")
        print(f"  ρ = {mi_error_correlation['rho']:.4f}")
        print(f"  p-value = {mi_error_correlation['pvalue']:.4f}")
        if mi_error_correlation['rho'] < 0:
            print("  (Negative ρ indicates lower MI → fewer errors, as expected)")
        print("=" * 90)
