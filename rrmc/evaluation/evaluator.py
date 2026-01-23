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
from ..methods import get_method
from ..methods.stopping_rules import (
    BaseStoppingRule,
    FixedTurnsStopping,
    SelfConsistencyStopping,
    SemanticEntropyStopping,
    MIOnlyStopping,
    RobustMIStopping,
)


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
        """
        self.llm = llm
        self.data_path = data_path
        self.task_type = task_type
        self.max_turns = max_turns
        self.output_dir = output_dir
        self.regime = regime
        self.max_workers = max_workers

        # Initialize environment (for sequential runs and metadata)
        self.env = ARBenchEnv(
            task_type=task_type,
            data_path=data_path,
            llm=llm,
        )

        # Initialize clusterer (without sentence-transformers dependency for now)
        self.clusterer = SemanticClusterer(use_entailment=False)

        os.makedirs(output_dir, exist_ok=True)

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
            question_generator: Question generator (uses simple prompting if None)

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

        mi_scores = []
        decisions = []
        turn = 0

        while not self.env.done and turn < self.max_turns:
            turn += 1

            # Get current state
            state = self.env.get_state()

            # Make stopping decision
            decision = stopping_rule.should_stop(task_type_str, state, turn)
            mi_scores.append(decision.score)
            decisions.append({
                "turn": turn,
                "score": decision.score,
                "should_stop": decision.should_stop,
                "reason": decision.reason,
            })

            if decision.should_stop:
                # Submit answer
                answer = decision.prediction or stopping_rule.get_best_answer(task_type_str, state)
                result = self.env.step(ActionANSWER(answer=answer))
                break
            else:
                # Generate and ask question
                question, suspect = self._generate_question(state)
                if self.task_type == TaskType.DC:
                    result = self.env.step(ActionASK(question=question, suspect=suspect))
                else:
                    result = self.env.step(ActionASK(question=question))

        # If we exhausted turns without stopping, force answer
        if not self.env.done:
            state = self.env.get_state()
            answer = stopping_rule.get_best_answer(task_type_str, state)
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
        )

    def _generate_question(self, state: Dict[str, Any]) -> tuple:
        """Generate next question using simple prompting."""
        history = state.get("history_string", "")
        initial_info = state.get("initial_info", "")

        if self.task_type == TaskType.DC:
            suspect_names = state.get("suspect_names", [])
            suspect_list = ", ".join(suspect_names)

            prompt = f"""You are investigating a murder case.

Case background:
{initial_info}

Previous interrogation:
{history}

Choose a suspect to question and formulate a question.
Available suspects: {suspect_list}

Format your response as:
Suspect: [name]
Question: [your question]"""

            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )

            # Parse response
            text = response.content
            suspect_match = None
            for name in suspect_names:
                if name.lower() in text.lower():
                    suspect_match = name
                    break
            suspect = suspect_match or suspect_names[0] if suspect_names else ""

            question_match = text.split("Question:")[-1].strip() if "Question:" in text else text
            question = question_match.split("\n")[0].strip()

            return question, suspect

        elif self.task_type == TaskType.SP:
            prompt = f"""You are playing a situation puzzle game.

Situation: {state.get('surface', '')}

Previous questions and answers:
{history}

Ask a yes/no question to uncover the hidden story:"""

            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=128,
            )
            return response.content.strip(), None

        else:  # GN
            prompt = f"""You are playing a number guessing game.
The secret is a 4-digit number with unique digits.

Previous guesses:
{history}

Make your next guess (4 unique digits):"""

            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=32,
            )
            import re
            digits = re.sub(r'[^0-9]', '', response.content)[:4]
            return digits, None

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
            )
            
            # Run episode with the thread-local environment
            result = self._run_episode_with_env(idx, stopping_rule, thread_env)
            
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

        mi_scores = []
        decisions = []
        turn = 0

        while not env.done and turn < self.max_turns:
            turn += 1

            # Get current state
            state = env.get_state()

            # Make stopping decision
            decision = stopping_rule.should_stop(task_type_str, state, turn)
            mi_scores.append(decision.score)
            decisions.append({
                "turn": turn,
                "score": decision.score,
                "should_stop": decision.should_stop,
                "reason": decision.reason,
            })

            if decision.should_stop:
                # Submit answer
                answer = decision.prediction or stopping_rule.get_best_answer(task_type_str, state)
                result = env.step(ActionANSWER(answer=answer))
                break
            else:
                # Generate and ask question
                question, suspect = self._generate_question_with_env(state, env)
                if self.task_type == TaskType.DC:
                    result = env.step(ActionASK(question=question, suspect=suspect))
                else:
                    result = env.step(ActionASK(question=question))

        # If we exhausted turns without stopping, force answer
        if not env.done:
            state = env.get_state()
            answer = stopping_rule.get_best_answer(task_type_str, state)
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
        )

    def _generate_question_with_env(self, state: Dict[str, Any], env: ARBenchEnv) -> tuple:
        """Generate next question using simple prompting (thread-safe version)."""
        history = state.get("history_string", "")
        initial_info = state.get("initial_info", "")

        if self.task_type == TaskType.DC:
            suspect_names = state.get("suspect_names", [])
            suspect_list = ", ".join(suspect_names)

            prompt = f"""You are investigating a murder case.

Case background:
{initial_info}

Previous interrogation:
{history}

Choose a suspect to question and formulate a question.
Available suspects: {suspect_list}

Format your response as:
Suspect: [name]
Question: [your question]"""

            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=256,
            )

            # Parse response
            text = response.content
            suspect_match = None
            for name in suspect_names:
                if name.lower() in text.lower():
                    suspect_match = name
                    break
            suspect = suspect_match or suspect_names[0] if suspect_names else ""

            question_match = text.split("Question:")[-1].strip() if "Question:" in text else text
            question = question_match.split("\n")[0].strip()

            return question, suspect

        elif self.task_type == TaskType.SP:
            prompt = f"""You are playing a situation puzzle game.

Situation: {state.get('surface', '')}

Previous questions and answers:
{history}

Ask a yes/no question to uncover the hidden story:"""

            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=128,
            )
            return response.content.strip(), None

        else:  # GN
            prompt = f"""You are playing a number guessing game.
The secret is a 4-digit number with unique digits.

Previous guesses:
{history}

Make your next guess (4 unique digits):"""

            response = self.llm.generate(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=32,
            )
            digits = re.sub(r'[^0-9]', '', response.content)[:4]
            return digits, None

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

        if verbose:
            print(f"  Results: Accuracy={accuracy:.2%}, Avg Turns={avg_turns:.1f}")

        return EvaluationResult(
            method=method_name,
            task_type=self.task_type.value,
            n_episodes=len(puzzle_indices),
            accuracy=accuracy,
            avg_turns=avg_turns,
            std_turns=std_turns,
            avg_mi=avg_mi,
            episode_results=results,
        )

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
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{self.task_type.value}_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Convert to serializable format
        output = {}
        for method, result in results.items():
            output[method] = {
                "method": result.method,
                "task_type": result.task_type,
                "n_episodes": result.n_episodes,
                "accuracy": result.accuracy,
                "avg_turns": result.avg_turns,
                "std_turns": result.std_turns,
                "avg_mi": result.avg_mi,
                "episodes": [
                    {
                        "puzzle_idx": ep.puzzle_idx,
                        "correct": ep.correct,
                        "prediction": str(ep.prediction),
                        "ground_truth": str(ep.ground_truth),
                        "turns_used": ep.turns_used,
                        "mi_scores": ep.mi_scores,
                        "total_time": ep.total_time,
                    }
                    for ep in result.episode_results
                ],
            }

        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {filepath}")

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

                # Ask a question to continue the episode
                question, suspect = self._generate_question(state)
                if self.task_type == TaskType.DC:
                    result = self.env.step(ActionASK(question=question, suspect=suspect))
                else:
                    result = self.env.step(ActionASK(question=question))

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
    """Print a comparison table of results."""
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Method':<25} {'Accuracy':>10} {'Avg Turns':>12} {'Std Turns':>10}")
    print("-" * 70)

    for method, result in sorted(results.items(), key=lambda x: -x[1].accuracy):
        print(f"{result.method:<25} {result.accuracy:>10.2%} {result.avg_turns:>12.1f} {result.std_turns:>10.2f}")

    print("=" * 70)

    # Print MI-error correlation if available
    if mi_error_correlation is not None:
        print(f"\nMI-Error Correlation (Spearman):")
        print(f"  ρ = {mi_error_correlation['rho']:.4f}")
        print(f"  p-value = {mi_error_correlation['pvalue']:.4f}")
        if mi_error_correlation['rho'] < 0:
            print("  (Negative ρ indicates lower MI → fewer errors, as expected)")
        print("=" * 70)
