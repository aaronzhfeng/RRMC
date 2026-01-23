"""
Risk-Controlled Threshold Calibration for RRMC.

Implements Clopper-Pearson binomial UCB for threshold selection.
"""

import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import Counter
import numpy as np
from scipy import stats


def _char_f1(prediction: str, ground_truth: str) -> float:
    """
    Character-level F1 score using bag-of-characters.
    """
    pred_chars = list(prediction)
    true_chars = list(ground_truth)
    if not pred_chars or not true_chars:
        return 0.0
    common = Counter(pred_chars) & Counter(true_chars)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_chars)
    recall = num_same / len(true_chars)
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0


@dataclass
class CalibrationState:
    """A single calibration state."""
    puzzle_idx: int
    turn: int
    score: float  # MI or other uncertainty score
    prediction: Any  # Predicted answer
    ground_truth: Any  # True answer
    error: int  # 1 if wrong, 0 if correct
    homogeneity: Optional[float] = None
    task_type: str = "DC"


@dataclass
class CalibrationResult:
    """Result of threshold calibration."""
    threshold: float
    n_states: int
    n_covered: int  # States with score <= threshold
    empirical_error: float  # Error rate among covered states
    ucb_error: float  # Upper confidence bound on error
    target_error: float  # Target error rate (delta)


class RiskControlledCalibrator:
    """
    Risk-controlled threshold calibration using Clopper-Pearson UCB.

    Finds threshold tau such that error rate among states with
    score <= tau is controlled at target level delta.
    """

    def __init__(
        self,
        target_error: float = 0.10,
        confidence_level: float = 0.95,
        sp_f1_threshold: float = 0.5,
    ):
        """
        Initialize calibrator.

        Args:
            target_error: Target maximum error rate (delta)
            confidence_level: Confidence level for UCB (1 - alpha)
        """
        self.target_error = target_error
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.sp_f1_threshold = sp_f1_threshold

        # Calibration data
        self.states: List[CalibrationState] = []

    def add_state(
        self,
        puzzle_idx: int,
        turn: int,
        score: float,
        prediction: Any,
        ground_truth: Any,
        homogeneity: Optional[float] = None,
        task_type: str = "DC",
    ):
        """
        Add a calibration state.

        Args:
            puzzle_idx: Puzzle index
            turn: Turn number
            score: Uncertainty score (MI)
            prediction: Model's prediction
            ground_truth: True answer
            homogeneity: Homogeneity score (optional)
            task_type: Task type
        """
        # Determine if error
        if task_type == "DC" or task_type == "GN":
            error = int(prediction != ground_truth)
        else:  # SP - use char-level F1 threshold
            pred_text = str(prediction).lower()
            gt_text = str(ground_truth).lower()
            f1 = _char_f1(pred_text, gt_text)
            error = int(f1 < self.sp_f1_threshold)

        state = CalibrationState(
            puzzle_idx=puzzle_idx,
            turn=turn,
            score=score,
            prediction=prediction,
            ground_truth=ground_truth,
            error=error,
            homogeneity=homogeneity,
            task_type=task_type,
        )
        self.states.append(state)

    def calibrate(self) -> CalibrationResult:
        """
        Calibrate threshold using Clopper-Pearson UCB.

        Returns:
            CalibrationResult with optimal threshold
        """
        if not self.states:
            return CalibrationResult(
                threshold=-np.inf,
                n_states=0,
                n_covered=0,
                empirical_error=0.0,
                ucb_error=0.0,
                target_error=self.target_error,
            )

        # Sort states by score ascending (lower score = more confident)
        sorted_states = sorted(self.states, key=lambda s: s.score)
        scores = [s.score for s in sorted_states]
        errors = [s.error for s in sorted_states]

        # Find best threshold
        best_threshold = -np.inf
        n = len(sorted_states)

        # Cumulative error count
        cum_err = np.cumsum(errors)

        for j in range(1, n + 1):
            k = int(cum_err[j - 1])  # Number of errors in first j states
            ucb = self._clopper_pearson_upper(k, j, self.alpha)

            if ucb <= self.target_error:
                best_threshold = scores[j - 1]

        # Compute result statistics
        if best_threshold == -np.inf:
            # No valid threshold found
            n_covered = 0
            empirical_error = 0.0
            ucb_error = 1.0
        else:
            # Find how many states are covered
            n_covered = sum(1 for s in scores if s <= best_threshold)
            if n_covered > 0:
                k = sum(1 for i, s in enumerate(scores) if s <= best_threshold and errors[i])
                empirical_error = k / n_covered
                ucb_error = self._clopper_pearson_upper(k, n_covered, self.alpha)
            else:
                empirical_error = 0.0
                ucb_error = 0.0

        return CalibrationResult(
            threshold=best_threshold,
            n_states=n,
            n_covered=n_covered,
            empirical_error=empirical_error,
            ucb_error=ucb_error,
            target_error=self.target_error,
        )

    def _clopper_pearson_upper(self, k: int, n: int, alpha: float) -> float:
        """
        Compute Clopper-Pearson upper confidence bound.

        Args:
            k: Number of successes (errors)
            n: Number of trials
            alpha: Significance level

        Returns:
            Upper bound on true proportion
        """
        if n == 0:
            return 1.0
        if k == n:
            return 1.0
        if k == 0:
            return 1 - alpha ** (1 / n)

        # Use beta distribution quantile
        return stats.beta.ppf(1 - alpha, k + 1, n - k)

    def get_threshold_for_regime(
        self,
        regime: str = "normal",
        homogeneity_bucket: Optional[Tuple[float, float]] = None,
    ) -> CalibrationResult:
        """
        Get threshold for a specific regime.

        Args:
            regime: Decoding regime (normal, homogeneous)
            homogeneity_bucket: Optional homogeneity range (min, max)

        Returns:
            CalibrationResult for the regime
        """
        # Filter states by regime/bucket
        filtered_states = self.states.copy()

        if homogeneity_bucket is not None:
            h_min, h_max = homogeneity_bucket
            filtered_states = [
                s for s in filtered_states
                if s.homogeneity is not None and h_min <= s.homogeneity < h_max
            ]

        # Create temporary calibrator with filtered states
        temp_calibrator = RiskControlledCalibrator(
            target_error=self.target_error,
            confidence_level=self.confidence_level,
        )
        temp_calibrator.states = filtered_states

        return temp_calibrator.calibrate()

    def save(self, path: str):
        """Save calibration data to file."""
        data = {
            "target_error": self.target_error,
            "confidence_level": self.confidence_level,
            "sp_f1_threshold": self.sp_f1_threshold,
            "states": [
                {
                    "puzzle_idx": s.puzzle_idx,
                    "turn": s.turn,
                    "score": s.score,
                    "prediction": str(s.prediction),
                    "ground_truth": str(s.ground_truth),
                    "error": s.error,
                    "homogeneity": s.homogeneity,
                    "task_type": s.task_type,
                }
                for s in self.states
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str):
        """Load calibration data from file."""
        with open(path, "r") as f:
            data = json.load(f)

        self.target_error = data["target_error"]
        self.confidence_level = data["confidence_level"]
        self.alpha = 1 - self.confidence_level
        self.sp_f1_threshold = data.get("sp_f1_threshold", self.sp_f1_threshold)

        self.states = [
            CalibrationState(
                puzzle_idx=s["puzzle_idx"],
                turn=s["turn"],
                score=s["score"],
                prediction=s["prediction"],
                ground_truth=s["ground_truth"],
                error=s["error"],
                homogeneity=s.get("homogeneity"),
                task_type=s.get("task_type", "DC"),
            )
            for s in data["states"]
        ]


class StoppingController:
    """
    Stopping controller that uses calibrated threshold.

    Decides whether to stop (answer) or continue (ask) based on
    uncertainty score and calibrated threshold.
    """

    def __init__(
        self,
        threshold: float,
        max_turns: int = 25,
    ):
        """
        Initialize stopping controller.

        Args:
            threshold: Calibrated MI threshold (answer if MI <= threshold)
            max_turns: Maximum turns before forced answer
        """
        self.threshold = threshold
        self.max_turns = max_turns

    def should_stop(
        self,
        mi_score: float,
        current_turn: int,
    ) -> Tuple[bool, str]:
        """
        Decide whether to stop and answer.

        Args:
            mi_score: Current robust MI score
            current_turn: Current turn number

        Returns:
            Tuple of (should_stop, reason)
        """
        # Check turn limit
        if current_turn >= self.max_turns:
            return True, "max_turns_reached"

        # Check MI threshold
        if mi_score <= self.threshold:
            return True, "mi_below_threshold"

        return False, "continue"

    def update_threshold(self, new_threshold: float):
        """Update the stopping threshold."""
        self.threshold = new_threshold
