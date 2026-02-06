"""
Trajectory Ensemble for RRMC.

Runs N independent investigation trajectories from start to finish,
then aggregates final answers via majority vote.

This addresses the Turn-1 stopping problem by providing diversity
across full trajectories rather than just at the sampling level.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from ..methods.stopping_rules import BaseStoppingRule


@dataclass
class TrajectoryResult:
    """Result from a single trajectory run."""
    trajectory_id: int
    final_answer: str
    turns_used: int
    history: List[Dict[str, Any]]
    confidence: float
    stopping_reason: str


@dataclass
class EnsembleResult:
    """Aggregated result from trajectory ensemble."""
    final_answer: str
    consensus_count: int
    total_trajectories: int
    confidence: float
    trajectory_results: List[TrajectoryResult]
    avg_turns: float

    @property
    def consensus_ratio(self) -> float:
        return self.consensus_count / self.total_trajectories if self.total_trajectories > 0 else 0.0


class TrajectoryEnsemble:
    """
    Run multiple independent trajectories and aggregate results.

    Each trajectory runs a full episode (reset -> question loop -> answer)
    independently, with temperature variation for diversity. The final
    answer is determined by majority vote across trajectories.
    """

    def __init__(
        self,
        n_trajectories: int = 6,
        consensus_threshold: float = 0.5,
        early_consensus: bool = True,
        temperature_range: tuple = (0.5, 1.0),
    ):
        self.n_trajectories = n_trajectories
        self.consensus_threshold = consensus_threshold
        self.early_consensus = early_consensus
        self.temperature_range = temperature_range

    def generate_temperatures(self) -> List[float]:
        """Generate evenly spaced temperatures for trajectory diversity."""
        lo, hi = self.temperature_range
        if self.n_trajectories == 1:
            return [(lo + hi) / 2]
        step = (hi - lo) / (self.n_trajectories - 1)
        return [lo + i * step for i in range(self.n_trajectories)]

    def check_early_consensus(self, results: List[TrajectoryResult]) -> bool:
        """Check if we can stop early due to consensus."""
        if len(results) < 3:
            return False

        answers = [r.final_answer for r in results]
        counter = Counter(answers)
        _, count = counter.most_common(1)[0]

        # Can stop if current leader can't be overtaken
        return count > (self.n_trajectories - count)

    def aggregate_results(self, results: List[TrajectoryResult]) -> EnsembleResult:
        """Aggregate trajectory results via majority vote."""
        if not results:
            return EnsembleResult(
                final_answer="",
                consensus_count=0,
                total_trajectories=0,
                confidence=0.0,
                trajectory_results=[],
                avg_turns=0.0,
            )

        answers = [r.final_answer for r in results]
        counter = Counter(answers)
        winner, count = counter.most_common(1)[0]

        return EnsembleResult(
            final_answer=winner,
            consensus_count=count,
            total_trajectories=len(results),
            confidence=count / len(results),
            trajectory_results=results,
            avg_turns=sum(r.turns_used for r in results) / len(results),
        )
