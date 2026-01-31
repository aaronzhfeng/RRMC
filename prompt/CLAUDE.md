# RRMC — Claude Code Task Board (Active)

This file is the **current active implementation plan** for Claude Code runs.  
Older plans are archived in this folder (see `prompt/06_step1to6_baseline_methods.md`).

## Context: The Turn-1 Stopping Problem

**Problem:** All stopping methods suffer from "Turn 1 stopping" — the LLM stops before asking any questions because:
- MI = 0 at Turn 1 (all k samples give identical answers → false confidence)
- 70-100% of puzzles stop at Turn 1 across all methods
- Accuracy is 25-45%, barely better than random

**Solution:** Two complementary improvements:
1. **Force information gathering** — require questioning all suspects before allowing stop
2. **Trajectory diversity** — run multiple independent investigations, decide by consensus

See `docs/08_proposed_improvements.md` for full analysis.

---

## Ground Rules (Don't Skip)

- **Do not refactor** unless required for the step you're executing.
- Keep changes **small and reviewable**: prefer 1–3 files per step.
- After each step: run a **tiny smoke run** (1–2 puzzles, 2–3 turns) to ensure nothing broke.
- **Test both enabled and disabled states** for configurable features.

---

## Current Repo Reality

- Config runner: `run.py` + `config.py` + `pipeline.py`
- Core tasks + env wrapper: `rrmc/core/environment.py` (DC/SP/GN)
- RRMC components: `rrmc/core/mi_estimator.py`, `rrmc/core/calibration.py`
- Baselines + RRMC stopping: `rrmc/methods/stopping_rules.py`
- Evaluation harness: `rrmc/evaluation/evaluator.py`
- DC method configs: `configs/experiments/dc_methods/*.yaml`

---

## Step 1 — AllSuspectsWrapper (Code-Enforced Suspect Questioning)

### Goal
Create a **wrapper** stopping rule that forces the LLM to question all suspects before allowing any stopping rule to trigger. Must be:
- **Modular:** Works with any stopping rule
- **Configurable:** Enable/disable via config
- **DC-specific:** Only applies to detective cases

### Implementation Tasks

#### 1.1 Add Config Option
In `configs/base.yaml` (or experiment configs):
```yaml
require_all_suspects: true  # Enable/disable the requirement
```

#### 1.2 Create AllSuspectsWrapper Class
Add to `rrmc/methods/stopping_rules.py`:

```python
class AllSuspectsWrapper(BaseStoppingRule):
    """
    Wrapper that enforces questioning all suspects before allowing stop.
    
    Wraps any existing stopping rule and overrides should_stop() to
    return False until all suspects have been questioned at least once.
    """
    
    def __init__(
        self,
        inner_rule: BaseStoppingRule,
        suspects: List[str],
        enabled: bool = True,
    ):
        super().__init__(inner_rule.max_turns)
        self.inner_rule = inner_rule
        self.suspects = set(suspects)  # Full list of suspects in puzzle
        self.enabled = enabled
    
    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        # Only apply to DC task when enabled
        if self.enabled and task_type.lower() in ["dc", "detective_cases"]:
            questioned = self._get_questioned_suspects(state)
            remaining = self.suspects - questioned
            
            if remaining:
                # Not all suspects questioned — force continue
                return StoppingDecision(
                    should_stop=False,
                    reason="not_all_suspects_questioned",
                    score=len(remaining),
                    prediction=None,
                    metadata={"remaining_suspects": list(remaining)},
                )
        
        # All suspects questioned (or disabled) — delegate to inner rule
        return self.inner_rule.should_stop(task_type, state, turn)
    
    def _get_questioned_suspects(self, state: Dict[str, Any]) -> Set[str]:
        """Extract suspects that have been questioned from history."""
        record = state.get("record", [])
        return {r.get("suspect") for r in record if r.get("suspect")}
    
    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        return self.inner_rule.get_best_answer(task_type, state)
```

#### 1.3 Update Environment to Expose Suspects
In `rrmc/core/environment.py`, ensure state includes suspect list:

```python
def _get_state(self) -> Dict[str, Any]:
    return {
        ...
        "suspects": self._get_suspect_list(),  # ADD THIS
    }

def _get_suspect_list(self) -> List[str]:
    """Extract suspect names/IDs from current puzzle."""
    # Parse from puzzle data — check AR-Bench DC format
    # Typically suspects are in puzzle["suspects"] or parsed from background
    if "suspects" in self.current_puzzle:
        return list(self.current_puzzle["suspects"].keys())
    # Fallback: parse from initial_info if needed
    return []
```

#### 1.4 Wire Wrapper in Pipeline
In `pipeline.py`, wrap stopping rules when config enabled:

```python
def _create_method(self, method_name: str, method_cfg: Dict[str, Any]):
    # Create base stopping rule
    stopping_rule = get_method(method_name, **kwargs)
    
    # Wrap with AllSuspectsWrapper if configured
    if self.config.get("require_all_suspects", False):
        suspects = self._get_current_puzzle_suspects()  # Need puzzle context
        stopping_rule = AllSuspectsWrapper(
            inner_rule=stopping_rule,
            suspects=suspects,
            enabled=True,
        )
    
    return stopping_rule
```

**Note:** The wrapper needs puzzle-specific suspect list. This may require refactoring how methods are created (per-puzzle vs once). Consider:
- Option A: Create wrapper once, update suspects on each `env.reset()`
- Option B: Pass suspects via state (already in step 1.3)
- Option C: Create wrapper inside evaluation loop

### Smoke Test
```bash
# Test with wrapper disabled (should behave as before)
python run.py dc_methods/mi_only --n_puzzles 2 --max_turns 5 --require_all_suspects false

# Test with wrapper enabled (should force 5+ turns)
python run.py dc_methods/mi_only --n_puzzles 2 --max_turns 10 --require_all_suspects true
```

### Expected Outcome
- With wrapper: minimum 5 turns per puzzle (one per suspect)
- Should see improved accuracy from better information gathering

---

## Step 2 — TrajectoryEnsemble (Full Trajectory Sampling)

### Goal
Run N **independent** investigation trajectories from start to finish. Each trajectory:
- Makes its own decisions about questions to ask
- Uses its own stopping criterion
- Produces a final answer

Final answer is determined by **majority vote** across trajectories.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    TrajectoryEnsemble                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Traj 1   │  │ Traj 2   │  │ Traj 3   │  │ Traj N   │    │
│  │ temp=0.5 │  │ temp=0.7 │  │ temp=0.9 │  │ temp=1.0 │    │
│  │          │  │          │  │          │  │          │    │
│  │ T1→T5→T8 │  │ T1→T2    │  │ T1→T3→T6 │  │ ...      │    │
│  │ Ans: A   │  │ Ans: A   │  │ Ans: B   │  │ Ans: A   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                              │
│  Consensus: 3/4 say A → Final Answer: A                     │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Tasks

#### 2.1 Add Config Options
```yaml
trajectory_ensemble:
  enabled: true
  n_trajectories: 6           # Number of parallel trajectories
  consensus_threshold: 0.5    # Fraction needed for consensus
  early_consensus: true       # Stop if consensus before all finish
  temperature_range: [0.5, 1.0]  # Temperature variation for diversity
  inner_stopping_rule: "mi_only"  # Stopping rule for each trajectory
  inner_threshold: 0.1        # Threshold for inner stopping rule
```

#### 2.2 Create TrajectoryEnsemble Class
Create new file `rrmc/methods/trajectory_ensemble.py`:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Callable
from collections import Counter
import copy

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
        return self.consensus_count / self.total_trajectories


class TrajectoryEnsemble:
    """
    Run multiple independent trajectories and aggregate results.
    """
    
    def __init__(
        self,
        env,                        # Environment instance
        stopping_rule_factory: Callable,  # Factory: (temp) -> StoppingRule
        llm,                        # LLM wrapper
        n_trajectories: int = 6,
        consensus_threshold: float = 0.5,
        early_consensus: bool = True,
        temperature_range: tuple = (0.5, 1.0),
        max_turns: int = 25,
    ):
        self.env = env
        self.stopping_rule_factory = stopping_rule_factory
        self.llm = llm
        self.n_trajectories = n_trajectories
        self.consensus_threshold = consensus_threshold
        self.early_consensus = early_consensus
        self.temperature_range = temperature_range
        self.max_turns = max_turns
    
    def run(self, puzzle_idx: int) -> EnsembleResult:
        """Run ensemble of trajectories on a single puzzle."""
        results: List[TrajectoryResult] = []
        
        # Generate temperatures for diversity
        temps = self._generate_temperatures()
        
        for traj_id in range(self.n_trajectories):
            # Reset environment to same puzzle
            self.env.reset(puzzle_idx)
            
            # Create stopping rule with this trajectory's temperature
            stopping_rule = self.stopping_rule_factory(temperature=temps[traj_id])
            
            # Run single trajectory
            result = self._run_single_trajectory(
                stopping_rule=stopping_rule,
                traj_id=traj_id,
                temperature=temps[traj_id],
            )
            results.append(result)
            
            # Early consensus check
            if self.early_consensus and self._check_early_consensus(results):
                break
        
        return self._aggregate_results(results)
    
    def _generate_temperatures(self) -> List[float]:
        """Generate evenly spaced temperatures for trajectory diversity."""
        lo, hi = self.temperature_range
        if self.n_trajectories == 1:
            return [(lo + hi) / 2]
        step = (hi - lo) / (self.n_trajectories - 1)
        return [lo + i * step for i in range(self.n_trajectories)]
    
    def _run_single_trajectory(
        self,
        stopping_rule,
        traj_id: int,
        temperature: float,
    ) -> TrajectoryResult:
        """Run one complete trajectory from start to finish."""
        turn = 0
        history = []
        
        while turn < self.max_turns:
            turn += 1
            state = self.env.get_state()
            
            # Check stopping
            decision = stopping_rule.should_stop(
                task_type=self.env.task_type,
                state=state,
                turn=turn,
            )
            
            if decision.should_stop:
                return TrajectoryResult(
                    trajectory_id=traj_id,
                    final_answer=decision.prediction,
                    turns_used=turn,
                    history=history,
                    confidence=1.0 - (decision.score if decision.score < 1 else 1.0),
                    stopping_reason=decision.reason,
                )
            
            # Generate question (with temperature for diversity)
            question = self._generate_question(state, temperature)
            
            # Execute question
            feedback = self.env.step(question)
            history.append({
                "turn": turn,
                "question": question,
                "feedback": feedback,
            })
        
        # Hit max turns
        final_answer = stopping_rule.get_best_answer(
            self.env.task_type, 
            self.env.get_state()
        )
        return TrajectoryResult(
            trajectory_id=traj_id,
            final_answer=final_answer,
            turns_used=self.max_turns,
            history=history,
            confidence=0.0,
            stopping_reason="max_turns",
        )
    
    def _generate_question(self, state: Dict[str, Any], temperature: float) -> str:
        """Generate next question using LLM with given temperature."""
        # Use existing question generation logic
        # Temperature affects diversity of question choice
        # Implementation depends on current question generation approach
        pass  # TODO: Wire to existing question generation
    
    def _check_early_consensus(self, results: List[TrajectoryResult]) -> bool:
        """Check if we can stop early due to consensus."""
        if len(results) < 3:
            return False
        
        answers = [r.final_answer for r in results]
        counter = Counter(answers)
        _, count = counter.most_common(1)[0]
        
        # If majority already achieved, stop
        needed = int(self.n_trajectories * self.consensus_threshold) + 1
        remaining = self.n_trajectories - len(results)
        
        # Can stop if current leader can't be overtaken
        return count > (self.n_trajectories - count)
    
    def _aggregate_results(self, results: List[TrajectoryResult]) -> EnsembleResult:
        """Aggregate trajectory results via majority vote."""
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
```

#### 2.3 Create Stopping Rule Factory
Add helper in `pipeline.py`:

```python
def _create_stopping_rule_factory(self, method_name: str, base_kwargs: dict):
    """Create a factory that produces stopping rules with varying temperature."""
    def factory(temperature: float):
        kwargs = base_kwargs.copy()
        kwargs["temperature"] = temperature
        return get_method(method_name, **kwargs)
    return factory
```

#### 2.4 Wire into Evaluation
In `rrmc/evaluation/evaluator.py`, add ensemble mode:

```python
def evaluate_puzzle_ensemble(
    self,
    puzzle_idx: int,
    ensemble_config: dict,
) -> EpisodeResult:
    """Run trajectory ensemble on a single puzzle."""
    ensemble = TrajectoryEnsemble(
        env=self.env,
        stopping_rule_factory=self._create_stopping_rule_factory(...),
        llm=self.llm,
        **ensemble_config,
    )
    
    result = ensemble.run(puzzle_idx)
    
    # Convert to EpisodeResult format
    return EpisodeResult(
        puzzle_idx=puzzle_idx,
        correct=result.final_answer == self.env.current_puzzle["label"],
        prediction=result.final_answer,
        ground_truth=self.env.current_puzzle["label"],
        turns_used=result.avg_turns,
        # ... additional fields
    )
```

#### 2.5 Add Ensemble Config Files
Create `configs/experiments/dc_methods/ensemble_mi_only.yaml`:

```yaml
experiment_name: dc_methods/ensemble_mi_only
task: dc
data_subset: test
n_puzzles: 20
max_turns: 25

trajectory_ensemble:
  enabled: true
  n_trajectories: 6
  consensus_threshold: 0.5
  early_consensus: true
  temperature_range: [0.5, 1.0]
  inner_stopping_rule: mi_only
  inner_threshold: 0.1
```

### Smoke Test
```bash
# Test ensemble with 3 trajectories on 2 puzzles
python run.py dc_methods/ensemble_mi_only --n_puzzles 2 --trajectory_ensemble.n_trajectories 3
```

### Expected Outcome
- More diverse exploration of question space
- Consensus-based final answer more robust than single trajectory
- Higher accuracy at cost of ~6x API calls

---

## Step 3 — Combine Both Improvements

### Goal
Test the combination: AllSuspectsWrapper + TrajectoryEnsemble.

Each trajectory is wrapped with AllSuspectsWrapper, ensuring:
- Every trajectory questions all suspects
- Then each applies its stopping rule
- Final answer via consensus

### Config
```yaml
require_all_suspects: true

trajectory_ensemble:
  enabled: true
  n_trajectories: 6
  inner_stopping_rule: mi_only
```

### Expected Outcome
- Best of both: forced information gathering + diversity
- Highest expected accuracy improvement
- Highest API cost (~30 calls per puzzle)

---

## Testing Checklist

After implementation, run these experiments:

1. **Baseline** (no improvements):
   ```bash
   python run.py dc_methods/mi_only --n_puzzles 20
   ```

2. **AllSuspectsWrapper only**:
   ```bash
   python run.py dc_methods/mi_only --n_puzzles 20 --require_all_suspects true
   ```

3. **TrajectoryEnsemble only**:
   ```bash
   python run.py dc_methods/ensemble_mi_only --n_puzzles 20
   ```

4. **Both combined**:
   ```bash
   python run.py dc_methods/ensemble_mi_only --n_puzzles 20 --require_all_suspects true
   ```

Compare accuracy, avg turns, and API cost across all four.

---

## Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `rrmc/methods/stopping_rules.py` | Modify | Add `AllSuspectsWrapper` class |
| `rrmc/methods/trajectory_ensemble.py` | Create | New `TrajectoryEnsemble` class |
| `rrmc/core/environment.py` | Modify | Expose suspect list in state |
| `pipeline.py` | Modify | Wire wrapper and ensemble |
| `rrmc/evaluation/evaluator.py` | Modify | Add ensemble evaluation mode |
| `configs/base.yaml` | Modify | Add `require_all_suspects` option |
| `configs/experiments/dc_methods/ensemble_*.yaml` | Create | Ensemble experiment configs |
