# Proposed Improvements for Stopping Rules

**Date:** 2026-01-30  
**Status:** Planning  
**Context:** Current stopping methods all suffer from "Turn 1 stopping" where the LLM stops before asking any questions.

## Problem Summary

From validation runs on 20 puzzles:
- 70-100% of puzzles stop at Turn 1 across all methods
- MI = 0 at Turn 1 because all k samples give identical answers
- No information is gathered before making predictions
- Accuracy ranges from 25-45%, barely better than random guessing

**Root Cause:** The LLM is "confidently guessing" without evidence. Stopping rules detect this false confidence and stop immediately.

---

## Proposed Solutions

### Overview of 4 Options

| Option | Approach | Difficulty | API Cost | Chosen? |
|--------|----------|------------|----------|---------|
| A | Ask all suspects (prompt) | ⭐ Easy | Same | No |
| **B** | **Ask all suspects (code)** | ⭐⭐ Easy-Medium | Same | **✓ Yes** |
| C | Partial trajectory sampling | ⭐⭐ Medium | 2-3x | No |
| **D** | **Full trajectory sampling** | ⭐⭐⭐ Hard | 6x | **✓ Yes** |

---

## Option A: Ask All Suspects (Prompt Engineering)

### Description
Modify the question-asking prompt to instruct the LLM to interview all suspects before concluding.

### Implementation
Add to the question prompt:
```
You MUST ask at least one question to each suspect before making your final guess.
Suspects not yet questioned: {unquestioned_suspects}
Do NOT make a final prediction until you have questioned everyone.
```

### Pros
- No code changes needed
- Quick to test

### Cons
- LLM might ignore instructions
- Hard to enforce strictly
- Relies on prompt following ability

### Not Chosen Because
Unreliable enforcement. The LLM may still try to guess early despite instructions.

---

## Option B: Ask All Suspects (Code Enforcement) ✓ CHOSEN

### Description
Track which suspects have been questioned and programmatically prevent stopping until all suspects are interviewed at least once.

### Design Goals
1. **Modular:** Can be enabled/disabled via config
2. **Wrapper pattern:** Works with any stopping rule
3. **Task-specific:** Only applies to DC (detective cases) where suspects exist

### Implementation Plan

#### 1. Config Addition
```yaml
# In configs/base.yaml or experiment config
require_all_suspects: true  # Enable/disable the requirement
```

#### 2. New Wrapper Class
Create `AllSuspectsWrapper` in `rrmc/methods/stopping_rules.py`:

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
        enabled: bool = True,
    ):
        super().__init__(inner_rule.max_turns)
        self.inner_rule = inner_rule
        self.enabled = enabled
    
    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        # Only apply to DC task
        if self.enabled and task_type.lower() in ["dc", "detective_cases"]:
            suspects_questioned = self._get_questioned_suspects(state)
            all_suspects = self._get_all_suspects(state)
            
            if suspects_questioned != all_suspects:
                # Not all suspects questioned yet - force continue
                remaining = all_suspects - suspects_questioned
                return StoppingDecision(
                    should_stop=False,
                    reason=f"not_all_suspects_questioned",
                    score=len(remaining),
                    prediction=None,
                    metadata={"remaining_suspects": list(remaining)},
                )
        
        # All suspects questioned (or not DC task) - delegate to inner rule
        return self.inner_rule.should_stop(task_type, state, turn)
    
    def _get_questioned_suspects(self, state: Dict[str, Any]) -> Set[str]:
        """Extract set of suspects that have been questioned from history."""
        record = state.get("record", [])
        return {r.get("suspect") for r in record if r.get("suspect")}
    
    def _get_all_suspects(self, state: Dict[str, Any]) -> Set[str]:
        """Get full set of suspects from puzzle info."""
        # Option 1: From state directly
        if "suspects" in state:
            return set(state["suspects"])
        
        # Option 2: Parse from initial_info (puzzle description)
        # This requires extracting suspect names from the puzzle text
        initial_info = state.get("initial_info", "")
        # Implementation depends on puzzle format
        # For now, assume 5 suspects labeled by index or name
        return self._parse_suspects_from_puzzle(initial_info)
    
    def _parse_suspects_from_puzzle(self, initial_info: str) -> Set[str]:
        """Parse suspect names/IDs from puzzle description."""
        # TODO: Implement based on actual puzzle format
        # For DC task, suspects are typically named characters
        # May need regex or structured parsing
        pass
    
    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        return self.inner_rule.get_best_answer(task_type, state)
```

#### 3. Integration in Pipeline
Modify `pipeline.py` to wrap stopping rules:

```python
def _create_method(self, method_name: str, method_cfg: Dict[str, Any]):
    # Create the base stopping rule
    stopping_rule = get_method(method_name, **kwargs)
    
    # Optionally wrap with AllSuspectsWrapper
    if self.config.get("require_all_suspects", False):
        from rrmc.methods.stopping_rules import AllSuspectsWrapper
        stopping_rule = AllSuspectsWrapper(
            inner_rule=stopping_rule,
            enabled=True,
        )
    
    return stopping_rule
```

#### 4. Environment State Enhancement
Ensure the environment provides suspect information:

```python
# In rrmc/core/environment.py
def _get_state(self) -> Dict[str, Any]:
    return {
        "initial_info": self.current_puzzle["background"],
        "history": self.history,
        "history_string": self._format_history(),
        "record": self.record,
        "suspects": self._get_suspect_list(),  # ADD THIS
        ...
    }

def _get_suspect_list(self) -> List[str]:
    """Extract suspect names from current puzzle."""
    # Parse from puzzle data structure
    return self.current_puzzle.get("suspects", [])
```

### Testing Plan
1. Run with `require_all_suspects: false` - should behave as before
2. Run with `require_all_suspects: true` - should force 5+ turns
3. Compare accuracy between the two modes

### Expected Outcome
- Minimum 5 turns per puzzle (one per suspect)
- Better information gathering before decision
- Hopefully improved accuracy

---

## Option C: Partial Trajectory Sampling

### Description
Run multiple parallel "games" for a fixed number of turns, then check if answers converge.

### Implementation Sketch
```python
def partial_trajectory_sample(puzzle, n_trajectories=6, turns_per_batch=3):
    trajectories = [Trajectory() for _ in range(n_trajectories)]
    
    while not converged(trajectories):
        # Run each trajectory for `turns_per_batch` turns
        for traj in trajectories:
            for _ in range(turns_per_batch):
                traj.step()
        
        # Check convergence
        answers = [t.current_answer for t in trajectories]
        if majority_agrees(answers, threshold=4):
            return majority_answer(answers)
    
    return majority_answer([t.final_answer for t in trajectories])
```

### Not Chosen Because
- Middle-ground complexity without full benefits
- Still requires significant refactoring
- Full trajectory sampling is more principled

---

## Option D: Full Trajectory Sampling ✓ CHOSEN

### Description
Run N complete, independent investigation trajectories from start to finish. Each trajectory makes its own decisions about when to stop. Final answer is determined by majority vote across trajectory outcomes.

### Design Goals
1. **True diversity:** Each trajectory explores different question paths
2. **Independent stopping:** Each trajectory uses its own stopping criterion
3. **Ensemble decision:** Final answer from trajectory consensus
4. **Configurable:** Number of trajectories, consensus threshold

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TrajectoryEnsemble                        │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Traj 1   │  │ Traj 2   │  │ Traj 3   │  │ Traj N   │    │
│  │ seed=0   │  │ seed=1   │  │ seed=2   │  │ seed=N-1 │    │
│  │          │  │          │  │          │  │          │    │
│  │ T1→T2→T5 │  │ T1→T8    │  │ T1→T3→T6 │  │ ...      │    │
│  │ Ans: A   │  │ Ans: A   │  │ Ans: B   │  │ Ans: A   │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
│                                                              │
│  Consensus: 3/4 say A → Final Answer: A                     │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Plan

#### 1. Config Options
```yaml
# Trajectory ensemble settings
trajectory_ensemble:
  enabled: true
  n_trajectories: 6           # Number of parallel trajectories
  consensus_threshold: 0.5    # Fraction needed for consensus (0.5 = majority)
  early_consensus: true       # Stop early if consensus reached before all finish
  diversity_seeds: true       # Use different random seeds per trajectory
  inner_stopping_rule: "mi_only"  # Which stopping rule each trajectory uses
  inner_threshold: 0.1        # Threshold for inner stopping rule
```

#### 2. New TrajectoryEnsemble Class
Create `rrmc/methods/trajectory_ensemble.py`:

```python
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import Counter
import random

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
    
    @property
    def consensus_ratio(self) -> float:
        return self.consensus_count / self.total_trajectories


class TrajectoryEnsemble:
    """
    Run multiple independent trajectories and aggregate results.
    
    Each trajectory is a complete investigation from start to finish,
    using its own stopping rule instance and random seed for diversity.
    """
    
    def __init__(
        self,
        env_factory,           # Factory to create fresh environment instances
        stopping_rule_factory, # Factory to create stopping rule instances
        n_trajectories: int = 6,
        consensus_threshold: float = 0.5,
        early_consensus: bool = True,
        max_turns: int = 25,
    ):
        self.env_factory = env_factory
        self.stopping_rule_factory = stopping_rule_factory
        self.n_trajectories = n_trajectories
        self.consensus_threshold = consensus_threshold
        self.early_consensus = early_consensus
        self.max_turns = max_turns
    
    def run(self, puzzle_idx: int) -> EnsembleResult:
        """Run ensemble of trajectories on a single puzzle."""
        results: List[TrajectoryResult] = []
        
        for traj_id in range(self.n_trajectories):
            # Create fresh environment and stopping rule for each trajectory
            env = self.env_factory()
            env.reset(puzzle_idx)
            
            stopping_rule = self.stopping_rule_factory(seed=traj_id)
            
            # Run single trajectory
            result = self._run_single_trajectory(
                env=env,
                stopping_rule=stopping_rule,
                traj_id=traj_id,
            )
            results.append(result)
            
            # Check for early consensus
            if self.early_consensus and len(results) >= 3:
                answers = [r.final_answer for r in results]
                counter = Counter(answers)
                most_common, count = counter.most_common(1)[0]
                
                # If we already have majority, can stop early
                remaining = self.n_trajectories - len(results)
                if count > (self.n_trajectories * self.consensus_threshold):
                    # Already have consensus, fill remaining with None
                    break
        
        return self._aggregate_results(results)
    
    def _run_single_trajectory(
        self,
        env,
        stopping_rule,
        traj_id: int,
    ) -> TrajectoryResult:
        """Run a single trajectory from start to finish."""
        turn = 0
        history = []
        
        while turn < self.max_turns:
            turn += 1
            state = env.get_state()
            
            # Check stopping condition
            decision = stopping_rule.should_stop(
                task_type=env.task_type,
                state=state,
                turn=turn,
            )
            
            if decision.should_stop:
                return TrajectoryResult(
                    trajectory_id=traj_id,
                    final_answer=decision.prediction,
                    turns_used=turn,
                    history=history,
                    confidence=1.0 - decision.score if decision.score < 1 else 0.0,
                    stopping_reason=decision.reason,
                )
            
            # Generate and execute question
            question = self._generate_question(env, state, seed=traj_id + turn)
            feedback = env.step(question)
            history.append({"turn": turn, "question": question, "feedback": feedback})
        
        # Hit max turns
        final_answer = stopping_rule.get_best_answer(env.task_type, env.get_state())
        return TrajectoryResult(
            trajectory_id=traj_id,
            final_answer=final_answer,
            turns_used=self.max_turns,
            history=history,
            confidence=0.0,
            stopping_reason="max_turns",
        )
    
    def _generate_question(self, env, state, seed: int) -> str:
        """Generate next question with seed-based diversity."""
        # Use seed to influence question generation
        # This could be temperature variation, prompt variation, etc.
        random.seed(seed)
        # ... question generation logic
        pass
    
    def _aggregate_results(self, results: List[TrajectoryResult]) -> EnsembleResult:
        """Aggregate trajectory results into final answer."""
        answers = [r.final_answer for r in results]
        counter = Counter(answers)
        most_common, count = counter.most_common(1)[0]
        
        return EnsembleResult(
            final_answer=most_common,
            consensus_count=count,
            total_trajectories=len(results),
            confidence=count / len(results),
            trajectory_results=results,
        )
```

#### 3. Integration Points

**Environment Factory:**
```python
def create_env_factory(config, puzzle_data):
    def factory():
        return ARBenchEnv(
            task_type=config["task"],
            data_path=puzzle_data,
        )
    return factory
```

**Stopping Rule Factory:**
```python
def create_stopping_rule_factory(config, llm, clusterer):
    def factory(seed: int):
        # Create stopping rule with seed-based variation
        rule = get_method(
            config["inner_stopping_rule"],
            llm=llm,
            clusterer=clusterer,
            threshold=config["inner_threshold"],
        )
        # Optionally set seed for any random components
        return rule
    return factory
```

#### 4. Diversity Mechanisms

To ensure trajectories explore different paths:

1. **Temperature variation:** Different temperatures per trajectory
   ```python
   temperatures = [0.5, 0.7, 0.9, 1.0, 0.6, 0.8]
   ```

2. **Question selection variation:** Randomly select among top-k questions
   ```python
   # Instead of always picking best question, sample from top-3
   question = random.choice(top_k_questions[:3])
   ```

3. **Prompt variation:** Use different prompt phrasings per trajectory

4. **Suspect order variation:** Query suspects in different orders

### API Cost Analysis

| Config | Trajectories | Avg Turns | Samples/Turn | Total API Calls |
|--------|--------------|-----------|--------------|-----------------|
| Current | 1 | 1 | 6 | 6 |
| Ensemble (6 traj) | 6 | 5 | 1 | 30 |
| Ensemble + sampling | 6 | 5 | 6 | 180 |

**Recommendation:** Use single-sample per turn within each trajectory to control costs.

### Testing Plan
1. Unit test `TrajectoryEnsemble` with mock environment
2. Integration test on 5 puzzles with n_trajectories=3
3. Full validation on 20 puzzles with n_trajectories=6
4. Compare accuracy vs single-trajectory baseline

### Expected Benefits
- Better exploration of question space
- More robust final answers through consensus
- Reduced impact of unlucky early guesses
- Natural uncertainty quantification (consensus ratio)

---

## Implementation Priority

1. **Phase 1:** Option B (Ask All Suspects) - Quick win, low cost
2. **Phase 2:** Option D (Trajectory Ensemble) - Bigger improvement, higher cost

## Next Steps

- [ ] Implement `AllSuspectsWrapper` class
- [ ] Add `require_all_suspects` config option
- [ ] Update environment to expose suspect list
- [ ] Test Option B on 20 puzzles
- [ ] Design `TrajectoryEnsemble` class
- [ ] Implement trajectory diversity mechanisms
- [ ] Benchmark API cost vs accuracy trade-off
