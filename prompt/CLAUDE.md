# Task: Restructure RRMC for Config-Based Experiments

## Goal

Restructure RRMC to use a modular, YAML-based config system like AR-Bench_Test and PersonaBench. This enables:
- Easy experiment presets (`python run.py mve_dc_normal`)
- Swappable methods via config (not code changes)
- Reproducible experiments via frozen configs
- Simple ablation studies

## Current Structure

```
RRMC/
├── run_rrmc.py              # All config via CLI args
├── rrmc/
│   ├── core/                # MI, calibration, env, clustering
│   ├── baselines/           # stopping_rules.py with hardcoded methods
│   └── evaluation/          # evaluator.py with _create_stopping_rule()
└── AR-Bench/
```

## Target Structure

```
RRMC/
├── run.py                   # Thin entry point
├── config.py                # YAML loading + merge logic
├── pipeline.py              # Orchestration (extracted from run_rrmc.py)
├── configs/
│   ├── base.yaml            # Defaults (model, sampling, max_turns, etc.)
│   ├── providers/
│   │   └── openrouter.yaml  # API settings
│   ├── methods/
│   │   ├── fixed_turns.yaml
│   │   ├── self_consistency.yaml
│   │   ├── semantic_entropy.yaml
│   │   ├── mi_only.yaml
│   │   └── robust_mi.yaml
│   └── experiments/
│       ├── mve_dc_normal.yaml
│       ├── mve_dc_homogeneous.yaml
│       ├── mve_sp_normal.yaml
│       └── ablation_no_diversity.yaml
├── rrmc/
│   ├── core/                # (unchanged)
│   ├── methods/             # Renamed from baselines/
│   │   ├── __init__.py      # METHODS registry dict
│   │   ├── base.py          # BaseStoppingRule (moved)
│   │   └── stopping_rules.py
│   └── evaluation/
│       ├── evaluator.py     # Simplified (uses registry)
│       └── pipeline.py      # (optional, or keep at root)
├── AR-Bench/
├── docs/
├── prompt/
└── results/
```

## Tasks

### 1. Create configs/ Directory Structure

Create the following files:

**configs/base.yaml:**
```yaml
# Default settings
task: dc
max_turns: 25
data_subset: test

# Model settings
policy_model: qwen/qwen3-8b
api_key: ${OPENROUTER_API_KEY}
base_url: https://openrouter.ai/api/v1

# Sampling defaults
temperature: 0.7
top_p: 0.95

# MI estimation defaults
k_samples: 4
variants:
  - base
  - skeptical

# Calibration defaults
target_error: 0.10
confidence_level: 0.95

# Output
output_dir: results
```

**configs/providers/openrouter.yaml:**
```yaml
base_url: https://openrouter.ai/api/v1
api_key: ${OPENROUTER_API_KEY}
```

**configs/methods/robust_mi.yaml:**
```yaml
method: robust_mi
use_diversity_sampling: true
regime: normal
threshold: 0.3  # Default, overridden by calibration
```

**configs/experiments/mve_dc_normal.yaml:**
```yaml
# MVE: Detective Cases with normal decoding
experiment_name: mve_dc_normal
task: dc
data_subset: test

# Methods to compare
methods:
  - fixed_turns
  - self_consistency
  - semantic_entropy
  - mi_only
  - robust_mi

# Calibration settings
calibrate: true
n_train: 20
n_test: 10
regime: normal
```

### 2. Create config.py

Create `config.py` with:
- `load_yaml(path)` - Load single YAML file
- `load_config(experiment_path)` - Merge base → provider → experiment → CLI
- `Config` class with dot-notation access

Reference: AR-Bench_Test/runner.py lines 44-76

### 3. Create Method Registry

In `rrmc/methods/__init__.py`:
```python
from .stopping_rules import (
    FixedTurnsStopping,
    SelfConsistencyStopping,
    SemanticEntropyStopping,
    MIOnlyStopping,
    RobustMIStopping,
)

METHODS = {
    "fixed_turns": FixedTurnsStopping,
    "self_consistency": SelfConsistencyStopping,
    "semantic_entropy": SemanticEntropyStopping,
    "mi_only": MIOnlyStopping,
    "robust_mi": RobustMIStopping,
}

def get_method(name: str, **kwargs):
    """Get stopping rule by name."""
    if name not in METHODS:
        raise ValueError(f"Unknown method: {name}. Available: {list(METHODS.keys())}")
    return METHODS[name](**kwargs)
```

### 4. Create pipeline.py

Extract orchestration logic from `run_rrmc.py` into `pipeline.py`:
- `Pipeline` class that takes config
- `run()` method that handles calibration + evaluation
- Methods loaded via registry, not switch statement

### 5. Create run.py (New Entry Point)

Simple entry point:
```python
#!/usr/bin/env python3
"""
RRMC - Risk-Controlled Robust-MI Active Inquiry

Usage:
    python run.py mve_dc_normal
    python run.py configs/experiments/mve_dc_normal.yaml
    python run.py mve_dc_normal --task sp --max-turns 10
"""
import fire
from config import load_config
from pipeline import Pipeline

def main(
    config: str = None,
    task: str = None,
    max_turns: int = None,
    # ... other CLI overrides
):
    cfg = load_config(config, overrides={...})
    pipeline = Pipeline(cfg)
    results = pipeline.run()
    return results

if __name__ == "__main__":
    fire.Fire(main)
```

### 6. Update evaluator.py

Remove `_create_stopping_rule()` switch statement. Instead:
```python
from rrmc.methods import get_method

# In evaluate loop:
stopping_rule = get_method(method_name, llm=self.llm, clusterer=self.clusterer, **method_config)
```

### 7. Rename baselines/ to methods/

```bash
mv rrmc/baselines/ rrmc/methods/
```

Update imports in evaluator.py accordingly.

## File Changes Summary

| Action | File |
|--------|------|
| Create | `config.py` |
| Create | `pipeline.py` |
| Create | `run.py` |
| Create | `configs/base.yaml` |
| Create | `configs/providers/openrouter.yaml` |
| Create | `configs/methods/*.yaml` (5 files) |
| Create | `configs/experiments/*.yaml` (2-3 files) |
| Move | `rrmc/baselines/` → `rrmc/methods/` |
| Modify | `rrmc/methods/__init__.py` (add registry) |
| Modify | `rrmc/evaluation/evaluator.py` (use registry) |
| Keep | `run_rrmc.py` (deprecated, keep for reference) |

## Verification

After restructure:
```bash
# Run MVE with experiment config
python run.py mve_dc_normal

# Override via CLI
python run.py mve_dc_normal --task sp --max-turns 5

# Should produce same results as:
# python run_rrmc.py --calibrate --n_train 20 --n_test 10 --task dc --regime normal
```

## Notes

- Keep `run_rrmc.py` temporarily for reference (rename to `run_rrmc_legacy.py`)
- Core `rrmc/core/` modules should NOT need changes
- The key is the config layer + method registry
- Follow AR-Bench_Test pattern for config merging
