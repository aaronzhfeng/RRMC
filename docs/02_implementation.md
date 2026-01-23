# RRMC Implementation Progress

**Last updated:** 2026-01-23

This document tracks the implementation progress of the Risk-Controlled Robust-MI Active Inquiry system.

---

## Current State: MVE Infrastructure Complete

The core infrastructure for the Minimum Viable Experiment (MVE) is implemented and functional. The system can:
- Run AR-Bench tasks (DC, SP, GN)
- Estimate self-revision MI with multiple prompt variants
- Calibrate stopping thresholds via Clopper-Pearson UCB
- Compare multiple stopping rules on test puzzles

---

## Code Map

| Module | File | Description |
|--------|------|-------------|
| **Entry Point** | `run.py` | Config-based CLI entry point |
| **Config** | `config.py` | YAML loading, merging, and Config class |
| **Pipeline** | `pipeline.py` | Experiment orchestration |
| **LLM Wrapper** | `rrmc/core/llm.py` | OpenRouter API client with token tracking |
| **Environment** | `rrmc/core/environment.py` | AR-Bench wrapper (DC/SP/GN) with reset/step/get_state |
| **Clustering** | `rrmc/core/clustering.py` | Semantic clustering + optional entailment refinement |
| **MI Estimator** | `rrmc/core/mi_estimator.py` | Self-revision MI + Robust MI (max over variants) |
| **Calibration** | `rrmc/core/calibration.py` | Clopper-Pearson UCB threshold selection |
| **Methods** | `rrmc/methods/` | Stopping rules + registry |
| **Evaluator** | `rrmc/evaluation/evaluator.py` | Episode runner, calibration collection, comparison |

### Configuration Structure

```
configs/
├── base.yaml              # Default settings
├── providers/
│   └── openrouter.yaml    # API settings
├── methods/
│   ├── fixed_turns.yaml
│   ├── self_consistency.yaml
│   ├── semantic_entropy.yaml
│   ├── mi_only.yaml
│   └── robust_mi.yaml
└── experiments/
    ├── mve_dc_normal.yaml
    ├── mve_dc_homogeneous.yaml
    ├── mve_sp_normal.yaml
    └── ablation_no_diversity.yaml
```

---

## Implementation Progress

### Phase 1: Infrastructure ✅
- [x] AR-Bench environment wrapper (DC/SP/GN)
- [x] LLM wrapper (OpenRouter)
- [x] Semantic clustering with entailment support

### Phase 2: Core MI Estimator ✅
- [x] Self-revision MI with task-specific prompts
- [x] Prompt variants (base/skeptical/alternative)
- [x] Robust MI aggregation (max over variants)
- [x] Diversity sampling integration with importance weighting
- [x] Homogeneous decoding regime for stress testing

### Phase 3: Risk-Controlled Calibration ✅
- [x] Calibration state collection on train puzzles
- [x] Clopper-Pearson UCB threshold selection
- [x] Runtime stopping controller

### Phase 4: Baselines (Partial)
- [x] Fixed turns
- [x] Self-consistency gate
- [x] Verbalized confidence gate
- [x] Semantic entropy gate
- [x] MI-only (no robustness)
- [ ] KnowNo-style (conformal set-size)
- [ ] C-IP (conformal information pursuit)
- [ ] UoT (uncertainty of thoughts)
- [ ] EIG (expected information gain)

### Phase 5: Full Method ❌
- [ ] VoI question selection (expected MI reduction)
- [ ] DPP slate selection
- [ ] Candidate question generation with validity filters

### Phase 6: Evaluation (Partial)
- [x] Accuracy/turns metrics
- [x] Comparison table output
- [x] MI-error Spearman correlation analysis
- [ ] Homogeneity diagnostics
- [ ] Risk-coverage curves

---

## Execution History

Detailed task execution logs are in `prompt/`:

| File | Date | Summary |
|------|------|---------|
| `01_implementation_status.md` | 2026-01-22 | Initial implementation audit and code map |
| `02_fix_calibration_issues.md` | 2026-01-23 | Fixed DC prediction normalization, SP F1 threshold, MI caching |
| `03_complete_mve.md` | 2026-01-23 | Wired diversity sampling, added homogeneous regime, MI-error correlation |
| `04_requirements_and_fixes.md` | 2026-01-23 | Created requirements.txt, fixed correlation output in results dict |

---

## Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

Required dependencies:
- `numpy` — Numerical computations
- `scipy` — Statistical functions (Spearman correlation, Clopper-Pearson UCB)
- `requests` — HTTP client for OpenRouter API
- `pyyaml` — YAML configuration loading
- `fire` — CLI argument parsing (optional, has argparse fallback)
- `sentence-transformers` — Semantic embeddings for clustering (optional, falls back to exact matching)

---

## How to Run

### Using Config-Based Entry Point (Recommended)

```bash
# List available experiments
python run.py --list

# Run MVE on Detective Cases
python run.py mve_dc_normal

# Run with CLI overrides
python run.py mve_dc_normal --n_train 5 --n_test 3 --max_turns 5

# Run stress test with homogeneous decoding
python run.py mve_dc_homogeneous

# Run on Situation Puzzles
python run.py mve_sp_normal
```

### Using Legacy CLI (run_rrmc_legacy.py)

```bash
# Quick test
python3 run_rrmc_legacy.py --n_puzzles 1 --methods fixed_turns --max_turns 3 --task dc

# Calibrated pipeline
python3 run_rrmc_legacy.py --calibrate --n_train 10 --n_test 5 --task dc
```

---

## Next Steps

1. **Run MVE evaluation** — Generate comparison numbers for publication
2. **Add conformal baselines** — KnowNo, C-IP for comparison
3. **VoI question selection** — Expected MI reduction for question ranking
4. **Risk-coverage curves** — Visualize calibration performance

See `prompt/CLAUDE.md` for current active task.

---

## References

- **Proposal:** `docs/00_proposal.md` — Full specification with algorithms and baselines
- **Literature:** `docs/01_literature.md` — 157 papers organized by theme
- **Checklist:** `prompt/00_implementation_plan.md` — Canonical build order
