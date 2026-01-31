# RRMC Documentation

## Specification & Research

| File | Description |
|------|-------------|
| **`00_proposal.md`** | Implementation-ready spec (algorithms, baselines, metrics, ablations). See Section 7 for the implementation checklist. |
| **`01_literature.md`** | Curated literature map (157 papers) supporting the design and baselines. |

## Implementation

| File | Description |
|------|-------------|
| **`02_implementation.md`** | Current implementation progress, code map, and how to run. |
| **`03_debug_rrmc_issues.md`** | Debug notes for RRMC issues. |

## Task-Specific Fixes

| File | Description |
|------|-------------|
| **`04_gn_task_fixes.md`** | GN (Guessing Numbers) task fixes - format, extraction, history tracking. |
| **`05_model_comparison.md`** | Model comparison guide - which models work with RRMC. |

## Session Logs

| File | Description |
|------|-------------|
| **`06_session_progress_20260127.md`** | Session progress (Jan 26-27, 2026) - GN fixes, model testing. |
| **`07_session_progress_20260130.md`** | Session progress (Jan 30, 2026) - Grid search, validation analysis. |

## Proposed Improvements

| File | Description |
|------|-------------|
| **`08_proposed_improvements.md`** | Proposed fixes for Turn-1 stopping problem: (A) prompt-based, (B) code-enforced suspect questioning, (C) partial trajectory, (D) full trajectory ensemble. |

## Execution Logs

Detailed task execution history is tracked in `../prompt/`:
- `00_implementation_plan.md` — Canonical build checklist
- `01_implementation_status.md` — Initial implementation audit
- `02_fix_calibration_issues.md` — Calibration bug fixes
- `03_complete_mve.md` — Diversity sampling, homogeneous regime, correlation
- `04_requirements_and_fixes.md` — Dependencies and output fixes

---

## Quick Start for New Agents

### Current Status
- **GN task**: Pipeline works, 0% accuracy (model capability limit)
- **DC task**: 35% accuracy with Qwen3-8B
- **Model**: `mistralai/mistral-small-3.1-24b-instruct:free`

### Key Commands
```bash
# Run GN baseline
cd /root/RRMC && python run.py fixed_turns --task gn --n_puzzles 5 --max_turns 10

# Run DC baseline
cd /root/RRMC && python run.py fixed_turns --task dc --n_puzzles 10 --max_turns 15

# Run all methods
cd /root/RRMC && python run.py all_methods --task gn --n_puzzles 20 --max_turns 25
```

### Model Configuration
Edit `/root/RRMC/configs/base.yaml` to change `policy_model`.

Recommended models (no thinking mode):
- `qwen/qwen-2.5-7b-instruct` - Fast, no rate limits
- `mistralai/mistral-small-3.1-24b-instruct:free` - Larger
- `meta-llama/llama-3.3-70b-instruct:free` - Best but rate limited

