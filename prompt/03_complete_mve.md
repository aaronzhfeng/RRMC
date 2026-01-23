# Task: Complete MVE with Diversity Sampling and Evaluation Report

## Context

The RRMC implementation has core infrastructure working:
- Self-revision MI estimator with prompt variants (base/skeptical/alternative)
- Risk-controlled calibration via Clopper-Pearson UCB
- Baselines: fixed_turns, self_consistency, semantic_entropy, mi_only, robust_mi
- Evaluation harness with calibration pipeline

**What's missing for MVE completion:**
1. Diversity sampling is implemented (`DiversitySampler`) but not wired into MI estimation
2. No "homogeneous decoding regime" for stress-testing (low temp to induce collapse)
3. No MI-error correlation analysis in the evaluation output

## Tasks

### 1. Wire Diversity Sampling into MI Estimation

In `rrmc/core/mi_estimator.py`, the `RobustMI` class has `use_diversity_sampling` flag but `DiversitySampler` isn't actually used during sampling.

**Changes needed:**
- In `SelfRevisionMI.estimate()`, optionally use diversity-filtered sampling for generating initial/revised answers
- Track importance weights from rejection sampling
- Use weighted counts in `_compute_mi()` (the `weights` parameter already exists)

Keep it simple: just add a `diversity_sampler` parameter and use it when provided.

### 2. Add Homogeneous Decoding Regime

Add support for a "collapse" regime that induces mode collapse for stress testing:
- `temperature=0.1` (or 0.0)
- `top_p=1.0`
- Optional: system prompt "Be decisive. Provide one best answer. Do not hedge."

**Changes needed:**
- Add `regime` parameter to `RobustMI` and `SelfRevisionMI` (default: "normal", options: "normal", "homogeneous")
- When regime="homogeneous", use low temperature settings
- Update `run_rrmc.py` CLI to accept `--regime` flag

### 3. Add MI-Error Correlation to Evaluation

In `rrmc/evaluation/evaluator.py`, add analysis that computes:
- Spearman correlation between MI scores and error indicators across all visited states
- Output this in the results JSON and print in comparison table

**Changes needed:**
- After calibration state collection, compute `scipy.stats.spearmanr(scores, errors)`
- Add to calibration result output
- Print correlation in the summary

### 4. Update run_rrmc.py for MVE

Add CLI options:
- `--regime normal|homogeneous` (default: normal)
- Ensure the evaluation report includes MI-error correlation

## Verification

After changes, run a small test:
```bash
python3 run_rrmc.py --calibrate --n_train 5 --n_test 3 --task dc --max_turns 5 --regime normal
```

Should complete without errors and output:
- Calibration summary with threshold τ
- Comparison table with accuracy/turns for each method
- MI-error correlation coefficient

## Files to Modify

1. `rrmc/core/mi_estimator.py` — wire diversity sampling, add regime parameter
2. `rrmc/evaluation/evaluator.py` — add correlation analysis
3. `run_rrmc.py` — add --regime CLI flag

## Notes

- Keep changes minimal and focused
- Don't add conformal baselines yet (future task)
- The goal is a working MVE that can produce publishable comparison numbers
