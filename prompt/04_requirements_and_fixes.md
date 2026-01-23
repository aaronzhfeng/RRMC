# Task: Add Requirements and Fix MI-Error Correlation Output

## Context

The MVE implementation is complete (see `03_complete_mve.md`), but review found:
1. No `requirements.txt` documenting dependencies
2. MI-error correlation is computed but not included in `results["calibration"]` dict in `run_rrmc.py`

## Tasks

### 1. Create requirements.txt

Create `/root/RRMC/requirements.txt` with the required dependencies:

```
numpy
scipy
requests
sentence-transformers
```

Note: `sentence-transformers` is optional (clustering falls back to exact matching), but include it for full functionality.

### 2. Fix MI-Error Correlation in run_rrmc.py

In `run_rrmc.py`, the manual calibration path builds `results["calibration"]` without `mi_error_correlation`, but then tries to read it for the comparison table.

**Location:** Around line 336-365 in `run_rrmc.py`

**Fix:** After calling `calibrate_threshold()`, compute the correlation and add it to the results dict:

```python
# After calibration_result = train_evaluator.calibrate_threshold(...)
mi_corr = train_evaluator._compute_mi_error_correlation(calibrator)

results = {
    "calibration": {
        "threshold": calibration_result.threshold,
        "n_states": calibration_result.n_states,
        "n_covered": calibration_result.n_covered,
        "empirical_error": calibration_result.empirical_error,
        "ucb_error": calibration_result.ucb_error,
        "mi_error_correlation": mi_corr,  # ADD THIS
    },
    "evaluation": eval_results,
}
```

### 3. Update docs/02_implementation.md

Add a "Dependencies" section listing the requirements.

## Verification

```bash
# Check requirements.txt exists
cat /root/RRMC/requirements.txt

# Run calibration and verify MI-error correlation is printed
python3 run_rrmc.py --calibrate --n_train 3 --n_test 2 --task dc --max_turns 3
```

Should output MI-error correlation (ρ and p-value) in the summary.

## Files to Modify

1. Create: `requirements.txt`
2. Modify: `run_rrmc.py` — add mi_error_correlation to results dict
3. Modify: `docs/02_implementation.md` — add Dependencies section
