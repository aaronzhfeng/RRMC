# Calibration Fix Instructions (RRMC)

These changes fix three correctness issues in the calibration pipeline.

## 1) DC prediction normalization (index vs letter/name)

**Problem:** DC ground truth is an index (0–4), but predictions are often letters or names.  
**Fix:** Normalize prediction to an index before `add_state()`.

**Applied in:** `rrmc/evaluation/evaluator.py`  
Inside `collect_calibration_states()`:
- Convert DC prediction to index using `self.env._parse_dc_answer(...)`.
- Normalize GN digits to ensure consistent comparison.

## 2) SP calibration should not use exact string match

**Problem:** SP uses exact string equality, which is too strict.  
**Fix:** Use character‑level F1 with a threshold (default 0.5).

**Applied in:** `rrmc/core/calibration.py`  
Adds `_char_f1()` and `sp_f1_threshold` in `RiskControlledCalibrator`.

## 3) MI score vs prediction mismatch

**Problem:** `get_best_answer()` re‑samples, so `(score, prediction)` are inconsistent.  
**Fix:** Reuse cached `_last_estimates` from the last MI call.

**Applied in:** `rrmc/baselines/stopping_rules.py`  
`RobustMIStopping.get_best_answer()` now uses cached estimates when available.

---


## Proposal alignment checklist

- Sequential questioning is expected. The agent can ask many questions over multiple turns.
- Calibration uses on-policy visited states. Keep asking during calibration to collect states.
- Question generation is fixed in the MVE. Only the stop/answer rule changes.

---

## Optional next implementation steps (to improve performance)

1. **Use diversity sampling in robust MI**
   - Wire `DiversitySampler` into `RobustMI.estimate()` so the samples used for MI
     are diversity‑filtered and importance‑weighted.
2. **SP evaluation with AR‑Bench F1**
   - Replace the simplified SP F1 with AR‑Bench’s `f1_score` logic for evaluation.
3. **Calibration sanity metrics**
   - Log `n_states`, `n_covered`, and UCB error by task to catch calibration drift.

---

## Next sanity check

Run a tiny calibration pass to verify:

```bash
python3 run_rrmc.py --calibrate --n_train 2 --n_test 1 --task dc --max_turns 3 --k_samples 2
```

If the run completes and produces a calibration JSON, the wiring is correct.

## Performance knobs (safe defaults)

- `--k_samples 2` and `--variants base` for fast smoke tests.
- `--max_turns 3` to reduce total API calls.
- Scale up gradually once the pipeline completes end‑to‑end.
