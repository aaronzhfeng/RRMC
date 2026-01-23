# RRMC Implementation Status

This repo implements the backbone of **Risk-Controlled Robust‑MI Active Inquiry**.
Below is a more detailed snapshot of the current implementation, gaps vs the proposal,
and the most efficient next steps to reach a publishable MVE.

## Code Map (Where Things Live)

- `rrmc/core/llm.py` — OpenRouter LLM wrapper + simple `.env` loader.
- `rrmc/core/environment.py` — AR‑Bench environment wrapper (DC/SP/GN) + `get_ground_truth()`.
- `rrmc/core/clustering.py` — semantic clustering + optional entailment refinement.
- `rrmc/core/mi_estimator.py` — self‑revision MI + robust MI (prompt variants) + `compute_homogeneity_score()`.
- `rrmc/core/calibration.py` — Clopper–Pearson risk‑controlled thresholding + `StoppingController`.
- `rrmc/baselines/stopping_rules.py` — fixed turns, self‑consistency, entropy, MI‑only, robust MI.
- `rrmc/evaluation/evaluator.py` — evaluation harness + **calibration pipeline** + comparison runner.
- `run_rrmc.py` — CLI entrypoint with **calibration mode** support.
- `test_*.py` — minimal local sanity tests.

## Implemented (Core Infrastructure)

- LLM wrapper (OpenRouter) and AR‑Bench environment wrapper (DC/SP/GN).
- Self‑revision MI estimator and robust MI (prompt variants: base/skeptical/alternative).
- Risk‑controlled calibration class (Clopper–Pearson UCB).
- Baseline stopping rules and evaluation harness.
- ✅ **Risk‑controlled calibration pipeline** — fully wired into evaluation.
- ✅ **Calibration state collection** — `collect_calibration_states()` method.
- ✅ **End-to-end calibrated evaluation** — `run_calibrated_evaluation()` method.
- ✅ **CLI support for calibration** — `--calibrate`, `--n_train`, `--target_error`, `--load_calibration`.

## Procedure Correctness (Already Fixed)

- `get_state()` no longer resets episodes (history now persists across turns).
- Task type strings are mapped correctly to `"DC" | "SP" | "GN"` for MI and clustering.
- `get_ground_truth()` added to environment for calibration correctness checks.

## Now Implemented ✅

- **Risk‑controlled stopping is now active** in calibration mode.
  Use `--calibrate` to collect states on train split, compute optimal τ via Clopper-Pearson UCB,
  and evaluate with the calibrated threshold on test split.
- **Calibration state persistence** — saves/loads calibration data to JSON.
- **Homogeneity tracking** — recorded during calibration for diagnostics.

## Still Missing vs Proposal

- **VoI question selection** (expected MI reduction) and **DPP slate selection**.
- **C‑IP / KnowNo / UoT / EIG** baselines.
- **Homogeneity diagnostics** (effective rank, repetition, prompt sensitivity plots).
- **Proper SP scoring** (current SP evaluation is simplified).
- **Diversity sampling and importance weights** in MI sampling (code exists but not applied).

## Performance / Cost Notes (Why tests sometimes die)

- MI is expensive: each decision uses `k_samples × variants × 2` LLM calls.
- OpenRouter responses are slow (often 10–30s per call).
- For dev, keep small `k_samples`, low `max_turns`, and tiny puzzle subsets.
- `sentence-transformers` is optional; clustering falls back to exact matching if missing.

## How to Run

### Quick Test (Fixed Thresholds)
```bash
python3.13 run_rrmc.py --n_puzzles 1 --methods fixed_turns --max_turns 3 --task dc
```

### Full Calibrated RRMC Pipeline ✅
```bash
# Run calibration on train split, then evaluate on test split
python3.13 run_rrmc.py --calibrate --n_train 10 --n_test 5 --task dc --target_error 0.10

# Load existing calibration and evaluate
python3.13 run_rrmc.py --load_calibration results/calibration_dc_xxx.json --n_test 5
```

### Calibration Options
- `--calibrate` — Enable calibration mode
- `--n_train N` — Number of train puzzles for collecting calibration states
- `--n_test N` — Number of test puzzles for evaluation
- `--target_error RATE` — Target error rate δ for Clopper-Pearson UCB (default: 0.10)
- `--load_calibration PATH` — Load existing calibration data instead of collecting new
- `--k_samples N` — Number of MI samples (default: 4)
- `--variants LIST` — Prompt variants for robust MI (default: base,skeptical)

## Recommended Next Steps (MVE Path)

1. ✅ ~~Wire risk-controlled calibration into evaluation~~ **DONE**
2. **Run calibrated DC evaluation** with 10-20 train puzzles, 10 test puzzles
3. **Compare methods**: `rrmc_calibrated` vs `fixed_turns`, `self_consistency`, `mi_only`
4. **Report**: accuracy vs average turns + calibration statistics

## After MVE (Full System)

- Implement VoI question selection with expected MI reduction
- Add C-IP/KnowNo baselines for direct comparison
- Implement homogeneity diagnostics and stratified analysis
- Scale to SP task with proper F1 evaluation

