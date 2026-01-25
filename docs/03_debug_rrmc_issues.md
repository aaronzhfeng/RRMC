# RRMC Debug Notes

Date: 2026-01-23

## Summary
This note captures the main implementation issues discovered in RRMC and the
corresponding fixes applied in the codebase.

## Issues Diagnosed

1) Parallel evaluation shared a single stopping rule instance across threads.
   - Impact: data races and cross-puzzle contamination for stateful rules
     such as `SelfConsistencyStopping`, `SemanticEntropyStopping`, and
     `RobustMIStopping`.

2) `base_url` from config was never passed to `LLMWrapper`.
   - Impact: API requests always went to the OpenRouter default even if
     `base_url` was set in configs.

3) `question_generator` argument in `run_episode()` was ignored.
   - Impact: external question generators could not be injected as intended.

4) GN solved via `ActionASK` did not populate `prediction`/`ground_truth`.
   - Impact: result logs were missing fields for those episodes.

## Fixes Applied

1) Added per-thread cloning of stopping rules in parallel evaluation.
   - A new `_clone_stopping_rule()` helper constructs a fresh instance for each
     puzzle to avoid shared mutable state.

2) Passed `base_url` into `LLMWrapper` during pipeline initialization.

3) Wired `question_generator` into `run_episode()` with a safe fallback to the
   default generator.

4) Included `prediction` and `ground_truth` in GN `ActionASK` success info.

## Files Changed

- `RRMC/pipeline.py`: pass `base_url` to `LLMWrapper`.
- `RRMC/rrmc/evaluation/evaluator.py`: clone stopping rules for parallel runs;
  honor `question_generator`.
- `RRMC/rrmc/core/environment.py`: add GN `prediction`/`ground_truth` on solve.

## Follow-up Checks

- Run a small parallel evaluation and confirm stable metrics across runs.
- Verify custom `base_url` targets the intended API endpoint.
- If adding new stopping rules, ensure they can be cloned safely (either
  implement `clone()` or update `_clone_stopping_rule()`).
