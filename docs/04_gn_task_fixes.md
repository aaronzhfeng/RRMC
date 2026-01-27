# GN (Guessing Numbers) Task Fixes

## Date: January 26-27, 2026

## Overview

This document captures the fixes made to get the Guessing Numbers (GN) task working correctly in RRMC.

---

## Problems Identified

### 1. Qwen3-8B "Thinking Mode" Issue
- **Problem**: Qwen3-8B outputs reasoning text before answers regardless of prompts
- **Symptom**: Model outputs "Okay, let's see. The user wants me to..." instead of "Guess: 1234"
- **Root Cause**: Qwen3 models are trained with thinking-first paradigm (similar to o1/DeepSeek-R1)
- **Solution**: Switch to Qwen 2.5-7B (same family, pre-thinking era)

### 2. Feedback Format Misalignment
- **Problem**: RRMC used custom feedback format, not AR-Bench's standard
- **Location**: `rrmc/evaluation/evaluator.py` line ~705
- **Fix**: Changed to use `GN_EVAL_PROMPT.format(same_pos=bulls, diff_pos=cows)`

### 3. Final Answer Extraction
- **Problem**: Extraction picked up example numbers from reasoning (e.g., "Turn 1 (0123)..." → extracted "0123")
- **Location**: `rrmc/methods/stopping_rules.py` `_parse_answer()` method
- **Fix**: 
  - Added explicit patterns for "Guess: XXXX" and "Final Answer: XXXX"
  - Changed to return LAST valid 4-digit number (after analysis)
  - Added uniqueness validation

### 4. CLI `--n_puzzles` Argument Ignored
- **Problem**: `--n_puzzles 10` was ignored, running 20 puzzles instead
- **Root Cause**: CLI used `n_test`, config used `n_puzzles` - mismatch
- **Location**: `run.py` and `pipeline.py`
- **Fix**: Modified `run.py` to set both `n_test` and `n_puzzles` in overrides

### 5. History Not Saved in Results
- **Problem**: Episode history was empty in JSON output despite being tracked
- **Location**: `rrmc/evaluation/evaluator.py` JSON serialization
- **Fix**: Added `"history": ep.history` to the episode serialization

---

## Code Changes Summary

### `rrmc/evaluation/evaluator.py`

1. **GN Feedback Format** (~line 705):
```python
# Old:
feedback = f"{h.get('bulls', 0)} Bulls, {h.get('cows', 0)} Cows"

# New:
same_pos = h.get('bulls', 0)
diff_pos = h.get('cows', 0)
feedback = GN_EVAL_PROMPT.format(same_pos=same_pos, diff_pos=diff_pos)
```

2. **Guess Extraction** (`_extract_guess` method):
```python
def _extract_guess(self, response_text: str) -> Optional[str]:
    # 1. Look for "Guess: XXXX" pattern
    # 2. Look for intent phrases ("I'll guess", "my guess is")
    # 3. Return LAST 4-digit number with unique digits
    # 4. Fallback to any 4-digit number
```

3. **History Serialization**:
```python
"episodes": [
    {
        # ... other fields ...
        "history": ep.history,  # Added this line
    }
    for ep in result.episode_results
],
```

4. **Few-shot Template**: Updated `GN_PROPOSE_TEMPLATE` to include example trajectory

### `rrmc/methods/stopping_rules.py`

1. **GN Answer Prompt** (`_get_answer_prompt`):
```python
# Includes game rules + previous guesses + "Guess: [number]" format
```

2. **Answer Parsing** (`_parse_answer`):
```python
# Added explicit pattern matching for final answers
# Returns LAST valid 4-digit unique number
# Fallback to random valid guess if needed
```

### `run.py`

1. **CLI Alias Handling**:
```python
# Handle n_puzzles as alias for n_test
if n_puzzles is not None and n_test is None:
    n_test = n_puzzles

# Set both in overrides
if n_test is not None:
    overrides["n_test"] = n_test
    overrides["n_puzzles"] = n_test
```

---

## Current State

### Working Correctly:
- ✅ Format compliance - model outputs clean "Guess: XXXX"
- ✅ Guesses vary each turn (9/9 unique per episode)
- ✅ History tracked and saved in results
- ✅ CLI arguments work correctly
- ✅ Token usage reduced (222K → 25K with Qwen 2.5)

### Model Performance:
| Model | Format | Guessing | Accuracy |
|-------|--------|----------|----------|
| Qwen3-8B | ❌ Thinking | Repeated "0123" | 0% |
| Qwen 2.5-7B | ✅ Clean | Varies correctly | 0% |
| Llama 3.3 70B | ✅ Clean | Smarter | 0% (rate limited) |

### Why 0% Accuracy?
- GN (Bulls & Cows) is a hard logical deduction task
- Requires inferring 4-digit number from feedback
- Small models (7B-8B) struggle with this reasoning
- Best bulls achieved: 1-2 out of 4 needed
- **This is a model capability limitation, not a code bug**

---

## Recommended Next Steps

1. **For GN experiments**: Use larger models (70B+) but expect rate limits on free tiers
2. **For baselines**: DC task achieves 35% accuracy and works well
3. **For production**: Consider paid API tiers to avoid rate limits

---

## Files Modified

- `/root/RRMC/rrmc/evaluation/evaluator.py`
- `/root/RRMC/rrmc/methods/stopping_rules.py`
- `/root/RRMC/run.py`
- `/root/RRMC/configs/base.yaml`
