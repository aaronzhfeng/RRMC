# Model Comparison for RRMC

## Date: January 26-27, 2026

## Overview

This document captures findings from testing different LLM models with RRMC, specifically focusing on instruction following and reasoning capabilities.

---

## Models Tested

### 1. Qwen3-8B (`qwen/qwen3-8b`)

**Status**: ❌ Not recommended for RRMC

**Issues**:
- Has "thinking mode" enabled by default
- Always outputs reasoning before answers regardless of prompts
- Example: Asked for "Guess: [number]", outputs "Okay, let's see. The user wants me to..."
- Reasoning appears in `content` field, not separated

**Fix Attempted**:
- Tried `/no_think` suffix - not supported on OpenRouter
- Few-shot prompting - still outputs reasoning
- Explicit instructions - ignored

**Recommendation**: Use Qwen 2.5 instead (same family, no thinking mode)

---

### 2. Qwen 2.5-7B (`qwen/qwen-2.5-7b-instruct`)

**Status**: ✅ Recommended for RRMC

**Strengths**:
- Follows instruction format perfectly
- Outputs clean "Guess: 1234" without reasoning
- Good response speed
- No rate limits on free tier
- Similar size to Qwen3-8B (7B vs 8B)

**Performance on GN**:
- Format compliance: ✅
- Guess variation: ✅ (9/9 unique per episode)
- Token usage: ~25K for 10 puzzles × 10 turns
- Accuracy: 0% (model capability limitation)

**Performance on DC**:
- Accuracy: 35%
- Questions are coherent and relevant
- Suspect selection works correctly

---

### 3. Llama 3.3 70B (`meta-llama/llama-3.3-70b-instruct:free`)

**Status**: ⚠️ Good but rate limited

**Strengths**:
- Excellent instruction following
- Smarter reasoning on GN task
- Better guess strategies (not just sequential)

**Issues**:
- Rate limit: 8 requests per minute on free tier
- Makes parallel evaluation impractical
- Many 429 errors during runs

**Performance on GN**:
- Format compliance: ✅
- Guess quality: Better (uses feedback more effectively)
- Practical for experiments: ❌ (rate limits)

---

### 4. Mistral Small 3.1 24B (`mistralai/mistral-small-3.1-24b-instruct:free`)

**Status**: 🔄 Testing (currently configured)

**Expected**:
- 3x larger than Qwen 2.5-7B
- Should have better reasoning
- Rate limits TBD

---

## Model Selection Guide

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| Quick experiments | `qwen/qwen-2.5-7b-instruct` | Fast, no rate limits |
| Better reasoning | `meta-llama/llama-3.3-70b-instruct:free` | Smarter, but rate limited |
| Production | Paid tier of any model | No rate limits |
| DC task | `qwen/qwen-2.5-7b-instruct` | 35% accuracy, stable |
| GN task | Larger model (24B+) | Small models struggle |

---

## Models to Avoid

| Model | Reason |
|-------|--------|
| `qwen/qwen3-*` | Thinking mode breaks format |
| `deepseek/deepseek-r1-*` | Reasoning models with forced thinking |
| `tngtech/*-chimera` | Based on DeepSeek R1 |
| `*-vl-*` | Vision-language models, not optimal for text-only |

---

## Key Insight: "Instruct" vs "Thinking" Models

**Instruct Models** (recommended):
- Fine-tuned to follow instructions
- Output directly matches requested format
- Examples: Llama-instruct, Qwen 2.5-instruct, Mistral-instruct

**Thinking/Reasoning Models** (avoid):
- Trained to reason before answering
- Always output chain-of-thought first
- Examples: Qwen3, DeepSeek-R1, o1-style models

---

## Configuration

Current model is set in `/root/RRMC/configs/base.yaml`:

```yaml
policy_model: mistralai/mistral-small-3.1-24b-instruct:free
```

To change, update this line to any supported model from OpenRouter.

---

## Rate Limits (Free Tier)

| Model | Limit |
|-------|-------|
| `qwen/qwen-2.5-7b-instruct` | ~20 req/min |
| `meta-llama/llama-3.3-70b-instruct:free` | 8 req/min |
| `mistralai/mistral-small-3.1-24b-instruct:free` | TBD |

---

## API Configuration

OpenRouter API is configured in `/root/RRMC/configs/providers/openrouter.yaml`:

```yaml
provider: openrouter
base_url: https://openrouter.ai/api/v1
api_key: sk-or-v1-xxx  # Your API key
```

To query available models:
```python
import requests
headers = {'Authorization': f'Bearer {api_key}'}
resp = requests.get('https://openrouter.ai/api/v1/models', headers=headers)
models = resp.json().get('data', [])
```
