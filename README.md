# RRMC

**RRMC (Robust Revision-MI Control)** is the working name for the method described in the Robust-MI Active Inquiry proposal: use **robust self-revision mutual information** as an uncertainty signal, then apply **risk-controlled thresholding** (Clopper–Pearson UCB) to decide **ask vs answer** in interactive tasks like AR-Bench.

## Quick Start

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set API key (OpenRouter)
export OPENROUTER_API_KEY="sk-or-v1-..."
```

### Run Experiments

```bash
# Run DC (Detective Cases) - 5 suspects, identify murderer
python run.py fixed_turns --task dc --n-puzzles 5

# Run GN (Guessing Numbers) - Bulls & Cows style game
python run.py fixed_turns --task gn --n-puzzles 5

# Run SP (Situation Puzzles) - Yes/no questions to explain story
python run.py fixed_turns --task sp --n-puzzles 5
```

### Run with Different Methods

```bash
# Available methods: fixed_turns, self_consistency, semantic_entropy, mi_only, robust_mi
python run.py self_consistency --task dc --n-puzzles 10
python run.py semantic_entropy --task dc --n-puzzles 10

# Run all methods for comparison
python run.py all_methods --task dc --n-puzzles 10
```

### Configuration Options

```bash
# Adjust max turns per episode
python run.py fixed_turns --task gn --n-puzzles 5 --max-turns 50

# List available experiment configs
python run.py --list
```

### Change Model

Edit `configs/base.yaml`:
```yaml
policy_model: qwen/qwen-2.5-7b-instruct  # Recommended (no rate limits)
# policy_model: meta-llama/llama-3.3-70b-instruct:free  # Better but rate limited
```

## Results

Results are saved to `results/` as JSON files with:
- Accuracy and average turns per method
- Full episode history (questions, answers, feedback)
- Token usage statistics

## Docs

- **00_proposal (implementation-ready spec)**: `docs/00_proposal.md`
- **01_literature (curated references)**: `docs/01_literature.md`
- **Docs index / reading order**: `docs/README.md`

## Secrets / API keys

- Put secrets in a local `.env` file in the repo root (this file is git-ignored).
- Template: `configs/env.example`