# RRMC — Claude Code Task Board (Active)

This file is the **current active implementation plan** for Claude Code runs.  
Older plans are archived in this folder (see `prompt/99_archive_CLAUDE_20260130.md`).

## Ground rules (don’t skip)
- **Do not refactor** unless required for the step you’re executing.
- Keep changes **small and reviewable**: prefer 1–3 files per step.
- After each step: run a **tiny smoke run** (1–2 puzzles, 2–3 turns) to ensure nothing broke.

## Current repo reality (what already exists)
- Config runner: `run.py` + `config.py` + `pipeline.py`
- Core tasks + env wrapper: `rrmc/core/environment.py` (DC/SP/GN)
- RRMC components: `rrmc/core/mi_estimator.py`, `rrmc/core/calibration.py`
- Baselines + RRMC stopping: `rrmc/methods/stopping_rules.py`
- Evaluation harness: `rrmc/evaluation/evaluator.py`
- AR-Bench submodule checked out under `AR-Bench/`

## Step 0 — Standardized DC method configs (done by agent, keep aligned)
We are creating a standardized set of single-method configs under:
- `configs/experiments/dc_methods/*.yaml`

They are intended to be runnable via:
- `python run.py dc_methods/fixed_turns`

If you add new methods, add a matching config there.

---

## Step 1 — Fix SP fidelity to AR-Bench (high value correctness)

### Problem
Our SP implementation is intentionally simplified:
- SP “referee” answers are derived from a single hidden-story prompt and return `YES/NO/IRRELEVANT`.
- SP scoring uses a simplistic character-set F1 in `rrmc/core/environment.py` and char bag F1 in `rrmc/core/calibration.py`.

AR-Bench uses:
- Referee template and answer space `{"Yes","No","Unknown"}` with few-shot system prompt (see `AR-Bench/arbench/reasoner/sp/sp_evaluator.py`)
- F1 via `arbench.reasoner.utils.f1_score` (bag-of-tokens) and logs both char/word F1.

### Goal
Make RRMC’s SP evaluation and calibration **match AR-Bench scoring semantics**, while keeping the RRMC architecture intact.

### Implementation tasks
- Update **SP scoring** in RRMC:
  - Replace the current SP correctness heuristic with **bag-of-characters F1 using counts** (Counter-based), not set-based.
  - Keep a threshold (default 0.5) configurable.
  - Optional: also compute word-level F1 for logging parity.
- Update **SP referee outputs**:
  - Match labels to `Yes/No/Unknown` (case-insensitive), and normalize outputs.
  - Consider porting AR-Bench’s referee system prompt (with 2 shots) to reduce noise.

### Files likely touched
- `rrmc/core/environment.py` (SP `_ask_sp` and `_evaluate_sp_answer`)
- `rrmc/core/calibration.py` (SP scoring used in `add_state`)
- (optional) `rrmc/evaluation/evaluator.py` to log SP metrics

### Smoke test
Run a tiny SP config:
- `python run.py mve_sp_normal --n_train 2 --n_test 1 --max_turns 3 --k_samples 2`

---

## Step 2 — Implement VoI question selection (expected robust-MI reduction)

### Goal
Replace “ask next question” generation with a **candidate generation + scoring** pipeline:
1) generate M candidate questions
2) simulate outcomes per candidate
3) score by expected robust MI reduction (or expected posterior entropy reduction proxy)
4) pick top-1 to ask (or top-m with DPP in next step)

### Practical constraints
- Must be **cheap enough** to run on small configs. Start with tiny M and tiny rollout depth.
- Keep DC only at first.

### Minimal DC version (start here)
- Candidate generation:
  - Use policy LLM to generate `M=5` candidate questions in plain text.
  - Also select a suspect per candidate (or generate questions conditioned on a chosen suspect).
  - Normalize/deduplicate questions.
- Outcome simulation:
  - For each candidate question, sample `R=2` simulated suspect responses using the existing DC suspect role-play mechanism (same LLM wrapper).
  - For each simulated response, construct a “next state” by appending to history.
  - Compute robust MI on the next state.
- Score:
  - `score(q) = E[ robust_mi(current_state) - robust_mi(next_state | simulated_feedback) ]`
  - Pick question maximizing score.

### Where to implement
Keep it modular:
- Add a `QuestionSelector` (or similar) module under `rrmc/core/` or `rrmc/methods/`.
- Wire it in `rrmc/evaluation/evaluator.py` where questions are generated when `decision.should_stop == False`.

### Smoke test
DC tiny run:
- `python run.py dc_methods/robust_mi --n_puzzles 2 --max_turns 3 --k_samples 2`

---

## Step 3 — DPP slate selection (diversify top questions)

### Goal
When you have many high-scoring candidate questions, select a diverse slate:
- Start with `m=3` chosen from `M=10` candidates.
- Ask the top-1, but keep the slate for logging/ablation.

### Minimal implementation
- Compute embeddings for each candidate question (reuse sentence-transformers if installed; else simple TF-IDF or normalized n-gram fallback).
- Greedy DPP:
  - maximize `det(L_S)` approximately (standard greedy marginal gain).
- Combine quality and diversity:
  - `L = diag(quality) * K * diag(quality)`, where `K` is cosine-sim kernel.

### Output
Log:
- chosen slate questions
- their quality scores
- pairwise similarities

---

## Step 4 — Missing baselines (port minimal versions)

### Needed (per proposal / AR-Bench comparisons)
- KnowNo-style set-size gate
- C-IP-lite (conformal set-size stopping)
- C-IP-full (expected set-size / querying)
- UoT-style planning baseline
- EIG-based question selection baseline

### Strategy
Do them in increasing complexity:
1) **KnowNo / set-size** (uses clustering output counts)
2) **C-IP-lite** (conformal thresholding on set size)
3) **UoT-lite** (generate answer set + ask question to reduce it, no deep simulation)

Keep implementation **inside `rrmc/methods/`** so config-driven runs can pick them up.

---

## Step 5 — Evaluation/metrics upgrades (publishable reporting)
- Risk–coverage curves (coverage vs error under varying τ)
- ECE (calibration error) on visited states
- Bootstrap CIs for accuracy/turns
- Homogeneity diagnostics:
  - effective rank of cluster distribution
  - repetition metrics
  - prompt sensitivity index (variant spread)

---

## Step 6 — Performance and reliability
- Add **caching** for MI estimation per `(task, state_hash, variant)` to avoid repeated calls in the same episode.
- Add structured retries/backoff for OpenRouter timeouts.
- Keep `max_workers` sane to avoid rate limits.

