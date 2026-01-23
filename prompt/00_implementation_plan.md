# RRMC Implementation Plan (Checklist)

This is the full step‑by‑step implementation checklist derived from `docs/00_proposal.md`
(Section 7: Implementation Checklist). Use it as the canonical build order.

## Phase 1 — Infrastructure (Week 1)

### Step 1: AR‑Bench environment wrapper
- [x] Parse AR‑Bench datasets (DC/SP/GN)
- [x] Implement `reset/step`
- [x] Standardize action schema `{type: ASK/ANSWER, ...}`
- [ ] Logging: JSONL per episode with full transcript

### Step 2: LLM wrapper
- [x] `generate(prompt, temperature, top_p, max_tokens) -> text`
- [x] `sample_n(prompt, n, ...) -> list[text]`
- [ ] Optional: `score_labels(prompt, labels) -> probs` (for C‑IP)
- [x] Support: OpenAI / Anthropic / vLLM / OpenRouter (as needed)

### Step 3: Semantic clustering module
- [x] SentenceTransformer embedder (`all-mpnet-base-v2` or similar)
- [x] Agglomerative clustering with cosine distance (`d_thr=0.25`)
- [x] Entailment‑aware clustering for SP (NLI split)
- [x] Task shortcuts (DC labels; GN digit strings)
- [x] Edge cases: all‑identical → 1 cluster; all‑unique → max entropy

## Phase 2 — Core MI Estimator (Week 2)

### Step 4: Self‑revision MI estimator
- [x] `prompt_initial(task, history)`
- [x] `prompt_revision(task, history, initial_answer)`
- [x] MI computation with additive smoothing (`eps=1e-6`)

### Step 5: Diversity‑steered sampling (with reweighting)
- [x] Embedding‑based rejection sampling (code exists in `DiversitySampler`)
- [x] Track rejection counts → importance weights
- [x] Use weighted counts in MI computation
- [x] Apply to MI sampling + simulated outcomes

### Step 6: Prompt‑variant generation
- [x] Base / Skeptical / Alternative variants
- [x] `robust_mi(...) = max_v MI_v`

## Phase 3 — Risk‑Controlled Calibration (Week 3)

### Step 7: Calibration data collection
- [x] Run questioning policy on train puzzles
- [x] For each visited state: `score`, `pred`, `err`, `homogeneity`, `task`, `turn`

### Step 8: Threshold selection (Clopper–Pearson UCB)
- [x] Implement `calibrate_tau(scores, errors, delta, alpha)`
- [x] Calibrate per task + per decoding regime (optional per homogeneity bucket)

### Step 9: Runtime stopping rule
- [x] If `robust_mi(h_t) <= tau` → answer
- [x] Else → ask (unless max turns reached)

## Phase 4 — Baselines (Week 3–4)

### Step 10: Implement all baselines
- [x] Naive prompting (fixed turns)
- [x] Self‑consistency gate (`p_max ≥ 0.8`)
- [x] Verbalized confidence gate
- [x] Semantic entropy gate
- [x] MI‑only (no robustness)
- [ ] KnowNo‑style set‑size gate
- [ ] C‑IP‑lite (set‑size stopping)
- [ ] C‑IP‑full (expected set‑size querying)
- [ ] UoT‑style planning
- [ ] EIG‑based question selection

## Phase 5 — Full Method (Week 4–5)

### Step 11: VoI scoring via expected robust MI reduction
- [ ] Simulate outcomes (R samples per candidate)
- [ ] Score candidate by expected MI drop

### Step 12: DPP slate selection
- [ ] Embed candidates
- [ ] Greedy DPP selection (m from M)

### Step 13: Candidate question generation
- [ ] DC: one suspect per question
- [ ] SP: yes/no form
- [ ] GN: 4 unique digits
- [ ] Deduplicate by normalized text

## Phase 6 — Evaluation (Week 5–6)

### Step 14: Evaluation harness + metrics
- [x] Accuracy / F1 / efficiency
- [x] MI-error Spearman correlation
- [ ] Calibration metrics on visited states (ECE, risk–coverage)
- [ ] Bootstrap CIs

### Step 15: Homogeneity diagnostics
- [ ] Effective rank
- [ ] Repeat‑curse metrics
- [ ] Prompt sensitivity index
- [ ] Stratified performance by homogeneity bucket

## MVE Sprint (Minimum Viable Experiment)

- [x] DC‑only, direct prompting question generator
- [x] Compare: self‑consistency, semantic entropy, MI‑only, robust MI + τ
- [x] Homogeneous decoding regime (`--regime homogeneous`)
- [x] MI–error correlation in output
- [ ] Run full evaluation and generate report
