# Session Progress - January 30, 2026

## Session Goal

Implement all remaining steps from the CLAUDE.md task board: SP scoring fidelity, VoI question selection, DPP slate selection, missing baselines, evaluation metrics upgrades, and performance/reliability improvements.

---

## What Was Accomplished

### Step 1: Fix SP Fidelity to AR-Bench Scoring

**Status:** Complete

**Changes:**
- `rrmc/core/environment.py`:
  - Replaced set-based F1 with `Counter`-based bag-of-characters F1 matching AR-Bench's `utils_sp.py`
  - Added `_bag_f1()` static method supporting both character-level and word-level F1
  - Result now includes `f1_score_char` and `f1_score_word` in episode info
  - Replaced simple NPC prompt with AR-Bench's official 2-shot referee template (`system_prompt_with_2shots`)
  - Referee labels normalised to `Yes`/`No`/`Unknown` (was `YES`/`NO`/`IRRELEVANT`)
  - Conversation history now threaded through multi-turn referee messages

**Key difference:** Old implementation used `set` intersection (lost frequency info), new uses `Counter` intersection preserving duplicate character/word counts.

---

### Step 2: VoI Question Selection

**Status:** Complete

**New file:** `rrmc/methods/question_selector.py`

**Algorithm:**
1. Generate M=5 candidate questions (via LLM with structured JSON output)
2. For each candidate, simulate R=2 suspect responses (role-play prompt)
3. Compute robust MI on each simulated next-state
4. Score = E[current_MI - next_MI] (expected MI reduction)
5. Pick candidate with highest expected MI reduction

**Changes:**
- `rrmc/methods/question_selector.py` - New module with `VoIQuestionSelector` class
- `rrmc/evaluation/evaluator.py` - Added `question_selector` parameter; evaluator uses VoI when available on DC
- `pipeline.py` - Wires VoI selector when `use_voi: true` in config
- `configs/experiments/dc_methods/voi_robust_mi.yaml` - Experiment config

**Limitations:** DC task only. SP and GN would need different candidate generation templates.

---

### Step 3: DPP Slate Selection

**Status:** Complete

**Added to:** `rrmc/methods/question_selector.py`

**Algorithm:**
1. After VoI scoring, build similarity kernel K from question embeddings
2. Construct L-ensemble: L = diag(q) * K * diag(q) where q = quality scores
3. Greedy selection: iteratively pick item maximising marginal log det(L_S)
4. Select top-1 from the diverse slate

**Classes:**
- `DPPSlateSelector` - Greedy DPP selector with cosine similarity kernel
- `SlateInfo` - Metadata dataclass (indices, quality scores, pairwise similarities)

**Embedding strategy:** Uses sentence-transformer model if available via clusterer, falls back to character 3-gram TF vectors.

**Changes:**
- `rrmc/methods/question_selector.py` - Added `DPPSlateSelector`, integrated into `VoIQuestionSelector`
- `pipeline.py` - Added `use_dpp` and `dpp_slate_size` config params
- `configs/experiments/dc_methods/voi_dpp_robust_mi.yaml` - Experiment config

---

### Step 4: Missing Baselines (KnowNo, C-IP, UoT)

**Status:** Complete

**New stopping rules in** `rrmc/methods/stopping_rules.py`:

| Method | Class | Signal | Default Threshold |
|--------|-------|--------|------------------|
| KnowNo | `KnowNoStopping` | Prediction set size (# clusters) | set_size <= 1 |
| C-IP-lite | `CIPLiteStopping` | Conformal prediction set size | set_size <= 2 (calibratable) |
| UoT-lite | `UoTLiteStopping` | # plausible answers from explicit enumeration | n_plausible <= 1 |

**Registry entries:** `knowno`, `cip_lite`, `uot_lite` added to `rrmc/methods/__init__.py`

**Changes:**
- `rrmc/methods/stopping_rules.py` - Three new stopping rule classes
- `rrmc/methods/__init__.py` - Registry updated
- `pipeline.py` - `_create_method` handles new methods
- `rrmc/evaluation/evaluator.py` - Import updated
- `configs/methods/knowno.yaml`, `cip_lite.yaml`, `uot_lite.yaml` - Method configs

---

### Step 5: Evaluation Metrics Upgrades

**Status:** Complete

**New file:** `rrmc/evaluation/metrics.py`

**Metrics implemented:**

| Metric | Function | Description |
|--------|----------|-------------|
| Risk-coverage curve | `compute_risk_coverage_curve()` | Error rate vs. coverage at varying thresholds, plus AUC |
| ECE | `compute_ece()` | Expected Calibration Error with per-bin accuracy/confidence |
| Bootstrap CI (accuracy) | `bootstrap_accuracy_ci()` | 95% CI via 1000 bootstrap resamples |
| Bootstrap CI (turns) | `bootstrap_turns_ci()` | 95% CI for average turns |
| Effective rank | `compute_effective_rank()` | exp(H(p)) over cluster distribution |
| Repetition rate | `compute_repetition_rate()` | Fraction of duplicate answers |
| Prompt sensitivity index | `compute_prompt_sensitivity_index()` | CoV of MI across prompt variants |
| Full report | `generate_metrics_report()` | Aggregates all above |

**Integration:**
- `evaluator.py` prints bootstrap CIs in results line and comparison table
- Comparison table now shows `[lower, upper]` CI columns for accuracy and turns

---

### Step 6: Performance and Reliability

**Status:** Complete

**Changes:**

1. **Structured retries with exponential backoff** (`rrmc/core/llm.py`):
   - `max_retries=3` with base delay 1s, max delay 30s
   - Exponential backoff: 1s, 2s, 4s (capped at 30s)
   - Retries on any exception (timeout, rate limit, server error)

2. **MI estimation cache** (`rrmc/core/mi_estimator.py`):
   - Cache key: SHA-256 hash of (task_type, history_string, variant)
   - Avoids redundant MI estimation for the same state+variant
   - Enabled by default; disable with `estimator._cache_enabled = False`

---

## Files Changed

### New Files
| File | Purpose |
|------|---------|
| `rrmc/methods/question_selector.py` | VoI question selection + DPP slate selection |
| `rrmc/evaluation/metrics.py` | Risk-coverage, ECE, bootstrap CIs, homogeneity diagnostics |
| `configs/experiments/dc_methods/voi_robust_mi.yaml` | VoI experiment config |
| `configs/experiments/dc_methods/voi_dpp_robust_mi.yaml` | VoI + DPP experiment config |
| `configs/methods/knowno.yaml` | KnowNo method config |
| `configs/methods/cip_lite.yaml` | C-IP-lite method config |
| `configs/methods/uot_lite.yaml` | UoT-lite method config |

### Modified Files
| File | Changes |
|------|---------|
| `rrmc/core/environment.py` | SP bag-of-chars F1, AR-Bench referee prompt, Yes/No/Unknown labels |
| `rrmc/core/llm.py` | Structured retry with exponential backoff |
| `rrmc/core/mi_estimator.py` | MI estimation cache (state hash based) |
| `rrmc/methods/stopping_rules.py` | KnowNo, C-IP-lite, UoT-lite stopping rules |
| `rrmc/methods/__init__.py` | Registry updated with 3 new methods |
| `rrmc/evaluation/evaluator.py` | VoI integration, bootstrap CIs, new method imports |
| `pipeline.py` | VoI/DPP config wiring, new method creation |

---

## Method Registry (Complete)

| Name | Class | Type |
|------|-------|------|
| `fixed_turns` | `FixedTurnsStopping` | Baseline |
| `self_consistency` | `SelfConsistencyStopping` | Baseline |
| `verbalized_confidence` | `VerbalizedConfidenceStopping` | Baseline |
| `semantic_entropy` | `SemanticEntropyStopping` | Baseline |
| `mi_only` | `MIOnlyStopping` | Baseline |
| `robust_mi` | `RobustMIStopping` | **RRMC** |
| `knowno` | `KnowNoStopping` | Baseline (new) |
| `cip_lite` | `CIPLiteStopping` | Baseline (new) |
| `uot_lite` | `UoTLiteStopping` | Baseline (new) |

---

## Commands for Next Session

### Run All 9 Methods on DC
```bash
cd /root/RRMC && python run.py all_methods --task dc --n_puzzles 20 --max_turns 25
```

### Run VoI + DPP Experiment
```bash
cd /root/RRMC && python run.py dc_methods/voi_dpp_robust_mi --n_puzzles 10
```

### Run Calibration Pipeline
```bash
cd /root/RRMC && python run.py mve_dc_normal --calibrate true --n_train 20 --n_test 10
```

### Run SP with Fixed Scoring
```bash
cd /root/RRMC && python run.py fixed_turns --task sp --n_puzzles 10 --max_turns 15
```

---

## Next Steps

1. **Run experiments** to validate all new code paths end-to-end
2. **Add EIG baseline** (Expected Information Gain) - requires different architecture
3. **SP/GN VoI support** - extend question selection beyond DC
4. **Plotting utilities** - risk-coverage curves, calibration diagrams for paper figures
5. **Full paper results** - run all 9 methods on DC/SP/GN with 50+ puzzles each
