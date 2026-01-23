# Risk-Controlled Robust-MI Active Inquiry: Final Proposal

**Version:** 05 — Final (Publication-Ready Framing)  
**Based on:** `04_literature_refined.md` + `12_section_rewrites_.md`  
**Last updated:** 2025-01-11

---

## Document Purpose

This is the **final implementation-ready specification** for the Risk-Controlled Robust-MI Active Inquiry project. It contains:
- Exact algorithms with pseudocode
- Hyperparameter defaults
- Baseline implementations (including C-IP, KnowNo, UoT)
- Evaluation metrics and ablation plan
- Publication-quality framing for all key sections

**Implementation checklist:** Included in this document (see **Section 7**).

---

## Executive Summary

Interactive active-reasoning benchmarks require an agent to decide, at each dialogue state, whether to ask another question or commit to a final answer. In AR-Bench, this decision is complicated by two practical pathologies of instruction-tuned LLMs: (i) **overconfidence** (often reinforced by RLHF-style training) and (ii) **homogeneity / mode-collapse**, where sampling yields highly redundant answers and questions. Both effects can degrade standard uncertainty proxies, including verbalized confidence, self-consistency, and probability-based expected information gain (EIG).

We propose **Risk-Controlled Robust-MI Active Inquiry**, which uses **self-revision mutual information (MI)** as a black-box uncertainty signal and converts this signal into an "ask vs answer" policy via **risk-controlled thresholding**. At a state \(h_t\), the policy generates an initial answer and a revised answer under an explicit reconsideration prompt, and estimates \(\widehat{I}(Y^{(0)};Y^{(1)})\) after clustering generations into semantic equivalence classes (Paper 09). To reduce underestimation of uncertainty under collapse, we robustify MI using multiple meaning-preserving prompt variants and diversity-steered sampling with bias-aware estimation (Paper 36). Our central empirical hypothesis is that **robust self-revision MI** is more predictive of error in high-homogeneity regimes than conformal set size, self-consistency, or probability-based EIG.

**Risk-controlled thresholding (not conformal prediction).** We do *not* claim classical conformal prediction guarantees. Instead, we calibrate a stopping threshold \(\tau\) on a held-out set of **visited states** by applying a **Clopper–Pearson binomial upper confidence bound** to the empirical error rate among states with \(\widehat{I}_{rob}(h)\le\tau\). This yields a finite-sample confidence statement about **calibration risk** for a fixed score-thresholding rule under standard binomial/i.i.d. assumptions. It does not provide the distribution-free coverage guarantees associated with conformal prediction or conformal risk control (Papers 05, 26–28), and it can degrade under distribution shift or strong sequential dependence between trajectory states. Accordingly, we treat \(\tau\) as a conservative on-policy calibration mechanism and empirically audit robustness via episode-level dependence checks and stratification by homogeneity.

**Relation to conformal baselines.** KnowNo-style and C‑IP‑style baselines use **conformal prediction sets** as their primary uncertainty object: the agent triggers querying/assistance when the prediction set is non-singleton, and (in C‑IP) selects questions to minimize expected prediction-set size. In contrast, our approach uses **mutual information** between initial and revised generations as the uncertainty signal, and applies **risk-controlled thresholding** only to map that signal to a stopping rule rather than to construct prediction sets.

---  

## 0. Problem, hypothesis, and target setting

**Target benchmark:** AR-Bench has 3 interactive active-reasoning tasks—Detective Cases (DC), Situation Puzzles (SP), Guessing Numbers (GN)—with train/test sizes 400/100 (DC), 400/100 (SP), 4940/100 (GN). Interactions run up to **25 rounds**; SP/DC feedback is produced by an LLM “NPC” and GN feedback is provided by an oracle function.

**Task mechanics (needed for implementation):**

* **DC:** interrogate 5 suspects with noisy/narrative feedback; final output is which suspect is the murderer.
* **SP:** ask **yes/no** questions to uncover a hidden story; final output is a free-text explanation scored by character-level F1 vs ground truth.
* **GN:** deduce a **4-digit number with unique digits** from symbolic feedback about correct digits/positions (Bulls-and-Cows-like).

**Competitors that must be beaten (baselines):**

* **KnowNo**: conformal prediction sets; trigger help when set is non-singleton; set size is “help” to minimize.
* **Conformal Information Pursuit (C‑IP)**: sequential querying that uses **average conformal prediction set size** as an uncertainty proxy tied to entropy; selects queries by minimizing expected set size (instead of entropy).
* **Uncertainty of Thoughts (UoT)**: generates candidate questions, simulates futures (tree), uses information-gain–motivated rewards, and propagates reward to pick the next question.
* **EIG-based query selection**: choose questions maximizing expected information gain; commonly measured and optimized in 20Q-style games.

**Core hypothesis to validate:**

> Under **homogeneity / mode-collapse** and **RLHF-style overconfidence**, *self-revision mutual information* (MI)—especially in a **robust** form—predicts error and controls “ask vs answer” better than conformal set size, self-consistency, or probability-based EIG.

Homogeneity/mode-collapse is documented as an “Artificial Hivemind” effect in open-ended LMs (intra-model repetition and inter-model homogeneity).

---

# 1. Minimal Viable Experiment

The MVE should isolate the **uncertainty signal** (MI vs conformal-set-size vs self-consistency, etc.) without requiring the full VoI question-selection pipeline.

## 1.1 Simplest demonstrator: “Stopping / abstention controller” on DC + SP (plus GN diagnostics)

### Why DC + SP first (despite GN being diagnostic)

* **DC**: final answer is **finite** (5 suspects), so conformal prediction sets and set-size are clean and directly comparable to KnowNo/C‑IP.
* **SP**: question answers are **yes/no**, making it a direct analogue to 20Q-style settings used by UoT and C‑IP; also good for homogeneity stress tests (models often repeat similar question patterns).
* **GN**: extremely useful for **high-power statistics** and homogeneity diagnostics because the answer space is large (5040 possible secrets) and the environment is symbolic.
  However GN lacks a clean “ask vs answer” separation (each guess is both probe and final attempt), so we use it in MVE mainly for **homogeneity analysis** and efficiency.

### Base models (policy model)

Pick **one open model** for reproducibility + logits access (optional), and **one API model** to stress RLHF overconfidence/homogeneity.

**Recommended default:**

* **Open-source policy model:** Llama-3.1-8B-Instruct (matches common AR-Bench setup and is cheap).
* **Optional stronger open model:** Llama-3.1-70B-Instruct or a similar 70B-class model (only if GPU budget allows).
* **Optional closed model for homogeneity stress:** GPT-4o or similar. AR-Bench reports GPT-4o in evaluation.

**NPC/judge models (environment side):**

* AR-Bench uses **Llama-3.1-405B** to provide interactive feedback for SP/DC for reproducibility, and uses an oracle for GN.
* Under the compute constraint, for MVE you can replace the NPC with a smaller model **as long as all methods share the same NPC**.

### Task subsets and sample sizes

* **DC:** use official **100 test puzzles** for evaluation; use the 400 train puzzles to build calibration states.
* **SP:** same: 100 test / 400 train.
* **GN (diagnostic):** use official 100 test for headline numbers, plus **≥1000 random secrets** (sampled from the 5040 valid secrets) for homogeneity/efficiency plots.

### Statistical significance plan (practical)

* DC/SP have only 100 official test puzzles; rely on:

  * **paired bootstrap** over puzzles to compare methods (difference in accuracy/F1).
  * **state-level** analysis (visited states per episode × episodes ≈ up to 25×100 = 2500 states) for MI–error correlation and homogeneity stratification.
* GN provides **large N** cheaply; use it for high-confidence improvements in efficiency and collapse-regime behavior.

### Control condition

Hold **question generation** fixed across all stopping rules:

* Use **Direct Prompting (DP)** to generate the next question (one question per step).
* Only change the **stop/answer rule** based on each uncertainty signal.

This isolates the core claim: *which uncertainty signal best decides whether to keep asking.*

### Homogeneity / RLHF-overconfidence stress test (required)

Run every method under two decoding regimes:

1. **Normal sampling:** temperature=0.7, top_p=0.95
2. **Homogeneous (collapse) regime:** temperature=0.0–0.1, top_p=1.0, plus a system instruction like:

   * “Be decisive. Provide one best answer. Do not hedge.”

This regime is designed to induce repeated outputs / mode collapse consistent with the “Artificial Hivemind” effect.

**MVE success criterion:** Under regime (2), our **robust self-revision MI gate** should:

* reduce premature stopping vs set-size/self-consistency
* improve accuracy/F1 given the same question generator

---

# 2. Exact Algorithms

## 2.1 Core interfaces (what to implement)

### Environment wrapper

Create a uniform interface over AR-Bench tasks:

```python
class ARBenchEnv:
    def reset(self, puzzle_id: str) -> dict:
        """Returns initial observation/state dict including rules + initial clues."""
    def step(self, action: dict) -> dict:
        """
        action = {"type": "ASK", "question": str, ...} or {"type": "ANSWER", "answer": str}
        Returns dict with keys:
          - "obs": npc answer / feedback
          - "done": bool
          - "reward": optional
          - "info": includes ground_truth for evaluation only
        """
```

* **DC/SP:** `ASK` sends a question; env returns NPC response (text).
* **GN:** `ASK` is a numeric guess; env returns symbolic feedback (e.g., bulls/cows counts) per AR-Bench definition.

### LLM wrapper

You need:

* `generate(prompt, temperature, top_p, max_tokens) -> text`
* `sample_n(prompt, n, ...) -> list[text]`
* Optional: `score_labels(prompt, labels) -> probs` (for C‑IP/KnowNo if using logits)

## 2.2 Semantic primitives used across methods

### 2.2.1 Canonicalization/parsing

Implement task-specific answer parsing:

* **DC:** map output to one of the 5 suspects (exact match via regex / alias table).
* **SP:** free-text explanation, but for uncertainty computations use semantic clustering of explanations.
* **GN:** parse 4-digit guess as string; enforce unique digits; if invalid, repair by:

  * keep first occurrence of digits, then fill remaining with random unused digits.

### 2.2.2 Semantic clustering (task-dependent; entailment-aware for SP)

Several components in this proposal (semantic entropy, self-revision MI, and set-based baselines) require aggregating sampled generations into "unique answers" or "semantic hypotheses." A purely embedding-distance clustering is efficient, but it can over-merge generations that are topically similar yet **logically incompatible** (e.g., near-contradictions with high lexical overlap), which can distort entropy and MI estimates. Semantic Entropy addresses this by defining semantic classes via **bidirectional entailment**: two generations are treated as equivalent only if they mutually entail each other (Paper 09).

We therefore use a **two-mode clustering strategy** tailored to the output structure of AR-Bench tasks:

* **DC and GN (discrete-output tasks):** The relevant outputs are naturally canonicalized (DC: suspect identity; GN: exact 4-digit string). Here, clustering is primarily used for auxiliary free-text artifacts (e.g., deduplicating candidate questions), and the main failure mode is superficial paraphrase rather than semantically meaningful contradiction. We therefore use embedding-based clustering as a fast deduplication heuristic.

* **SP (free-text explanations):** Explanations are open-ended and semantically rich; embedding-only clustering is unreliable for separating contradictory hypotheses. For SP, we use an **entailment-aware clustering** procedure in the style of Paper 09: we first form coarse candidate neighborhoods with embeddings (to avoid quadratic comparisons), then refine clusters using an NLI model by requiring **mutual entailment** within clusters and separating contradictions into different clusters.

#### Implementation

```python
def cluster_with_entailment(texts, embedder, nli_model, d_thr=0.25):
    # Step 1: Embedding-based candidate clusters
    embs = embedder.encode(texts)
    candidate_clusters = agglomerative_cluster(embs, d_thr)
    
    # Step 2: Split clusters where NLI detects contradiction
    final_clusters = []
    for cluster in candidate_clusters:
        if len(cluster) <= 1:
            final_clusters.append(cluster)
            continue
        # Check pairwise entailment within cluster
        subclusters = split_by_contradiction(cluster, texts, nli_model)
        final_clusters.extend(subclusters)
    
    return assign_cluster_ids(texts, final_clusters)
```

**NLI model:** Use `roberta-large-mnli` or similar. If entailment score < 0.3 AND contradiction score > 0.5, split.

This design explicitly trades compute for semantic fidelity. NLI inference adds non-trivial overhead relative to embedding-only clustering; we mitigate by restricting entailment checks to SP (where misclustering is most damaging) and to embedding-local candidate sets. We report runtime overhead and include an ablation comparing entailment-aware versus embedding-only clustering for SP.

Edge cases:

* If all embeddings are near-identical → 1 cluster (but still check entailment for SP)
* If clustering yields all singletons → treat as max-entropy case

---

## 2.3 Self-revision MI uncertainty estimator (core component)

### 2.3.1 Definition

At a conversation state/history (h), define two random variables:

* (Y^{(0)}): initial model answer
* (Y^{(1)}): revised model answer after a self-revision prompt

We estimate mutual information:
[
\widehat{I}(Y^{(0)};Y^{(1)})=\sum_{a,b}\hat{p}(a,b)\log\frac{\hat{p}(a,b)}{\hat{p}(a)\hat{p}(b)}
]
where (a,b) are **semantic cluster IDs** of answers.

### 2.3.2 Two-step self-revision sampling procedure

**Inputs:** history (h), samples (k), prompt variant (v)

**Prompt templates (must be implemented)**

1. **Initial answer prompt** (task-specific):

* DC: “Given the case + transcript so far, who is the murderer? Output JSON: {suspect: ..., rationale: ...}”
* SP: “Give your current best explanation. Output JSON: {explanation: ...}”
* GN: “Give the next guess as 4 distinct digits. Output JSON: {guess: ....}”

2. **Revision prompt**:

* Provide the model’s own initial answer.
* Ask it to *reconsider*, explicitly looking for contradictions and alternative hypotheses.
* Force it to output a revised final answer in the same JSON schema.

### 2.3.3 Pseudocode

```python
def estimate_self_revision_mi(llm, history, task, k, temp, top_p, clusterer):
    pairs = []
    for i in range(k):
        y0 = llm.generate(prompt_initial(history, task),
                          temperature=temp, top_p=top_p)
        y0_ans = parse_answer(y0, task)

        y1 = llm.generate(prompt_revision(history, task, y0),
                          temperature=temp, top_p=top_p)
        y1_ans = parse_answer(y1, task)

        pairs.append((y0_ans, y1_ans))

    # cluster union of answers (for SP); DC/GN can be exact-label clusters
    all_ans = [a for (a,_) in pairs] + [b for (_,b) in pairs]
    cluster_ids = cluster_answers(all_ans, task, clusterer)
    c0 = cluster_ids[:k]
    c1 = cluster_ids[k:]

    return mutual_information(c0, c1)
```

### 2.3.4 MI estimator details

Implement `mutual_information(c0, c1)` via a joint-count table with smoothing:

* Let (n_{ab}) be counts of ((c0=a, c1=b))
* (n_a=\sum_b n_{ab}), (n_b=\sum_a n_{ab}), (N=k)
* Use additive smoothing `eps = 1e-6`:

  * (\hat{p}(a,b)=(n_{ab}+\epsilon)/(N+\epsilon|A||B|))
  * (\hat{p}(a)=(n_{a}+\epsilon)/(N+\epsilon|A|))
  * (\hat{p}(b)=(n_{b}+\epsilon)/(N+\epsilon|B|))

Edge cases:

* If there is only 1 cluster in both steps, MI≈0 (return 0.0 exactly).
* If clustering fails (NaNs), fall back to exact string match clusters.

### 2.3.5 Hyperparameters (MVE defaults)

* `k=8` for DC, `k=6` for SP, `k=4` for GN (GN is cheap; MI mainly diagnostic)
* `temp=0.7`, `top_p=0.95` (normal); plus collapse regime `temp=0.1` and `temp=0`
* `max_tokens`: small (DC: 128, SP: 256, GN: 32)

---

## 2.4 Robust MI (homogeneity-resistant MI)

### 2.4.1 Motivation

Under homogeneity, sampled answers may collapse to one cluster, making MI and entropy both near zero—even when the model is wrong. The robustification explicitly probes sensitivity to “nearby prompts” and forces diversity in sampling.

### 2.4.2 Robust MI definition

Let (v \in {1,\dots,V}) index prompt variants (different wording / skepticism / counterfactual instructions). Compute MI per variant:
[
\widehat{I}_v = \widehat{I}(Y^{(0)}_v;Y^{(1)}_v)
]
Then define robust MI as:

* **Conservative:** (\widehat{I}_{\text{rob}}=\max_v \widehat{I}_v)
* **Less sensitive:** (\widehat{I}*{\text{rob}}=\mathrm{Quantile}*{0.75}({\widehat{I}_v}))

Use **max** for the core claim (don’t miss uncertainty under collapse).

### 2.4.3 Prompt variants (implement exactly)

Implement at least 3 variants (V=3):

1. **Base**: normal instructions
2. **Skeptical**: “Assume your current answer might be wrong; find counterevidence.”
3. **Alternative hypothesis**: “Generate a plausible alternative explanation; then decide.”

### 2.4.4 Diversity-steered sampling and estimator bias (Paper 36)

Under homogeneity/mode-collapse, naive sampling can produce near-duplicate generations, yielding spuriously low entropy and low self-revision MI. Following diversity-steered semantic uncertainty estimation (Paper 36), we augment sampling with an **embedding-based rejection sampler**: candidates are drawn from the base policy, and are accepted only if they are sufficiently dissimilar (by embedding cosine distance) from previously accepted samples.

This procedure is intentionally adversarial to collapse, but it also changes the effective sampling distribution. Unweighted cluster frequencies computed over accepted samples estimate uncertainty under the **diversity-filtered proposal distribution**, not under the model's original sampling distribution. Consequently, plug-in estimators of entropy and MI computed from accepted samples can be **biased**, with the bias depending on the rejection threshold and the model's collapse behavior (Paper 36). This matters whenever MI is interpreted as an uncertainty proxy rather than a stress-test statistic.

To mitigate distributional distortion, we apply **importance reweighting** (Paper 36). Each accepted sample is assigned a weight inversely proportional to its acceptance probability under the diversity filter; weighted cluster counts are then used in the entropy/MI estimators. Because acceptance probabilities are not analytically available in black-box LLM sampling, we estimate them from observed rejection rates (e.g., number of attempts required to obtain an accepted sample) and report the resulting variance. Importantly, we treat reweighting as a controlled design choice: we report both **unweighted** (biased, diversity-stressed) and **importance-weighted** (bias-mitigated) MI estimates as an explicit ablation.

#### Implementation

```python
def diversity_sample_with_weights(llm, prompt, k, sim_thr, embedder):
    samples = []
    weights = []
    embeddings = []
    
    for i in range(k):
        for attempt in range(max_tries := 5):
            candidate = llm.generate(prompt, temp=0.7)
            emb = embedder.encode(candidate)
            
            # Compute acceptance probability
            if len(embeddings) == 0:
                accept_prob = 1.0
            else:
                max_sim = max(cosine_sim(emb, e) for e in embeddings)
                accept_prob = 1.0 if max_sim <= sim_thr else 0.0
            
            if accept_prob > 0 or attempt == max_tries - 1:
                samples.append(candidate)
                # Importance weight = 1 / P(accept) ≈ 1 / (1 - rejection_rate)
                # Approximate: track rejection count
                weight = 1.0 + attempt * 0.2  # Simple heuristic
                weights.append(weight)
                embeddings.append(emb)
                break
    
    # Normalize weights
    weights = [w / sum(weights) * len(weights) for w in weights]
    return samples, weights
```

Defaults: `sim_thr=0.90` (answers), `sim_thr=0.85` (questions)

When computing MI, use weighted counts:
* `n_ab = sum(weights[i] for i where (c0[i], c1[i]) == (a, b))`

### 2.4.5 Pseudocode

```python
def robust_mi(llm, history, task, k, variants):
    mi_vals = []
    for v in variants:
        mi_v = estimate_self_revision_mi_diverse_weighted(llm, history, task, k, v)
        mi_vals.append(mi_v)
    return max(mi_vals)
```

Failure mode:

* If robust MI is always high, the controller may never answer. Mitigation: risk-controlled thresholding (next section) + cap number of questions.

---

## 2.5 Risk-Controlled Thresholding for "answer vs ask" (Clopper-Pearson UCB)

> **Note:** This is *not* classical conformal prediction. We use a binomial UCB approach to select a threshold τ that controls risk on a score-sorted prefix. See Papers 05, 26-28 for actual conformal methods.

This is the key “implementable” replacement for hand-tuned thresholds.

### 2.5.1 Goal

Choose a threshold (\tau) such that when the policy answers at states with (\widehat{I}_{rob}(h) \le \tau), the **error rate** is controlled at a target level (\delta) on the **visited-state distribution**.

### 2.5.2 Calibration data construction (on-policy)

For each training puzzle:

1. Run a *questioning policy* (use DP or your full method with a loose threshold).
2. At each visited state (h_t), compute:

   * `score_t = robust_mi(h_t)`
   * `pred_t = answer_from_revision(h_t)` (e.g., majority cluster of (Y^{(1)}))
   * `err_t = 1[pred_t != ground_truth]` (DC/GN exact; SP judged using AR-Bench evaluator)

Collect `(score_t, err_t, task, t, puzzle_id, homogeneity_t)`.

This is aligned with AR-Bench’s emphasis on **state-level** evaluation along trajectories.

### 2.5.3 Risk-controlled threshold selection (exact algorithm)

Use a binomial upper confidence bound on empirical risk among the *most confident* states.

**Algorithm (per task, per regime):**

* Sort calibration states by `score` ascending.
* For each prefix size (j), compute empirical error rate (\hat{r}*j = \frac{1}{j}\sum*{i=1}^j err_i).
* Compute Clopper–Pearson upper bound (U_j) for (Binomial(j, r)) at confidence (1-\alpha).
* Choose the **largest** (j^*) such that (U_{j^*} \le \delta).
* Set (\tau = score_{j^*}).

Pseudocode:

```python
def calibrate_tau(scores, errors, delta=0.10, alpha=0.05):
    idx = np.argsort(scores)
    scores = scores[idx]; errors = errors[idx]
    cum_err = np.cumsum(errors)
    best = None
    for j in range(1, len(scores)+1):
        k = int(cum_err[j-1])
        u = clopper_pearson_upper(k=k, n=j, alpha=alpha)
        if u <= delta:
            best = scores[j-1]
    if best is None:
        return -np.inf  # never answer (or relax delta)
    return best
```

**Notes:**

* Calibrate **separately** for:

  * each task (DC, SP, GN),
  * each decoding regime (normal vs homogeneous),
  * optionally each homogeneity bucket (multi-group calibration).

This addresses policy-induced shift by calibrating on visited states and is consistent with AR-Bench’s “process” emphasis.

### 2.5.4 Runtime stopping rule

At test time (DC/SP):

* If `robust_mi(h_t) <= tau`: answer now.
* Else: ask another question (unless max turns reached).

Edge cases:

* If at max turns and still `MI > tau`, answer anyway using best available answer; log as “forced answer”.

---

## 2.6 Question generation (shared across methods)

### 2.6.1 Candidate generation prompt (task-specific)

Implement `generate_candidates(history, M)`:

* **DC:** ask for a list of `{"suspect": name, "question": "...?"}`; enforce one suspect per question.
* **SP:** ask for M yes/no questions.
* **GN:** ask for M candidate guesses (4 unique digits), ideally info-seeking.

Output format: **JSON list**. Parse robustly with fallback regex.

### 2.6.2 Validity filters

* Remove duplicates by normalized text
* Enforce:

  * SP: must be yes/no form (classifier or heuristic: starts with “Is/Are/Did/Do/Can/Was/Were/Have/Has”)
  * GN: exactly 4 digits, all distinct
  * DC: mentions exactly one suspect

---

## 2.7 Full method (beyond MVE): VoI scoring by expected robust MI reduction + DPP slate selection

This is the “Ours (full)” system to compare to C‑IP and UoT end-to-end.

### 2.7.1 DPP slate selection (diversity)

Given M candidates (q_1,\dots,q_M), pick a diverse slate of size (m).

Implementation:

1. Embed questions: (e_i = Emb(q_i))
2. Similarity matrix: (S_{ij} = \exp(-|e_i - e_j|^2/\sigma^2))
3. Greedy DPP selection (approximate):

   * start with highest-quality question
   * iteratively add question maximizing marginal determinant gain

Hyperparameters:

* `m=8`, `sigma=0.5` (tune)
* quality weight: optional (e.g., length/validity score)

### 2.7.2 Outcome simulation for a candidate question

To score expected MI, you need a distribution over possible NPC answers **as the model believes it could happen** (not ground truth).

Implement a cheap simulation using the **policy model** as an “answerer simulator” (UoT uses simulated futures and an answerer simulator to branch outcomes).

For each candidate question (q):

* Prompt: “You are the NPC. Given the current puzzle and conversation so far, produce a plausible answer to q. (SP: output YES/NO/UNKNOWN.)”
* Sample `R=8` answers, cluster them into discrete answer-types.
* Use frequencies as probabilities (p(o|h,q)).

### 2.7.3 Expected robust MI after asking q

For each answer cluster (o):

* Construct hypothetical next history (h' = h \oplus (q, o))
* Compute `robust_mi(h')` using smaller `k_future=4` to save compute

Score:
[
\text{VoI}(q) = \widehat{I}*{rob}(h) - \sum_o p(o|h,q),\widehat{I}*{rob}(h')
]
Select (q^* = \arg\max \text{VoI}(q)).

Pseudocode:

```python
def score_question_by_expected_mi(llm, history, q, R=8):
    outcomes = [simulate_answer(llm, history, q) for _ in range(R)]
    clusters = cluster_outcomes(outcomes)
    p = cluster_frequencies(clusters)

    exp_mi = 0.0
    for o, prob in p.items():
        h2 = history_plus(history, q, canonicalize_outcome(o))
        exp_mi += prob * robust_mi(llm, h2, k_future=4)

    return robust_mi(llm, history) - exp_mi
```

Compute-saving tricks (mandatory under budget):

* Only run VoI scoring if `robust_mi(h) > tau` (i.e., we will ask anyway)
* Use `k_future=3–4` and only 1–2 prompt variants for hypothetical scoring
* Cache `robust_mi(h)` across candidates within the same step

Failure mode:

* Simulation collapses under homogeneity (all simulated answers identical). Mitigation: use **diversity-steered sampling** in simulation.

---

# 3. Baselines: Implementation Fidelity and Exact Specifications

We evaluate against multiple baseline families intended to represent (i) conformal prediction-set gating and set-size-driven querying, and (ii) information-gain-driven planning. Because AR-Bench is interactive and some policy models are accessed as black boxes (logits unavailable), several baselines require **adaptation** (e.g., estimating probabilities by sampling, constraining hypothesis spaces). To avoid over-claiming implementation fidelity, we refer to such methods as "**-style**" baselines: they are **inspired by** the original decision principle but are not guaranteed to reproduce every engineering choice of the canonical implementation.

All baselines share **the same environment wrapper, same NPC/judge, same maximum turn budget, same decoding regime**, and (for stopping-only comparisons) the same question generator.

| Baseline                  | What It Tests                   | Expected Result                |
| ------------------------- | ------------------------------- | ------------------------------ |
| Naive prompting           | Raw LLM behavior                | Premature guessing             |
| Self-consistency gate     | “Answer when high agreement”    | Collapses under homogeneity    |
| Verbalized confidence     | “Rate confidence 1-10”          | Overconfident due to RLHF      |
| Semantic entropy gate     | Semantic uncertainty proxy      | Collapses under homogeneity    |
| **KnowNo-style**          | Conformal set-size gate (baseline) | Set-size proxy fragile      |
| **C-IP-style**            | Conformal set-size for VoI (baseline) | Direct competitor        |
| **UoT-style**             | Uncertainty-aware planning      | Probability/EIG fragile        |
| EIG-based query selection | Expected info gain objective    | Fragile w/ miscalibrated probs |
| MI-only (no robustness)   | Naive MI                        | Fails under homogeneity        |
| **Ours (full)**           | Robust MI + risk-controlled τ   | Best under homogeneity         |

## 3.1 Naive prompting

Policy:

* Ask questions for a fixed number of turns (e.g., 10), then answer.
* Or “free-form”: single prompt that tells model to decide when to answer.

## 3.2 Self-consistency gate

At state (h):

1. Sample `k=10` answers (no revision).
2. Cluster answers semantically; let (p_{\max}) = largest cluster frequency.
3. Answer if (p_{\max} \ge \rho) (default (\rho=0.8)), else ask.

## 3.3 Verbalized confidence gate

Prompt:

* “Give your best answer and a confidence from 1–10.”
  Answer if confidence ≥ θ (e.g., 8).

Include calibration: map reported confidence to empirical accuracy (isotonic regression) and optionally use threshold on calibrated confidence.

## 3.4 Semantic entropy gate (Paper 09-style)

At state (h):

1. Sample `k` answers.
2. Cluster → distribution (p(c)).
3. Semantic entropy:
   [
   H_{sem}(h) = -\sum_c p(c)\log p(c)
   ]
   Answer if (H_{sem} \le \tau_H) (tune/calibrate).

## 3.5 KnowNo-style (adapted conformal prediction-set gate; inspired by Papers 26–28)

**Uncertainty object:** conformal prediction set \(C(h)\) over a discrete label space.

**Label space:**
* **DC:** \(Y = \{\text{5 suspects}\}\)
* **SP (adapted):** \(Y = \{\text{K semantic clusters of sampled explanations}\}\), where clusters are constructed via entailment-aware clustering (Paper 09). This is an approximation because the "true" explanation may not be present in the sampled support.

**Probability estimation:** If logits are available, use normalized softmax probabilities over labels. Otherwise, estimate \(\hat{p}(y \mid h)\) from \(k\) i.i.d. samples and semantic clustering (Paper 09).

**Split conformal calibration:** On a held-out calibration set of visited states, compute nonconformity scores \(s_i = 1 - \hat{p}(y_i \mid h_i)\) (for the ground-truth label \(y_i\)) and set \(q\) to the standard \((1-\alpha)\) split-conformal quantile.

**Prediction set:** \(C(h) = \{y : 1 - \hat{p}(y \mid h) \le q\}\) (equivalently \(\hat{p}(y \mid h) \ge 1 - q\)).

**Stopping rule:** Answer iff \(|C(h)| = 1\); otherwise ask another question.

## 3.6 C‑IP-style (adapted set-size–driven querying; inspired by Papers 05, 26–28)

C‑IP-style methods operationalize uncertainty via **prediction-set size** and select questions that minimize expected set size (as a surrogate for entropy).

### C‑IP-lite (stopping-only; for MVE isolation)

At each state, compute \(|C(h)|\) as above. Ask if \(|C(h)| > 1\); answer if \(|C(h)| = 1\) (or if the turn cap is reached).

### C‑IP-full (query selection)

1. Generate a candidate set of questions \(Q = \{q_1, \dots, q_M\}\) using the shared question generator.
2. For each \(q \in Q\), estimate a distribution over possible outcomes via simulation (same simulator used elsewhere): sample \(R\) NPC responses, cluster them into discrete outcomes \(o\), and set \(\hat{p}(o \mid h, q)\) by frequency.
3. For each outcome \(o\), form the hypothetical next history \(h' = h \oplus (q, o)\), recompute the conformal prediction set \(C(h')\), and record \(|C(h')|\).
4. Compute expected set size \(\mathbb{E}[|C(h')|] = \sum_o \hat{p}(o \mid h, q) \cdot |C(h')|\).
5. Select \(q^* = \arg\min_q \mathbb{E}[|C(h')|\).

We report this baseline explicitly as an *adapted* implementation because it depends on sampling-based probability estimation, semantic clustering for SP labels, and simulator-driven outcome modeling.

## 3.7 UoT-style (adapted uncertainty-of-thought planning; Paper 96)

We implement UoT-style planning for SP (yes/no questions), where outcomes can be discretized.

**Hypothesis/possibility set construction (fixed):** At state \(h\), construct a hypothesis set \(H = \{z_1, \dots, z_K\}\) by sampling \(K\) candidate SP explanations from the policy model and clustering them with entailment-aware clustering (Paper 09). Initialize a prior \(p(z)\) proportional to cluster frequency (uniform within cluster).

**Candidate questions:** Generate \(M\) candidate questions from the shared question generator (after validity filtering).

**Branch alphabet (fixed):** Outcomes are discretized to \(\{\textbf{YES}, \textbf{NO}, \textbf{UNKNOWN}\}\). The UNKNOWN branch includes refusals or ambiguous answers.

**NLI method (fixed):** To update hypotheses under a branch outcome, we use an NLI-based entailment scorer to predict the outcome implied by a hypothesis \(z\) for a question \(q\). Concretely, we template three statements ("Under story \(z\), the answer to \(q\) is YES/NO/UNKNOWN") and use a fixed NLI model to select the most entailed option; low-confidence cases map to UNKNOWN.

**Planning depth (fixed):** Depth \(d=2\): for each candidate question, compute expected posterior entropy after one question; for each branch, compute the best follow-up question using the same scoring rule and back up expected entropy reduction to select the first action (Paper 96).

**Stopping/answering:** Stop when posterior entropy \(H(p(z))\) drops below a fixed threshold or when the turn cap is reached; output the MAP hypothesis converted to the required final explanation format.

## 3.8 EIG-based query selection (Papers 97, 101)

We implement an EIG-style baseline that selects the next question by maximizing expected entropy reduction over a hypothesis set.

**Hypothesis set construction (fixed):**
* **GN:** hypotheses are the exact set of candidate secrets consistent with feedback (uniform prior).
* **SP:** hypotheses are \(K\) explanation clusters constructed as in §3.7 (Paper 09).

**Outcome simulation (fixed):**
* **GN:** outcomes are computed deterministically by the oracle feedback function for each hypothesis (exact partitioning).
* **SP:** outcomes are discretized to YES/NO/UNKNOWN. For each hypothesis \(z\) and question \(q\), we predict the implied outcome using the same fixed NLI method as in §3.7. This induces a partition of \(H\) and a predictive distribution \(\hat{p}(o \mid q) = \sum_z p(z) \cdot \mathbf{1}[\text{outcome}(z,q) = o]\).

**EIG objective:** Select \(q^* = \arg\max_q \left( H(p(z)) - \sum_o \hat{p}(o \mid q) H(p(z \mid o, q)) \right)\), where \(p(z \mid o, q) \propto p(z) \cdot \mathbf{1}[\text{outcome}(z,q) = o]\) with smoothing for empty branches.

**Stopping/answering:** Stop when hypothesis entropy is below a threshold or at the turn cap; answer with the MAP hypothesis (formatted appropriately).

## 3.9 MI-only (no robustness)

Use `estimate_self_revision_mi` with 1 variant and no diversity enforcement; tune a fixed threshold on dev.

## 3.10 Ours (full)

* Robust MI (`max` over prompt variants + diversity sampling)
* Conformal calibrated (\tau) on visited states
* VoI question selection by expected robust MI reduction
* DPP slate selection

---

# 4. Evaluation Metrics (exact definitions)

## 4.1 Primary

1. **Task accuracy**

   * DC: accuracy of identifying true murderer.
   * SP: character-level F1 between predicted explanation and ground truth.
   * GN: exact match rate of final predicted number.

2. **Question efficiency**

   * mean # questions asked until stop (DC/SP) or until solve (GN)
   * GN: compare to a near-optimal heuristic baseline (or a known solver) as “oracle efficiency”

## 4.2 Secondary

1. **Uncertainty calibration on visited states**

   * Convert uncertainty score (u(h)) to confidence (\hat{c}(h)) via monotone mapping (fit isotonic regression on calibration states):

     * for MI: lower MI → higher confidence
     * for set-size: smaller set → higher confidence
   * Compute **ECE** over visited states:
     [
     \text{ECE}=\sum_{b=1}^B \frac{|B_b|}{N}\left|\text{acc}(B_b)-\text{conf}(B_b)\right|
     ]
   * Reliability diagrams: bins on (\hat{c}(h))

2. **Question quality**

   * **Redundancy rate:** fraction of questions whose embedding cosine similarity to any previous question ≥ 0.9
   * **Information gain proxy:**

     * GN: true reduction in candidate space size (using constraint propagation)
     * SP/DC: reduction in uncertainty score or reduction in hypothesis-set entropy (if hypothesis filter is used)

3. **Abstention quality**

   * Risk–coverage curves: vary stopping threshold and plot accuracy vs fraction answered (DC/SP)

## 4.3 Multi-axis homogeneity characterization (diagnostics)

We treat "homogeneity" as a **multi-dimensional failure mode**, not a single scalar quantity. Two states can share similar mean embedding cosine similarity while reflecting qualitatively different degeneracies: one may exhibit low-dimensional semantic collapse, another may exhibit surface-level repetition, and a third may be brittle to meaning-preserving prompt changes. Because these failure modes affect uncertainty estimation differently, relying only on \(p_{\max}\) or mean cosine similarity risks conflating distinct phenomena.

**Reference implementation:** See `artificial-hivemind/src/model_calibration_analysis/` for baseline utilities.

We report a diagnostic suite spanning complementary axes:

### 1. Semantic collapse / concentration (Papers 06, 08)

* **Cluster max-frequency** \(p_{\max}\): largest cluster frequency from sampled answers
* **Mean pairwise similarity**: average cosine similarity across sample pairs
* **Effective rank**: compute eigenspectrum of sample–sample similarity matrix; summarize via \(\text{eff\_rank} = \exp(\text{entropy}(\lambda / \sum\lambda))\). Low effective rank indicates that sampled outputs occupy a low-dimensional semantic subspace, consistent with collapse even when pairwise similarity statistics are noisy (Paper 06).

### 2. Surface repetition ("repeat curse") — Paper 154

* **Exact-string repetition rate**: fraction of sample pairs with identical strings
* **N-gram overlap**: mean Jaccard similarity of 3-grams across samples
* **Paragraph-level repetition**: for SP explanations, sentence-level repetition rates

These measures capture degenerate copying and templating that embeddings can under-detect.

### 3. Prompt sensitivity — Papers 64-65 (FormatSpread/POSIX)

* Sample answers under V=5 meaning-preserving prompt variants
* **Sensitivity score**: \(1 - \text{agreement\_rate}\) across variants
* This axis helps distinguish epistemic uncertainty from instability induced by prompt phrasing
* **Key test:** Is robust MI measuring epistemic uncertainty or just prompt fragility?

### 4. Simulator-NPC divergence

* When VoI simulates NPC responses, track:
  * KL divergence between simulated outcome distribution and empirical NPC outcomes
  * Cluster overlap between simulated and actual responses
* Report: mean divergence per task, correlation with VoI effectiveness

### 5. Task-functional diversity — Paper 151

Task-specific diversity beyond embedding similarity:
* DC: distribution over accused suspects
* GN: distribution over digit constraints explored
* SP: diversity of question types (factual/causal/clarification)

### 6. Stratified evaluation

* **MI–error correlation**: Spearman correlation between robust MI and error indicator
* **Set-size–error correlation**: same but with conformal set size
* **Performance by homogeneity bucket**: High (\(H \ge 0.9\)), Medium (\(0.6 \le H < 0.9\)), Low (\(H < 0.6\))

These diagnostics are used for **analysis and stratification**, not as primary control signals: (i) to validate that "homogeneous regimes" correspond to measurable collapse, (ii) to interpret whether robust MI increases under true ambiguity versus prompt fragility, and (iii) to report performance as a function of homogeneity along multiple axes.

---

# 5. Ablation Studies (implementable)

| Ablation                           | Question Answered                      | Implementation                                        |
| ---------------------------------- | -------------------------------------- | ----------------------------------------------------- |
| − Risk-controlled τ                | Does calibration improve stability?    | Replace τ with a tuned fixed threshold                |
| − Robust MI (variants + diversity) | Does robustification prevent collapse? | Use MI with V=1 and no diversity                      |
| − Importance reweighting           | Does bias correction matter?           | Use rejection sampling without weights                |
| − Entailment clustering (SP)       | Does NLI help semantic clustering?     | Use embedding-only clustering for SP                  |
| − DPP diversification              | Does slate diversity help?             | Replace DPP with top-M or random subset               |
| − MI-based selection (use random)  | Does VoI scoring matter?               | Ask random valid question when uncertain              |
| MI → Self-consistency              | Is MI better than agreement?           | Swap MI score with (1-p_max)                          |
| MI → Set-size (C-IP style)         | Is MI better than set-size?            | Swap MI score with \|C(h)\| and keep rest fixed       |
| Episode-level vs state-level τ     | Sequential dependence impact?          | Calibrate with one state per episode (block bootstrap)|

---

# 6. Expected Results (and concrete success/failure criteria)

## 6.1 Success criteria

1. **Ours > C‑IP on DC/SP under homogeneous regime**

   * Higher accuracy/F1 at equal or fewer questions
2. **Lower premature-guess rate**

   * Define premature guess: answering at a state where calibration-set empirical error is high (or where key questions unresolved if you use AR-Bench key-question process scoring)
3. **MI–error correlation > set-size–error correlation** in high-homogeneity bucket
4. **Robust MI > Naive MI** (ablation shows clear gain in homogeneous regime)

## 6.2 Failure indicators

1. **C‑IP matches or beats ours** in high-homogeneity regime → MI not a better control signal
2. **Homogeneity rarely occurs** in practice (low mass in high-H bucket) → need stronger stress tests or the motivation weakens
3. **Robust MI ≈ Naive MI** → robustification not needed (or implemented incorrectly)
4. **Conformal calibration breaks under policy shift**

   * large gap between calibration risk and test risk → need on-policy / group calibration

---

# 7. Implementation Checklist (build order, concrete deliverables)

## Step 1: AR-Bench environment wrapper

* Parse AR-Bench datasets (DC/SP/GN)
* Implement `reset/step`
* Standardize action schema `{type: ASK/ANSWER, ...}`
* Logging: JSONL per episode with full transcript and timestamps

## Step 2: Semantic clustering module

* SentenceTransformer embedder
* Agglomerative/DBSCAN clustering
* Task-specific shortcuts (DC labels; GN exact digit string)

## Step 3: MI estimator (self-revision)

* Implement:

  * `prompt_initial(task, history)`
  * `prompt_revision(task, history, initial_answer)`
* Implement MI estimator with smoothing and robust parsing

## Step 4: Diversity-steered sampling

* Embedding-based rejection sampling utility
* Plug into MI sampling and into outcome simulation sampling

## Step 5: Prompt-variant generation

* Hardcode 3 variants for now (base/skeptical/alternative)
* Later: auto-generate variants with a meta-prompt (optional)

## Step 6: Risk-controlled calibration on visited states

* Run a data-collection policy on train puzzles
* Store (score, correctness) per state
* Implement `calibrate_tau` (Clopper–Pearson UCB)
* Save τ per task + per decoding regime + optional per homogeneity bucket

## Step 7: VoI scoring with simulated answers (full method)

* Implement answer-simulation prompts for SP/DC/GN
* Implement `score_question_by_expected_mi`
* Add caching + small k for hypothetical MI

## Step 8: DPP slate selection

* Implement greedy DPP selection
* Integrate into candidate pipeline

## Step 9: C‑IP baseline implementation

* Implement conformal prediction sets:

  * closed label set for DC
  * sampled-label set for SP
* Implement:

  * C‑IP-lite stopping
  * C‑IP-full query selection by minimizing expected set size

C‑IP uses prediction sets defined by probability thresholding and split conformal coverage.

## Step 10: Evaluation harness + metrics

* Batch-run methods across tasks and regimes
* Compute:

  * primary metrics
  * calibration metrics on visited states
  * risk–coverage curves
  * redundancy and efficiency
* Bootstrap CIs

## Step 11: Homogeneity analysis pipeline

* Implement homogeneity score (H) per state
* Stratified plots:

  * accuracy vs homogeneity bucket
  * MI vs error correlation vs bucket
  * set-size vs error correlation vs bucket

---

# 8. Risk Mitigation

| Risk                                 | Likelihood | Impact | Mitigation                                                                                                         |
| ------------------------------------ | ---------: | -----: | ------------------------------------------------------------------------------------------------------------------ |
| C‑IP beats us                        |     Medium |   High | Ensure homogeneous-regime stress tests; verify MI robustification actually increases variance when collapse occurs |
| Homogeneity rare                     |    Low–Med |   High | Evaluate both normal + forced homogeneous decoding; include explicit collapse prompts; report stratified results   |
| MI estimation too slow               |     Medium | Medium | Adaptive k (k=3 default, increase near threshold); cache clustering; use fewer variants for hypothetical VoI       |
| Risk-control shift on-policy         |     Medium | Medium | On-policy calibration (collect states under the same controller); group calibration by homogeneity bucket/task     |
| SP evaluation noise (judge variance) |     Medium | Medium | Fix judge model + decoding; run multiple seeds for judge; report mean±std                                          |
| Parsing / format failures            |     Medium |    Low | Enforce JSON outputs; implement regex fallbacks; validate and auto-repair actions                                  |

---

## Compute budget sanity check (fits <100 A100 hours)

A realistic MVE (DC+SP, stopping-only):

* per visited state MI: V=3 variants × k=6 samples × 2 generations (initial+revision) = 36 LLM calls/state
* but you only compute MI at the **decision point** (each turn) and you can reduce:

  * k=4, V=2 for MVE
* DC/SP test: 200 episodes × up to 25 turns = 5000 state-evals worst-case; with early stopping it’s much less.

Full VoI scoring is more expensive; constrain with:

* M=20 candidates → DPP to m=8 → score only 8
* outcome simulation R=6
* hypothetical MI uses k_future=3 and V_future=1

---

## Deliverable: what a coding agent should implement first (MVE sprint)

1. **DC-only** with DP question generation and compare stopping rules:

   * self-consistency gate
   * semantic entropy gate
   * KnowNo-style set-size gate
   * C‑IP-lite set-size gate
   * MI-only
   * **robust MI + risk-controlled τ**
2. Add homogeneity-regime decoding and produce:

   * accuracy vs average questions
   * MI–error vs set-size–error correlations
   * stratified by homogeneity bucket

That is the minimal, clean test of the core hypothesis before building full VoI selection.

---

# Limitations / Threats to Validity

**Disagreement-based uncertainty can miss confident errors.** Self-revision MI is fundamentally a disagreement-based proxy: it is high when the model's answer changes under revision and low when the model is self-consistent. This implies an important failure mode: the model can be **confidently wrong yet stable**, producing low MI even after revision. Robustification (prompt variants, diversity-steered sampling) increases the chance of surfacing alternative hypotheses, but it does not guarantee variability when the model's confabulation is entrenched. As a result, the controller can still answer prematurely with low MI. We therefore emphasize risk–coverage evaluation and qualitative failure analysis for low-MI/high-error states.

**Risk-controlled thresholding is not conformal prediction.** Our stopping policy calibrates a threshold \(\tau\) using a Clopper–Pearson binomial upper confidence bound on a held-out set of visited states. This provides a finite-sample confidence statement about calibration risk under approximate binomial/i.i.d. assumptions, but it is **not** a classical conformal prediction guarantee. In interactive settings, sequential dependence along trajectories and policy-induced distribution shift can violate these assumptions and weaken the validity of the bound. We therefore treat the procedure as conservative on-policy calibration and include sensitivity analyses (e.g., episode-level calibration / block-bootstrap checks) rather than claiming distribution-free validity.

**Simulator mismatch can make VoI anti-informative.** For question selection, VoI methods rely on a simulator (typically the policy model) to predict outcome distributions for candidate questions. If the simulator is misspecified relative to the actual NPC/environment dynamics, the estimated VoI can be noisy or even **anti-informative**, selecting questions that reduce model uncertainty while harming task performance. We explicitly monitor simulator–NPC divergence and report ablations that remove or weaken simulator dependence (e.g., DP question selection, random valid question when uncertain).

**More interaction is not always beneficial.** Additional questions do not monotonically improve performance: follow-up questions can introduce misleading information, amplify confabulations, or distract the agent from correct reasoning. MediQ reports that clarification dialogues can reduce performance when the questioning strategy is misaligned (Paper 95). Accordingly, we report both accuracy and interaction cost, enforce a strict turn cap, and evaluate whether the marginal utility of additional questions becomes negative in certain regimes.
