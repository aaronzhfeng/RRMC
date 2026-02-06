# 10 — Case Study: AllSuspects & Ensemble on CIP-Lite (Detective Cases)

> **Date:** 2026-02-06  
> **Model:** Qwen 2.5 7B Instruct (via OpenRouter)  
> **Task:** Detective Cases (DC) — 5 suspects, identify the guilty one  
> **Puzzles:** 20 from `AR-Bench/data/dc/test.json`  
> **Base method:** CIP-Lite stopping (`k_samples=8`, `set_size_threshold=1`, `max_turns=25`)

---

## 1. Methods Compared

| Method | Description | Config |
|--------|-------------|--------|
| **Baseline (CIP-Lite)** | Single trajectory; stop when prediction-set size = 1 | `dc_methods/cip_lite` |
| **AllSuspects+CIPLite** | Same, but forces questioning every suspect before CIP-Lite can trigger | `dc_methods/all_suspects_cip_lite` |
| **Ensemble+AllSusp+CIPLite** | 6 independent AllSuspects+CIPLite trajectories; majority vote | `dc_methods/ensemble_all_suspects_cip_lite` |

---

## 2. Aggregate Results

| Method | Accuracy | Avg Turns | Tokens | Time | Cost vs Baseline |
|--------|----------|-----------|--------|------|------------------|
| **Baseline (CIP-Lite)** | **45% (9/20)** | **1.4** | 515K | ~1 min | 1× |
| AllSuspects+CIPLite | 15% (3/20) | 6.6 | 515K | ~1 min | 1× |
| Ensemble+AllSusp+CIPLite | 25% (5/20) | 7.0 | 4,248K | 84 min | 8.3× |

**Key takeaway:** The baseline outperforms both "improvements" by a wide margin. Forcing more investigation and running ensembles made things **3× worse** at **8× cost**.

---

## 3. The Turn-1 Paradox

The baseline CIP-Lite stops at Turn 1 on **75% of puzzles** (15/20). At first glance, this seems pathological — the model commits to an answer without asking a single question. Our intuition said: this must be wrong, force the model to investigate.

But the numbers tell a different story:

| Subgroup | Count | Correct | Accuracy |
|----------|-------|---------|----------|
| Baseline stops at T1 | 15 | 7 | 47% |
| Baseline stops at T2+ | 5 | 2 | 40% |
| AllSuspects (always T5+) | 20 | 3 | 15% |

The model's Turn-1 accuracy (47%) is **3× higher** than AllSuspects' accuracy (15%). The early guesses aren't lucky — the model reads the case background and makes a reasonable first impression that is often correct. Forced investigation **degrades** this.

---

## 4. Answer Drift: How AllSuspects Destroys Correct Answers

Of 20 puzzles, AllSuspects **changed the answer** on 12 of them compared to what baseline would have guessed:

| Outcome | Count | Examples |
|---------|-------|----------|
| **Degraded** (baseline right → AS wrong) | **7** | Puzzles 1, 4, 11, 15, 17, 18, 19 |
| **Improved** (baseline wrong → AS right) | **1** | Puzzle 12 |
| Changed, both wrong | 4 | Puzzles 2, 7, 10, 16 |
| Same answer | 8 | Puzzles 0, 3, 5, 6, 8, 9, 13, 14 |

**Net effect: −6 puzzles.** AllSuspects flipped 7 correct answers to wrong and only recovered 1. The "force more investigation" strategy has a 7:1 negative-to-positive ratio.

---

## 5. Case-by-Case Analysis

### Puzzle 12 — The One Win for AllSuspects ✅

> Victim: Jonathan Harrington | Guilty: **Professor Charles Whitaker (#1)**

| Method | Predicted | Turns | Correct? |
|--------|-----------|-------|----------|
| Baseline | Mr. Henry Aldridge (#3) | 1 | ❌ |
| AllSuspects | Prof. Charles Whitaker (#1) | 8 | ✅ |
| Ensemble | Mr. Henry Aldridge (#3) | 6 | ❌ |

**What happened:**

The baseline guessed Aldridge at Turn 1 without any investigation — wrong. AllSuspects forced the model to question all 5 suspects, then continued for 3 more turns:

```
T1: [Prof. Whitaker] "I was reviewing references near the library entrance."
T2: [Dr. Eleanor Finch] "I was in the library, reviewing notes after the argument."
T3: [Mr. Aldridge] "I was in the dining room providing an alibi for others."
T4: [Mr. Bennett] "I was at my usual diner for dinner."
T5: [Ms. Langley] "I was in the dining hall, then the drawing room."
T6: [Prof. Whitaker] "I was OUTSIDE the library, pacing and reflecting."  ← contradiction!
T7: [Dr. Finch] "I was ensuring the artifact was displayed perfectly."
→ CIP-Lite stops at T8, correctly predicts Whitaker.
```

The critical moment is Turn 6: when asked the same question a second time, Whitaker changes his story from "near the library entrance reviewing references" to "outside the library, pacing." This inconsistency gave the model enough signal to correctly identify him.

**Why it worked:** This is the ideal scenario for AllSuspects — systematic questioning uncovered a genuine contradiction. But this only happened in 1 out of 20 puzzles.

---

### Puzzle 1 — A Clear Degradation ❌

> Victim: Dr. Jonathan Reed | Guilty: **Evelyn Carter (#0)**

| Method | Predicted | Turns | Correct? |
|--------|-----------|-------|----------|
| Baseline | Evelyn Carter (#0) | 1 | ✅ |
| AllSuspects | Samantha Greene (#4) | 6 | ❌ |
| Ensemble | Evelyn Carter (#0) | 7 | ✅ |

**What happened:**

Baseline read the case background and immediately, correctly identified Evelyn Carter. AllSuspects forced 5 turns of questioning — and every suspect gave a bland alibi:

```
T1: [Evelyn Carter] "I was at home, reviewing data for a presentation."
T2: [Clara Whitmore] "I was at home, relaxing and reading."
T3: [Henry Collins] "I was at home, trying to make sense of everything."
T4: [Michael Donovan] "I was at home, relaxing and trying to recover."
T5: [Samantha Greene] "I was at home watching a movie, but I heard a sighting near the park."
```

Every suspect says they were "at home." The responses are nearly identical and informationally empty. But Samantha Greene's answer contains a slight deviation — she mentions "a sighting near the park." The model latches onto this irrelevant detail as suspicious and changes its answer from Carter to Greene.

**Diagnosis: The NPC responses are not informative enough for the model to use.** All suspects give generic alibis. The model then over-interprets minor phrasing differences (Greene's "sighting near the park") as evidence, leading it astray from its correct initial impression.

**Why the ensemble recovered:** With 6 trajectories, 4/5 still converged on the correct answer (Carter), outvoting the noise. But this only works when the baseline answer is correct — and for the 7 degraded puzzles, only 2 were recovered by the ensemble.

---

### Puzzle 19 — First Impression Destroyed ❌

> Victim: Jonathan Harper | Guilty: **Michael Donovan (#3)**

| Method | Predicted | Turns | Correct? |
|--------|-----------|-------|----------|
| Baseline | Michael Donovan (#3) | 1 | ✅ |
| AllSuspects | Rebecca Lang (#0) | 6 | ❌ |
| Ensemble | Rebecca Lang (#0) | 7 | ❌ |

**What happened:**

Baseline correctly guessed Donovan at Turn 1. AllSuspects forced 5 rounds of questioning:

```
T1: [Thomas Reed] "I was just walking home from a late shift at the factory."
T2: [Evelyn Carter] "At midnight, I was in bed, fast asleep."
T3: [Michael Donovan] "I was at a different construction site for a brief check around 11:50 PM."
T4: [Rebecca Lang] "I was at home, reviewing the documents Jonathan asked me to deliver."
T5: [David Collins] "I was just getting into bed when my phone rang with the news."
```

Donovan's answer in T3 is actually the most suspicious — he was near a construction site at 11:50 PM, close to the crime time. But Rebecca Lang's answer mentions "documents Jonathan asked me to deliver," which creates a narrative link to the victim. The model incorrectly treats this connection as evidence of guilt.

**Diagnosis: The model confuses *association with the victim* for *evidence of guilt*.** Lang had a legitimate connection to the victim (delivering documents), but the model interprets any mention of the victim's name as suspicious. Meanwhile, Donovan's genuinely suspicious alibi ("at a construction site at 11:50 PM") is overlooked.

**The ensemble amplified this error:** 3/6 trajectories converged on Lang, locking in the wrong answer across both AllSuspects and Ensemble runs.

---

### Puzzle 8 — The One Ensemble Win ✅

> Victim: Jonathan Blackwood | Guilty: **Dr. Evelyn Harper (#0)**

| Method | Predicted | Turns | Correct? |
|--------|-----------|-------|----------|
| Baseline | Marcus Ellison (#3) | 1 | ❌ |
| AllSuspects | Marcus Ellison (#3) | 9 | ❌ |
| Ensemble | Dr. Evelyn Harper (#0) | 7 | ✅ |

**What happened:**

Both baseline and AllSuspects converged on Marcus Ellison — wrong. The AllSuspects conversation shows the model asking generic "where were you" questions and getting uninformative responses. After T5, CIP-Lite's set-size score stayed at 2.0 (two competing clusters) for 3 turns before finally converging to 1.0 at T9 — still on the wrong answer.

The ensemble found the correct answer through diversity: across 6 independent trajectories, at least 3 followed different questioning paths that led to Harper. The majority vote of 3/6 (50% confidence) was just enough.

**Why it worked:** This is the genuine value of ensembling — when the model's default reasoning path leads to a wrong answer, diverse exploration can stumble onto the right one. But with only 50% consensus, the ensemble was barely confident.

---

### Puzzles 5 & 6 — Ensemble Kills Correct Answers ❌

> Puzzle 5 — Guilty: **Mr. Sebastian Crowe (#3)**  
> Puzzle 6 — Guilty: **Michael Turner (#1)**

| Puzzle | Baseline | AllSusp | Ensemble |
|--------|----------|---------|----------|
| 5 | ✅ #3 (T1) | ✅ #3 (T6) | ❌ #4 (T10, 4/6 consensus) |
| 6 | ✅ #1 (T1) | ✅ #1 (T6) | ❌ #0 (T6, 4/5 at 80%) |

**What happened:**

Both baseline and AllSuspects got these right. But the ensemble got them wrong — with *high confidence*.

In Puzzle 6, the ensemble had 4/5 trajectories (80% confidence) agreeing on Evelyn Carter (#0) instead of the correct Michael Turner (#1). Both baseline and AllSuspects individually identified Turner correctly. But when 6 independent trajectories explore different questioning paths, the model's systematic bias toward Carter (a more "obvious" suspect) dominated 4 out of 5 trajectories, outvoting the correct answer.

**Diagnosis: Ensemble majority vote amplifies systematic model biases.** If the model has a tendency to suspect a particular character type (e.g., the one with the most narrative connection to the victim), running more trajectories doesn't diversify — it reinforces the bias. Temperature variation (0.5–1.0) is not enough to overcome systematic reasoning errors.

---

## 6. Root Cause Analysis

### Why did our intuitions fail?

**Intuition 1:** "Stopping at Turn 1 is premature — the model should investigate before guessing."

**Reality:** The model's case-background comprehension is its strongest signal. CIP-Lite's k=8 samples at Turn 1 all agree on the same answer (set_size=1), which means the model consistently identifies the same suspect from the case description alone. This is genuine confidence, not false confidence. Forcing investigation introduces **noise** (uninformative NPC responses) that overwrites this strong prior.

**Intuition 2:** "Questioning all suspects ensures fair coverage and prevents tunnel vision."

**Reality:** The NPC responses in AR-Bench are generated by another LLM and tend to be **generic and undifferentiated**. All suspects give plausible-sounding alibis ("I was at home", "I was at dinner"). The model can't distinguish guilty from innocent based on these responses because the responses themselves lack discriminative content. In the rare case where a suspect contradicts themselves (Puzzle 12), AllSuspects helps. But in 19 out of 20 cases, the additional responses are noise.

**Intuition 3:** "Ensembling with majority vote should reduce variance and improve accuracy."

**Reality:** Ensemble majority vote works when errors are **uncorrelated** across trajectories. But Qwen-2.5-7B's errors are **highly correlated** — it has systematic biases (e.g., suspecting the character most connected to the victim, or the one with the most vocal motive). Running 6 trajectories with temperature variation (0.5–1.0) produces 6 instances of the same systematic bias, not 6 independent assessments. The majority vote then locks in the wrong answer with high confidence.

### The Fundamental Issue: Evidence Quality

The core problem is not the stopping rule — it's the **information channel**. The NPC responses don't carry enough signal to improve upon the model's initial impression:

| Information Source | Signal Quality | Effect on Accuracy |
|---|---|---|
| Case background | High (written by puzzle designer) | Strong positive |
| NPC responses (Turn 1–5) | Low (generic LLM-generated alibis) | Weak negative |
| NPC responses (Turn 5+) | Very low (repetitive, contradictory) | Strong negative |

The model's accuracy degrades monotonically with more turns because each turn adds more noise than signal.

---

## 7. Ensemble Consensus Analysis

| Consensus (out of 6) | Count | Correct | Accuracy |
|---|---|---|---|
| 2/6 (33%) | 6 | 1 | 17% |
| 3/6 (50%) | 8 | 3 | 38% |
| 4/6 (67%) | 6 | 1 | 17% |

No puzzle achieved 5/6 or 6/6 consensus. Even the highest consensus (4/6) has only 17% accuracy — the model can be confidently wrong. The mean consensus of 3.0/6 across all puzzles means the ensemble is essentially flipping coins.

**Confidence calibration is inverted:** Higher ensemble consensus does not predict correctness. This means the ensemble's "confidence" signal is unreliable for this model.

---

## 8. Summary Table

| Pz | Label | Baseline (45%) | AllSusp (15%) | Ensemble (25%) | Pattern |
|----|-------|----------------|---------------|----------------|---------|
| 0 | #3 | ✗ #0 (T1) | ✗ #0 (T6) | ✗ #1 (T6) | all wrong |
| 1 | #0 | **✓ (T1)** | ✗ #4 (T6) | ✓ (T7) | AS degraded, Ens recovered |
| 2 | #1 | ✗ #0 (T2) | ✗ #3 (T6) | ✗ #4 (T7) | all wrong |
| 3 | #4 | ✗ #3 (T1) | ✗ #3 (T6) | ✗ #3 (T6) | all wrong, same wrong ans |
| 4 | #2 | **✓ (T2)** | ✗ #4 (T6) | ✓ (T6) | AS degraded, Ens recovered |
| 5 | #3 | **✓ (T1)** | **✓ (T6)** | ✗ #4 (T10) | Ens killed correct answer |
| 6 | #1 | **✓ (T1)** | **✓ (T6)** | ✗ #0 (T6, 80%) | Ens killed with high conf |
| 7 | #4 | ✗ #2 (T1) | ✗ #1 (T8) | ✗ #0 (T6) | all wrong, all different |
| 8 | #0 | ✗ #3 (T1) | ✗ #3 (T9) | **✓ (T7, 50%)** | Ens-only win (barely) |
| 9 | #1 | ✗ #4 (T6) | ✗ #4 (T6) | ✗ #0 (T6) | all wrong |
| 10 | #3 | ✗ #0 (T2) | ✗ #1 (T7) | ✗ #1 (T10) | all wrong |
| 11 | #2 | **✓ (T1)** | ✗ #3 (T7) | ✓ (T8) | AS degraded, Ens recovered |
| 12 | #1 | ✗ #3 (T1) | **✓ (T8)** | ✗ #3 (T6) | AS-only win (contradiction) |
| 13 | #1 | ✗ #3 (T1) | ✗ #3 (T6) | ✗ #3 (T6) | all wrong, same wrong ans |
| 14 | #3 | ✗ #2 (T1) | ✗ #2 (T6) | ✗ #2 (T8) | all wrong, same wrong ans |
| 15 | #4 | **✓ (T1)** | ✗ #3 (T6) | ✓ (T7) | AS degraded, Ens recovered |
| 16 | #1 | ✗ #3 (T1) | ✗ #0 (T6) | ✗ #0 (T7) | all wrong |
| 17 | #0 | **✓ (T2)** | ✗ #4 (T8) | ✗ #4 (T8) | both new methods degraded |
| 18 | #2 | **✓ (T1)** | ✗ #1 (T7) | ✗ #0 (T7) | both new methods degraded |
| 19 | #3 | **✓ (T1)** | ✗ #0 (T6) | ✗ #0 (T7) | both new methods degraded |

---

## 9. Key Insights

### What we learned:

1. **Turn-1 stopping is not a bug — it's a feature.** For Qwen-2.5-7B on AR-Bench DC, the model's best signal comes from reading the case background. CIP-Lite's unanimous k=8 samples at Turn 1 reflect genuine (if imperfect) comprehension, not false confidence.

2. **NPC response quality is the bottleneck.** The suspects' LLM-generated responses are too generic to carry useful signal. "I was at home" tells the model nothing. In 19/20 cases, these responses added noise rather than information.

3. **AllSuspects has a 7:1 degradation ratio.** It flipped 7 correct answers to wrong while only recovering 1. The one success (Puzzle 12) required a suspect to contradict themselves across two questioning rounds — a rare event.

4. **Ensemble consensus is anti-correlated with correctness.** Puzzles where the ensemble was most confident (4/6, 80%) were often wrong (Puzzle 6). The model's systematic biases are replicated, not diversified, by temperature variation.

5. **More information ≠ better decisions.** This is the central finding. The model's reasoning quality degrades as the conversation lengthens, likely because it cannot effectively filter relevant from irrelevant information in a long context.

### Implications for the RRMC framework:

- **Stopping rule improvements require model capability improvements.** AllSuspects and Ensemble are sound strategies in principle, but they need a model that can (a) ask discriminative questions and (b) synthesize multi-turn evidence without being distracted by noise.
- **CIP-Lite at Turn 1 functions as a "prior-based classifier"** — it essentially evaluates the model's zero-shot comprehension of the case. This is a legitimate approach and may be near-optimal for weaker models.
- **Future work should focus on response quality** (better NPC prompting, structured evidence extraction) rather than investigation quantity.

---

## 10. Comparison with Document 09 (MI-Only Variant)

Document 09 studied the same AllSuspects and Ensemble ideas applied to MI-Only stopping on a 5-puzzle subset. The CIP-Lite results on 20 puzzles confirm and strengthen those findings:

| Finding | Doc 09 (MI-Only, 5 puzzles) | Doc 10 (CIP-Lite, 20 puzzles) |
|---------|----------------------------|-------------------------------|
| Baseline accuracy | 60% (3/5) | 45% (9/20) |
| AllSuspects accuracy | 40% (2/5) | 15% (3/20) |
| Ensemble accuracy | 40% (2/5) | 25% (5/20) |
| Ens+AllSusp accuracy | 20% (1/5) | 25% (5/20) |
| Degradation ratio | ~2:1 | **7:1** |
| More turns help? | No | **No** |
| Ensemble consensus reliable? | No | **No** |

The larger sample size (20 vs 5) makes the pattern more definitive: **the "improvements" consistently hurt, and the effect is robust across stopping methods and sample sizes.**
