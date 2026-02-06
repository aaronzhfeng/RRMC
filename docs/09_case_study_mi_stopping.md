# 09 — Case Study: Stopping Rule Improvements on Detective Cases

> **Date:** 2026-02-06  
> **Model:** Qwen 2.5 7B Instruct (via OpenRouter)  
> **Task:** Detective Cases (DC) — 5 suspects, identify the guilty one  
> **Puzzles:** First 5 from `AR-Bench/data/dc/test.json`  
> **Base method:** MI-Only stopping (`mi_threshold=0.01`, `k_samples=6`, `max_turns=25`)

---

## 1. Methods Compared

| Method | Description | Config |
|--------|-------------|--------|
| **Baseline (MI-Only)** | Single trajectory; stop when MI drops below 0.01 | `dc_methods/mi_only` |
| **AllSuspects** | Same as baseline but forces the model to question every suspect at least once before MI-based stopping can trigger | `dc_methods/all_suspects_mi_only` |
| **Ensemble(MI)** | Runs 6 independent trajectories per puzzle, each using MI-Only stopping; final answer is majority vote | `dc_methods/ensemble_mi_only` |
| **Ensemble+AllSuspects** | Runs 6 independent trajectories per puzzle, each wrapped with AllSuspects; majority vote | `dc_methods/ensemble_all_suspects_mi_only` |

---

## 2. Aggregate Results

| Method | Accuracy | Avg Turns | API Calls | Relative Cost |
|--------|----------|-----------|-----------|---------------|
| **Baseline (MI-Only)** | **60% (3/5)** | 9.2 | 627 | 1× |
| AllSuspects | 40% (2/5) | 18.8 | ~1,000 | ~1.6× |
| Ensemble(MI) | 40% (2/5) | 4.6 | 1,851 | ~3× |
| Ensemble+AllSuspects | 20% (1/5) | 15.2 | 4,548 | ~7× |

**Key takeaway:** The simple baseline outperforms all "improved" variants. More turns and more trajectories made things _worse_, not better.

---

## 3. Ground Truth (First 5 Puzzles)

| Puzzle | Suspects | Guilty |
|--------|----------|--------|
| 0 | Prof. Harold Fenwick, Dr. Margaret Langley, Mr. Oliver Grant, **Ms. Clara Whitmore**, Mr. Theodore Blake | **#3 — Ms. Clara Whitmore** |
| 1 | **Evelyn Carter**, Michael Donovan, Clara Whitmore, Henry Collins, Samantha Greene | **#0 — Evelyn Carter** |
| 2 | Eleanor Blackwood, **Henry Caldwell**, Charlotte Everly, Victor Langston, Beatrice Wainwright | **#1 — Henry Caldwell** |
| 3 | Alex Turner, Emily Davis, David Mitchell, Sophia Reynolds, **Michael Harris** | **#4 — Michael Harris** |
| 4 | Professor Harold Greene, Ms. Clara Thompson, **Mr. Jonathan Reed**, Ms. Vivian Clarke, Mr. Peter Langley | **#2 — Mr. Jonathan Reed** |

---

## 4. Case-by-Case Analysis

### Puzzle 0 — Guilty: Ms. Clara Whitmore (#3) — ❌ All Methods Wrong

| Method | Predicted | Turns | Confidence / MI |
|--------|-----------|-------|-----------------|
| Baseline | Dr. Margaret Langley (#1) | 3 | MI: 0.13 → 0.87 → 0.0 (stopped) |
| AllSuspects | Prof. Harold Fenwick (#0) | 25 (max) | MI never settled; ranged 0.4–1.3 for 20 turns |
| Ensemble(MI) | Prof. Harold Fenwick (#0) | 2 | 4/5 vote, 80% confidence |
| Ens+AllSuspects | Mr. Theodore Blake (#4) | 14 | 2/6 vote, 33% confidence |

**What happened:**

- **Baseline** asked only 2 suspects (Blake and Langley) before MI collapsed to 0.0 at Turn 3. It locked onto Dr. Langley (#1) — wrong, but at least it stopped quickly.
- **AllSuspects** was forced to question all 5 suspects (Turns 1–5 show MI counting down as a "not yet" signal). After Turn 5, MI stayed high (0.4–1.3) for 20 more turns, meaning the model remained _uncertain_ throughout. It hit the 25-turn maximum and guessed Prof. Fenwick (#0) — also wrong.
- **Ensemble(MI)** ran 6 short trajectories averaging 2 turns each. Four of five valid trajectories agreed on Fenwick (#0) with 80% confidence — wrong with high certainty.
- **Ens+AllSuspects** is the only method that shows _low_ confidence (33%), correctly signaling uncertainty. But it still picked wrong (Blake, #4).

**Diagnosis:** This is a genuinely hard puzzle. No method found Clara Whitmore. The AllSuspects decision log is revealing — MI oscillated between 0.4 and 1.3 for 20 turns without ever converging, meaning the model was perpetually confused. More questioning didn't help.

---

### Puzzle 1 — Guilty: Evelyn Carter (#0) — ✅ All Methods Correct

| Method | Predicted | Turns | Confidence / MI |
|--------|-----------|-------|-----------------|
| Baseline | Evelyn Carter (#0) ✓ | 17 | MI fluctuated, dropped to 0.0 at T17 |
| AllSuspects | Evelyn Carter (#0) ✓ | 9 | MI dropped to 0.0 at T9 |
| Ensemble(MI) | Evelyn Carter (#0) ✓ | 3 | 4/4 vote, 100% confidence |
| Ens+AllSuspects | Evelyn Carter (#0) ✓ | 17 | 4/5 vote, 80% confidence |

**What happened:**

This is an "easy" puzzle — the model quickly identifies Evelyn Carter across all methods.

- **Baseline** took 17 turns with MI bouncing around (0.13 → 0.87 → 0.55 → ... → 1.33 → 0.0), but converged correctly.
- **AllSuspects** was actually _faster_ here (9 turns vs 17), because after questioning all 5 suspects by Turn 5, the MI needed only 4 more turns to reach 0.
- **Ensemble** was the fastest at 3 turns with unanimous agreement across trajectories.

**Diagnosis:** When the puzzle is easy, all methods work. Ensemble is most efficient; AllSuspects is sometimes faster than baseline because it systematically gathers evidence from all suspects.

---

### Puzzle 2 — Guilty: Henry Caldwell (#1) — 🔴 AllSuspects+Ensemble Breaks It

| Method | Predicted | Turns | Confidence / MI |
|--------|-----------|-------|-----------------|
| Baseline | Henry Caldwell (#1) ✓ | 7 | MI: 0.13 → 1.33 → 0.64 → 1.1 → 0.22 → 1.56 → 0.0 |
| AllSuspects | Henry Caldwell (#1) ✓ | 10 | MI dropped to 0.0 at T10 |
| Ensemble(MI) | Henry Caldwell (#1) ✓ | 1 | 4/6 vote, 67% confidence |
| **Ens+AllSuspects** | **Eleanor Blackwood (#0) ✗** | **19** | **4/5 vote, 80% confidence — wrong** |

**What happened:**

- **Baseline** correctly identified Caldwell in 7 turns. The MI trace shows genuine oscillation — the model gathered evidence and converged.
- **AllSuspects** also got it right in 10 turns (5 forced + 5 MI-based).
- **Ensemble(MI)** got it right in just 1 turn — 4/6 trajectories agreed.
- **Ens+AllSuspects** is the critical failure case: after forcing 6 independent trajectories to each question all 5 suspects, 4 out of 5 trajectories converged on Eleanor Blackwood (#0) with **80% confidence**. The model had the right answer early, but after extensive questioning, it _changed its mind_ and committed to the wrong answer with high confidence.

**Diagnosis: This is the "smoking gun" for why more turns can hurt.** The model's reasoning degrades with more information. Early on, the model correctly suspects Caldwell. But after 19 turns of questioning, contradictory and irrelevant responses from suspects muddy the model's reasoning, and it flip-flops to Blackwood. The ensemble then amplifies this wrong answer through majority vote.

**Conversation excerpt from the Baseline (correct):**
```
T1: Asked Victor Langston — "What were you doing with the antique marble bust?"
    → "I was discussing art pieces with Charlotte Everly, far from the bust."
T2: Asked Charlotte Everly — "Where were you at 8pm?"
    → "I was in the dining hall for dinner with the other guests."
T3: Asked Eleanor Blackwood — "Who were you discussing the hostile takeover with?"
    → "I was alone in the library, contemplating business prospects."
T5: Asked Henry Caldwell — "What were you doing in the library around 8pm?"
    → "I was just there to get away from everyone, catch my breath."
T6: Asked Victor Langston — "Did you have disputes with Reginald Thorne?"
    → "Not that I recall; our conversations were professional."
→ MI drops to 0.0 at Turn 7, correctly predicts Caldwell.
```

The baseline gathered just enough evidence and stopped. The ensemble variants over-investigated.

---

### Puzzle 3 — Guilty: Michael Harris (#4) — ❌ All Methods Wrong (Same Answer)

| Method | Predicted | Turns | Confidence / MI |
|--------|-----------|-------|-----------------|
| Baseline | Sophia Reynolds (#3) ✗ | 9 | MI: 0.13 → 0.17 → 0.78 → ... → 0.0 |
| AllSuspects | Alex Turner (#0) ✗ | 25 (max) | MI oscillated 0.3–1.1, never converged |
| Ensemble(MI) | Sophia Reynolds (#3) ✗ | 10 | 3/6 vote, 50% confidence |
| Ens+AllSuspects | Sophia Reynolds (#3) ✗ | 14 | 4/6 vote, 67% confidence |

**What happened:**

- **Baseline** converged on Sophia Reynolds (#3) in 9 turns.
- **AllSuspects** picked a _different_ wrong answer (Alex Turner, #0) after hitting max turns. Its MI never dropped below 0.01, oscillating between 0.3 and 1.1 for 20 turns — the model was perpetually uncertain.
- **Ensemble** and **Ens+AllSuspects** both picked Sophia Reynolds with low-to-moderate confidence (50% and 67%).

**Conversation excerpt (Baseline):**
```
T1: Asked Michael Harris — "Did you see anyone suspicious near the warehouse?"
    → "I didn't see anything out of the ordinary; it was dark and quiet."
T2: Asked Alex Turner — "What were you doing near the warehouse at 8pm?"
    → "I was already long gone by then, heading home."
T5: Asked David Mitchell — "Were you the last person to leave the warehouse?"
    → "Yes, I left at 7:45 PM after completing my audit."
T6: Asked Alex Turner — "Do you have an alibi?"
    → "I was at a nearby café getting work done around 8pm."
```

**Diagnosis: Systematic reasoning failure.** The model never picks up on the right clues about Michael Harris (#4). Harris's response in Turn 1 is evasive ("didn't see anything, dark and quiet"), which should be suspicious, but the model doesn't recognize it. All methods converge on wrong answers because the model fundamentally misreads the evidence. This is a model capability limitation, not a stopping rule problem.

---

### Puzzle 4 — Guilty: Mr. Jonathan Reed (#2) — 🔴 Ensemble Kills the Correct Answer

| Method | Predicted | Turns | Confidence / MI |
|--------|-----------|-------|-----------------|
| **Baseline** | **Mr. Jonathan Reed (#2) ✓** | **10** | MI: 0.13 → 0.45 → 0.78 → ... → 1.1 → 0.0 |
| AllSuspects | Prof. Harold Greene (#0) ✗ | 25 (max) | MI stuck at 1.0 for turns 6–8, then oscillated |
| Ensemble(MI) | Prof. Harold Greene (#0) ✗ | 7 | 4/5 vote, 80% confidence |
| Ens+AllSuspects | Prof. Harold Greene (#0) ✗ | 12 | 4/6 vote, 67% confidence |

**What happened:**

- **Baseline** correctly identified Reed in 10 turns. MI oscillated and then dropped to 0.0.
- **AllSuspects** hit max turns (25) — MI was stuck exactly at 1.0 for Turns 6–8 (while still inside the "not all suspects questioned" phase), then oscillated 0.2–1.2 without converging. Ended up guessing Prof. Greene (#0).
- **Ensemble** ran 6 trajectories; 4/5 agreed on Prof. Greene with 80% confidence — wrong.
- **Ens+AllSuspects** similarly converged on Greene.

**Conversation excerpt (Baseline, correct):**
```
T1: Asked Mr. Jonathan Reed — "What were you doing with the antique marble bust at 7pm?"
    → "I was ensuring the bust was properly displayed for Dr. Blackwood's presentation."
T3: Asked Professor Harold Greene — "Have you ever felt threatened by Dr. Blackwood's book?"
    → "Absolutely, her book has been a constant source of frustration and threat to my research."
T5: Asked Ms. Vivian Clarke — "What were you researching before and after 7pm?"
    → "I was excited to attend Dr. Blackwood's presentation and take notes."
T9: Asked Mr. Peter Langley — "Do you have an alibi?"
    → "I was at a café a few blocks away, but I could have left early if asked."
→ MI drops to 0.0 at Turn 10, correctly predicts Reed.
```

**Diagnosis: The ensemble's majority vote amplified the wrong answer.** The single baseline trajectory happened to follow a line of questioning that led to the correct conclusion. But when 6 trajectories are run independently, 4 out of 5 valid ones converge on Prof. Greene — the model's "default" suspect due to his vocal hostility about Dr. Blackwood's book. Greene is the "obvious" suspect (he admitted feeling threatened), but Reed is the actual culprit. The single correct trajectory was outvoted.

---

## 5. MI Behavior Analysis

### AllSuspects Decision Logs

The AllSuspects wrapper uses a countdown mechanism for the first 5 turns (MI = 5, 4, 3, 2, 1), blocking the stop signal until all suspects are questioned. After Turn 5, real MI kicks in:

| Puzzle | Turns 6–25 MI Range | Converged? | Final Outcome |
|--------|---------------------|------------|---------------|
| 0 | 0.41–1.33 | ❌ Never (hit max) | Wrong |
| 1 | 0.64–0.69 → 0.0 | ✅ Turn 9 | Correct |
| 2 | 0.45–1.10 → 0.0 | ✅ Turn 10 | Correct |
| 3 | 0.32–1.10 | ❌ Never (hit max) | Wrong |
| 4 | 0.22–1.24 | ❌ Never (hit max) | Wrong |

**Pattern:** When MI converges to 0 (Puzzles 1, 2), the model is correct. When MI oscillates forever without converging (Puzzles 0, 3, 4), the model is confused and hits max turns — and the final answer at max turns is wrong.

### Baseline MI Traces

In contrast, the baseline's MI _always_ converges to 0 (because it doesn't have the forced questioning overhead that introduces noise):

| Puzzle | Turns to MI=0 | Correct? |
|--------|---------------|----------|
| 0 | 3 | ❌ (too fast) |
| 1 | 17 | ✅ |
| 2 | 7 | ✅ |
| 3 | 9 | ❌ |
| 4 | 10 | ✅ |

The baseline converges faster but sometimes prematurely (Puzzle 0: only 3 turns). However, its 60% accuracy is still the best across all methods.

---

## 6. Root Cause Analysis

### Why do "improvements" make things worse?

**1. Ensemble amplifies systematic bias (Puzzles 3, 4)**

When the model has a systematic blind spot (e.g., always suspects the "most vocal" character), running 6 trajectories doesn't create diversity — all 6 make the same mistake. Majority voting then locks in the wrong answer with high confidence. In Puzzle 4, the one correct trajectory was outvoted 4-to-1.

**2. More turns → more confusion (Puzzles 0, 2, 4)**

Qwen 7B doesn't synthesize multi-turn evidence well. After questioning all suspects, the model often flip-flops. In Puzzle 2, the model had the right answer early but changed its mind after 19 turns of questioning. The suspects' answers contain distractors (irrelevant details, evasive non-answers) that the model can't filter out.

**3. MI oscillation as a "confusion signal" (Puzzles 0, 3, 4)**

When MI stays high (0.3–1.3) and never converges, it means the model's initial and revised answers keep disagreeing — the model is persistently uncertain. In the AllSuspects variant, this leads to hitting max turns (25) and guessing — effectively random.

**4. Temperature sampling doesn't create meaningful diversity**

The ensemble's 6 trajectories should ideally explore different investigative strategies. In practice, the model asks similar questions in a similar order across trajectories, leading to correlated errors. True diversity would require different prompting strategies, not just temperature variation.

---

## 7. Key Insights

### What works:
- **MI-based stopping is effective when it converges.** In Puzzles 1, 2, 4, the baseline's MI dropping to 0 coincided with correct answers.
- **Low confidence signals are accurate.** Ens+AllSuspects showed 33% confidence on Puzzle 0 — correctly indicating uncertainty.

### What doesn't work with Qwen 7B:
- **Ensembling** — correlated errors across trajectories mean majority vote doesn't help
- **Forced extensive questioning** — the model can't synthesize 20+ turns of evidence
- **Combining both** — the worst of both worlds: 7× cost for 3× worse accuracy

### Implications for future work:
1. **Stronger models** (GPT-4, Claude) may benefit from these techniques because they can better synthesize multi-turn evidence and produce more diverse reasoning paths
2. **Confidence calibration** is needed — the ensemble votes 80% on wrong answers
3. **Adaptive stopping** could use MI oscillation as a "I'm confused" signal, triggering a different strategy rather than continuing to question
4. **Diverse prompting** (e.g., chain-of-thought vs. direct, different suspect orderings) might create the trajectory diversity that temperature alone cannot

---

## 8. Summary Table

| Puzzle | Label | Baseline (60%) | AllSuspects (40%) | Ens(MI) (40%) | Ens+All (20%) |
|--------|-------|-----------------|-------------------|---------------|---------------|
| 0 | #3 Clara Whitmore | ✗ #1 (3 turns) | ✗ #0 (25 turns) | ✗ #0 (2 turns) | ✗ #4 (14 turns) |
| 1 | #0 Evelyn Carter | ✓ (17 turns) | ✓ (9 turns) | ✓ (3 turns) | ✓ (17 turns) |
| 2 | #1 Henry Caldwell | ✓ (7 turns) | ✓ (10 turns) | ✓ (1 turn) | **✗ #0 (19 turns)** |
| 3 | #4 Michael Harris | ✗ #3 (9 turns) | ✗ #0 (25 turns) | ✗ #3 (10 turns) | ✗ #3 (14 turns) |
| 4 | #2 Jonathan Reed | **✓ (10 turns)** | ✗ #0 (25 turns) | **✗ #0 (7 turns)** | **✗ #0 (12 turns)** |

**The simple baseline wins.** Each "improvement" added cost and reduced accuracy.

---

## Appendix: Bug Fixes Applied Before These Runs

Two critical bugs were fixed before the final Ensemble and Ens+AllSuspects runs:

1. **Double-parsing bug** (`evaluator.py`): `evaluate_method_ensemble` called `_parse_dc_answer()` on predictions that were already 0-indexed integers, shifting every answer by −1. Fix: use `int(ensemble_result.final_answer)` directly.

2. **List-in-answer bug** (`mi_estimator.py`): `_parse_answer()` could return a list `["Suspect Name"]` instead of a string when the LLM's JSON had `"suspect": ["name"]`. This crashed the clustering module's `.strip()` call. Fix: unwrap single-element lists.
