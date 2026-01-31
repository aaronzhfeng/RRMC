# Conversation Analysis: MI-Only Validation Run

## Summary of Findings

### Turn-1 Cases (14/20 puzzles, 28.6% accuracy)

**What happens:**
- MI = 0.0 at Turn 1 → Immediate stop
- **No questions asked** - conversation record is empty
- Model predicts based on puzzle description alone (pure guessing)

**Example - Puzzle 0 (WRONG):**
```
Prediction: 0 | Ground Truth: 3
MI Scores: [0.0]
Conversation: [] (EMPTY - no questions asked!)
```

**Example - Puzzle 1 (CORRECT):**
```
Prediction: 0 | Ground Truth: 0  
MI Scores: [0.0]
Conversation: [] (EMPTY - lucky guess)
```

**Why MI = 0:**
All k=6 samples at Turn 1 give identical predictions → zero entropy → MI = 0
The model is "confident" but has no information.

---

### Multi-Turn Cases (6/20 puzzles, 16.7% accuracy)

**What happens:**
- MI > 0 at Turn 1 → Continue asking questions
- Model asks suspects about their whereabouts/alibis
- MI fluctuates but eventually drops to 0 or hits max turns

**Example - Puzzle 17 (CORRECT, 8 turns):**
```
MI Scores: [0.22, 0.64, 0.78, 0.41, 0.87, 0.64, 1.01, 0.0]

Turn 1: Asked Eleanor Whitaker - "What were you doing near Jonathan Blake's studio?"
Turn 2: Asked Michael Turner - "Who was with you at the real estate project?"  
Turn 3: Asked Clara Mitchell - "Did you have disagreements with Jonathan Blake?"
...
Turn 8: MI drops to 0 → Stop → Correct prediction!
```

**Example - Puzzle 18 (WRONG, 25 turns = max):**
```
MI Scores: [0.32, 0.26, 1.24, 0.87, 0.87, 1.10, 0.22, ...]

Turn 1: Asked Clara Bennett - "What were you doing in office 9-10pm?"
Turn 2: Asked Marcus Langley - "Did you have disputes with Jonathan?"
Turn 3: Asked Dr. Evelyn Harper - "Were you in office 9-10pm?"
...
Turn 22: Asked Marcus Langley - SAME QUESTION as earlier!
Turn 23: Asked Dr. Evelyn Harper - SAME QUESTION as earlier!
```

**Problems identified:**
1. Model asks repetitive questions (same suspect, same question)
2. MI never stabilizes → runs to max turns
3. Still gets wrong answer despite 25 turns of questioning

---

## Key Insights

### 1. MI = 0 Does NOT Mean "Enough Information"
- At Turn 1 with no questions asked, MI = 0 means "all samples agree"
- But this is just confident guessing, not informed decision-making
- 28.6% accuracy on Turn-1 stops = slightly worse than random (1/4 suspects)

### 2. High MI Correlates with Hard Puzzles
- Puzzles where samples disagree (MI > 0) are genuinely harder
- More turns don't help because the LLM lacks reasoning ability
- Only 1/6 multi-turn puzzles was correct (16.7%)

### 3. Question Quality is Poor
- Model asks repetitive questions
- Doesn't synthesize information across turns
- Doesn't follow up on suspicious responses

### 4. The Paradox
| Scenario | What It Means | Accuracy |
|----------|---------------|----------|
| MI = 0 at Turn 1 | Easy puzzle OR lucky guess | 28.6% |
| MI > 0, continues | Hard puzzle, uncertain | 16.7% |

**More information gathering = WORSE accuracy** because multi-turn cases are harder puzzles.

---

## Recommended Fixes

1. **Minimum turns**: Force at least 3-5 questions before checking MI
2. **Question diversity**: Detect and prevent repetitive questions  
3. **Separate uncertainty types**: Distinguish "confident guess" from "informed decision"
4. **Better question selection**: Use VoI to pick informative questions
