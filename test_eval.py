#!/usr/bin/env python3
"""Minimal evaluation test for RRMC."""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("RRMC MINIMAL EVALUATION TEST")
print("=" * 60)

from rrmc.core.llm import LLMWrapper, load_api_key_from_env_file
from rrmc.core.environment import ARBenchEnv, TaskType, ActionASK, ActionANSWER
from rrmc.core.clustering import SemanticClusterer

# Load API key
env_path = os.path.join(os.path.dirname(__file__), "openrouter.env")
api_key = load_api_key_from_env_file(env_path)

# Initialize components
llm = LLMWrapper(api_key=api_key, model="qwen/qwen3-8b")
data_path = os.path.join(os.path.dirname(__file__), "AR-Bench", "data", "dc", "test.json")
env = ARBenchEnv(task_type=TaskType.DC, data_path=data_path, llm=llm)
clusterer = SemanticClusterer(use_entailment=False)

# Test parameters
N_PUZZLES = 3
MAX_TURNS = 3
K_SAMPLES = 4  # For self-consistency

results = []

for puzzle_idx in range(N_PUZZLES):
    print(f"\n{'=' * 40}")
    print(f"PUZZLE {puzzle_idx + 1}/{N_PUZZLES}")
    print("=" * 40)

    obs = env.reset(puzzle_idx)
    print(f"Suspects: {obs['suspect_names']}")
    print(f"Ground truth: {obs['ground_truth_name']}")

    # Run simple Q&A turns
    for turn in range(MAX_TURNS):
        history = env.get_history_string()

        # Generate question
        prompt = f"""You are investigating a murder case.
Case: {obs['initial_info'][:500]}...
Previous: {history if history else 'None'}
Suspects: {', '.join(obs['suspect_names'])}

Pick one suspect and ask them a brief question.
Format: Suspect: [name]
Question: [question]"""

        response = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=128,
        )

        # Parse response
        text = response.content
        suspect = None
        for name in obs['suspect_names']:
            if name.lower() in text.lower():
                suspect = name
                break
        suspect = suspect or obs['suspect_names'][0]

        question = text.split("Question:")[-1].strip().split("\n")[0] if "Question:" in text else "What were you doing?"

        print(f"\nTurn {turn + 1}: Asking {suspect[:20]}...")
        result = env.step(ActionASK(question=question, suspect=suspect))
        print(f"  Response: {result.observation[:80]}...")

    # Self-consistency voting for final answer
    print(f"\nGenerating {K_SAMPLES} answer samples for voting...")
    history = env.get_history_string()
    answers = []

    answer_prompt = f"""Based on this investigation, who is the murderer?
Case: {obs['initial_info'][:500]}...
Interrogation: {history}
Suspects: A={obs['suspect_names'][0]}, B={obs['suspect_names'][1]}, C={obs['suspect_names'][2]}, D={obs['suspect_names'][3]}, E={obs['suspect_names'][4]}
Answer with just the letter (A, B, C, D, or E):"""

    for i in range(K_SAMPLES):
        resp = llm.generate(
            messages=[{"role": "user", "content": answer_prompt}],
            temperature=0.7,
            max_tokens=32,
        )
        # Extract letter
        import re
        match = re.search(r'\b([A-E])\b', resp.content.upper())
        ans = match.group(1) if match else "A"
        answers.append(ans)

    # Majority vote
    from collections import Counter
    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0]
    consistency = vote_counts[best_answer] / len(answers)

    print(f"Votes: {dict(vote_counts)}")
    print(f"Best answer: {best_answer} (consistency={consistency:.0%})")

    # Submit final answer
    result = env.step(ActionANSWER(answer=best_answer))
    correct = result.info.get('correct', False)
    print(f"Correct: {correct}")

    results.append({
        "puzzle_idx": puzzle_idx,
        "correct": correct,
        "prediction": best_answer,
        "ground_truth": obs['ground_truth'],
        "consistency": consistency,
        "turns_used": MAX_TURNS,
    })

# Summary
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
accuracy = sum(r['correct'] for r in results) / len(results)
avg_consistency = sum(r['consistency'] for r in results) / len(results)
print(f"Accuracy: {accuracy:.0%} ({sum(r['correct'] for r in results)}/{len(results)})")
print(f"Avg Consistency: {avg_consistency:.0%}")
print(f"Total LLM calls: ~{len(results) * (MAX_TURNS * 2 + K_SAMPLES)}")

# Token usage
usage = llm.get_token_usage()
print(f"\nToken usage:")
print(f"  Prompt: {usage['prompt_tokens']:,}")
print(f"  Completion: {usage['completion_tokens']:,}")
print(f"  Total: {usage['total_tokens']:,}")

# Save results
os.makedirs("results", exist_ok=True)
output_file = f"results/minimal_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "avg_consistency": avg_consistency,
        "results": results,
        "token_usage": usage,
    }, f, indent=2)
print(f"\nResults saved to: {output_file}")
