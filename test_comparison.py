#!/usr/bin/env python3
"""
RRMC Method Comparison Test.

Compares different stopping rules on a small subset of DC puzzles.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("RRMC METHOD COMPARISON TEST")
print("=" * 60)

from rrmc.core.llm import LLMWrapper, load_api_key_from_env_file
from rrmc.core.environment import ARBenchEnv, TaskType, ActionASK, ActionANSWER
from rrmc.core.clustering import SemanticClusterer
from rrmc.core.mi_estimator import SelfRevisionMI, compute_homogeneity_score
from rrmc.baselines.stopping_rules import (
    FixedTurnsStopping,
    SelfConsistencyStopping,
    MIOnlyStopping,
)

# Configuration
N_PUZZLES = 3
MAX_TURNS = 5

# Load API key
env_path = os.path.join(os.path.dirname(__file__), "openrouter.env")
api_key = load_api_key_from_env_file(env_path)

# Initialize components
print("\nInitializing components...")
llm = LLMWrapper(api_key=api_key, model="qwen/qwen3-8b")
data_path = os.path.join(os.path.dirname(__file__), "AR-Bench", "data", "dc", "test.json")
clusterer = SemanticClusterer(use_entailment=False)

# Methods to compare
methods = {
    "fixed_5": FixedTurnsStopping(llm, fixed_turns=5, max_turns=MAX_TURNS),
    "self_consistency": SelfConsistencyStopping(
        llm, clusterer, k_samples=4, consistency_threshold=0.7, max_turns=MAX_TURNS
    ),
    "mi_only": MIOnlyStopping(
        llm, clusterer, mi_threshold=0.3, k_samples=3, max_turns=MAX_TURNS
    ),
}

print(f"Methods: {list(methods.keys())}")
print(f"Testing on {N_PUZZLES} puzzles with max {MAX_TURNS} turns")

# Results storage
all_results = {name: [] for name in methods}

def generate_question(llm, obs, history_string):
    """Generate a question for the current state."""
    prompt = f"""You are investigating a murder case.

Case background (summary):
- Time: {obs.get('initial_info', '')[:200]}...
- Suspects: {', '.join(obs.get('suspect_names', []))}

Previous questions:
{history_string if history_string else 'None yet'}

Pick ONE suspect and ask them ONE short question to help identify the murderer.
Format:
Suspect: [full name]
Question: [your question]"""

    response = llm.generate(
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=128,
    )

    # Parse response
    text = response.content
    suspect = None
    for name in obs.get('suspect_names', []):
        if name.lower() in text.lower():
            suspect = name
            break
    suspect = suspect or (obs.get('suspect_names', [''])[0])

    if "Question:" in text:
        question = text.split("Question:")[-1].strip().split("\n")[0]
    else:
        question = "What do you know about the victim?"

    return question, suspect


def run_episode(env, llm, stopping_rule, puzzle_idx, obs):
    """Run a single episode with a stopping rule."""
    turn = 0
    mi_scores = []

    while not env.done and turn < MAX_TURNS:
        turn += 1
        state = env.get_state()

        # Make stopping decision
        decision = stopping_rule.should_stop("DC", state, turn)
        mi_scores.append(decision.score)

        if decision.should_stop:
            # Submit answer
            answer = decision.prediction or stopping_rule.get_best_answer("DC", state)
            result = env.step(ActionANSWER(answer=answer))
            break
        else:
            # Generate and ask question
            history_string = env.get_history_string()
            question, suspect = generate_question(llm, obs, history_string)
            result = env.step(ActionASK(question=question, suspect=suspect))

    # Force answer if not done
    if not env.done:
        state = env.get_state()
        answer = stopping_rule.get_best_answer("DC", state)
        result = env.step(ActionANSWER(answer=answer))

    return {
        "correct": result.info.get("correct", False),
        "turns_used": turn,
        "mi_scores": mi_scores,
        "prediction": result.info.get("prediction", ""),
        "ground_truth": result.info.get("ground_truth", ""),
    }


# Run comparison
for puzzle_idx in range(N_PUZZLES):
    print(f"\n{'=' * 50}")
    print(f"PUZZLE {puzzle_idx + 1}/{N_PUZZLES}")
    print("=" * 50)

    for method_name, stopping_rule in methods.items():
        # Create fresh environment for each method
        env = ARBenchEnv(task_type=TaskType.DC, data_path=data_path, llm=llm)
        obs = env.reset(puzzle_idx)

        print(f"\n  Method: {method_name}")
        result = run_episode(env, llm, stopping_rule, puzzle_idx, obs)

        all_results[method_name].append(result)
        print(f"    Turns: {result['turns_used']}, Correct: {result['correct']}")

# Summary
print("\n" + "=" * 60)
print("COMPARISON RESULTS")
print("=" * 60)
print(f"{'Method':<20} {'Accuracy':>10} {'Avg Turns':>12} {'Avg MI':>10}")
print("-" * 60)

for method_name, results in all_results.items():
    accuracy = sum(r['correct'] for r in results) / len(results)
    avg_turns = sum(r['turns_used'] for r in results) / len(results)
    all_mi = [s for r in results for s in r['mi_scores'] if r['mi_scores']]
    avg_mi = sum(all_mi) / len(all_mi) if all_mi else 0.0

    print(f"{method_name:<20} {accuracy:>10.0%} {avg_turns:>12.1f} {avg_mi:>10.2f}")

print("=" * 60)

# Token usage
usage = llm.get_token_usage()
print(f"\nTotal API usage:")
print(f"  Prompt tokens: {usage['prompt_tokens']:,}")
print(f"  Completion tokens: {usage['completion_tokens']:,}")
print(f"  Total: {usage['total_tokens']:,}")

# Save results
os.makedirs("results", exist_ok=True)
output = {
    "config": {
        "n_puzzles": N_PUZZLES,
        "max_turns": MAX_TURNS,
        "methods": list(methods.keys()),
    },
    "results": all_results,
    "token_usage": usage,
}
output_file = f"results/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to: {output_file}")
