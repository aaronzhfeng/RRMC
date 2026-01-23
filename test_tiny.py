#!/usr/bin/env python3
"""Tiny test for RRMC - single puzzle, single method."""

import sys, os
sys.stdout = sys.stderr  # Force unbuffered output
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("RRMC TINY TEST", flush=True)
print("=" * 40, flush=True)

from rrmc.core.llm import LLMWrapper, load_api_key_from_env_file
from rrmc.core.environment import ARBenchEnv, TaskType, ActionASK, ActionANSWER
from rrmc.core.clustering import SemanticClusterer
from rrmc.baselines.stopping_rules import FixedTurnsStopping

# Load components
print("Loading...", flush=True)
api_key = load_api_key_from_env_file("openrouter.env")
llm = LLMWrapper(api_key=api_key, model="qwen/qwen3-8b")
env = ARBenchEnv(task_type=TaskType.DC, data_path="AR-Bench/data/dc/test.json", llm=llm)
clusterer = SemanticClusterer(use_entailment=False)

# Create stopping rule
stopper = FixedTurnsStopping(llm, fixed_turns=3, max_turns=5)

# Run single puzzle
print("\nRunning puzzle 0...", flush=True)
obs = env.reset(0)
print(f"Suspects: {obs['suspect_names'][:2]}...", flush=True)
print(f"Ground truth: {obs['ground_truth_name']}", flush=True)

turn = 0
while not env.done and turn < 3:
    turn += 1
    state = env.get_state()

    # Check stopping
    decision = stopper.should_stop("DC", state, turn)
    print(f"Turn {turn}: score={decision.score:.2f}, stop={decision.should_stop}", flush=True)

    if decision.should_stop:
        answer = decision.prediction or stopper.get_best_answer("DC", state)
        result = env.step(ActionANSWER(answer=answer))
        break

    # Ask a question
    suspect = obs['suspect_names'][turn % len(obs['suspect_names'])]
    result = env.step(ActionASK(question="What were you doing?", suspect=suspect))
    print(f"  Asked {suspect[:15]}...", flush=True)

# Get final answer if not done
if not env.done:
    state = env.get_state()
    answer = stopper.get_best_answer("DC", state)
    result = env.step(ActionANSWER(answer=answer))

print(f"\nResult: correct={result.info.get('correct')}", flush=True)
print(f"Tokens: {llm.get_token_usage()['total_tokens']}", flush=True)
print("DONE", flush=True)
