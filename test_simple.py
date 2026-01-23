#!/usr/bin/env python3
"""Simple test to validate RRMC components."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing RRMC components...")

# Test 1: Load API key
print("\n1. Testing API key loading...")
from rrmc.core.llm import load_api_key_from_env_file

env_path = os.path.join(os.path.dirname(__file__), "openrouter.env")
api_key = load_api_key_from_env_file(env_path)
print(f"   API key loaded: {api_key[:20]}...")

# Test 2: Initialize LLM
print("\n2. Testing LLM initialization...")
from rrmc.core.llm import LLMWrapper

llm = LLMWrapper(
    api_key=api_key,
    model="qwen/qwen3-8b",
)
print(f"   LLM initialized: model={llm.model}")

# Test 3: Simple LLM call
print("\n3. Testing simple LLM call...")
response = llm.generate(
    messages=[{"role": "user", "content": "Say 'Hello RRMC' and nothing else."}],
    temperature=0.1,
    max_tokens=32,
)
print(f"   Response: {response.content}")
print(f"   Tokens: {response.prompt_tokens} prompt, {response.completion_tokens} completion")

# Test 4: Load environment
print("\n4. Testing environment loading...")
from rrmc.core.environment import ARBenchEnv, TaskType

data_path = os.path.join(os.path.dirname(__file__), "AR-Bench", "data", "dc", "test.json")
env = ARBenchEnv(
    task_type=TaskType.DC,
    data_path=data_path,
    llm=llm,
)
print(f"   Environment loaded: {len(env)} puzzles")

# Test 5: Reset environment
print("\n5. Testing environment reset...")
obs = env.reset(0)
print(f"   Puzzle 0 loaded")
print(f"   Suspect names: {obs['suspect_names']}")
print(f"   Ground truth: {obs['ground_truth_name']}")

# Test 6: Test semantic clustering
print("\n6. Testing semantic clustering...")
from rrmc.core.clustering import SemanticClusterer

clusterer = SemanticClusterer(use_entailment=False)
test_answers = ["A", "B", "A", "A", "B"]
result = clusterer.cluster(test_answers, "DC")
print(f"   Clustered {len(test_answers)} answers into {result.n_clusters} clusters")
print(f"   Cluster sizes: {result.cluster_sizes}")

# Test 7: Simple question generation and answer
print("\n7. Testing question/answer flow...")
from rrmc.core.environment import ActionASK, ActionANSWER

# Ask a simple question
suspect = obs['suspect_names'][0]
result = env.step(ActionASK(question="Where were you at the time of the murder?", suspect=suspect))
print(f"   Asked {suspect}: 'Where were you at the time of the murder?'")
print(f"   Response: {result.observation[:100]}...")

# Submit answer
result = env.step(ActionANSWER(answer="A"))
print(f"   Submitted answer: A")
print(f"   Correct: {result.info.get('correct', 'unknown')}")

print("\n" + "=" * 50)
print("All basic tests passed!")
print("=" * 50)
