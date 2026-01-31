#!/usr/bin/env python3
"""
Grid search over stopping thresholds on a small puzzle subset.
Saves detailed conversation logs for inspection.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from pipeline import Pipeline


def run_grid_search(
    method: str,
    threshold_param: str,
    thresholds: list,
    n_puzzles: int = 5,
    max_workers: int = 2,
):
    """Run grid search over thresholds for a single method."""
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/grid_search") / f"{method}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Grid Search: {method}")
    print(f"Threshold param: {threshold_param}")
    print(f"Thresholds: {thresholds}")
    print(f"Puzzles: {n_puzzles}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    for thresh in thresholds:
        print(f"\n--- Testing {threshold_param}={thresh} ---")
        
        # Build config with this threshold
        config = load_config(
            experiment=f"dc_methods/{method}",
            overrides={
                "n_puzzles": n_puzzles,
                "max_workers": max_workers,
                threshold_param: thresh,
            }
        )
        
        # Run pipeline
        try:
            pipeline = Pipeline(config)
            run_results = pipeline.run()
            
            # Pipeline returns {"evaluation": {method_name: {...}}}
            eval_results = run_results.get("evaluation", run_results)
            
            # Extract metrics - key might be method name or stopping rule class name
            r = None
            for key in eval_results:
                # Take first result (single-method run)
                r = eval_results[key]
                break
            
            if r is None:
                print(f"  WARNING: No results found, keys={list(eval_results.keys())}")
                results.append({"threshold": thresh, "error": "no_results"})
                continue
            
            # Handle both dict and dataclass results
            if hasattr(r, 'accuracy'):
                # Dataclass EvaluationResult
                accuracy = r.accuracy
                avg_turns = r.avg_turns
                episodes = r.episode_results if hasattr(r, 'episode_results') else []
            else:
                # Dict format
                accuracy = r.get("accuracy", 0)
                avg_turns = r.get("avg_turns", 0)
                episodes = r.get("episodes", [])
            
            # Collect detailed episode info
            episode_details = []
            for ep in episodes:
                # Handle both dict and dataclass EpisodeResult
                if hasattr(ep, 'puzzle_idx'):
                    episode_details.append({
                        "puzzle_idx": ep.puzzle_idx,
                        "correct": ep.correct,
                        "prediction": ep.prediction,
                        "ground_truth": ep.ground_truth,
                        "turns_used": ep.turns_used,
                        "mi_scores": getattr(ep, 'mi_scores', []),
                        "history": getattr(ep, 'history', []),
                        "raw_samples": getattr(ep, 'raw_samples', []),
                    })
                else:
                    episode_details.append({
                        "puzzle_idx": ep.get("puzzle_idx"),
                        "correct": ep.get("correct"),
                        "prediction": ep.get("prediction"),
                        "ground_truth": ep.get("ground_truth"),
                        "turns_used": ep.get("turns_used"),
                        "mi_scores": ep.get("mi_scores", []),
                        "history": ep.get("history", []),
                        "raw_samples": ep.get("raw_samples", []),
                    })
            
            result_entry = {
                "threshold": thresh,
                "accuracy": accuracy,
                "avg_turns": avg_turns,
                "n_episodes": len(episodes),
                "episodes": episode_details,
            }
            results.append(result_entry)
            
            print(f"  Accuracy: {accuracy*100:.1f}%, Avg Turns: {avg_turns:.1f}")
            
            # Save individual threshold result
            thresh_file = output_dir / f"thresh_{thresh:.3f}.json"
            with open(thresh_file, "w") as f:
                json.dump(result_entry, f, indent=2)
                
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "threshold": thresh,
                "error": str(e),
            })
    
    # Save summary
    summary = {
        "method": method,
        "threshold_param": threshold_param,
        "thresholds_tested": thresholds,
        "n_puzzles": n_puzzles,
        "results": [
            {
                "threshold": r["threshold"],
                "accuracy": r.get("accuracy", "error"),
                "avg_turns": r.get("avg_turns", "error"),
            }
            for r in results
        ],
    }
    
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Avg Turns':<12}")
    print("-" * 36)
    for r in results:
        if "error" in r:
            print(f"{r['threshold']:<12.3f} ERROR")
        else:
            print(f"{r['threshold']:<12.3f} {r['accuracy']*100:<12.1f} {r['avg_turns']:<12.1f}")
    
    print(f"\nResults saved to: {output_dir}")
    return results, output_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Grid search over stopping thresholds")
    parser.add_argument("--method", default="mi_only", 
                        choices=["mi_only", "robust_mi", "semantic_entropy", "self_consistency",
                                 "verbalized_confidence", "knowno", "cip_lite"])
    parser.add_argument("--n_puzzles", type=int, default=5)
    parser.add_argument("--max_workers", type=int, default=2)
    args = parser.parse_args()
    
    # Method-specific threshold configs
    GRID_CONFIGS = {
        "mi_only": {
            "param": "mi_threshold",
            "values": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        },
        "robust_mi": {
            "param": "threshold",
            "values": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
        },
        "semantic_entropy": {
            "param": "entropy_threshold",
            "values": [0.05, 0.1, 0.2, 0.3, 0.5],
        },
        "self_consistency": {
            "param": "consistency_threshold",
            "values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        "verbalized_confidence": {
            "param": "confidence_threshold",
            "values": [5.0, 6.0, 7.0, 8.0, 9.0],
        },
        "knowno": {
            "param": "set_size_threshold",
            "values": [1, 2, 3],
        },
        "cip_lite": {
            "param": "set_size_threshold",
            "values": [1, 2, 3, 4],
        },
    }
    
    config = GRID_CONFIGS[args.method]
    
    run_grid_search(
        method=args.method,
        threshold_param=config["param"],
        thresholds=config["values"],
        n_puzzles=args.n_puzzles,
        max_workers=args.max_workers,
    )
