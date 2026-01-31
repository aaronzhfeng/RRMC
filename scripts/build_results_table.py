#!/usr/bin/env python3
"""
Build comparison tables from experiment results.
Outputs CSV files to results_table/ directory.
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def find_latest_comparison_files(results_dir: Path) -> Dict[str, Path]:
    """Find the latest comparison file for each method."""
    comparison_files = list(results_dir.glob("comparison_*.json"))
    
    # Group by method and find latest
    method_files: Dict[str, List[Path]] = {}
    
    for f in comparison_files:
        with open(f) as fp:
            data = json.load(fp)
            for method_name in data.keys():
                if method_name not in method_files:
                    method_files[method_name] = []
                method_files[method_name].append(f)
    
    # Take the latest file for each method (by modification time)
    latest = {}
    for method, files in method_files.items():
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest[method] = files[0]
    
    return latest


def extract_method_stats(data: Dict[str, Any], method_name: str) -> Dict[str, Any]:
    """Extract all relevant statistics for a method."""
    method_data = data[method_name]
    episodes = method_data.get("episodes", [])
    
    # Basic stats
    n_episodes = len(episodes)
    accuracy = method_data.get("accuracy", 0)
    avg_turns = method_data.get("avg_turns", 0)
    std_turns = method_data.get("std_turns", 0)
    avg_mi = method_data.get("avg_mi", 0)
    
    # Compute additional stats from episodes
    turns_list = [ep.get("turns_used", 0) for ep in episodes]
    min_turns = min(turns_list) if turns_list else 0
    max_turns_used = max(turns_list) if turns_list else 0
    
    # Count stopping reasons
    correct_count = sum(1 for ep in episodes if ep.get("correct", False))
    
    # MI scores analysis
    all_mi_scores = []
    for ep in episodes:
        all_mi_scores.extend(ep.get("mi_scores", []))
    
    avg_mi_all = sum(all_mi_scores) / len(all_mi_scores) if all_mi_scores else 0
    min_mi = min(all_mi_scores) if all_mi_scores else 0
    max_mi = max(all_mi_scores) if all_mi_scores else 0
    
    # First-turn MI (initial uncertainty)
    first_turn_mis = [ep.get("mi_scores", [0])[0] for ep in episodes if ep.get("mi_scores")]
    avg_first_turn_mi = sum(first_turn_mis) / len(first_turn_mis) if first_turn_mis else 0
    
    # Episodes that stopped at turn 1
    stopped_turn_1 = sum(1 for ep in episodes if ep.get("turns_used", 0) == 1)
    pct_stopped_turn_1 = stopped_turn_1 / n_episodes * 100 if n_episodes > 0 else 0
    
    return {
        "method": method_name,
        "method_class": method_data.get("method", ""),
        "task_type": method_data.get("task_type", ""),
        "n_episodes": n_episodes,
        "accuracy": accuracy,
        "accuracy_pct": accuracy * 100,
        "correct_count": correct_count,
        "avg_turns": avg_turns,
        "std_turns": std_turns,
        "min_turns": min_turns,
        "max_turns_used": max_turns_used,
        "stopped_turn_1": stopped_turn_1,
        "pct_stopped_turn_1": pct_stopped_turn_1,
        "avg_mi": avg_mi,
        "avg_mi_all_turns": avg_mi_all,
        "avg_first_turn_mi": avg_first_turn_mi,
        "min_mi": min_mi,
        "max_mi": max_mi,
    }


def build_summary_table(results_dir: Path, output_dir: Path) -> Path:
    """Build summary table from all comparison files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all unique methods across comparison files
    all_stats = []
    seen_methods = set()
    
    comparison_files = sorted(results_dir.glob("comparison_*.json"), 
                              key=lambda p: p.stat().st_mtime, reverse=True)
    
    for comp_file in comparison_files:
        with open(comp_file) as f:
            data = json.load(f)
        
        for method_name in data.keys():
            if method_name not in seen_methods:
                seen_methods.add(method_name)
                stats = extract_method_stats(data, method_name)
                stats["source_file"] = comp_file.name
                all_stats.append(stats)
    
    if not all_stats:
        print("No comparison files found!")
        return None
    
    # Sort by method name
    all_stats.sort(key=lambda x: x["method"])
    
    # Define column order
    columns = [
        "method",
        "method_class", 
        "task_type",
        "n_episodes",
        "accuracy_pct",
        "correct_count",
        "avg_turns",
        "std_turns",
        "min_turns",
        "max_turns_used",
        "stopped_turn_1",
        "pct_stopped_turn_1",
        "avg_mi",
        "avg_first_turn_mi",
        "min_mi",
        "max_mi",
        "source_file",
    ]
    
    # Write CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"methods_summary_{timestamp}.csv"
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"Summary table saved to: {output_file}")
    
    # Also print to console
    print("\n" + "=" * 120)
    print("METHODS COMPARISON SUMMARY")
    print("=" * 120)
    
    # Print header
    header = f"{'Method':<25} {'Accuracy':<10} {'Avg Turns':<12} {'Std Turns':<12} {'Stop@T1':<10} {'Avg MI':<12} {'1st MI':<12}"
    print(header)
    print("-" * 120)
    
    for s in all_stats:
        row = f"{s['method']:<25} {s['accuracy_pct']:>7.1f}%   {s['avg_turns']:>8.2f}     {s['std_turns']:>8.2f}     {s['pct_stopped_turn_1']:>6.1f}%    {s['avg_mi']:>10.4f}   {s['avg_first_turn_mi']:>10.4f}"
        print(row)
    
    print("=" * 120)
    
    return output_file


def build_episode_table(results_dir: Path, output_dir: Path, method: Optional[str] = None) -> Path:
    """Build detailed episode-level table."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_episodes = []
    comparison_files = sorted(results_dir.glob("comparison_*.json"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
    
    seen_method_puzzles = set()
    
    for comp_file in comparison_files:
        with open(comp_file) as f:
            data = json.load(f)
        
        for method_name, method_data in data.items():
            if method and method_name != method:
                continue
                
            for ep in method_data.get("episodes", []):
                key = (method_name, ep.get("puzzle_idx"))
                if key in seen_method_puzzles:
                    continue
                seen_method_puzzles.add(key)
                
                mi_scores = ep.get("mi_scores", [])
                
                episode_row = {
                    "method": method_name,
                    "puzzle_idx": ep.get("puzzle_idx"),
                    "correct": ep.get("correct"),
                    "prediction": ep.get("prediction"),
                    "ground_truth": ep.get("ground_truth"),
                    "turns_used": ep.get("turns_used"),
                    "n_history": len(ep.get("history", [])),
                    "mi_first": mi_scores[0] if mi_scores else None,
                    "mi_last": mi_scores[-1] if mi_scores else None,
                    "mi_max": max(mi_scores) if mi_scores else None,
                    "mi_min": min(mi_scores) if mi_scores else None,
                    "mi_scores": str(mi_scores),
                    "raw_answer": ep.get("raw_answer"),
                    "total_time": ep.get("total_time"),
                }
                all_episodes.append(episode_row)
    
    if not all_episodes:
        print("No episodes found!")
        return None
    
    # Sort by method then puzzle
    all_episodes.sort(key=lambda x: (x["method"], x["puzzle_idx"]))
    
    # Write CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{method}" if method else ""
    output_file = output_dir / f"episodes{suffix}_{timestamp}.csv"
    
    columns = list(all_episodes[0].keys())
    
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_episodes)
    
    print(f"Episode table saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build results tables from experiment runs")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output_dir", default="results_table", help="Output directory for tables")
    parser.add_argument("--episodes", action="store_true", help="Also build episode-level table")
    parser.add_argument("--method", default=None, help="Filter to specific method (for episodes)")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    # Build summary table
    build_summary_table(results_dir, output_dir)
    
    # Optionally build episode table
    if args.episodes:
        build_episode_table(results_dir, output_dir, method=args.method)
