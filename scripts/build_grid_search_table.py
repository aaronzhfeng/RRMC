#!/usr/bin/env python3
"""
Build comparison tables from grid search results.
Outputs CSV files to results_table/ directory.
"""

import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


def find_grid_search_dirs(results_dir: Path) -> Dict[str, List[Path]]:
    """Find all grid search result directories grouped by method."""
    grid_dir = results_dir / "grid_search"
    if not grid_dir.exists():
        return {}
    
    method_dirs: Dict[str, List[Path]] = {}
    
    for d in grid_dir.iterdir():
        if d.is_dir():
            # Parse method name from directory name (e.g., "mi_only_20260130_174749")
            parts = d.name.rsplit("_", 2)
            if len(parts) >= 3:
                method = parts[0]
            else:
                method = d.name
            
            if method not in method_dirs:
                method_dirs[method] = []
            method_dirs[method].append(d)
    
    # Sort by modification time (latest first)
    for method in method_dirs:
        method_dirs[method].sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return method_dirs


def load_grid_search_results(grid_dir: Path) -> List[Dict[str, Any]]:
    """Load all threshold results from a grid search directory."""
    results = []
    
    # Load summary if exists
    summary_file = grid_dir / "summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
    else:
        summary = {}
    
    # Load individual threshold files
    for thresh_file in sorted(grid_dir.glob("thresh_*.json")):
        with open(thresh_file) as f:
            data = json.load(f)
        
        # Extract stats
        threshold = data.get("threshold", 0)
        accuracy = data.get("accuracy", 0)
        avg_turns = data.get("avg_turns", 0)
        n_episodes = data.get("n_episodes", 0)
        episodes = data.get("episodes", [])
        
        # Compute additional stats
        turns_list = [ep.get("turns_used", 0) for ep in episodes]
        std_turns = (sum((t - avg_turns)**2 for t in turns_list) / len(turns_list))**0.5 if turns_list else 0
        min_turns = min(turns_list) if turns_list else 0
        max_turns = max(turns_list) if turns_list else 0
        
        # MI stats
        all_mi = []
        first_mi = []
        for ep in episodes:
            mi_scores = ep.get("mi_scores", [])
            all_mi.extend(mi_scores)
            if mi_scores:
                first_mi.append(mi_scores[0])
        
        avg_mi = sum(all_mi) / len(all_mi) if all_mi else 0
        avg_first_mi = sum(first_mi) / len(first_mi) if first_mi else 0
        
        # Stopped at turn 1
        stopped_t1 = sum(1 for t in turns_list if t == 1)
        pct_stopped_t1 = stopped_t1 / len(turns_list) * 100 if turns_list else 0
        
        results.append({
            "threshold": threshold,
            "accuracy": accuracy,
            "accuracy_pct": accuracy * 100,
            "n_episodes": n_episodes,
            "avg_turns": avg_turns,
            "std_turns": std_turns,
            "min_turns": min_turns,
            "max_turns": max_turns,
            "stopped_t1": stopped_t1,
            "pct_stopped_t1": pct_stopped_t1,
            "avg_mi": avg_mi,
            "avg_first_mi": avg_first_mi,
        })
    
    # Sort by threshold
    results.sort(key=lambda x: x["threshold"])
    return results


def build_grid_search_table(
    results_dir: Path, 
    output_dir: Path,
    method: Optional[str] = None,
) -> List[Path]:
    """Build grid search comparison tables."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    method_dirs = find_grid_search_dirs(results_dir)
    
    if not method_dirs:
        print("No grid search results found!")
        return []
    
    output_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter to specific method if requested
    if method:
        if method not in method_dirs:
            print(f"Method '{method}' not found in grid search results")
            return []
        method_dirs = {method: method_dirs[method]}
    
    # Build combined summary across all methods
    all_results = []
    
    for method_name, dirs in method_dirs.items():
        # Use latest directory for each method
        latest_dir = dirs[0]
        print(f"\nProcessing: {method_name} ({latest_dir.name})")
        
        results = load_grid_search_results(latest_dir)
        
        for r in results:
            r["method"] = method_name
            r["source_dir"] = latest_dir.name
            all_results.append(r)
        
        # Print method-specific table
        print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Avg Turns':<12} {'Std':<10} {'Stop@T1':<10}")
        print("-" * 60)
        for r in results:
            print(f"{r['threshold']:<12.3f} {r['accuracy_pct']:>8.1f}%    {r['avg_turns']:>8.2f}     {r['std_turns']:>6.2f}     {r['pct_stopped_t1']:>6.1f}%")
        
        # Find best threshold
        if results:
            best = max(results, key=lambda x: (x["accuracy"], -x["avg_turns"]))
            print(f"\n  → Best: threshold={best['threshold']:.3f} → {best['accuracy_pct']:.1f}% acc, {best['avg_turns']:.1f} turns")
    
    # Write combined CSV
    if all_results:
        columns = [
            "method", "threshold", "accuracy_pct", "n_episodes",
            "avg_turns", "std_turns", "min_turns", "max_turns",
            "stopped_t1", "pct_stopped_t1", "avg_mi", "avg_first_mi",
            "source_dir",
        ]
        
        output_file = output_dir / f"grid_search_summary_{timestamp}.csv"
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_results)
        
        print(f"\n\nCombined table saved to: {output_file}")
        output_files.append(output_file)
        
        # Also write per-method CSV files
        for method_name in method_dirs:
            method_results = [r for r in all_results if r["method"] == method_name]
            if method_results:
                method_file = output_dir / f"grid_search_{method_name}_{timestamp}.csv"
                with open(method_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(method_results)
                output_files.append(method_file)
    
    return output_files


def print_best_thresholds(results_dir: Path):
    """Print recommended thresholds for each method."""
    method_dirs = find_grid_search_dirs(results_dir)
    
    if not method_dirs:
        print("No grid search results found!")
        return
    
    print("\n" + "=" * 70)
    print("RECOMMENDED THRESHOLDS (based on grid search)")
    print("=" * 70)
    print(f"{'Method':<25} {'Best Threshold':<18} {'Accuracy':<12} {'Avg Turns':<12}")
    print("-" * 70)
    
    recommendations = []
    
    for method_name, dirs in sorted(method_dirs.items()):
        latest_dir = dirs[0]
        results = load_grid_search_results(latest_dir)
        
        if results:
            # Best = highest accuracy, then lowest turns
            best = max(results, key=lambda x: (x["accuracy"], -x["avg_turns"]))
            print(f"{method_name:<25} {best['threshold']:<18.3f} {best['accuracy_pct']:>8.1f}%    {best['avg_turns']:>8.2f}")
            recommendations.append({
                "method": method_name,
                "threshold": best["threshold"],
                "accuracy": best["accuracy_pct"],
                "avg_turns": best["avg_turns"],
            })
    
    print("=" * 70)
    return recommendations


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build tables from grid search results")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--output_dir", default="results_table", help="Output directory")
    parser.add_argument("--method", default=None, help="Filter to specific method")
    parser.add_argument("--best", action="store_true", help="Only show best thresholds")
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    if args.best:
        print_best_thresholds(results_dir)
    else:
        build_grid_search_table(results_dir, output_dir, method=args.method)
        print("\n")
        print_best_thresholds(results_dir)
