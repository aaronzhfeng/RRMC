#!/usr/bin/env python3
"""
Extract grid search results from comparison JSON files.
Creates one CSV table per method with all threshold results.
"""

import json
import csv
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def find_comparison_files_for_method(results_dir: Path, method_name: str) -> list:
    """
    Find comparison JSON files created during grid search for a method.
    Parse timestamps and match to grid search runs.
    Returns list of (threshold, filepath) tuples.
    """
    grid_search_dir = results_dir / "grid_search"
    
    # Known threshold ranges for each method
    threshold_ranges = {
        "mi_only": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5],
        "verbalized_confidence": [5.0, 6.0, 7.0, 8.0, 9.0],
        "self_consistency": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "semantic_entropy": [0.05, 0.1, 0.2, 0.3, 0.5],
        "knowno": [1, 2, 3],
        "cip_lite": [1, 2, 3, 4],
    }
    
    # Find all grid search directories for this method
    method_dirs = sorted([
        d for d in grid_search_dir.iterdir()
        if d.is_dir() and d.name.startswith(f"{method_name}_")
    ])
    
    if not method_dirs:
        return []
    
    # Use the latest grid search directory
    latest_dir = method_dirs[-1]
    print(f"  Using grid search dir: {latest_dir.name}")
    
    # Check if there are threshold result files (new format)
    thresh_files = sorted(latest_dir.glob("thresh_*.json"))
    if thresh_files:
        return [("thresh_file", f) for f in thresh_files]
    
    # Otherwise, try to find comparison files by timestamp range
    match = re.search(r'_(\d{8}_\d{6})$', latest_dir.name)
    if not match:
        return []
    
    start_time = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    
    # Find comparison files created after this timestamp
    comparison_files = []
    for f in results_dir.glob("comparison_detective_cases_*.json"):
        match = re.search(r'_(\d{8}_\d{6})\.json$', f.name)
        if match:
            file_time = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            # Files created within 30 minutes of grid search start
            if 0 <= (file_time - start_time).total_seconds() <= 1800:
                comparison_files.append((file_time, f))
    
    # Sort by timestamp
    comparison_files.sort(key=lambda x: x[0])
    
    # Match to thresholds by order
    thresholds = threshold_ranges.get(method_name, [])
    result = []
    for i, (_, f) in enumerate(comparison_files):
        thresh = thresholds[i] if i < len(thresholds) else f"idx_{i}"
        result.append((thresh, f))
    
    return result


def parse_comparison_file(filepath: Path) -> dict:
    """Parse a comparison JSON file and extract method results."""
    with open(filepath) as f:
        data = json.load(f)
    
    results = {}
    for method_name, method_data in data.items():
        if isinstance(method_data, dict):
            results[method_name] = {
                "accuracy": method_data.get("accuracy", 0),
                "avg_turns": method_data.get("avg_turns", 0),
                "std_turns": method_data.get("std_turns", 0),
                "n_episodes": method_data.get("n_episodes", 0),
            }
    return results


def parse_thresh_file(filepath: Path) -> dict:
    """Parse a threshold result file from grid search."""
    with open(filepath) as f:
        data = json.load(f)
    
    return {
        "threshold": data.get("threshold", 0),
        "accuracy": data.get("accuracy", 0),
        "avg_turns": data.get("avg_turns", 0),
        "n_episodes": data.get("n_episodes", 0),
    }


def extract_grid_search_results(results_dir: Path, output_dir: Path):
    """Extract and save grid search results for each method."""
    
    # Methods to process (skip robust_mi as requested)
    methods = [
        "mi_only",
        "verbalized_confidence", 
        "self_consistency",
        "semantic_entropy",
        "knowno",
        "cip_lite",
    ]
    
    # Threshold parameter names for each method
    threshold_params = {
        "mi_only": "mi_threshold",
        "verbalized_confidence": "confidence_threshold",
        "self_consistency": "consistency_threshold",
        "semantic_entropy": "entropy_threshold",
        "knowno": "set_size_threshold",
        "cip_lite": "set_size_threshold",
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for method in methods:
        print(f"\nProcessing {method}...")
        
        files_with_thresh = find_comparison_files_for_method(results_dir, method)
        
        if not files_with_thresh:
            print(f"  No results found for {method}")
            continue
        
        results = []
        
        # Check if these are thresh_*.json files or comparison files
        if files_with_thresh[0][0] == "thresh_file":
            # Parse threshold files directly (new format)
            for _, f in files_with_thresh:
                data = parse_thresh_file(f)
                results.append(data)
        else:
            # Parse comparison files with threshold mapping
            for thresh, f in files_with_thresh:
                parsed = parse_comparison_file(f)
                # Find the method result (may have different class name)
                for key, data in parsed.items():
                    if method.lower() in key.lower() or key.lower() in method.lower():
                        results.append({
                            "threshold": thresh,
                            "file": f.name,
                            **data
                        })
                        break
        
        if not results:
            print(f"  No valid results parsed for {method}")
            continue
        
        # Sort by threshold if available, otherwise by accuracy
        if "threshold" in results[0]:
            # Handle mixed types (float, int, str)
            def sort_key(x):
                t = x["threshold"]
                if isinstance(t, (int, float)):
                    return (0, float(t))
                return (1, str(t))
            results.sort(key=sort_key)
        
        # Save to CSV
        csv_path = output_dir / f"grid_search_{method}.csv"
        fieldnames = list(results[0].keys())
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"  Saved {len(results)} results to {csv_path.name}")
        
        # Store for summary
        all_results[method] = results
        
        # Print table
        print(f"\n  {'Threshold':<12} {'Accuracy':<12} {'Avg Turns':<12} {'N':<6}")
        print(f"  {'-'*42}")
        for r in results:
            thresh = r.get("threshold", "?")
            acc = r.get("accuracy", 0)
            turns = r.get("avg_turns", 0)
            n = r.get("n_episodes", "?")
            print(f"  {thresh:<12} {acc*100:>8.1f}%    {turns:>8.2f}     {n:<6}")
    
    # Create summary of best results per method
    summary_path = output_dir / "grid_search_summary.csv"
    summary_rows = []
    
    for method, results in all_results.items():
        if not results:
            continue
        
        # Find best by accuracy, then by turns
        best = max(results, key=lambda x: (x.get("accuracy", 0), -x.get("avg_turns", float("inf"))))
        
        summary_rows.append({
            "method": method,
            "threshold_param": threshold_params.get(method, "threshold"),
            "best_threshold": best.get("threshold", "?"),
            "accuracy": best.get("accuracy", 0),
            "avg_turns": best.get("avg_turns", 0),
            "n_episodes": best.get("n_episodes", "?"),
        })
    
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["method", "threshold_param", "best_threshold", "accuracy", "avg_turns", "n_episodes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"\n\n{'='*60}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Param':<20} {'Best':<8} {'Acc':<10} {'Turns':<8}")
    print(f"{'-'*60}")
    for row in summary_rows:
        print(f"{row['method']:<25} {row['threshold_param']:<20} {row['best_threshold']:<8} {row['accuracy']*100:>6.1f}%   {row['avg_turns']:>6.2f}")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / "results"
    output_dir = Path(__file__).parent.parent / "results_table" / "grid_search"
    
    extract_grid_search_results(results_dir, output_dir)
