#!/usr/bin/env python3
"""
Analyze 20-puzzle validation runs to understand stopping behavior.
Outputs tables for slides showing the early-stopping problem.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def analyze_episode(ep: dict) -> dict:
    """Extract key metrics from a single episode."""
    mi_scores = ep.get('mi_scores', [])
    turns = ep.get('turns_used', 0)
    correct = ep.get('correct', False)
    
    return {
        'turns': turns,
        'correct': correct,
        'first_mi': mi_scores[0] if mi_scores else None,
        'final_mi': mi_scores[-1] if mi_scores else None,
        'mi_zero_at_start': mi_scores[0] == 0.0 if mi_scores else None,
        'stopped_turn_1': turns == 1,
    }


def analyze_comparison_file(filepath: Path) -> dict:
    """Analyze a comparison JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    results = {}
    for method_name, method_data in data.items():
        if not isinstance(method_data, dict):
            continue
        
        episodes = method_data.get('episodes', [])
        if not episodes:
            continue
        
        # Analyze each episode
        analyzed = [analyze_episode(ep) for ep in episodes]
        
        n_total = len(analyzed)
        n_correct = sum(1 for ep in analyzed if ep['correct'])
        n_turn_1 = sum(1 for ep in analyzed if ep['stopped_turn_1'])
        n_mi_zero_start = sum(1 for ep in analyzed if ep['mi_zero_at_start'])
        
        # Turn distribution
        turn_dist = defaultdict(int)
        for ep in analyzed:
            turn_dist[ep['turns']] += 1
        
        # Accuracy by turn count
        correct_by_turns = defaultdict(lambda: {'correct': 0, 'total': 0})
        for ep in analyzed:
            t = ep['turns']
            correct_by_turns[t]['total'] += 1
            if ep['correct']:
                correct_by_turns[t]['correct'] += 1
        
        results[method_name] = {
            'n_episodes': n_total,
            'accuracy': n_correct / n_total if n_total > 0 else 0,
            'n_correct': n_correct,
            'avg_turns': sum(ep['turns'] for ep in analyzed) / n_total if n_total > 0 else 0,
            'n_stopped_turn_1': n_turn_1,
            'pct_stopped_turn_1': n_turn_1 / n_total * 100 if n_total > 0 else 0,
            'n_mi_zero_start': n_mi_zero_start,
            'pct_mi_zero_start': n_mi_zero_start / n_total * 100 if n_total > 0 else 0,
            'turn_distribution': dict(sorted(turn_dist.items())),
            'correct_by_turns': {k: v for k, v in sorted(correct_by_turns.items())},
            'episodes': analyzed,
        }
    
    return results


def find_validation_runs(results_dir: Path) -> list:
    """Find 20-puzzle validation run files."""
    validation_files = []
    
    for f in results_dir.glob('comparison_detective_cases_*.json'):
        with open(f) as fp:
            data = json.load(fp)
        
        for method_name, method_data in data.items():
            if isinstance(method_data, dict):
                n = method_data.get('n_episodes', 0)
                if n == 20:  # 20-puzzle runs
                    validation_files.append((f, method_name))
                    break
    
    return validation_files


def main():
    results_dir = Path(__file__).parent.parent / 'results'
    output_dir = Path(__file__).parent.parent / 'results_table' / 'validation_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("VALIDATION RUN ANALYSIS (20 puzzles)")
    print("="*70)
    
    # Find and analyze validation runs
    validation_files = find_validation_runs(results_dir)
    
    all_results = {}
    for filepath, expected_method in validation_files:
        analysis = analyze_comparison_file(filepath)
        for method, data in analysis.items():
            if data['n_episodes'] == 20:
                all_results[method] = {
                    'file': filepath.name,
                    **data
                }
    
    # Print summary table
    print(f"\n{'Method':<30} {'Acc':<8} {'AvgTurns':<10} {'Turn1%':<10} {'MI=0%':<10}")
    print("-"*70)
    
    summary_rows = []
    for method, data in sorted(all_results.items()):
        acc = data['accuracy'] * 100
        turns = data['avg_turns']
        t1_pct = data['pct_stopped_turn_1']
        mi0_pct = data['pct_mi_zero_start']
        
        print(f"{method:<30} {acc:>5.1f}%   {turns:>6.1f}     {t1_pct:>6.1f}%    {mi0_pct:>6.1f}%")
        
        summary_rows.append({
            'method': method,
            'accuracy_pct': acc,
            'avg_turns': turns,
            'stopped_turn_1_pct': t1_pct,
            'mi_zero_start_pct': mi0_pct,
            'n_correct': data['n_correct'],
            'n_episodes': data['n_episodes'],
            'file': data['file'],
        })
    
    # Save summary CSV
    summary_path = output_dir / 'validation_summary.csv'
    with open(summary_path, 'w', newline='') as f:
        fieldnames = ['method', 'accuracy_pct', 'avg_turns', 'stopped_turn_1_pct', 
                      'mi_zero_start_pct', 'n_correct', 'n_episodes', 'file']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    
    # Detailed per-method analysis
    print("\n" + "="*70)
    print("DETAILED ANALYSIS: Turn Distribution & Accuracy by Turns")
    print("="*70)
    
    for method, data in sorted(all_results.items()):
        print(f"\n{method}")
        print("-"*50)
        
        # Turn distribution
        turn_dist = data['turn_distribution']
        correct_by_turns = data['correct_by_turns']
        
        print(f"{'Turns':<8} {'Count':<8} {'Correct':<10} {'Acc%':<8}")
        for turns, count in sorted(turn_dist.items()):
            cb = correct_by_turns.get(turns, {'correct': 0, 'total': count})
            acc = cb['correct'] / cb['total'] * 100 if cb['total'] > 0 else 0
            print(f"{turns:<8} {count:<8} {cb['correct']:<10} {acc:>5.1f}%")
        
        # Save per-method detail
        detail_path = output_dir / f'detail_{method.lower().replace(" ", "_")}.csv'
        with open(detail_path, 'w', newline='') as f:
            fieldnames = ['puzzle_idx', 'turns', 'correct', 'first_mi', 'final_mi', 'mi_zero_at_start']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, ep in enumerate(data['episodes']):
                writer.writerow({
                    'puzzle_idx': i,
                    'turns': ep['turns'],
                    'correct': ep['correct'],
                    'first_mi': ep['first_mi'],
                    'final_mi': ep['final_mi'],
                    'mi_zero_at_start': ep['mi_zero_at_start'],
                })
    
    print(f"\n\nResults saved to: {output_dir}")
    
    # Generate LaTeX table for slides
    print("\n" + "="*70)
    print("LATEX TABLE FOR SLIDES")
    print("="*70)
    
    latex = """
\\begin{table}
  \\centering
  \\begin{tabular}{lcccc}
    \\toprule
    \\textbf{Method} & \\textbf{Accuracy} & \\textbf{Avg Turns} & \\textbf{Turn 1 \\%} & \\textbf{MI=0 \\%} \\\\
    \\midrule
"""
    for row in summary_rows:
        latex += f"    {row['method'].replace('_', '\\_')} & {row['accuracy_pct']:.1f}\\% & {row['avg_turns']:.1f} & {row['stopped_turn_1_pct']:.1f}\\% & {row['mi_zero_start_pct']:.1f}\\% \\\\\n"
    
    latex += """    \\bottomrule
  \\end{tabular}
  \\caption{Validation results on 20 puzzles showing early-stopping problem}
\\end{table}
"""
    print(latex)
    
    # Save LaTeX
    latex_path = output_dir / 'validation_table.tex'
    with open(latex_path, 'w') as f:
        f.write(latex)


if __name__ == '__main__':
    main()
