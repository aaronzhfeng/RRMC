#!/usr/bin/env python3.13
"""
RRMC - Risk-Controlled Robust-MI Active Inquiry

A config-based experiment runner for uncertainty-guided stopping in LLM games.

Usage:
    # Run with experiment config
    python run.py mve_dc_normal

    # Run with config file path
    python run.py configs/experiments/mve_dc_normal.yaml

    # Override settings via CLI
    python run.py mve_dc_normal --task sp --max_turns 10

    # List available experiments
    python run.py --list

Examples:
    # MVE on Detective Cases with normal decoding
    python run.py mve_dc_normal

    # MVE on Detective Cases with homogeneous decoding (stress test)
    python run.py mve_dc_homogeneous

    # MVE on Situation Puzzles
    python run.py mve_sp_normal

    # Quick test with fewer puzzles
    python run.py mve_dc_normal --n_train 3 --n_test 2 --max_turns 5
"""

import sys
from typing import Optional, List

from config import load_config, list_experiments
from pipeline import Pipeline


def main(
    config: Optional[str] = None,
    # Task settings
    task: Optional[str] = None,
    max_turns: Optional[int] = None,
    # Calibration settings
    calibrate: Optional[bool] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_puzzles: Optional[int] = None,  # Alias for n_test
    target_error: Optional[float] = None,
    # MI settings
    k_samples: Optional[int] = None,
    regime: Optional[str] = None,
    # Concurrency settings
    max_workers: Optional[int] = None,
    # Improvement features
    require_all_suspects: Optional[bool] = None,
    # Output settings
    output_dir: Optional[str] = None,
    verbose: bool = True,
    quiet: bool = False,
    # Utility flags
    list: bool = False,
    return_results: bool = False,
):
    """
    Run RRMC experiment.

    Args:
        config: Experiment name or path to config YAML
        task: Task type (dc, sp, gn)
        max_turns: Maximum turns per episode
        calibrate: Enable calibration mode
        n_train: Number of train puzzles for calibration
        n_test: Number of test puzzles for evaluation
        target_error: Target error rate for calibration
        k_samples: Number of samples for MI estimation
        regime: Decoding regime (normal, homogeneous)
        max_workers: Max concurrent API calls (default: 8)
        output_dir: Output directory for results
        verbose: Print verbose output
        quiet: Suppress output
        list: List available experiments and exit
        return_results: Return results dict (suppresses Fire stdout spam by default)
    """
    # Handle n_puzzles as alias for n_test
    if n_puzzles is not None and n_test is None:
        n_test = n_puzzles
    
    # Handle list flag
    if list:
        experiments = list_experiments()
        print("Available experiments:")
        for exp in experiments:
            print(f"  - {exp}")
        return

    # Default to mve_dc_normal if no config specified
    if config is None:
        print("Usage: python run.py <experiment_name> [options]")
        print("\nRun 'python run.py --list' to see available experiments.")
        print("\nExample: python run.py mve_dc_normal --n_train 5 --n_test 3")
        return

    # Build overrides from CLI args
    overrides = {}
    if task is not None:
        overrides["task"] = task
    if max_turns is not None:
        overrides["max_turns"] = max_turns
    if calibrate is not None:
        overrides["calibrate"] = calibrate
    if n_train is not None:
        overrides["n_train"] = n_train
    if n_test is not None:
        overrides["n_test"] = n_test
        overrides["n_puzzles"] = n_test  # Also set n_puzzles for comparison pipeline
    if target_error is not None:
        overrides["target_error"] = target_error
    if k_samples is not None:
        overrides["k_samples"] = k_samples
    if regime is not None:
        overrides["regime"] = regime
    if max_workers is not None:
        overrides["max_workers"] = max_workers
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if require_all_suspects is not None:
        overrides["require_all_suspects"] = require_all_suspects

    # Handle verbose/quiet
    overrides["verbose"] = verbose and not quiet

    # Load config with overrides
    try:
        cfg = load_config(experiment=config, overrides=overrides)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun 'python run.py --list' to see available experiments.")
        sys.exit(1)

    # Print experiment info
    if cfg.get("verbose", True):
        exp_name = cfg.get("experiment_name", config)
        print(f"\n{'='*60}")
        print(f"RRMC Experiment: {exp_name}")
        print(f"{'='*60}")

    # Run pipeline
    try:
        pipeline = Pipeline(cfg)
        results = pipeline.run()
        if return_results:
            return results
        return None
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Use fire for CLI parsing if available, otherwise argparse
    try:
        import fire
        fire.Fire(main)
    except ImportError:
        # Fallback: simple argparse
        import argparse
        parser = argparse.ArgumentParser(description="RRMC Experiment Runner")
        parser.add_argument("config", nargs="?", help="Experiment name or config path")
        parser.add_argument("--task", type=str, help="Task type (dc, sp, gn)")
        parser.add_argument("--max-turns", type=int, dest="max_turns", help="Max turns")
        parser.add_argument("--calibrate", action="store_true", help="Enable calibration")
        parser.add_argument("--n-train", type=int, dest="n_train", help="Train puzzles")
        parser.add_argument("--n-test", type=int, dest="n_test", help="Test puzzles")
        parser.add_argument("--target-error", type=float, dest="target_error", help="Target error")
        parser.add_argument("--k-samples", type=int, dest="k_samples", help="MI samples")
        parser.add_argument("--regime", type=str, help="Decoding regime")
        parser.add_argument("--output-dir", type=str, dest="output_dir", help="Output dir")
        parser.add_argument("--verbose", action="store_true", default=True, help="Verbose")
        parser.add_argument("--quiet", action="store_true", help="Quiet mode")
        parser.add_argument("--list", action="store_true", help="List experiments")
        parser.add_argument(
            "--return-results",
            action="store_true",
            dest="return_results",
            help="Return results dict (prints with Fire)",
        )

        args = parser.parse_args()
        main(
            config=args.config,
            task=args.task,
            max_turns=args.max_turns,
            calibrate=args.calibrate if args.calibrate else None,
            n_train=args.n_train,
            n_test=args.n_test,
            target_error=args.target_error,
            k_samples=args.k_samples,
            regime=args.regime,
            output_dir=args.output_dir,
            verbose=args.verbose,
            quiet=args.quiet,
            list=args.list,
            return_results=args.return_results,
        )
