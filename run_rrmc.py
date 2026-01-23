#!/usr/bin/env python3
"""
RRMC Small-Scale Validation Script.

Runs the RRMC approach on AR-Bench tasks with optional
risk-controlled threshold calibration.

Usage:
    # Basic comparison (fixed thresholds)
    python run_rrmc.py --n_puzzles 5 --methods robust_mi,self_consistency

    # Full calibrated RRMC pipeline
    python run_rrmc.py --calibrate --n_train 10 --n_test 5

    # Load existing calibration
    python run_rrmc.py --load_calibration results/calibration_dc_xxx.json --n_test 5
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rrmc.core.llm import LLMWrapper, load_api_key_from_env_file
from rrmc.core.environment import TaskType
from rrmc.evaluation.evaluator import RRMCEvaluator, print_comparison_table


def main():
    parser = argparse.ArgumentParser(description="RRMC Validation with Risk-Controlled Calibration")

    # Data and task settings
    parser.add_argument(
        "--task",
        type=str,
        default="dc",
        choices=["dc", "sp", "gn"],
        help="Task type (default: dc)",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum turns per episode (default: 10 for faster testing)",
    )

    # Calibration mode settings
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable calibration mode: collect states on train, calibrate tau, evaluate on test",
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=10,
        help="Number of train puzzles for calibration (default: 10)",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=5,
        help="Number of test puzzles for evaluation (default: 5)",
    )
    parser.add_argument(
        "--target_error",
        type=float,
        default=0.10,
        help="Target error rate for calibration (default: 0.10)",
    )
    parser.add_argument(
        "--load_calibration",
        type=str,
        default=None,
        help="Path to existing calibration file to load",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to train data file (default: auto-detect)",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data file (default: auto-detect)",
    )

    # Non-calibration mode settings (backwards compatible)
    parser.add_argument(
        "--n_puzzles",
        type=int,
        default=5,
        help="Number of puzzles for non-calibration mode (default: 5)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="fixed_turns,self_consistency,mi_only,robust_mi",
        help="Comma-separated list of methods to compare (non-calibration mode)",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="qwen/qwen3-8b",
        help="Model to use via OpenRouter (default: qwen/qwen3-8b)",
    )
    parser.add_argument(
        "--env_file",
        type=str,
        default="openrouter.env",
        help="Path to env file with API key",
    )

    # Output settings
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print verbose output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    # MI estimation settings
    parser.add_argument(
        "--k_samples",
        type=int,
        default=4,
        help="Number of samples for MI estimation (default: 4)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="base,skeptical",
        help="Comma-separated prompt variants for robust MI (default: base,skeptical)",
    )
    parser.add_argument(
        "--regime",
        type=str,
        default="normal",
        choices=["normal", "homogeneous"],
        help="Decoding regime: normal (default) or homogeneous (low temp for stress testing)",
    )

    args = parser.parse_args()

    # Handle verbose/quiet
    verbose = args.verbose and not args.quiet

    # Load API key
    env_path = os.path.join(os.path.dirname(__file__), args.env_file)
    if os.path.exists(env_path):
        api_key = load_api_key_from_env_file(env_path)
        if verbose:
            print(f"Loaded API key from {env_path}")
    else:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("Error: No API key found. Set OPENROUTER_API_KEY or provide --env_file")
            sys.exit(1)

    # Initialize LLM
    if verbose:
        print(f"\nInitializing LLM: {args.model}")
    llm = LLMWrapper(
        api_key=api_key,
        model=args.model,
        default_temperature=0.7,
        default_top_p=0.95,
    )

    # Determine task type
    task_type = {
        "dc": TaskType.DC,
        "sp": TaskType.SP,
        "gn": TaskType.GN,
    }[args.task]

    # Parse variants
    variants = [v.strip() for v in args.variants.split(",")]

    # Determine data paths
    base_dir = os.path.dirname(__file__)

    if args.calibrate or args.load_calibration:
        # Calibration mode - need train and test data
        if args.train_data:
            train_path = args.train_data
        else:
            train_path = os.path.join(base_dir, "AR-Bench", "data", args.task, "train.json")
            # Fallback to test if no train exists
            if not os.path.exists(train_path):
                train_path = os.path.join(base_dir, "AR-Bench", "data", args.task, "test.json")

        if args.test_data:
            test_path = args.test_data
        else:
            test_path = os.path.join(base_dir, "AR-Bench", "data", args.task, "test.json")

        if not os.path.exists(test_path):
            print(f"Error: Test data file not found: {test_path}")
            sys.exit(1)

        if verbose:
            print(f"\nTask: {task_type.value}")
            print(f"Train data: {train_path}")
            print(f"Test data: {test_path}")
            print(f"Target error rate: {args.target_error:.1%}")
            print(f"Decoding regime: {args.regime}")

        # Initialize evaluator with test data
        evaluator = RRMCEvaluator(
            llm=llm,
            data_path=test_path,
            task_type=task_type,
            max_turns=args.max_turns,
            output_dir=args.output_dir,
            regime=args.regime,
        )

        # Determine puzzle indices
        n_test_available = len(evaluator.env)
        test_indices = list(range(min(args.n_test, n_test_available)))

        if args.load_calibration:
            # Load existing calibration and evaluate
            if verbose:
                print(f"\nLoading calibration from: {args.load_calibration}")

            try:
                results = evaluator.load_calibration_and_evaluate(
                    calibration_path=args.load_calibration,
                    test_indices=test_indices,
                    target_error=args.target_error,
                    k_samples=args.k_samples,
                    variants=variants,
                    other_methods=["fixed_turns", "self_consistency", "mi_only"],
                    verbose=verbose,
                )
            except Exception as e:
                print(f"\nError loading calibration: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        else:
            # Full calibration pipeline
            # Need to create a separate evaluator for train data
            if train_path != test_path:
                train_evaluator = RRMCEvaluator(
                    llm=llm,
                    data_path=train_path,
                    task_type=task_type,
                    max_turns=args.max_turns,
                    output_dir=args.output_dir,
                    regime=args.regime,
                )
                n_train_available = len(train_evaluator.env)
            else:
                # Using same file for train/test (for small tests)
                train_evaluator = evaluator
                n_train_available = n_test_available
                # Use different indices for train
                test_indices = list(range(min(args.n_test, n_test_available // 2)))

            train_indices = list(range(min(args.n_train, n_train_available)))

            # Make sure train and test don't overlap when using same file
            if train_path == test_path:
                max_idx = n_train_available
                train_indices = list(range(0, min(args.n_train, max_idx // 2)))
                test_indices = list(range(max_idx // 2, max_idx // 2 + min(args.n_test, max_idx // 2)))

            if verbose:
                print(f"\nTrain puzzles: {len(train_indices)} (indices: {train_indices[:5]}...)")
                print(f"Test puzzles: {len(test_indices)} (indices: {test_indices[:5]}...)")

            try:
                # Collect calibration states from train data
                calibrator = train_evaluator.collect_calibration_states(
                    puzzle_indices=train_indices,
                    k_samples=args.k_samples,
                    variants=variants,
                    verbose=verbose,
                )

                # Calibrate threshold
                calibration_result = train_evaluator.calibrate_threshold(
                    calibrator=calibrator,
                    target_error=args.target_error,
                    verbose=verbose,
                )

                # Save calibration
                from datetime import datetime
                calibration_path = os.path.join(
                    args.output_dir,
                    f"calibration_{args.task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                os.makedirs(args.output_dir, exist_ok=True)
                calibrator.save(calibration_path)
                if verbose:
                    print(f"  Calibration saved to: {calibration_path}")

                # Evaluate with calibrated threshold
                calibrated_threshold = calibration_result.threshold
                if calibrated_threshold == -float('inf'):
                    calibrated_threshold = 0.1
                    if verbose:
                        print(f"  Using fallback threshold: {calibrated_threshold}")

                # Import here to avoid circular imports
                from rrmc.baselines.stopping_rules import RobustMIStopping

                if verbose:
                    print(f"\n{'='*60}")
                    print("EVALUATION PHASE")
                    print(f"{'='*60}")
                    print(f"Using calibrated threshold τ = {calibrated_threshold:.4f}")

                eval_results = {}

                # Calibrated RRMC
                calibrated_rule = RobustMIStopping(
                    llm=llm,
                    clusterer=evaluator.clusterer,
                    threshold=calibrated_threshold,
                    k_samples=args.k_samples,
                    max_turns=args.max_turns,
                    variants=variants,
                    regime=args.regime,
                )
                eval_results["rrmc_calibrated"] = evaluator.evaluate_method(
                    stopping_rule=calibrated_rule,
                    puzzle_indices=test_indices,
                    verbose=verbose,
                )

                # Compare against baselines
                for method in ["fixed_turns", "self_consistency", "mi_only"]:
                    stopping_rule = evaluator._create_stopping_rule(method)
                    eval_results[method] = evaluator.evaluate_method(
                        stopping_rule=stopping_rule,
                        puzzle_indices=test_indices,
                        verbose=verbose,
                    )

                # Save results
                evaluator._save_results(eval_results)

                # Get MI-error correlation from calibrator (computed during calibrate_threshold)
                mi_corr = getattr(calibrator, 'mi_error_correlation', None)

                results = {
                    "calibration": {
                        "threshold": calibration_result.threshold,
                        "n_states": calibration_result.n_states,
                        "n_covered": calibration_result.n_covered,
                        "empirical_error": calibration_result.empirical_error,
                        "ucb_error": calibration_result.ucb_error,
                        "mi_error_correlation": mi_corr,
                    },
                    "evaluation": eval_results,
                }

            except Exception as e:
                print(f"\nError during calibration: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        # Print results
        mi_error_corr = results["calibration"].get("mi_error_correlation")
        print_comparison_table(results["evaluation"], mi_error_correlation=mi_error_corr)

        if verbose:
            print(f"\nCalibration Summary:")
            print(f"  Threshold (τ): {results['calibration']['threshold']:.4f}")
            print(f"  States collected: {results['calibration']['n_states']}")
            print(f"  States covered: {results['calibration']['n_covered']}")
            print(f"  Empirical error: {results['calibration']['empirical_error']:.2%}")

    else:
        # Original non-calibration mode (backwards compatible)
        data_path = os.path.join(base_dir, "AR-Bench", "data", args.task, "test.json")

        if not os.path.exists(data_path):
            print(f"Error: Data file not found: {data_path}")
            sys.exit(1)

        if verbose:
            print(f"Task: {task_type.value}")
            print(f"Data: {data_path}")
            print(f"Evaluating {args.n_puzzles} puzzles with max {args.max_turns} turns")

        # Initialize evaluator
        evaluator = RRMCEvaluator(
            llm=llm,
            data_path=data_path,
            task_type=task_type,
            max_turns=args.max_turns,
            output_dir=args.output_dir,
            regime=args.regime,
        )

        # Parse methods
        methods = [m.strip() for m in args.methods.split(",")]
        if verbose:
            print(f"Methods: {methods}")

        # Run comparison
        if verbose:
            print("\n" + "=" * 60)
            print("STARTING EVALUATION")
            print("=" * 60)

        puzzle_indices = list(range(min(args.n_puzzles, len(evaluator.env))))

        try:
            results = evaluator.run_comparison(
                puzzle_indices=puzzle_indices,
                methods=methods,
                verbose=verbose,
            )

            # Print comparison table
            print_comparison_table(results)

        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"\nError during evaluation: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    # Print token usage
    usage = llm.get_token_usage()
    print(f"\nTotal API usage:")
    print(f"  Prompt tokens: {usage['prompt_tokens']:,}")
    print(f"  Completion tokens: {usage['completion_tokens']:,}")
    print(f"  Total tokens: {usage['total_tokens']:,}")


if __name__ == "__main__":
    main()
