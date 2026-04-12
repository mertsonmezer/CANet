#!/usr/bin/env python3
"""Example script to run robustness testing for the best MSE score.

Usage:
python run_robustness_test.py --best_mse 0.123456 --num_trials 5

This script will:
1. Run optuna optimization for the specified number of trials
2. If any trial achieves an MSE close to the best_mse, it will run robustness testing
3. The robustness test runs the same configuration with 5 different seeds: [original_seed, 0, 2, 42, 2025]
4. Print statistics about the robustness across different seeds
"""

import subprocess
import sys


def run_robustness_test():
  # Example best MSE score - replace with your actual best MSE
  best_mse = 0.123456

  # Command to run the modified script
  cmd = [
    sys.executable,
    "forecasting/train_tune_gpu0.py",
    "--best_mse",
    str(best_mse),
    "--num_trials",
    "10",  # Reduced for testing
    "--num_epochs",
    "5",  # Reduced for testing
    "--data_name",
    "ETTh1",
    "--seq_len",
    "96",
    "--pred_len",
    "96",
  ]

  print("Running robustness test with the following command:")
  print(" ".join(cmd))
  print("\nNote: Modify the --best_mse parameter to match your actual best MSE score")
  print("=" * 60)

  try:
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
      print("STDERR:")
      print(result.stderr)
  except subprocess.CalledProcessError as e:
    print(f"Command failed with return code {e.returncode}")
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)
  except FileNotFoundError:
    print("Error: Could not find the train_tune_gpu0.py script")
    print("Make sure you're running this from the correct directory")


if __name__ == "__main__":
  print(__doc__)
  run_robustness_test()
