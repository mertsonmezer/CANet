#!/usr/bin/env python3
"""Test script for the enhanced robustness testing functionality.

This script demonstrates:
1. Automatic database loading to find the best MSE
2. Robustness testing for both MSE and MAE
3. Excel output generation

Usage:
python test_robustness_enhanced.py
"""

import os
import sys

# Add the project root to the Python path
sys.path.append("/home/pc-amax-2/Desktop/Mert/Mert-recovered/Mert/Net_final")


def test_database_finding():
  """Test the database finding functionality."""
  from config.net_config import DataConfig, TrainConfig
  from forecasting.train_tune_gpu0 import find_optuna_database, load_best_result_from_database

  print("Testing Database Finding Functionality")
  print("=" * 50)

  # Create a sample configuration
  data_config = DataConfig(
    data_name="ETTh1",
    root_dir_path="ETT-small",
    data_path="ETTh1.csv",
    embed="timeF",
    features="M",
    target="OT",
    freq="h",
    num_channels=7,
  )

  config = TrainConfig(
    num_epochs=20,
    data_config=data_config,
    patch_size=16,
    seq_len=96,
    label_len=0,
    pred_len=96,
    first_norm_layer="layer",
    second_norm_layer="adain",
    blend_ratio=0.1,
    style_blend_gate_affine=True,
    use_multi_patching=True,
    use_rank_reduction=True,
    accumulate_grad_batches=1,
    seed=56,
  )

  # Test database finding
  db_path = find_optuna_database(config)
  if db_path:
    print(f"✓ Found database: {db_path}")

    # Test loading best result
    result = load_best_result_from_database(db_path)
    if result:
      best_mse, best_mae, hyperparams = result
      print("✓ Successfully loaded best result:")
      print(f"  - Best MSE: {best_mse:.6f}")
      print(f"  - Best MAE: {best_mae:.6f}")
      print(f"  - Hyperparameters: {hyperparams}")
      return True
    else:
      print("✗ Failed to load best result from database")
  else:
    print("✗ No database found for the configuration")

  return False


def test_robustness_functions():
  """Test the robustness testing functions without actual training."""
  import pandas as pd

  from config.net_config import DataConfig, TrainConfig
  from forecasting.train_tune_gpu0 import save_robustness_results

  print("\nTesting Robustness Functions")
  print("=" * 50)

  # Create sample data
  data_config = DataConfig(
    data_name="ETTh1",
    root_dir_path="ETT-small",
    data_path="ETTh1.csv",
    embed="timeF",
    features="M",
    target="OT",
    freq="h",
    num_channels=7,
  )

  config = TrainConfig(
    num_epochs=20,
    data_config=data_config,
    patch_size=16,
    seq_len=96,
    label_len=0,
    pred_len=96,
    first_norm_layer="layer",
    second_norm_layer="adain",
    blend_ratio=0.1,
    style_blend_gate_affine=True,
    use_multi_patching=True,
    use_rank_reduction=True,
    accumulate_grad_batches=1,
    seed=56,
  )

  # Sample results data
  mse_results = [
    {"test": 0.123456, "val": 0.098765, "seed": 56},
    {"test": 0.123789, "val": 0.098901, "seed": 0},
    {"test": 0.123234, "val": 0.098567, "seed": 2},
    {"test": 0.123678, "val": 0.098812, "seed": 42},
    {"test": 0.123445, "val": 0.098734, "seed": 2025},
  ]

  mae_results = [
    {"test": 0.098765, "val": 0.079012, "seed": 56},
    {"test": 0.099031, "val": 0.079121, "seed": 0},
    {"test": 0.098587, "val": 0.078854, "seed": 2},
    {"test": 0.098942, "val": 0.079050, "seed": 42},
    {"test": 0.098756, "val": 0.078987, "seed": 2025},
  ]

  try:
    # Test saving results
    output_path = save_robustness_results(mse_results, mae_results, config)

    if os.path.exists(output_path):
      print(f"✓ Excel file created successfully: {output_path}")

      # Verify the Excel file contents
      mse_df = pd.read_excel(output_path, sheet_name="MSE_Results")
      mae_df = pd.read_excel(output_path, sheet_name="MAE_Results")
      mse_stats = pd.read_excel(output_path, sheet_name="MSE_Statistics")
      mae_stats = pd.read_excel(output_path, sheet_name="MAE_Statistics")

      print(f"✓ Excel file contains {len(mse_df)} MSE results")
      print(f"✓ Excel file contains {len(mae_df)} MAE results")
      print(f"✓ MSE statistics sheet has {len(mse_stats)} rows")
      print(f"✓ MAE statistics sheet has {len(mae_stats)} rows")

      return True
    else:
      print("✗ Excel file was not created")

  except Exception as e:
    print(f"✗ Error creating Excel file: {e}")

  return False


def print_usage_example():
  """Print usage examples for the enhanced functionality."""
  print("\nUsage Examples")
  print("=" * 50)

  print("1. Run with automatic database detection:")
  print("   python forecasting/train_tune_gpu0.py \\")
  print("       --data_name ETTh1 \\")
  print("       --seq_len 96 \\")
  print("       --pred_len 96 \\")
  print("       --num_trials 10")

  print("\n2. Run with explicit best MSE:")
  print("   python forecasting/train_tune_gpu0.py \\")
  print("       --best_mse 0.123456 \\")
  print("       --data_name ETTh1 \\")
  print("       --seq_len 96 \\")
  print("       --pred_len 96 \\")
  print("       --num_trials 10")

  print("\n3. Key Features:")
  print("   ✓ Automatic optuna database detection")
  print("   ✓ Robustness testing with 5 seeds: [original, 0, 2, 42, 2025]")
  print("   ✓ Both MSE and MAE evaluation")
  print("   ✓ Excel output with multiple sheets")
  print("   ✓ Statistical analysis (mean, std, min, max)")


if __name__ == "__main__":
  print("Enhanced Robustness Testing - Test Suite")
  print("=" * 60)

  # Test database finding
  db_test_passed = test_database_finding()

  # Test robustness functions
  func_test_passed = test_robustness_functions()

  # Print usage examples
  print_usage_example()

  print("\n" + "=" * 60)
  print("TEST SUMMARY")
  print("=" * 60)
  print(f"Database Finding: {'✓ PASSED' if db_test_passed else '✗ FAILED'}")
  print(f"Robustness Functions: {'✓ PASSED' if func_test_passed else '✗ FAILED'}")

  if db_test_passed and func_test_passed:
    print("\n🎉 All tests passed! The enhanced robustness testing is ready to use.")
  else:
    print("\n❌ Some tests failed. Please check the implementation.")
