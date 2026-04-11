#!/usr/bin/env python3
"""
Test script for the standalone robustness testing functionality.

This script demonstrates and tests the separated robustness testing script.
"""

import sys

# Add the project root to the Python path
sys.path.append("/home/pc-amax-2/Desktop/Mert/Mert-recovered/Mert/Net_final")


def test_database_finding():
  """Test the database finding functionality."""
  from forecasting.robustness_test import find_optuna_database, load_best_result_from_database
  from config.net_config import DataConfig, TrainConfig

  print("Testing Standalone Robustness Script")
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
      return True, db_path, result
    else:
      print("✗ Failed to load best result from database")
  else:
    print("✗ No database found for the configuration")

  return False, None, None


def print_usage_examples():
  """Print usage examples for the standalone robustness testing."""
  print("\nUsage Examples for Standalone Robustness Testing")
  print("=" * 60)

  print("1. Basic usage (automatic database detection):")
  print("   python forecasting/robustness_test.py \\")
  print("       --data_name ETTh1 \\")
  print("       --seq_len 96 \\")
  print("       --pred_len 96")

  print("\n2. Specify custom database path:")
  print("   python forecasting/robustness_test.py \\")
  print("       --database_path results/ETTh1/study.db \\")
  print("       --data_name ETTh1")

  print("\n3. Custom seeds:")
  print("   python forecasting/robustness_test.py \\")
  print("       --data_name ETTh1 \\")
  print("       --custom_seeds 1,2,3,4,5")

  print("\n4. Quick test (fewer epochs):")
  print("   python forecasting/robustness_test.py \\")
  print("       --data_name ETTh1 \\")
  print("       --num_epochs 3 \\")
  print("       --custom_seeds 56,42")

  print("\n5. Different normalization settings:")
  print("   python forecasting/robustness_test.py \\")
  print("       --data_name ETTh1 \\")
  print("       --first_norm_layer layer \\")
  print("       --second_norm_layer adain")


def print_advantages():
  """Print advantages of the standalone robustness testing script."""
  print("\nAdvantages of Standalone Robustness Testing")
  print("=" * 60)

  advantages = [
    "🎯 **Focused Purpose**: Only performs robustness testing, no hyperparameter optimization",
    "⚡ **Faster Execution**: Skips the lengthy optuna optimization phase",
    "🔄 **Repeatable**: Can run robustness tests multiple times with different seeds",
    "📊 **Comprehensive Output**: Excel files with detailed statistics and hyperparameters",
    "🛠️ **Flexible**: Custom seeds, database paths, and output directories",
    "🔍 **Automatic Discovery**: Finds existing optuna databases automatically",
    "📁 **Clean Organization**: Separate logs and results for robustness testing",
    "📈 **Statistical Analysis**: Detailed statistics for both MSE and MAE metrics",
    "💾 **Persistent Results**: Excel output suitable for academic papers",
    "🔧 **Configurable**: All training parameters can be adjusted independently",
  ]

  for advantage in advantages:
    print(f"  {advantage}")


def demonstrate_workflow():
  """Demonstrate the typical workflow."""
  print("\nTypical Workflow")
  print("=" * 60)

  steps = [
    "1. **Run Hyperparameter Tuning**: Use train_tune_gpu0.py to find best hyperparameters",
    "2. **Database Creation**: Optuna creates .db file with all trial results",
    "3. **Robustness Testing**: Use robustness_test.py to test best hyperparameters",
    "4. **Results Analysis**: Review Excel output and console statistics",
    "5. **Paper Writing**: Use statistical results for robustness claims",
  ]

  for step in steps:
    print(f"  {step}")

  print("\nFile Structure After Both Steps:")
  print("  📁 logs/")
  print("    ├── ETTh1-SeqLen96-PredLen96-...-Trial{N}/     # Hyperparameter tuning")
  print("    └── ETTh1-SeqLen96-PredLen96-...-Seed{S}-RobustnessTest/  # Robustness testing")
  print("  📁 robustness_results/")
  print("    └── ETTh1-SeqLen96-PredLen96-...-RobustnessResults.xlsx")
  print("  📁 results/")
  print("    └── ETTh1/")
  print("        └── ETTh1-SeqLen96-PredLen96-....db")


if __name__ == "__main__":
  print("Standalone Robustness Testing - Information & Demo")
  print("=" * 70)

  # Test database finding
  success, db_path, result = test_database_finding()

  if success:
    print("\n✅ Standalone robustness testing is ready to use!")
    print(f"Database found: {db_path}")
    if result:
      print(f"Best MSE: {result[0]:.6f}")
  else:
    print("\n⚠️  No database found. Run hyperparameter tuning first:")
    print("   python forecasting/train_tune_gpu0.py --data_name ETTh1 --num_trials 10")

  # Print documentation
  print_usage_examples()
  print_advantages()
  demonstrate_workflow()

  print("\n" + "=" * 70)
  print("SUMMARY")
  print("=" * 70)
  print("The standalone robustness testing script provides:")
  print("✅ Independent robustness testing without hyperparameter optimization")
  print("✅ Automatic database detection and best result loading")
  print("✅ Comprehensive statistical analysis for both MSE and MAE")
  print("✅ Professional Excel output with multiple sheets")
  print("✅ Flexible configuration options and custom seeds")
  print("✅ Clean separation of concerns for better workflow management")
