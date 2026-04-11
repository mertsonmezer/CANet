#!/usr/bin/env python3
"""
Standalone Robustness Testing Script

This script finds the best hyperparameters from an existing optuna database
and performs robustness testing with multiple seeds for both MSE and MAE metrics.

Usage:
    python forecasting/robustness_test.py --data_name ETTh1 --seq_len 96 --pred_len 96
    python forecasting/robustness_test.py --database_path "path/to/study.db"
    python forecasting/robustness_test.py --data_name ETTh1 --custom_seeds 42,123,456
"""

import argparse
import os
import sys
import json
import glob
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import the project modules
import lightning as L
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from config.net_config import DataConfig, TrainConfig
from data.data_factory import data_provider
from models.net import TrainedNet
from utils.utils import str2bool


def find_optuna_database(config: TrainConfig, results_dir: str = "results", database_path: Optional[str] = None) -> Optional[str]:
  """Find the corresponding optuna database file for the given configuration."""

  # If explicit database path is provided, use it
  if database_path and os.path.exists(database_path):
    return database_path

  study_name = f"{config.data_config.data_name}-SeqLen{config.seq_len}-PredLen{config.pred_len}-FirstNorm{config.first_norm_layer}-SecondNorm{config.second_norm_layer}-MP{config.use_multi_patching}-RankReduction{config.use_rank_reduction}"

  # Check in the main directory first
  main_db_path = f"{study_name}.db"
  if os.path.exists(main_db_path):
    return main_db_path

  # Check in the results directory structure
  dataset_results_dir = os.path.join(results_dir, config.data_config.data_name)
  if os.path.exists(dataset_results_dir):
    db_path = os.path.join(dataset_results_dir, f"{study_name}.db")
    if os.path.exists(db_path):
      return db_path

  # Search for any matching database file in results directory
  pattern = os.path.join(results_dir, "**", f"{study_name}.db")
  matching_files = glob.glob(pattern, recursive=True)
  if matching_files:
    return matching_files[0]

  # If exact match not found, search for similar configurations
  pattern = os.path.join(results_dir, "**", f"{config.data_config.data_name}*.db")
  matching_files = glob.glob(pattern, recursive=True)
  for file_path in matching_files:
    if f"SeqLen{config.seq_len}" in file_path and f"PredLen{config.pred_len}" in file_path:
      print(f"Found similar database: {file_path}")
      return file_path

  return None


def load_best_result_from_database(db_path: str) -> Optional[Tuple[float, float, Dict[str, Any]]]:
  """Load the best result (MSE, MAE, and hyperparameters) from optuna database."""
  try:
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{db_path}")

    if len(study.trials) == 0:
      print(f"No trials found in database: {db_path}")
      return None

    best_trial = study.best_trial
    if best_trial.value is None:
      print(f"Best trial has no value in database: {db_path}")
      return None

    best_mse = float(best_trial.value)

    # Estimate MAE from MSE (rough approximation)
    best_mae = best_mse * 0.8

    print(f"Loaded best result from database: {db_path}")
    print(f"Best MSE: {best_mse:.6f}")
    print(f"Best trial number: {best_trial.number}")
    print(f"Total trials in study: {len(study.trials)}")
    print(f"Best hyperparameters: {best_trial.params}")

    return best_mse, best_mae, best_trial.params

  except Exception as e:
    print(f"Error loading database {db_path}: {e}")
    return None


def train_model(
  config: TrainConfig,
  train_dataloader: DataLoader[torch.dtype],
  val_dataloader: DataLoader[torch.dtype],
  test_dataloader: DataLoader[torch.dtype],
  checkpoint_dir_path: str,
  checkpoint_callback: ModelCheckpoint,
) -> Tuple[TrainedNet, Dict[str, float], Dict[str, float]]:
  """Train a model and return MSE and MAE results."""
  trainer = L.Trainer(
    default_root_dir=checkpoint_dir_path,
    accelerator="auto",
    num_sanity_val_steps=0,
    devices=1,
    max_epochs=config.num_epochs,
    accumulate_grad_batches=config.accumulate_grad_batches,
    callbacks=[
      checkpoint_callback,
      LearningRateMonitor("epoch"),
      TQDMProgressBar(refresh_rate=500),
    ],
  )

  # Disable computation graph and default hyperparameter metric logging
  trainer.logger._log_graph = False
  trainer.logger._default_hp_metric = None

  L.seed_everything(config.seed)
  model = TrainedNet(config)

  trainer.fit(model, train_dataloader, val_dataloader)

  # Load the best checkpoint after training
  model: TrainedNet = TrainedNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

  # Test the model on validation and test datasets
  val_result = trainer.test(model, dataloaders=val_dataloader, verbose=False)
  test_result = trainer.test(model, dataloaders=test_dataloader, verbose=False)

  mse_result: Dict[str, float] = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
  mae_result: Dict[str, float] = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

  return model, mse_result, mae_result


def run_single_training(config: TrainConfig, hyperparams: Dict[str, Any], seed: int, run_suffix: str = "") -> Tuple[Dict[str, float], Dict[str, float]]:
  """Run a single training with specified hyperparameters and seed. Returns (mse_result, mae_result)."""
  # Set hyperparameters
  config.batch_size = hyperparams["batch_size"]
  config.train_lr = hyperparams["train_lr"]
  config.dropout_rate = hyperparams["dropout_rate"]

  # Handle patch_sizes conversion if it's a string from optuna
  if isinstance(hyperparams["patch_sizes"], str):
    config.patch_sizes = [int(x) for x in hyperparams["patch_sizes"].split(",")]
  else:
    config.patch_sizes = hyperparams["patch_sizes"]

  config.embed_dim = hyperparams["embed_dim"]
  config.seed = seed

  # Load datasets
  _, train_dataloader = data_provider(config, flag="train")
  _, val_dataloader = data_provider(config, flag="val")
  _, test_dataloader = data_provider(config, flag="test")

  run_description: str = f"{config.data_config.data_name}-SeqLen{config.seq_len}-PredLen{config.pred_len}-FirstNorm{config.first_norm_layer}-SecondNorm{config.second_norm_layer}-MP{config.use_multi_patching}-RankReduction{config.use_rank_reduction}-Seed{seed}{run_suffix}"
  print(f"========== {run_description} ===========")

  checkpoint_dir_path: str = f"logs/{run_description}"
  checkpoint_callback = ModelCheckpoint(dirpath=checkpoint_dir_path, save_top_k=1, filename="train-{epoch}", monitor="val_mse", mode="min")

  model, mse_result, mae_result = train_model(
    config,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    checkpoint_dir_path,
    checkpoint_callback,
  )

  # Count total parameters in the model
  total_params: int = sum(p.numel() for p in model.parameters())

  # Save configuration and results
  with open(os.path.join(checkpoint_dir_path, "model.txt"), "w", encoding="utf-8") as f:
    f.write(f"Total Number of Parameters: {str(total_params)}\n\n")
    f.write("Training Configuration:\n")
    json.dump(asdict(config), f, indent=2)

  # Save both MSE and MAE results
  df = pd.DataFrame({"MSE": mse_result, "MAE": mae_result})
  df.to_excel(
    os.path.join(
      checkpoint_dir_path,
      "results.xlsx",
    )
  )

  return mse_result, mae_result


def run_robustness_test(config: TrainConfig, best_hyperparams: Dict[str, Any], seeds: List[int]) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
  """Run robustness test with multiple seeds and return both MSE and MAE results."""
  mse_results = []
  mae_results = []

  print(f"\n{'=' * 60}")
  print(f"ROBUSTNESS TEST: Running with {len(seeds)} different seeds")
  print(f"Seeds: {seeds}")
  print(f"{'=' * 60}")

  for i, seed in enumerate(seeds):
    print(f"\nRun {i + 1}/{len(seeds)} with seed {seed}")
    mse_result, mae_result = run_single_training(config, best_hyperparams, seed, "-RobustnessTest")

    # Store results with seed information
    mse_results.append({"test": mse_result["test"], "val": mse_result["val"], "seed": seed})
    mae_results.append({"test": mae_result["test"], "val": mae_result["val"], "seed": seed})

    print(f"Test MSE for seed {seed}: {mse_result['test']:.6f}")
    print(f"Test MAE for seed {seed}: {mae_result['test']:.6f}")

  return mse_results, mae_results


def save_robustness_results(mse_results: List[Dict[str, float]], mae_results: List[Dict[str, float]], config: TrainConfig, best_hyperparams: Dict[str, Any], output_dir: str = "robustness_results") -> str:
  """Save robustness test results to Excel file."""
  os.makedirs(output_dir, exist_ok=True)

  # Create filename based on configuration
  filename = f"{config.data_config.data_name}-SeqLen{config.seq_len}-PredLen{config.pred_len}-FirstNorm{config.first_norm_layer}-SecondNorm{config.second_norm_layer}-MP{config.use_multi_patching}-RankReduction{config.use_rank_reduction}-RobustnessResults.xlsx"
  filepath = os.path.join(output_dir, filename)

  # Create DataFrames
  mse_df = pd.DataFrame(mse_results)
  mae_df = pd.DataFrame(mae_results)

  # Create hyperparameters DataFrame
  hyperparams_df = pd.DataFrame([{"Parameter": k, "Value": str(v)} for k, v in best_hyperparams.items()])

  # Calculate statistics
  mse_stats = {"Metric": ["MSE"] * 4, "Type": ["Mean", "Std", "Min", "Max"], "Test": [mse_df["test"].mean(), mse_df["test"].std(), mse_df["test"].min(), mse_df["test"].max()], "Val": [mse_df["val"].mean(), mse_df["val"].std(), mse_df["val"].min(), mse_df["val"].max()]}

  mae_stats = {"Metric": ["MAE"] * 4, "Type": ["Mean", "Std", "Min", "Max"], "Test": [mae_df["test"].mean(), mae_df["test"].std(), mae_df["test"].min(), mae_df["test"].max()], "Val": [mae_df["val"].mean(), mae_df["val"].std(), mae_df["val"].min(), mae_df["val"].max()]}

  # Save to Excel with multiple sheets
  with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
    mse_df.to_excel(writer, sheet_name="MSE_Results", index=False)
    mae_df.to_excel(writer, sheet_name="MAE_Results", index=False)
    pd.DataFrame(mse_stats).to_excel(writer, sheet_name="MSE_Statistics", index=False)
    pd.DataFrame(mae_stats).to_excel(writer, sheet_name="MAE_Statistics", index=False)
    hyperparams_df.to_excel(writer, sheet_name="Best_Hyperparameters", index=False)

  print(f"\nRobustness results saved to: {filepath}")
  return filepath


def print_statistics(mse_results: List[Dict[str, float]], mae_results: List[Dict[str, float]]):
  """Print comprehensive statistics for MSE and MAE results."""
  # Calculate statistics for MSE
  test_mses = [result["test"] for result in mse_results]
  val_mses = [result["val"] for result in mse_results]
  mean_test_mse = sum(test_mses) / len(test_mses)
  std_test_mse = (sum((x - mean_test_mse) ** 2 for x in test_mses) / len(test_mses)) ** 0.5
  mean_val_mse = sum(val_mses) / len(val_mses)
  std_val_mse = (sum((x - mean_val_mse) ** 2 for x in val_mses) / len(val_mses)) ** 0.5

  # Calculate statistics for MAE
  test_maes = [result["test"] for result in mae_results]
  val_maes = [result["val"] for result in mae_results]
  mean_test_mae = sum(test_maes) / len(test_maes)
  std_test_mae = (sum((x - mean_test_mae) ** 2 for x in test_maes) / len(test_maes)) ** 0.5
  mean_val_mae = sum(val_maes) / len(val_maes)
  std_val_mae = (sum((x - mean_val_mae) ** 2 for x in val_maes) / len(val_maes)) ** 0.5

  print(f"\n{'=' * 60}")
  print("ROBUSTNESS TEST RESULTS SUMMARY")
  print(f"{'=' * 60}")

  # Print individual results
  seeds = [result["seed"] for result in mse_results]
  for seed, mse_result, mae_result in zip(seeds, mse_results, mae_results):
    print(f"Seed {seed:4d}: Test MSE = {mse_result['test']:.6f}, Val MSE = {mse_result['val']:.6f}")
    print(f"           Test MAE = {mae_result['test']:.6f}, Val MAE = {mae_result['val']:.6f}")

  print(f"\n{'=' * 30} MSE STATISTICS {'=' * 30}")
  print(f"Test MSE  - Mean: {mean_test_mse:.6f}, Std: {std_test_mse:.6f}, Min: {min(test_mses):.6f}, Max: {max(test_mses):.6f}")
  print(f"Val MSE   - Mean: {mean_val_mse:.6f}, Std: {std_val_mse:.6f}, Min: {min(val_mses):.6f}, Max: {max(val_mses):.6f}")

  print(f"\n{'=' * 30} MAE STATISTICS {'=' * 30}")
  print(f"Test MAE  - Mean: {mean_test_mae:.6f}, Std: {std_test_mae:.6f}, Min: {min(test_maes):.6f}, Max: {max(test_maes):.6f}")
  print(f"Val MAE   - Mean: {mean_val_mae:.6f}, Std: {std_val_mae:.6f}, Min: {min(val_maes):.6f}, Max: {max(val_maes):.6f}")
  print(f"{'=' * 60}")


def main():
  parser = argparse.ArgumentParser(description="Standalone Robustness Testing for Best Hyperparameters")

  # Data Configuration
  parser.add_argument("--data_name", type=str, default="ETTh1", help="Name of the dataset.")
  parser.add_argument("--root_path", type=str, default="ETT-small", help="Root path of the dataset.")
  parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Relative path to the data file.")
  parser.add_argument("--embed", type=str, default="timeF", help="Time features encoding (options: timeF, fixed, learned)")
  parser.add_argument("--features", type=str, default="M", help="Forecasting task type (options: M, S, MS)")
  parser.add_argument("--target", type=str, default="OT", help="Target feature")
  parser.add_argument("--freq", type=str, default="h", help="Frequency for time features encoding.")
  parser.add_argument("--num_channels", type=int, default=7, help="Number of data channels.")

  # Model Configuration
  parser.add_argument("--depth", type=int, default=2, help="Number of layers in the model.")
  parser.add_argument("--first_norm_layer", type=str, default="layer", help="Type of normalization technique used before noise reduction.")
  parser.add_argument("--second_norm_layer", type=str, default="adain", help="Type of normalization technique used after noise reduction.")
  parser.add_argument("--patch_size", type=int, default=16, help="Size of patches (used if multi-patching is disabled).")
  parser.add_argument("--blend_ratio", type=float, default=0.1, help="Indicates the ratio at which the style input will be combined.")
  parser.add_argument("--style_blend_gate_affine", type=str2bool, default=True, help="Whether to apply extra shifting and scaling.")
  parser.add_argument("--use_multi_patching", type=str2bool, default=True, help="Whether to use multi-patching technique.")
  parser.add_argument("--use_rank_reduction", type=str2bool, default=True, help="Whether to use rank reduction technique.")

  # Training Configuration
  parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs.")
  parser.add_argument("--seq_len", type=int, default=96, help="Length of the input sequence.")
  parser.add_argument("--label_len", type=int, default=0, help="Length of the label sequence.")
  parser.add_argument("--pred_len", type=int, default=96, help="Length of the prediction sequence.")
  parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulates gradients over specified number of batches.")

  # Robustness Testing Configuration
  parser.add_argument("--database_path", type=str, default=None, help="Explicit path to optuna database file.")
  parser.add_argument("--custom_seeds", type=str, default="56,0,2,42,2025", help="Comma-separated list of seeds for robustness testing.")
  parser.add_argument("--output_dir", type=str, default="robustness_results", help="Directory to save robustness results.")
  parser.add_argument("--results_dir", type=str, default="results", help="Directory to search for optuna databases.")

  args = parser.parse_args()

  # Parse custom seeds
  try:
    seeds = [int(s.strip()) for s in args.custom_seeds.split(",")]
  except ValueError:
    print(f"Error parsing custom seeds: {args.custom_seeds}")
    print("Using default seeds: [56, 0, 2, 42, 2025]")
    seeds = [56, 0, 2, 42, 2025]

  print("Standalone Robustness Testing Script")
  print("=" * 60)
  print(f"Dataset: {args.data_name}")
  print(f"Sequence Length: {args.seq_len}")
  print(f"Prediction Length: {args.pred_len}")
  print(f"Seeds for testing: {seeds}")
  print(f"Number of epochs per run: {args.num_epochs}")
  print("=" * 60)

  # Create configuration
  data_config = DataConfig(
    data_name=args.data_name,
    root_dir_path=args.root_path,
    data_path=args.data_path,
    embed=args.embed,
    features=args.features,
    target=args.target,
    freq=args.freq,
    num_channels=args.num_channels,
  )

  train_config = TrainConfig(
    num_epochs=args.num_epochs,
    data_config=data_config,
    patch_size=args.patch_size,
    seq_len=args.seq_len,
    label_len=args.label_len,
    pred_len=args.pred_len,
    first_norm_layer=args.first_norm_layer,
    second_norm_layer=args.second_norm_layer,
    blend_ratio=args.blend_ratio,
    style_blend_gate_affine=args.style_blend_gate_affine,
    use_multi_patching=args.use_multi_patching,
    use_rank_reduction=args.use_rank_reduction,
    accumulate_grad_batches=args.accumulate_grad_batches,
    seed=seeds[0],  # Use first seed as default
  )

  # Find optuna database
  db_path = find_optuna_database(train_config, args.results_dir, args.database_path)
  if not db_path:
    print("ERROR: No optuna database found!")
    print("Please ensure you have run hyperparameter tuning first, or provide a valid --database_path")
    return

  # Load best result from database
  result = load_best_result_from_database(db_path)
  if not result:
    print("ERROR: Could not load best result from database!")
    return

  best_mse, best_mae, best_hyperparams = result

  # Set up reproducibility
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  print(f"\nStarting robustness testing with best hyperparameters...")
  print(f"Best MSE from database: {best_mse:.6f}")

  # Run robustness test
  mse_results, mae_results = run_robustness_test(train_config, best_hyperparams, seeds)

  # Save results to Excel
  excel_path = save_robustness_results(mse_results, mae_results, train_config, best_hyperparams, args.output_dir)

  # Print comprehensive statistics
  print_statistics(mse_results, mae_results)

  print(f"\nRobustness testing completed!")
  print(f"Results saved to: {excel_path}")
  print(f"Individual run logs saved to: logs/")


if __name__ == "__main__":
  main()
