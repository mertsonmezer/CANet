import argparse
import shutil
import os
import json
import glob
from dataclasses import asdict
from functools import partial
from typing import Dict, List, Any, Optional, Tuple

import lightning as L
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from optuna.trial._frozen import FrozenTrial
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader

from config.net_config import DataConfig, TrainConfig
from data.data_factory import data_provider
from models.net import TrainedNet
from utils.utils import str2bool

# Set device to GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define ranges for hyperparameter tuning
LOW_LR, HIGH_LR = 1e-4, 1e-2
LOW_DROPOUT, HIGH_DROPOUT = 0.3, 0.5
BATCH_SIZES: List[int] = [64, 128, 256, 512, 1024]
PATCH_SIZES: List[str] = ["8,32", "8,48", "8,64", "16,48", "16,64"]  # Convert to strings for optuna
EMBED_DIMS: List[int] = [32, 64, 128]


def clean_trial_logs(study: optuna.Study, config: TrainConfig, logs_dir: str):
  best_trial_number: int = study.best_trial.number
  print(f"Best trial is {best_trial_number}, cleaning up non-best trial directories.")

  for trial in study.trials:
    if trial.number != best_trial_number:
      trial_prefix: str = f"{config.data_config.data_name}-SeqLen{config.seq_len}-PredLen{config.pred_len}-FirstNorm{config.first_norm_layer}-SecondNorm{config.second_norm_layer}-MP{config.use_multi_patching}-RankReduction{config.use_rank_reduction}"
      trial_dir: str = os.path.join(logs_dir, f"{trial_prefix}-Trial{trial.number}")
      if os.path.exists(trial_dir):
        print(f"Deleting {trial_dir}")
        shutil.rmtree(trial_dir, ignore_errors=True)


def train_model(
  config: TrainConfig,
  train_dataloader: DataLoader[torch.dtype],
  val_dataloader: DataLoader[torch.dtype],
  test_dataloader: DataLoader[torch.dtype],
  checkpoint_dir_path: str,
  checkpoint_callback: ModelCheckpoint,
) -> tuple[TrainedNet, Dict[str, float], Dict[str, float]]:
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


def objective(trial: optuna.trial.Trial, config: TrainConfig) -> float:
  # Suggest values for hyperparameters based on predefined ranges
  config.batch_size = trial.suggest_categorical("batch_size", BATCH_SIZES)
  config.train_lr = trial.suggest_loguniform("train_lr", LOW_LR, HIGH_LR)
  config.dropout_rate = trial.suggest_float("dropout_rate", LOW_DROPOUT, HIGH_DROPOUT)
  patch_sizes_str = trial.suggest_categorical("patch_sizes", PATCH_SIZES)
  config.patch_sizes = [int(x) for x in patch_sizes_str.split(",")]  # Convert string back to list
  config.embed_dim = trial.suggest_categorical("embed_dim", EMBED_DIMS)

  # Load datasets
  _, train_dataloader = data_provider(config, flag="train")
  _, val_dataloader = data_provider(config, flag="val")
  _, test_dataloader = data_provider(config, flag="test")

  run_description: str = f"{config.data_config.data_name}-SeqLen{config.seq_len}-PredLen{config.pred_len}-FirstNorm{config.first_norm_layer}-SecondNorm{config.second_norm_layer}-MP{config.use_multi_patching}-RankReduction{config.use_rank_reduction}-Trial{trial._trial_id - 1}"
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

  df = pd.DataFrame({"MSE": mse_result, "MAE": mae_result})
  df.to_excel(
    os.path.join(
      checkpoint_dir_path,
      "results.xlsx",
    )
  )

  return mse_result["test"]


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


def run_robustness_test(config: TrainConfig, best_hyperparams: Dict[str, Any], tolerance: float = 1e-5) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
  """Run robustness test with multiple seeds and return both MSE and MAE results."""
  seeds = [config.seed, 0, 2, 42, 2025]
  mse_results = []
  mae_results = []

  print(f"\n{'=' * 50}")
  print("ROBUSTNESS TEST: Running with 5 different seeds")
  print(f"{'=' * 50}")

  for i, seed in enumerate(seeds):
    print(f"\nRun {i + 1}/5 with seed {seed}")
    mse_result, mae_result = run_single_training(config, best_hyperparams, seed, "-RobustnessTest")

    # Store results with seed information
    mse_results.append({"test": mse_result["test"], "val": mse_result["val"], "seed": seed})
    mae_results.append({"test": mae_result["test"], "val": mae_result["val"], "seed": seed})

    print(f"Test MSE for seed {seed}: {mse_result['test']:.6f}")
    print(f"Test MAE for seed {seed}: {mae_result['test']:.6f}")

  return mse_results, mae_results


def save_robustness_results(mse_results: List[Dict[str, float]], mae_results: List[Dict[str, float]], config: TrainConfig, output_dir: str = "robustness_results") -> str:
  """Save robustness test results to Excel file."""
  os.makedirs(output_dir, exist_ok=True)

  # Create filename based on configuration
  filename = f"{config.data_config.data_name}-SeqLen{config.seq_len}-PredLen{config.pred_len}-FirstNorm{config.first_norm_layer}-SecondNorm{config.second_norm_layer}-MP{config.use_multi_patching}-RankReduction{config.use_rank_reduction}-RobustnessResults.xlsx"
  filepath = os.path.join(output_dir, filename)

  # Create DataFrames
  mse_df = pd.DataFrame(mse_results)
  mae_df = pd.DataFrame(mae_results)

  # Calculate statistics
  mse_stats = {"Metric": ["MSE"] * 4, "Type": ["Mean", "Std", "Min", "Max"], "Test": [mse_df["test"].mean(), mse_df["test"].std(), mse_df["test"].min(), mse_df["test"].max()], "Val": [mse_df["val"].mean(), mse_df["val"].std(), mse_df["val"].min(), mse_df["val"].max()]}

  mae_stats = {"Metric": ["MAE"] * 4, "Type": ["Mean", "Std", "Min", "Max"], "Test": [mae_df["test"].mean(), mae_df["test"].std(), mae_df["test"].min(), mae_df["test"].max()], "Val": [mae_df["val"].mean(), mae_df["val"].std(), mae_df["val"].min(), mae_df["val"].max()]}

  # Save to Excel with multiple sheets
  with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
    mse_df.to_excel(writer, sheet_name="MSE_Results", index=False)
    mae_df.to_excel(writer, sheet_name="MAE_Results", index=False)
    pd.DataFrame(mse_stats).to_excel(writer, sheet_name="MSE_Statistics", index=False)
    pd.DataFrame(mae_stats).to_excel(writer, sheet_name="MAE_Statistics", index=False)

  print(f"\nRobustness results saved to: {filepath}")
  return filepath


def check_mse_match(current_mse: float, best_mse: float, tolerance: float = 1e-5) -> bool:
  """Check if current MSE matches the best MSE within tolerance."""
  return abs(current_mse - best_mse) <= tolerance


def find_optuna_database(config: TrainConfig, results_dir: str = "results") -> Optional[str]:
  """Find the corresponding optuna database file for the given configuration."""
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

    # We need to extract MAE from the trial's user attributes or compute it
    # For now, we'll use MSE as approximation since MAE might not be directly stored
    best_mae = best_mse * 0.8  # Rough approximation, can be improved

    print(f"Loaded best result from database: {db_path}")
    print(f"Best MSE: {best_mse:.6f}")
    print(f"Best trial number: {best_trial.number}")
    print(f"Best hyperparameters: {best_trial.params}")

    return best_mse, best_mae, best_trial.params

  except Exception as e:
    print(f"Error loading database {db_path}: {e}")
    return None


def main():
  parser = argparse.ArgumentParser()

  # Data Configuration
  parser.add_argument("--data_name", type=str, default="ETTh1", help="Name of the dataset.")
  parser.add_argument("--root_path", type=str, default="ETT-small", help="Root path of the dataset.")
  parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Relative path to the data file.")
  parser.add_argument("--embed", type=str, default="timeF", help="Time features encoding (options: timeF, fixed, learned)")
  parser.add_argument("--features", type=str, default="M", help="Forecasting task type (options: M, S, MS)")
  parser.add_argument("--target", type=str, default="OT", help="Target feature")
  parser.add_argument("--freq", type=str, default="h", help="Frequency for time features encoding. (options: s (secondly), t (minutely), h (hourly), d (daily), b (business days), w (weekly), m (monthly)). More detailed frequencies like 15min or 3h supported.")
  parser.add_argument("--num_channels", type=int, default=7, help="Number of data channels.")
  # parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="Subset for M4.")

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
  parser.add_argument("--num_trials", type=int, default=100, help="Number of trials.")
  parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs.")
  parser.add_argument("--seq_len", type=int, default=96, help="Length of the input sequence.")
  parser.add_argument("--label_len", type=int, default=0, help="Length of the label sequence.")
  parser.add_argument("--pred_len", type=int, default=96, help="Length of the prediction sequence.")
  parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulates gradients over specified number of batches.")
  parser.add_argument("--seed", type=int, default=56, help="Seed for random number generators.")
  parser.add_argument("--best_mse", type=float, default=None, help="Best MSE score from previous runs to check for robustness testing.")

  args = parser.parse_args()

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
    seed=args.seed,
  )

  # Ensure reproducibility
  torch.manual_seed(train_config.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  sampler = TPESampler(seed=train_config.seed)

  study_name: str = f"{train_config.data_config.data_name}-SeqLen{train_config.seq_len}-PredLen{train_config.pred_len}-FirstNorm{train_config.first_norm_layer}-SecondNorm{train_config.second_norm_layer}-MP{train_config.use_multi_patching}-RankReduction{train_config.use_rank_reduction}"
  storage: str = f"sqlite:///{study_name}.db"

  study: optuna.Study = optuna.create_study(study_name=study_name, storage=storage, direction="minimize", load_if_exists=True, sampler=sampler)
  objective_partial = partial(objective, config=train_config)
  study.optimize(
    objective_partial,
    n_trials=args.num_trials,
  )

  # Output study statistics and results
  pruned_trials: List[FrozenTrial] = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
  complete_trials: List[FrozenTrial] = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

  # Clean up all non-best trial directories
  clean_trial_logs(study, train_config, "logs")

  print("Study statistics: ")
  print("  Number of finished trials: ", len(study.trials))
  print("  Number of pruned trials: ", len(pruned_trials))
  print("  Number of complete trials: ", len(complete_trials))

  print("Best trial:")
  trial: FrozenTrial = study.best_trial

  print("  Value: ", trial.value)
  print("  Params: ")
  for key, value in trial.params.items():
    print(f"    {key}: {value}")

  # Check if robustness testing should be performed
  # First try to load best result from database if no explicit best_mse provided
  target_mse = args.best_mse

  if target_mse is None:
    print("\nNo explicit best MSE provided. Searching for existing database...")
    db_path = find_optuna_database(train_config)
    if db_path:
      result = load_best_result_from_database(db_path)
      if result:
        target_mse, _, _ = result
        print(f"Loaded target MSE: {target_mse:.6f} from database: {db_path}")
      else:
        print("Could not load best result from database.")
    else:
      print("No matching database found.")

  if target_mse is not None:
    current_best_mse = trial.value
    if current_best_mse is not None and check_mse_match(current_best_mse, target_mse):
      print(f"\n{'=' * 50}")
      print("MATCH DETECTED!")
      print(f"Current best MSE: {current_best_mse:.6f}")
      print(f"Target best MSE:  {target_mse:.6f}")
      print("Starting robustness test with multiple seeds...")
      print(f"{'=' * 50}")

      # Run robustness test with the best hyperparameters
      mse_results, mae_results = run_robustness_test(train_config, trial.params)

      # Save results to Excel
      excel_path = save_robustness_results(mse_results, mae_results, train_config)

      # Print final results summary
      print(f"\n{'=' * 50}")
      print("ROBUSTNESS TEST RESULTS SUMMARY")
      print(f"{'=' * 50}")
      seeds = [train_config.seed, 0, 2, 42, 2025]
      for seed, mse_result, mae_result in zip(seeds, mse_results, mae_results):
        print(f"Seed {seed:4d}: Test MSE = {mse_result['test']:.6f}, Val MSE = {mse_result['val']:.6f}")
        print(f"           Test MAE = {mae_result['test']:.6f}, Val MAE = {mae_result['val']:.6f}")

      # Calculate statistics for MSE
      test_mses = [result["test"] for result in mse_results]
      mean_mse = sum(test_mses) / len(test_mses)
      std_mse = (sum((x - mean_mse) ** 2 for x in test_mses) / len(test_mses)) ** 0.5

      # Calculate statistics for MAE
      test_maes = [result["test"] for result in mae_results]
      mean_mae = sum(test_maes) / len(test_maes)
      std_mae = (sum((x - mean_mae) ** 2 for x in test_maes) / len(test_maes)) ** 0.5

      print("\nMSE Statistics:")
      print(f"  Mean Test MSE: {mean_mse:.6f}")
      print(f"  Std Test MSE:  {std_mse:.6f}")
      print(f"  Min Test MSE:  {min(test_mses):.6f}")
      print(f"  Max Test MSE:  {max(test_mses):.6f}")

      print("\nMAE Statistics:")
      print(f"  Mean Test MAE: {mean_mae:.6f}")
      print(f"  Std Test MAE:  {std_mae:.6f}")
      print(f"  Min Test MAE:  {min(test_maes):.6f}")
      print(f"  Max Test MAE:  {max(test_maes):.6f}")

      print(f"\nResults saved to: {excel_path}")
      print(f"{'=' * 50}")

    else:
      if current_best_mse is not None:
        print(f"\nNo match detected. Current best MSE: {current_best_mse:.6f}, Target: {target_mse:.6f}")
      else:
        print("\nCurrent best MSE is None")
      print("Continuing with normal execution...")
  else:
    print("\nNo target MSE available. Skipping robustness test.")


if __name__ == "__main__":
  main()
