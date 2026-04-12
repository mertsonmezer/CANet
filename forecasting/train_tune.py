"""Optuna-based hyperparameter tuning script for CANet."""

import argparse
import ast
import json
import logging
import os
import shutil
from dataclasses import asdict
from functools import partial
from typing import Dict, List

import lightning as L
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import (
  LearningRateMonitor,
  ModelCheckpoint,
  TQDMProgressBar,
)
from optuna.samplers import TPESampler
from optuna.trial._frozen import FrozenTrial
from torch.utils.data import DataLoader

from canet.canet import TrainedCANet
from config.canet_config import CANetDataConfig, CANetTrainConfig
from data.data_factory import data_provider
from utils.utils import DEVICE

logger = logging.getLogger(__name__)


def clean_trial_logs(study: optuna.Study, config: CANetTrainConfig, logs_dir: str):
  """
  Cleans up non-best trial directories from the log directory.

  This function removes the directories of trials that did not yield the best results.

  Args:
    study: The Optuna study containing all trials.
    config: Configuration object containing training settings.
    logs_dir: The directory where trial logs are stored.
  """

  best_trial_number: int = study.best_trial.number
  logger.info("Best trial is %d, cleaning up non-best trial directories.", best_trial_number)

  for trial in study.trials:
    if trial.number != best_trial_number:
      trial_prefix: str = (
        f"{config.data_config.data_name}-SeqLen{config.seq_len}-"
        f"PredLen{config.pred_len}-FirstNorm{config.first_norm_layer}-"
        f"SecondNorm{config.second_norm_layer}-RankReduction{config.use_rank_reduction}"
      )
      trial_dir: str = os.path.join(logs_dir, f"{trial_prefix}-Trial{trial.number}")

      # Delete the trial directory if it exists
      if os.path.exists(trial_dir):
        logger.info("Deleting %s", trial_dir)
        shutil.rmtree(trial_dir, ignore_errors=True)


def train_model(
  config: CANetTrainConfig,
  train_dataloader: DataLoader,
  val_dataloader: DataLoader,
  test_dataloader: DataLoader,
  checkpoint_dir: str,
  checkpoint_callback: ModelCheckpoint,
) -> tuple[TrainedCANet, Dict[str, float], Dict[str, float]]:
  """
  Trains the CANet model and evaluates on validation and test sets.

  Args:
    config: Training configuration.
    train_dataloader: DataLoader for the training dataset.
    val_dataloader: DataLoader for the validation dataset.
    test_dataloader: DataLoader for the test dataset.
    checkpoint_dir: Directory to save model checkpoints.
    checkpoint_callback: Callback for saving the best model checkpoint.

  Returns:
    A tuple containing the trained model, MSE results, and MAE results.
  """
  trainer = L.Trainer(
    default_root_dir=checkpoint_dir,
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
  model = TrainedCANet(config)

  trainer.fit(model, train_dataloader, val_dataloader)

  # Load the best checkpoint after training
  model: TrainedCANet = TrainedCANet.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path, weights_only=False
  )

  # Test the model on validation and test datasets
  val_result = trainer.test(model, dataloaders=val_dataloader, verbose=False)
  test_result = trainer.test(model, dataloaders=test_dataloader, verbose=False)

  mse_result: Dict[str, float] = {
    "test": test_result[0]["test_mse"],
    "val": val_result[0]["test_mse"],
  }
  mae_result: Dict[str, float] = {
    "test": test_result[0]["test_mae"],
    "val": val_result[0]["test_mae"],
  }

  return model, mse_result, mae_result


def objective(
  trial: optuna.trial.Trial, args: argparse.Namespace, config: CANetTrainConfig
) -> float:
  """
  Objective function for Optuna hyperparameter tuning.

  This function is used by Optuna to suggest hyperparameters and train the model. It evaluates the model
  using the Mean Squared Error (MSE) on the validation set, which is returned as the objective value.

  Args:
    trial: An Optuna trial instance used for suggesting hyperparameter values.
    args: ...
    config: A configuration object containing model and training parameters.

  Returns:
    float: The MSE result on the test set, which is used as the objective value.
  """
  # Suggest values for hyperparameters based on predefined ranges
  config.batch_size = trial.suggest_categorical("batch_size", args.batch_sizes)
  config.train_lr = trial.suggest_float("train_lr", args.low_lr, args.high_lr, log=True)
  config.dropout_rate = trial.suggest_float("dropout_rate", args.low_dropout, args.high_dropout)
  config.patch_sizes = trial.suggest_categorical("patch_sizes", args.patch_sizes)
  config.embed_dim = trial.suggest_categorical("embed_dim", args.embed_dims)

  # Load datasets
  _, train_dataloader = data_provider(config, flag="train")
  _, val_dataloader = data_provider(config, flag="val")
  _, test_dataloader = data_provider(config, flag="test")

  run_description: str = (
    f"{config.data_config.data_name}-SeqLen{config.seq_len}-PredLen{config.pred_len}-"
    f"FirstNorm{config.first_norm_layer}-SecondNorm{config.second_norm_layer}-"
    f"RankReduction{config.use_rank_reduction}-Trial{trial._trial_id - 1}"
  )
  logger.info("========== %s ===========", run_description)

  checkpoint_dir_path: str = f"logs/{run_description}"
  checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir_path,
    save_top_k=1,
    filename="train-{epoch}",
    monitor="val_mse",
    mode="min",
  )

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


def main():
  parser = argparse.ArgumentParser()

  # Data Configuration
  parser.add_argument("--data_name", type=str, default="ETTh1", help="Name of the dataset.")
  parser.add_argument(
    "--root_path", type=str, default="ETT-small", help="Root path of the dataset."
  )
  parser.add_argument(
    "--data_path",
    type=str,
    default="ETTh1.csv",
    help="Relative path to the data file.",
  )
  parser.add_argument(
    "--embed",
    type=str,
    default="timeF",
    choices=["timeF", "fixed", "learned"],
    help="Time feature encoding method.",
  )
  parser.add_argument(
    "--features",
    type=str,
    default="M",
    choices=["M", "S", "MS"],
    help="Forecasting task type.",
  )
  parser.add_argument("--target", type=str, default="OT", help="Target feature.")
  parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="Frequency for time features encoding (e.g., s (secondly), t (minutely), h (hourly), d (daily), b (business days), w (weekly), m (monthly)).",
  )
  parser.add_argument("--num_channels", type=int, default=7, help="Number of data channels.")
  # parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="Subset for M4.")

  # Model Configuration
  parser.add_argument(
    "--embed_dims",
    type=int,
    nargs="+",
    default=[32, 64, 128],
    help="List of embedding dimensions.",
  )
  parser.add_argument("--depth", type=int, default=2, help="Number of layers in the model.")
  parser.add_argument("--low_dropout", type=float, default=0.3, help="Lower bound of dropout rate.")
  parser.add_argument(
    "--high_dropout", type=float, default=0.5, help="Upper bound of dropout rate."
  )
  parser.add_argument(
    "--first_norm_layer",
    type=str,
    default="layer",
    choices=["nsan", "layer", "batch", "instance", "dain"],
    help="Pre-noise reduction normalization type.",
  )
  parser.add_argument(
    "--second_norm_layer",
    type=str,
    default="nsan",
    choices=["nsan", "layer", "batch", "instance", "dain"],
    help="Post-noise reduction normalization type.",
  )
  parser.add_argument(
    "--patch_sizes",
    type=str,
    default="[[8, 64]]",
    help="List of patch sizes in the form of a list of lists (e.g. [[8, 64], [16, 128]]).",
  )
  parser.add_argument("--blend_ratio", type=float, default=0.1, help="Ratio for style blending.")
  parser.add_argument(
    "--style_blend_gate_affine",
    action="store_true",
    help="Enable extra shifting and scaling.",
  )
  parser.add_argument(
    "--use_rank_reduction",
    action="store_true",
    help="Enable rank reduction technique.",
  )

  # Training Configuration
  parser.add_argument("--num_trials", type=int, default=100, help="Number of trials.")
  parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
  parser.add_argument(
    "--batch_sizes",
    type=int,
    nargs="+",
    default=[1024],
    help="List of batch sizes.",
  )
  parser.add_argument("--seq_len", type=int, default=96, help="Input sequence length.")
  parser.add_argument("--label_len", type=int, default=0, help="Label sequence length.")
  parser.add_argument("--pred_len", type=int, default=96, help="Prediction sequence length.")
  parser.add_argument("--low_lr", type=float, default=1e-4, help="Lower bound of learning rate.")
  parser.add_argument("--high_lr", type=float, default=1e-2, help="Upper bound of learning rate.")
  parser.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    help="Gradient accumulation steps.",
  )
  parser.add_argument("--seed", type=int, default=56, help="Random seed.")

  args: argparse.Namespace = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(name)s — %(levelname)s — %(message)s")

  try:
    args.patch_sizes = ast.literal_eval(args.patch_sizes)
  except (ValueError, SyntaxError) as exc:
    raise SystemExit(
      f"Invalid --patch_sizes value {args.patch_sizes!r}. "
      "Expected a list of lists, e.g. '[[8, 64], [16, 128]]'."
    ) from exc

  data_config = CANetDataConfig(
    data_name=args.data_name,
    root_dir_path=args.root_path,
    data_path=args.data_path,
    embed=args.embed,
    features=args.features,
    target=args.target,
    freq=args.freq,
    num_channels=args.num_channels,
  )

  train_config = CANetTrainConfig(
    num_epochs=args.num_epochs,
    data_config=data_config,
    seq_len=args.seq_len,
    label_len=args.label_len,
    pred_len=args.pred_len,
    first_norm_layer=args.first_norm_layer,
    second_norm_layer=args.second_norm_layer,
    blend_ratio=args.blend_ratio,
    style_blend_gate_affine=args.style_blend_gate_affine,
    use_rank_reduction=args.use_rank_reduction,
    accumulate_grad_batches=args.accumulate_grad_batches,
    seed=args.seed,
  )

  # Ensure reproducibility
  torch.manual_seed(train_config.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  sampler = TPESampler(seed=train_config.seed)

  study_name: str = f"{train_config.data_config.data_name}-SeqLen{train_config.seq_len}-PredLen{train_config.pred_len}-FirstNorm{train_config.first_norm_layer}-SecondNorm{train_config.second_norm_layer}-RankReduction{train_config.use_rank_reduction}"
  storage: str = f"sqlite:///{study_name}.db"

  study: optuna.Study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction="minimize",
    load_if_exists=True,
    sampler=sampler,
  )
  objective_partial = partial(objective, args=args, config=train_config)
  study.optimize(
    objective_partial,
    n_trials=args.num_trials,
  )

  # Output study statistics and results
  pruned_trials: List[FrozenTrial] = [
    t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
  ]
  complete_trials: List[FrozenTrial] = [
    t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
  ]

  # Clean up all non-best trial directories
  clean_trial_logs(study, train_config, "logs")

  logger.info("Study statistics:")
  logger.info("  Number of finished trials: %d", len(study.trials))
  logger.info("  Number of pruned trials: %d", len(pruned_trials))
  logger.info("  Number of complete trials: %d", len(complete_trials))

  trial: FrozenTrial = study.best_trial
  logger.info("Best trial:")
  logger.info("  Value: %s", trial.value)
  logger.info("  Params:")
  for key, value in trial.params.items():
    logger.info("    %s: %s", key, value)


if __name__ == "__main__":
  main()
