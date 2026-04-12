"""Single-run training script for CANet."""

import argparse
import datetime
import json
import logging
import os
from dataclasses import asdict
from typing import Any

import lightning as L
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from canet.canet import TrainedCANet
from config.canet_config import CANetDataConfig, CANetTrainConfig
from data.data_factory import data_provider
from utils.paths import TrainPaths, TunePaths, build_study_params

logger = logging.getLogger(__name__)


def _load_best_params(dataset: str, study_params: str) -> dict:
  """Load the best trial's tuned hyperparameters from a study DB.

  The study is identified by the dataset name and study_params string,
  which together determine the path to the Optuna SQLite database via
  TunePaths. The study name stored inside the DB is reconstructed as
  ``{dataset}-{study_params}``, matching the convention in train_tune.py.

  Args:
    dataset: Dataset identifier, e.g. ``"ETTh1"``.
    study_params: Output of :func:`build_study_params` for this configuration.

  Returns:
    A dict of the best trial's suggested hyperparameters, e.g.::

      {
        "train_lr": 0.0012,
        "batch_size": 128,
        "dropout_rate": 0.35,
        "embed_dim": 64,
        "patch_sizes": [8, 64],
      }

  Raises:
    FileNotFoundError: If the study DB does not exist at the expected path.
    RuntimeError: If the study has no completed trials.
  """
  paths = TunePaths(dataset=dataset, study_params=study_params)
  if not paths.db_path.exists():
    raise FileNotFoundError(
      f"No study DB found at {paths.db_path}. Run train_tune.py for this configuration first."
    )
  study_name = f"{dataset}-{study_params}"
  study = optuna.load_study(study_name=study_name, storage=paths.db_url)
  if not study.best_trial:
    raise RuntimeError(f"Study '{study_name}' has no completed trials.")
  return study.best_trial.params


def train_model(
  config: CANetTrainConfig,
  train_dataloader: DataLoader,
  val_dataloader: DataLoader,
  test_dataloader: DataLoader,
  checkpoint_dir: str,
  checkpoint_callback: ModelCheckpoint,
) -> tuple[Any, dict[str, float], dict[str, float]]:
  """Trains the CANet model and evaluates on validation and test sets.

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

  mse_result: dict[str, float] = {
    "test": test_result[0]["test_mse"],
    "val": val_result[0]["test_mse"],
  }
  mae_result: dict[str, float] = {
    "test": test_result[0]["test_mae"],
    "val": val_result[0]["test_mae"],
  }

  return model, mse_result, mae_result


def main() -> None:
  """Entry point — parse arguments and run the main workflow."""
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
    help="Frequency for time features encoding "
    "(e.g., s (secondly), t (minutely), h (hourly), d (daily), "
    "b (business days), w (weekly), m (monthly)).",
  )
  parser.add_argument("--num_channels", type=int, default=7, help="Number of data channels.")
  # parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="Subset for M4.")

  # Model Configuration
  parser.add_argument(
    "--embed_dim",
    type=int,
    default=64,
    help="Embedding dimensionality of the model.",
  )
  parser.add_argument("--depth", type=int, default=2, help="Number of layers in the model.")
  parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
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
    type=int,
    nargs="+",
    default=[8, 64],
    help="Sizes of multi-resolution patches.",
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
  parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs.")
  parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
  parser.add_argument("--seq_len", type=int, default=96, help="Input sequence length.")
  parser.add_argument("--label_len", type=int, default=0, help="Label sequence length.")
  parser.add_argument("--pred_len", type=int, default=96, help="Prediction sequence length.")
  parser.add_argument(
    "--accumulate_grad_batches",
    type=int,
    default=1,
    help="Gradient accumulation steps.",
  )
  parser.add_argument("--train_lr", type=float, default=1e-4, help="Learning rate.")
  parser.add_argument("--seed", type=int, default=56, help="Random seed.")
  parser.add_argument(
    "--from_study",
    action="store_true",
    help=(
      "Load the best hyperparameters from the Optuna study DB for this "
      "configuration and override --train_lr, --batch_size, --dropout, "
      "--embed_dim, and --patch_sizes. The study is identified automatically "
      "from --data_name, --seq_len, --pred_len, --first_norm_layer, "
      "--second_norm_layer, and --use_rank_reduction."
    ),
  )

  args: argparse.Namespace = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
  )

  if args.from_study:
    study_params = build_study_params(
      args.seq_len,
      args.pred_len,
      args.first_norm_layer,
      args.second_norm_layer,
      args.use_rank_reduction,
    )
    best_params = _load_best_params(args.data_name, study_params)
    logger.info("Loaded best hyperparameters from study: %s", best_params)
    args.train_lr = best_params["train_lr"]
    args.batch_size = best_params["batch_size"]
    args.dropout = best_params["dropout_rate"]
    args.embed_dim = best_params["embed_dim"]
    args.patch_sizes = best_params["patch_sizes"]

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
    batch_size=args.batch_size,
    train_lr=args.train_lr,
    data_config=data_config,
    patch_sizes=args.patch_sizes,
    embed_dim=args.embed_dim,
    depth=args.depth,
    seq_len=args.seq_len,
    label_len=args.label_len,
    dropout_rate=args.dropout,
    pred_len=args.pred_len,
    first_norm_layer=args.first_norm_layer,
    second_norm_layer=args.second_norm_layer,
    blend_ratio=args.blend_ratio,
    style_blend_gate_affine=args.style_blend_gate_affine,
    use_rank_reduction=args.use_rank_reduction,
    accumulate_grad_batches=args.accumulate_grad_batches,
    seed=args.seed,
  )

  # Generate run ID for logging and checkpointing.
  # The dataset name is encoded in the parent path, so it is omitted here
  # to keep the leaf directory name concise.
  run_description: str = (
    f"Embed{train_config.embed_dim}-"
    f"Depth{train_config.depth}-"
    f"{'MP' + '-'.join(map(str, train_config.patch_sizes))}-"
    f"PredLen{train_config.pred_len}-"
    f"Batch{train_config.batch_size}-"
    f"FirstNorm{train_config.first_norm_layer}-"
    f"SecondNorm{train_config.second_norm_layer}-"
    f"RankReduction{train_config.use_rank_reduction}-"
    f"{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
  )
  train_paths = TrainPaths(dataset=data_config.data_name, run_description=run_description)
  train_paths.run_dir.mkdir(parents=True, exist_ok=True)
  logger.info("========== %s ===========", train_paths.run_dir)

  checkpoint_dir_path: str = str(train_paths.run_dir)
  checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir_path,
    save_top_k=1,
    filename="train-{epoch}",
    monitor="val_mse",
    mode="min",
  )

  # Ensure reproducibility
  torch.manual_seed(train_config.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  # Load datasets
  _, train_dataloader = data_provider(train_config, flag="train")
  _, val_dataloader = data_provider(train_config, flag="val")
  _, test_dataloader = data_provider(train_config, flag="test")

  # Execute the training model function
  model, mse_result, mae_result = train_model(
    train_config,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    checkpoint_dir_path,
    checkpoint_callback,
  )
  logger.info("MSE results: %s", mse_result)
  logger.info("MAE results: %s", mae_result)

  # Save results and configuration
  df = pd.DataFrame({"MSE": mse_result, "MAE": mae_result})
  df.to_excel(train_paths.run_dir / "results.xlsx")

  total_params: int = sum(p.numel() for p in model.parameters() if p.requires_grad)
  with open(train_paths.run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump({"total_params": total_params, "config": asdict(train_config)}, f, indent=2)


if __name__ == "__main__":
  main()
  main()
  main()
