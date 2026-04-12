import argparse
import json
import os
import shutil
from dataclasses import asdict
from functools import partial

import lightning as L
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from optuna.samplers import TPESampler
from optuna.trial._frozen import FrozenTrial
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
BATCH_SIZES: list[int] = [64]
PATCH_SIZES: list[list[int]] = [[8, 48]]
EMBED_DIMS: list[int] = [128]


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
) -> tuple[TrainedNet, dict[str, float], dict[str, float]]:
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
  model: TrainedNet = TrainedNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, weights_only=False)

  # Test the model on validation and test datasets
  val_result = trainer.test(model, dataloaders=val_dataloader, verbose=False)
  test_result = trainer.test(model, dataloaders=test_dataloader, verbose=False)

  mse_result: dict[str, float] = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
  mae_result: dict[str, float] = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

  return model, mse_result, mae_result


def objective(trial: optuna.trial.Trial, config: TrainConfig) -> float:
  # Suggest values for hyperparameters based on predefined ranges
  config.batch_size = trial.suggest_categorical("batch_size", BATCH_SIZES)
  config.train_lr = trial.suggest_loguniform("train_lr", LOW_LR, HIGH_LR)
  config.dropout_rate = trial.suggest_float("dropout_rate", LOW_DROPOUT, HIGH_DROPOUT)
  config.patch_sizes = trial.suggest_categorical("patch_sizes", PATCH_SIZES)
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
  pruned_trials: list[FrozenTrial] = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
  complete_trials: list[FrozenTrial] = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

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


if __name__ == "__main__":
  main()
