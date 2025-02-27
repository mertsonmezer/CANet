import argparse
import datetime
import os
from typing import Any, Dict

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from torch.utils.data import DataLoader

from config.canet_config import CANetDataConfig, CANetTrainConfig
from data.data_factory import data_provider
from canet.canet import TrainedCANet

# Set device to GPU if available, else CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
  config: CANetTrainConfig,
  train_dataloader: DataLoader[torch.dtype],
  val_dataloader: DataLoader[torch.dtype],
  test_dataloader: DataLoader[torch.dtype],
  checkpoint_dir_path: str,
  checkpoint_callback: ModelCheckpoint,
) -> tuple[Any, Dict[str, float], Dict[str, float]]:
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
  model = TrainedCANet(config)

  trainer.fit(model, train_dataloader, val_dataloader)

  # Load the best checkpoint after training
  model: TrainedCANet = TrainedCANet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

  # Test the model on validation and test datasets
  val_result = trainer.test(model, dataloaders=val_dataloader, verbose=False)
  test_result = trainer.test(model, dataloaders=test_dataloader, verbose=False)

  mse_result: Dict[str, float] = {"test": test_result[0]["test_mse"], "val": val_result[0]["test_mse"]}
  mae_result: Dict[str, float] = {"test": test_result[0]["test_mae"], "val": val_result[0]["test_mae"]}

  return model, mse_result, mae_result


def main():
  parser = argparse.ArgumentParser()

  # Data Configuration
  parser.add_argument("--data_name", type=str, default="ETTh1", help="Name of the dataset.")
  parser.add_argument("--root_path", type=str, default="ETT-small", help="Root path of the dataset.")
  parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="Relative path to the data file.")
  parser.add_argument(
    "--embed", type=str, default="timeF", help="Time features encoding (options: timeF, fixed, learned)"
  )
  parser.add_argument("--features", type=str, default="M", help="Forecasting task type (options: M, S, MS)")
  parser.add_argument("--target", type=str, default="OT", help="Target feature")
  parser.add_argument(
    "--freq",
    type=str,
    default="h",
    help="Frequency for time features encoding. (options: s (secondly), t (minutely), h (hourly), d (daily), b (business days), w (weekly), m (monthly)). More detailed frequencies like 15min or 3h supported.",
  )
  parser.add_argument("--num_channels", type=int, default=7, help="Number of data channels.")
  # parser.add_argument("--seasonal_patterns", type=str, default="Monthly", help="Subset for M4.")

  # Model Configuration
  parser.add_argument("--emb_dim", type=int, default=64, help="Dimensionality of the model.")
  parser.add_argument("--depth", type=int, default=3, help="Number of layers in the model.")
  parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
  parser.add_argument(
    "--first_norm_layer", type=str, default="layer", help="Type of normalization technique used before noise reduction."
  )
  parser.add_argument(
    "--second_norm_layer", type=str, default="adain", help="Type of normalization technique used after noise reduction."
  )
  parser.add_argument(
    "--patch_size", type=int, default=16, help="Size of patches (used if multi-patching is disabled)."
  )
  parser.add_argument(
    "--patch_sizes",
    type=int,
    nargs="+",
    default=[8, 64],
    help="Sizes of multi-resolution patches (space-separated integers).",
  )
  parser.add_argument(
    "--blend_ratio", type=float, default=0.1, help="Indicates the ratio at which the style input will be combined."
  )
  parser.add_argument(
    "--style_blend_gate_affine", type="store_true", default=True, help="Whether to apply extra shifting and scaling."
  )
  parser.add_argument(
    "--use_multi_patching", type="store_true", default=True, help="Whether to use multi-patching technique."
  )
  parser.add_argument(
    "--use_rank_reduction", type="store_true", default=True, help="Whether to use rank reduction technique."
  )

  # Training Configuration
  parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs.")
  parser.add_argument("--batch_size", type=int, default=256, help="Training batch size.")
  parser.add_argument("--seq_len", type=int, default=96, help="Length of the input sequence.")
  parser.add_argument("--label_len", type=int, default=0, help="Length of the label sequence.")
  parser.add_argument("--pred_len", type=int, default=96, help="Length of the prediction sequence.")
  parser.add_argument(
    "--accumulate_grad_batches", type=int, default=1, help="Accumulates gradients over specified number of batches."
  )
  parser.add_argument("--seed", type=int, default=56, help="Seed for random number generators.")

  args = parser.parse_args()

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
    data_config=data_config,
    patch_size=args.patch_size,
    patch_sizes=args.patch_sizes,
    embed_dim=args.emb_dim,
    depth=args.depth,
    seq_len=args.seq_len,
    label_len=args.label_len,
    dropout_rate=args.dropout,
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

  # Generate run ID for logging and checkpointing
  run_description: str = (
    f"{data_config.data_name}-"
    f"Embed{train_config.embed_dim}-"
    f"Depth{train_config.depth}-"
    f"{'MP' + '-'.join(map(str, train_config.patch_sizes)) if train_config.use_multi_patching else 'PS' + str(train_config.patch_size)}-"
    f"PredLen{train_config.pred_len}-"
    f"Batch{train_config.batch_size}-"
    f"FirstNorm{train_config.first_norm_layer}-"
    f"SecondNorm{train_config.second_norm_layer}-"
    f"RankReduction{train_config.use_rank_reduction}"
    f"{datetime.datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}"
  )
  print(f"========== {run_description} ===========")

  checkpoint_dir_path: str = f"logs/{run_description}"
  checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir_path, save_top_k=1, filename="train-{epoch}", monitor="val_mse", mode="min"
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
  _, mse_result, mae_result = train_model(
    train_config,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    checkpoint_dir_path,
    checkpoint_callback,
  )
  print("MSE results", mse_result)
  print("MAE  results", mae_result)

  # Save results into an Excel sheet
  df = pd.DataFrame({"MSE": mse_result, "MAE": mae_result})
  df.to_excel(
    os.path.join(
      checkpoint_dir_path,
      "results.xlsx",
    )
  )


if __name__ == "__main__":
  main()
