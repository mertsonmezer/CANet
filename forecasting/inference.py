import matplotlib

matplotlib.use("TkAgg")

import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.utilities.model_summary import ModelSummary
from canet.canet import TrainedCANet
from data.data_factory import data_provider
import os
import optuna
import glob
from datetime import datetime
import argparse
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError


# Set up command-line arguments
def parse_args():
  parser = argparse.ArgumentParser(description="Find the best model and print its summary.")

  parser.add_argument("--data", type=str, default="ETTh1", help="Dataset name")
  parser.add_argument("--seq_len", type=int, default=96, help="Sequence length")
  parser.add_argument("--pred_len", type=int, default=96, help="Prediction length")
  parser.add_argument("--normalization", type=str, default="adain", help="Normalization method")
  parser.add_argument("--mrp", type=bool, default=True, help="MRP flag")

  return parser.parse_args()


# Main function to get the best model path
def get_best_model_path(data: str, seq_len: int, pred_len: int, normalization: str, mrp: bool) -> str:
  study_name = f"{data}_{str(seq_len)}_{str(pred_len)}_norm{str(normalization)}_mp{str(mrp)}"
  storage = f"sqlite:///{study_name}.db"

  study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)
  best_trial = study.best_trial
  best_trial_id = best_trial._trial_id

  model_folder_path_prefix: str = f"logs/{data}_{str(seq_len)}_{str(pred_len)}_trial{str(best_trial_id - 1)}_norm{str(normalization)}_mp{str(mrp)}"

  matching_folders = glob.glob(f"{model_folder_path_prefix}*")

  valid_folders = []
  for folder in matching_folders:
    try:
      time_str = folder.split("_")[-3:]  # the last 3 parts (H_M_S)
      folder_time = "_".join(time_str)
      datetime.strptime(folder_time, "%H_%M_%S")
      valid_folders.append(folder)
    except ValueError:
      continue

  valid_folders.sort(reverse=True)

  if not valid_folders:
    raise FileNotFoundError("No matching folder found with the required prefix and valid timestamp.")

  best_model_folder = valid_folders[0]
  ckpt_files = glob.glob(os.path.join(best_model_folder, "*.ckpt"))

  if not ckpt_files:
    raise FileNotFoundError(f"No .ckpt file found in {best_model_folder}")

  best_model_path: str = ckpt_files[0]
  return best_model_path


# Function to load the best model
def get_best_model(data: str, seq_len: int, pred_len: int, normalization: str, mrp: bool) -> TrainedCANet:
  best_model_path: str = get_best_model_path(data, seq_len, pred_len, normalization, mrp)
  best_model: TrainedCANet = TrainedCANet.load_from_checkpoint(best_model_path)
  return best_model


# Function to perform inference, measure time, compute MSE and MAE, and plot predictions vs actual target
def perform_inference(model: TrainedCANet, test_dataloader):
  model.eval()

  # Get a batch from the dataloader
  example = next(iter(test_dataloader))
  inputs, targets, _, _ = example

  # Move data to the appropriate device (GPU/CPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)
  inputs = inputs.to(device).float().contiguous()  # Ensure the tensor is contiguous
  targets = targets.to(device).float().contiguous()  # Ensure the tensor is contiguous

  # Measure inference time
  start_time = time.time()
  with torch.no_grad():
    outputs = model(inputs)
  inference_time = time.time() - start_time

  # Ensure outputs are contiguous as well
  outputs = outputs.contiguous()

  # Initialize MSE and MAE metrics
  mse_metric = MeanSquaredError().to(device)
  mae_metric = MeanAbsoluteError().to(device)

  # Compute MSE and MAE for the entire batch
  mse = mse_metric(outputs, targets).item()
  mae = mae_metric(outputs, targets).item()

  print(f"Inference Time: {inference_time:.4f} seconds")
  print(f"Mean Squared Error (MSE): {mse:.4f}")
  print(f"Mean Absolute Error (MAE): {mae:.4f}")

  # Select a single example from the batch (e.g., the 5th)
  last_feature_inputs = inputs[5, :, -1]  # Shape: [sequence_length]
  last_feature_targets = targets[5, :, -1]  # Shape: [sequence_length]
  last_feature_outputs = outputs[5, :, -1]  # Shape: [sequence_length]

  # Concatenate input and target (for continuous plotting)
  concatenated_input_target = torch.cat([last_feature_inputs, last_feature_targets], dim=0).cpu().numpy()

  # Plot the input + actual target as one continuous line
  plt.figure(figsize=(10, 6))

  # Plot the concatenated input and actual target data (continuous line)
  plt.plot(concatenated_input_target, label="Input + Actual Target (Ground Truth)", color="blue", linestyle="-")

  # Plot the predicted values for the target (right side)
  plt.plot(
    range(len(last_feature_inputs), len(last_feature_inputs) + len(last_feature_outputs)),
    last_feature_outputs.cpu().numpy(),
    label="Predicted Target (Right Side)",
    color="red",
    linestyle="-",
  )

  plt.title("Input + Actual Target vs Predicted Target")
  plt.xlabel("Time Steps")
  plt.ylabel("Value")
  plt.legend()
  plt.grid(True)
  plt.show()


# Main execution
if __name__ == "__main__":
  args = parse_args()

  # Pass the command-line parameters to the functions
  model: TrainedCANet = get_best_model(args.data, args.seq_len, args.pred_len, args.normalization, args.mrp)

  # Print model summary
  model_summary = ModelSummary(model, max_depth=-1)
  print(model_summary)

  # Load data
  _, test_dataloader = data_provider(model.config, flag="test")

  # Perform inference, measure time, compute MSE and plot results
  perform_inference(model, test_dataloader)
