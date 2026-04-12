"""Inference and result visualization for trained CANet models."""

import argparse
import logging
import time

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError

from canet.canet import TrainedCANet
from data.data_factory import data_provider
from utils.utils import DEVICE

logger = logging.getLogger(__name__)


def perform_inference(
  model: TrainedCANet, dataloader: DataLoader
) -> tuple[Tensor, Tensor, Tensor, float, float, float]:
  """Performs inference on the test dataset and evaluates performance.

  Args:
    model (TrainedCANet): The trained CANet model.
    dataloader (DataLoader): Dataloader providing input batches.

  Returns:
    Tuple[Tensor, Tensor, Tensor, float, float, float]: Inputs, targets,
      outputs, inference time, MSE, MAE.
  """
  model.eval()

  model.to(DEVICE)

  # Get a batch from the dataloader
  example = next(iter(dataloader))
  inputs, targets, _, _ = example

  # Move inputs and targets to the selected device and ensure they are float and contiguous
  inputs: Tensor = inputs.to(DEVICE).float().contiguous()
  targets: Tensor = targets.to(DEVICE).float().contiguous()

  # Measure inference time
  if DEVICE.type == "cuda":
    # GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(stream=None)
    with torch.no_grad():
      outputs = model(inputs)
    end_event.record(stream=None)

    torch.cuda.synchronize()
    inference_time: float = start_event.elapsed_time(end_event) / 1000  # Convert to seconds

  else:
    # CPU timing
    start_time: float = time.time()
    with torch.no_grad():
      outputs = model(inputs)
    end_time: float = time.time()

    inference_time: float = end_time - start_time

  # Ensure outputs are contiguous as well
  outputs: Tensor = outputs.contiguous()

  # Initialize evaluation metrics
  mse_metric: MeanSquaredError = MeanSquaredError().to(DEVICE)
  mae_metric: MeanAbsoluteError = MeanAbsoluteError().to(DEVICE)

  # Compute MSE and MAE for the entire batch
  mse: float = mse_metric(outputs, targets).item()
  mae: float = mae_metric(outputs, targets).item()

  # Display inference time and metrics
  logger.info("Inference Time: %.4f seconds", inference_time)
  logger.info("Mean Squared Error (MSE): %.4f", mse)
  logger.info("Mean Absolute Error (MAE): %.4f", mae)

  return inputs, targets, outputs, inference_time, mse, mae


def visualize_results(inputs: Tensor, targets: Tensor, outputs: Tensor, example_id: int) -> None:
  """Visualizes the input, actual target, and predicted target for a given example.

  Args:
    inputs (Tensor): Model inputs.
    targets (Tensor): Ground truth targets.
    outputs (Tensor): Model predictions.
    example_id (int): Index of the example to visualize.
  """
  # Extract the specific example
  last_feature_inputs: Tensor = inputs[example_id, :, -1]  # Shape: [sequence_length]
  last_feature_targets: Tensor = targets[example_id, :, -1]  # Shape: [sequence_length]
  last_feature_outputs: Tensor = outputs[example_id, :, -1]  # Shape: [sequence_length]

  # Concatenate input and target (for continuous plotting)
  concatenated_input_target: Tensor = (
    torch.cat([last_feature_inputs, last_feature_targets], dim=0).cpu().numpy()
  )

  # Plot input + actual target as one continuous line and predicted values separately
  plt.figure(figsize=(10, 6))
  plt.plot(
    concatenated_input_target,
    label="Input + Actual Target (Ground Truth)",
    color="blue",
    linestyle="-",
  )
  plt.plot(
    range(
      len(last_feature_inputs),
      len(last_feature_inputs) + len(last_feature_outputs),
    ),
    last_feature_outputs.cpu().numpy(),
    label="Predicted Target (Right Side)",
    color="red",
    linestyle="-",
  )

  plt.title(f"Example {example_id}: Input + Actual Target vs Predicted Target")
  plt.xlabel("Time Steps")
  plt.ylabel("Value")
  plt.legend()
  plt.grid(True)
  plt.show()


def main() -> None:
  """Entry point — parse arguments and run the main workflow."""
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
  )
  parser = argparse.ArgumentParser(description="Inference of CANet model.")
  parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the CANet model checkpoint (.ckpt file).",
  )
  parser.add_argument(
    "--example_id", type=int, default=0, help="Index of the example to visualize."
  )
  parser.add_argument("--visualize", action="store_true", help="Whether to visualize the results.")

  args: argparse.Namespace = parser.parse_args()

  # Load model
  model: TrainedCANet = TrainedCANet.load_from_checkpoint(args.model_path, weights_only=False)
  model.eval()

  # Load test data
  _, test_dataloader = data_provider(model.config, flag="test")

  # Perform inference
  inputs, targets, outputs, _, _, _ = perform_inference(model, test_dataloader)

  # Visualize results for the specified example
  if args.visualize:
    visualize_results(inputs, targets, outputs, args.example_id)


if __name__ == "__main__":
  main()
