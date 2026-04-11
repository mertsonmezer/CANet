import argparse
import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from canet.canet import TrainedCANet
from data.data_factory import data_provider


def calculate_model_size(model: TrainedCANet) -> int:
  """
  Calculates the total number of trainable parameters in a CANet model.

  Args:
    model (TrainedCANet): The model whose parameters are counted.

  Returns:
    int: Total number of parameters in the model.
  """
  total_params: int = sum(p.numel() for p in model.parameters())
  return total_params


def measure_inference_time(
  model: TrainedCANet, dataloader: DataLoader, num_trials: int = 25
) -> float:
  """
  Measures the average inference time of the model.

  Args:
    model (TrainedCANet): The trained model to be evaluated.
    dataloader (DataLoader): Dataloader providing input batches.
    num_trials (int, optional): Number of inference runs for averaging. Defaults to 25.

  Returns:
    float: Average inference time in milliseconds.
  """
  # Select device
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  # Prepare input tensor
  batch_x, _, _, _ = next(iter(dataloader))
  input_tensor: Tensor = batch_x.float().to(device)[0, :, :].unsqueeze(0)

  # Warm-up to avoid initialization overhead affecting timing
  for _ in range(10):
    _ = model(input_tensor)

  # Measure inference time
  if device.type == "cuda":
    # GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record(stream=None)
    for _ in range(num_trials):
      _ = model(input_tensor)
    end_event.record(stream=None)

    torch.cuda.synchronize()
    inference_time_ms: float = start_event.elapsed_time(end_event) / num_trials

  else:
    # CPU timing
    start_time: float = time.time()
    for _ in range(num_trials):
      _ = model(input_tensor)
    end_time: float = time.time()

    inference_time_ms = (end_time - start_time) * 1000 / num_trials

  return inference_time_ms


def main():
  parser = argparse.ArgumentParser(description="Evaluate CANet model performance.")
  parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the CANet model checkpoint (.ckpt file).",
  )
  args: argparse.Namespace = parser.parse_args()

  # Load model
  model: TrainedCANet = TrainedCANet.load_from_checkpoint(args.model_path, weights_only=False)
  model.eval()

  # Load test data
  _, test_dataloader = data_provider(model.config, flag="test")

  # Model size calculation
  total_params: int = calculate_model_size(model)
  print(f"Total Parameters: {total_params:,}")

  # Inference time calculation
  avg_inference_time: float = measure_inference_time(model, test_dataloader)
  print(f"Average Inference Time: {avg_inference_time:.4f} ms")


if __name__ == "__main__":
  main()
