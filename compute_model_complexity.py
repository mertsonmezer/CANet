from canet.canet import TrainedCANet
import torch
import time
from data.data_factory import data_provider

# Load the model from the .ckpt file
ckpt_path = "results/Exchange/custom-SeqLen96-PredLen720-FirstNormlayer-SecondNormadain-MPTrue-RankReductionTrue-Trial82/train-epoch=1.ckpt"
model = TrainedCANet.load_from_checkpoint(ckpt_path)  # Initialize your model architecture
model.eval()

test_data, test_dataloader = data_provider(model.config, flag="test")
sample_input: torch.Tensor = next(iter(test_dataloader))[0]
sample_input = sample_input.to(torch.float)

############# MODEL SIZE CALCULATION #############

total_params = sum(p.numel() for p in model.parameters())
print(f"Total Parameters: {total_params:,}")

##################################################

########### INFERENCE TIME CALCULATION ###########
# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

(batch_x, batch_y, batch_x_mark, batch_y_mark) = next(iter(test_dataloader))
batch_x = batch_x.float().to(device)
input_tensor = batch_x[0, :, :].unsqueeze(0)

# Warm-up (to avoid initialization overhead affecting timing)
for _ in range(10):
  _ = model(input_tensor)

# Measure inference time
num_trials = 25  # Number of inference runs for averaging

if device.type == "cuda":
  # GPU timing
  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  start_event.record()
  for _ in range(num_trials):
    _ = model(input_tensor)
  end_event.record()

  # Waits for everything to finish running
  torch.cuda.synchronize()

  # Calculate average time per inference
  inference_time_ms = start_event.elapsed_time(end_event) / num_trials
else:
  # CPU timing
  start_time = time.time()
  for _ in range(num_trials):
    _ = model(input_tensor)
  end_time = time.time()

  # Calculate average time per inference
  inference_time_ms = (end_time - start_time) * 1000 / num_trials  # ms per inference

print(f"Average Inference Time: {inference_time_ms:.4f} ms")

##################################################
