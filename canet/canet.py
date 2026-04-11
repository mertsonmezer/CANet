from typing import Any, Dict, List, Tuple

import lightning as L
import torch
from einops import rearrange
from torch import Tensor, nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from canet.modules.canet_layer import CANetLayer
from canet.modules.skp_layer import StackedKroneckerProductLayer
from config.canet_config import CANetTrainConfig

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CANet(L.LightningModule):
    def __init__(self, config: CANetTrainConfig):
        super().__init__()
        self.config: CANetTrainConfig = config

        drop_path_ratios: List[float] = [
            x.item() for x in torch.linspace(0, 0.5, self.config.depth)
        ]

        self.net_layers = nn.ModuleList(
            [
                CANetLayer(
                    patch_size=self.config.patch_sizes[i],
                    config=self.config,
                    drop_path_ratio=drop_path_ratios[i],
                )
                for i in range(self.config.depth)
            ]
        )

        total_num_pathes: int = sum(
            layer.patchifier.num_patches for layer in self.net_layers
        )

        # Optional rank-reducing projection using Stacked Kronecker Product Layer.
        self.fc1: StackedKroneckerProductLayer | nn.Linear = (
            StackedKroneckerProductLayer(
                self.config.embed_dim * total_num_pathes,
                self.config.output_features,
                scale=0.5,
            )
            if self.config.use_rank_reduction
            else nn.Linear(
                self.config.embed_dim * total_num_pathes, self.config.output_features
            )
        )
        self.drop = nn.Dropout(self.config.dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.config.output_features, self.config.pred_len)

    def forward(self, x: Tensor) -> Tensor:
        B, _, M = x.shape

        # Instance Normalization
        mean: Tensor = x.mean(dim=1, keepdim=True)
        std: Tensor = x.std(dim=1, keepdim=True)
        eps: float = 1e-5
        x = (x - mean) / (std + eps)

        xs_patched: List[Tensor] = []
        for layer in self.net_layers:
            x_patched: Tensor = layer(x, mean, std)
            x_patched = x_patched.reshape(B * M, -1)
            xs_patched.append(x_patched)

        x_concat: Tensor = torch.cat(xs_patched, dim=1)
        x_concat = self.fc1(x_concat)
        x_concat = self.drop(x_concat)
        x_concat = self.relu(x_concat)

        out: Tensor = self.fc2(x_concat)
        out = rearrange(out, "(b m) l -> b l m", b=B)

        # Denormalization
        out = (out * std) + mean

        return out


class TrainedCANet(L.LightningModule):
    def __init__(self, config: CANetTrainConfig):
        super().__init__()
        self.config: CANetTrainConfig = config
        self.model = CANet(self.config)

        self.criterion = nn.MSELoss()
        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()

        self.lr: float = self.config.train_lr

        self.preds: List[Any] = []
        self.trues: List[Any] = []

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.config.weight_decay
        )
        scheduler: Dict[str, Any] = {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=2
            ),
            "monitor": "val_mse",
            "interval": "epoch",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _calculate_loss(
        self, batch: Tensor, mode: str = "train"
    ) -> Tuple[Tensor, Tensor, Tensor]:
        batch_x, batch_y, _, _ = batch
        batch_x: Tensor = batch_x.float().to(DEVICE)
        batch_y: Tensor = batch_y.float().to(DEVICE)

        outputs: Tensor = self.model(batch_x)
        outputs = outputs[:, -self.config.pred_len :, :]
        batch_y = batch_y[:, -self.config.pred_len :, :].to(DEVICE)
        loss: Tensor = self.criterion(outputs, batch_y)

        pred: Tensor = outputs.detach().cpu()
        true: Tensor = batch_y.detach().cpu()

        mse: Tensor = self.mse(pred.contiguous(), true.contiguous())
        mae: Tensor = self.mae(pred.contiguous(), true.contiguous())

        self.log(
            f"{mode}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{mode}_mse", mse, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            f"{mode}_mae", mae, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return loss, pred, true

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        loss, _, _ = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> None:
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch: Tensor, batch_idx: int) -> Dict[str, Tensor]:
        loss, pred, true = self._calculate_loss(batch, mode="test")
        self.preds.append(pred)
        self.trues.append(true)
        return {"test_loss": loss, "pred": pred, "true": true}

    def on_train_batch_end(self, outputs: Any, batch: Tensor, batch_idx: int) -> None:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)

    def on_test_epoch_end(self) -> None:
        preds: Tensor = torch.cat(self.preds)
        trues: Tensor = torch.cat(self.trues)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mse: Tensor = self.mse(preds.contiguous(), trues.contiguous())
        mae: Tensor = self.mae(preds, trues)
        print(f"{mae, mse}")


def demo():
    # Create a dummy configuration
    config = CANetTrainConfig(
        batch_size=128,
        depth=2,
        patch_sizes=[8, 64],
        embed_dim=32,
        output_features=1024,
        seq_len=336,
        pred_len=96,
        use_rank_reduction=True,
    )

    # Generate a random input
    input: Tensor = torch.randn(
        config.batch_size, config.seq_len, config.data_config.num_channels
    )

    # Initialize model
    model = TrainedCANet(config)
    model.eval()

    # Run the model
    with torch.no_grad():
        output: Tensor = model(input)

    print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    demo()
