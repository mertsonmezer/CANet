"""Factory function for constructing dataset and DataLoader pairs."""

from __future__ import annotations

import logging

from torch.utils.data import DataLoader, Dataset

from config.canet_config import CANetTrainConfig
from data.data_loader import Dataset_Custom, Dataset_ETT_hour, Dataset_ETT_minute

logger = logging.getLogger(__name__)

data_dict = {
  "ETTh1": Dataset_ETT_hour,
  "ETTh2": Dataset_ETT_hour,
  "ETTm1": Dataset_ETT_minute,
  "ETTm2": Dataset_ETT_minute,
  "custom": Dataset_Custom,
}


def data_provider(config: CANetTrainConfig, flag: str) -> tuple[Dataset, DataLoader]:
  """Instantiate the appropriate dataset and wrap it in a DataLoader.

  Args:
    config: Full training configuration (uses ``data_config`` and batch/seq
      settings).
    flag: Split to load — ``"train"``, ``"val"``, or ``"test"``.

  Returns:
    A ``(dataset, dataloader)`` pair ready for Lightning training.
  """
  Data = data_dict[config.data_config.data_name]
  timeenc = 0 if config.data_config.embed != "timeF" else 1

  if flag == "test":
    shuffle_flag = False
    drop_last = True
    batch_size = config.batch_size  # 10  # bsz=1 for evaluation
    freq = config.data_config.freq
  else:
    shuffle_flag = True
    drop_last = True
    batch_size = config.batch_size  # bsz for train and valid
    freq = config.data_config.freq

  kwargs = dict(
    root_path=config.data_config.root_dir_path,
    data_path=config.data_config.data_path,
    flag=flag,
    size=[config.seq_len, config.label_len, config.pred_len],
    features=config.data_config.features,
    target=config.data_config.target,
    timeenc=timeenc,
    freq=freq,
  )
  if "ETT" not in config.data_config.data_name:
    kwargs["seasonal_patterns"] = config.data_config.seasonal_patterns

  data_set = Data(**kwargs)
  logger.info("%s dataset size: %d", flag, len(data_set))
  data_loader = DataLoader(
    data_set, batch_size=batch_size, shuffle=shuffle_flag, drop_last=drop_last
  )

  return data_set, data_loader
