from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from torch.utils.data import DataLoader
from config.net_config import TrainConfig

data_dict = {
  "ETTh1": Dataset_ETT_hour,
  "ETTh2": Dataset_ETT_hour,
  "ETTm1": Dataset_ETT_minute,
  "ETTm2": Dataset_ETT_minute,
  "custom": Dataset_Custom,
}


def data_provider(config: TrainConfig, flag: str):
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

  # print("=====================", args.data)
  if "ETT" in config.data_config.data_name:
    data_set = Data(
      root_path=config.data_config.root_dir_path,
      data_path=config.data_config.data_path,
      flag=flag,
      size=[config.seq_len, config.label_len, config.pred_len],
      features=config.data_config.features,
      target=config.data_config.target,
      timeenc=timeenc,
      freq=freq,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, drop_last=drop_last)
  else:
    data_set = Data(
      root_path=config.data_config.root_dir_path,
      data_path=config.data_config.data_path,
      flag=flag,
      size=[config.seq_len, config.label_len, config.pred_len],
      features=config.data_config.features,
      target=config.data_config.target,
      timeenc=timeenc,
      freq=freq,
      seasonal_patterns=config.data_config.seasonal_patterns,
    )
    print(flag, len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle_flag, drop_last=drop_last)

  return data_set, data_loader
