"""PyTorch Dataset classes for ETT and custom time-series benchmarks."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from data.timefeatures import time_features

warnings.filterwarnings("ignore")

_DEFAULT_SEQ_LEN: int = 24 * 4 * 4
_DEFAULT_LABEL_LEN: int = 24 * 4
_ETT_HOUR_TRAIN_BORDER: int = 12 * 30 * 24


class Dataset_ETT_hour(Dataset):
  """Hourly ETT (Electricity Transformer Temperature) dataset.

  Applies a fixed 12/4/4 month train/val/test split. Supports multivariate
  (M), univariate (S), and multivariate-with-single-target (MS) feature modes.

  Args:
    root_path: Directory containing the CSV file.
    flag: Split to load — ``"train"``, ``"val"``, or ``"test"``.
    size: ``[seq_len, label_len, pred_len]``; defaults to ``[384, 96, 96]``.
    features: Feature mode — ``"M"``, ``"S"``, or ``"MS"``.
    data_path: CSV file name; default ``"ETTh1.csv"``.
    target: Target column name; default ``"OT"``.
    scale: If True, applies StandardScaler fitted on the training split.
    timeenc: 0 for manual time encoding, 1 for Fourier time features.
    freq: Frequency string used for Fourier encoding; default ``"h"``.
  """

  def __init__(
    self,
    root_path: str,
    flag: str = "train",
    size: list[int] | None = None,
    features: str = "S",
    data_path: str = "ETTh1.csv",
    target: str = "OT",
    scale: bool = True,
    timeenc: int = 0,
    freq: str = "h",
  ) -> None:
    # size [seq_len, label_len, pred_len]
    # info
    if size is None:
      self.seq_len = _DEFAULT_SEQ_LEN
      self.label_len = _DEFAULT_LABEL_LEN
      self.pred_len = _DEFAULT_LABEL_LEN
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    assert flag in ["train", "test", "val"]
    type_map = {"train": 0, "val": 1, "test": 2}
    self.set_type = type_map[flag]

    self.features = features
    self.target = target
    self.scale = scale
    self.timeenc = timeenc
    self.freq = freq

    self.root_path = root_path
    self.data_path = data_path

    self.__read_data__()

  def __read_data__(self) -> None:
    """Load CSV, fit StandardScaler on train split, and build data/stamp arrays."""
    self.scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

    border1s = [
      0,
      _ETT_HOUR_TRAIN_BORDER - self.seq_len,
      _ETT_HOUR_TRAIN_BORDER + 4 * 30 * 24 - self.seq_len,
    ]
    border2s = [
      _ETT_HOUR_TRAIN_BORDER,
      _ETT_HOUR_TRAIN_BORDER + 4 * 30 * 24,
      _ETT_HOUR_TRAIN_BORDER + 8 * 30 * 24,
    ]
    border1 = border1s[self.set_type]
    border2 = border2s[self.set_type]

    if self.features == "M" or self.features == "MS":
      cols_data = df_raw.columns[1:]
      df_data = df_raw[cols_data]
    elif self.features == "S":
      df_data = df_raw[[self.target]]

    if self.scale:
      train_data = df_data[border1s[0] : border2s[0]]
      self.scaler.fit(train_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    df_stamp = df_raw[["date"]][border1:border2]
    df_stamp["date"] = pd.to_datetime(df_stamp.date)
    if self.timeenc == 0:
      df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
      data_stamp = df_stamp.drop(["date"], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data_stamp = data_stamp

  def __getitem__(
    self, index: int
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    # seq_x = zero_pad_sequence(torch.from_numpy(seq_x).permute(1,0), self.pad_len).permute(1,0)

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self) -> int:
    return len(self.data_x) - self.seq_len - self.pred_len + 1

  def inverse_transform(self, data: np.ndarray) -> np.ndarray:
    return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
  """Minute-resolution ETT dataset (15-minute intervals).

  Identical to ``Dataset_ETT_hour`` but uses a 4× timestep multiplier for
  border calculation and extracts a discretized minute feature.

  Args:
    root_path: Directory containing the CSV file.
    flag: Split to load — ``"train"``, ``"val"``, or ``"test"``.
    size: ``[seq_len, label_len, pred_len]``; defaults to ``[384, 96, 96]``.
    features: Feature mode — ``"M"``, ``"S"``, or ``"MS"``.
    data_path: CSV file name; default ``"ETTm1.csv"``.
    target: Target column name; default ``"OT"``.
    scale: If True, applies StandardScaler fitted on the training split.
    timeenc: 0 for manual time encoding, 1 for Fourier time features.
    freq: Frequency string used for Fourier encoding; default ``"t"``.
  """

  def __init__(
    self,
    root_path: str,
    flag: str = "train",
    size: list[int] | None = None,
    features: str = "S",
    data_path: str = "ETTm1.csv",
    target: str = "OT",
    scale: bool = True,
    timeenc: int = 0,
    freq: str = "t",
  ) -> None:
    # size [seq_len, label_len, pred_len]
    # info
    if size is None:
      self.seq_len = _DEFAULT_SEQ_LEN
      self.label_len = _DEFAULT_LABEL_LEN
      self.pred_len = _DEFAULT_LABEL_LEN
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    assert flag in ["train", "test", "val"]
    type_map = {"train": 0, "val": 1, "test": 2}
    self.set_type = type_map[flag]

    self.features = features
    self.target = target
    self.scale = scale
    self.timeenc = timeenc
    self.freq = freq

    self.root_path = root_path
    self.data_path = data_path
    self.__read_data__()

  def __read_data__(self) -> None:
    """Load CSV, fit StandardScaler on train split, and build data/stamp arrays."""
    self.scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

    _ett_min_border: int = _ETT_HOUR_TRAIN_BORDER * 4
    border1s = [
      0,
      _ett_min_border - self.seq_len,
      _ett_min_border + 4 * 30 * 24 * 4 - self.seq_len,
    ]
    border2s = [
      _ett_min_border,
      _ett_min_border + 4 * 30 * 24 * 4,
      _ett_min_border + 8 * 30 * 24 * 4,
    ]
    border1 = border1s[self.set_type]
    border2 = border2s[self.set_type]

    if self.features == "M" or self.features == "MS":
      cols_data = df_raw.columns[1:]
      df_data = df_raw[cols_data]
    elif self.features == "S":
      df_data = df_raw[[self.target]]

    if self.scale:
      train_data = df_data[border1s[0] : border2s[0]]
      self.scaler.fit(train_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    df_stamp = df_raw[["date"]][border1:border2]
    df_stamp["date"] = pd.to_datetime(df_stamp.date)
    if self.timeenc == 0:
      df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
      df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
      df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
      data_stamp = df_stamp.drop(["date"], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data_stamp = data_stamp

  def __getitem__(
    self, index: int
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self) -> int:
    return len(self.data_x) - self.seq_len - self.pred_len + 1

  def inverse_transform(self, data: np.ndarray) -> np.ndarray:
    return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
  """Generic CSV dataset with flexible 70/20/10 train/val/test splits.

  Args:
    root_path: Directory containing the CSV file.
    flag: Split to load — ``"train"``, ``"val"``, or ``"test"``.
    size: ``[seq_len, label_len, pred_len]``; defaults to ``[384, 96, 96]``.
    features: Feature mode — ``"M"``, ``"S"``, or ``"MS"``.
    data_path: CSV file name; default ``"ETTh1.csv"``.
    target: Target column name; default ``"OT"``.
    scale: If True, applies StandardScaler fitted on the training split.
    timeenc: 0 for manual time encoding, 1 for Fourier time features.
    freq: Frequency string used for Fourier encoding; default ``"h"``.
    seasonal_patterns: Unused; kept for API compatibility.
  """

  def __init__(
    self,
    root_path: str,
    flag: str = "train",
    size: list[int] | None = None,
    features: str = "S",
    data_path: str = "ETTh1.csv",
    target: str = "OT",
    scale: bool = True,
    timeenc: int = 0,
    freq: str = "h",
    seasonal_patterns: str | None = None,
  ) -> None:
    # size [seq_len, label_len, pred_len]
    # info
    if size is None:
      self.seq_len = _DEFAULT_SEQ_LEN
      self.label_len = _DEFAULT_LABEL_LEN
      self.pred_len = _DEFAULT_LABEL_LEN
    else:
      self.seq_len = size[0]
      self.label_len = size[1]
      self.pred_len = size[2]
    # init
    assert flag in ["train", "test", "val"]
    type_map = {"train": 0, "val": 1, "test": 2}
    self.set_type = type_map[flag]

    self.features = features
    self.target = target
    self.scale = scale
    self.timeenc = timeenc
    self.freq = freq

    self.root_path = root_path
    self.data_path = data_path
    self.__read_data__()

  def __read_data__(self) -> None:
    """Load CSV, fit StandardScaler on train split, and build data/stamp arrays."""
    self.scaler = StandardScaler()
    df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

    """
        df_raw.columns: ['date', ...(other features), target feature]
        """
    cols = list(df_raw.columns)
    cols.remove(self.target)
    cols.remove("date")
    df_raw = df_raw[["date"] + cols + [self.target]]
    num_train = int(len(df_raw) * 0.7)
    num_test = int(len(df_raw) * 0.2)
    num_vali = len(df_raw) - num_train - num_test
    border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
    border2s = [num_train, num_train + num_vali, len(df_raw)]
    border1 = border1s[self.set_type]
    border2 = border2s[self.set_type]

    if self.features == "M" or self.features == "MS":
      cols_data = df_raw.columns[1:]
      df_data = df_raw[cols_data]
    elif self.features == "S":
      df_data = df_raw[[self.target]]

    if self.scale:
      train_data = df_data[border1s[0] : border2s[0]]
      self.scaler.fit(train_data.values)
      data = self.scaler.transform(df_data.values)
    else:
      data = df_data.values

    df_stamp = df_raw[["date"]][border1:border2]
    df_stamp["date"] = pd.to_datetime(df_stamp.date)
    if self.timeenc == 0:
      df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
      df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
      df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
      df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
      data_stamp = df_stamp.drop(["date"], 1).values
    elif self.timeenc == 1:
      data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq)
      data_stamp = data_stamp.transpose(1, 0)

    self.data_x = data[border1:border2]
    self.data_y = data[border1:border2]
    self.data_stamp = data_stamp

  def __getitem__(
    self, index: int
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s_begin = index
    s_end = s_begin + self.seq_len
    r_begin = s_end - self.label_len
    r_end = r_begin + self.label_len + self.pred_len

    seq_x = self.data_x[s_begin:s_end]
    seq_y = self.data_y[r_begin:r_end]
    seq_x_mark = self.data_stamp[s_begin:s_end]
    seq_y_mark = self.data_stamp[r_begin:r_end]

    return seq_x, seq_y, seq_x_mark, seq_y_mark

  def __len__(self) -> int:
    return len(self.data_x) - self.seq_len - self.pred_len + 1

  def inverse_transform(self, data: np.ndarray) -> np.ndarray:
    return self.scaler.inverse_transform(data)
