"""Centralized experiment artifact path management for CANet."""

import os
from pathlib import Path

_DEFAULT_ROOT = "experiments"
_ENV_VAR = "CANET_EXPERIMENTS_DIR"


def get_experiments_root() -> Path:
    """Returns the experiments root directory.

    Reads CANET_EXPERIMENTS_DIR env var; falls back to 'experiments/'
    relative to CWD (the project root when invoked as
    ``uv run python -m forecasting.train_tune``).
    """
    return Path(os.environ.get(_ENV_VAR, _DEFAULT_ROOT))


def build_study_params(
    seq_len: int,
    pred_len: int,
    first_norm_layer: str,
    second_norm_layer: str,
    use_rank_reduction: bool,
) -> str:
    """Constructs the dataset-independent part of the study identity.

    This string is used as the leaf subdirectory name under
    ``experiments/{dataset}/tune/``, avoiding redundancy with the
    dataset component already encoded in the parent path.

    Example:
        ``SeqLen96-PredLen96-FirstNormlayer-SecondNormnsan-RankReductionTrue``
    """
    return (
        f"SeqLen{seq_len}-PredLen{pred_len}"
        f"-FirstNorm{first_norm_layer}-SecondNorm{second_norm_layer}"
        f"-RankReduction{use_rank_reduction}"
    )


class TunePaths:
    """Filesystem paths for one Optuna hyperparameter-tuning study.

    Directory layout::

        experiments/{dataset}/tune/{study_params}/
            optuna.db          ← Optuna SQLite storage
            trial_{N}/         ← per-trial artifacts (cleaned up after tuning)
            best/              ← best-trial artifacts (promoted after tuning)

    Args:
        dataset: Dataset identifier, e.g. ``"ETTh1"``.
        study_params: Output of :func:`build_study_params`.
        root: Override the experiments root (useful in tests); defaults to
            :func:`get_experiments_root`.
    """

    def __init__(self, dataset: str, study_params: str, root: Path | None = None) -> None:
        self._base = (root or get_experiments_root()) / dataset / "tune" / study_params

    @property
    def study_root(self) -> Path:
        """``experiments/{dataset}/tune/{study_params}/``"""
        return self._base

    def trial_dir(self, trial_number: int) -> Path:
        """``experiments/{dataset}/tune/{study_params}/trial_{N}/``"""
        return self._base / f"trial_{trial_number}"

    @property
    def best_dir(self) -> Path:
        """``experiments/{dataset}/tune/{study_params}/best/``"""
        return self._base / "best"

    @property
    def db_path(self) -> Path:
        """``experiments/{dataset}/tune/{study_params}/optuna.db``"""
        return self._base / "optuna.db"

    @property
    def db_url(self) -> str:
        """SQLite URL ready for ``optuna.create_study(storage=...)``."""
        return f"sqlite:///{self.db_path}"


class TrainPaths:
    """Filesystem paths for a single standalone training run.

    Directory layout::

        experiments/{dataset}/train/{run_description}/
            train-epoch=N.ckpt
            config.json
            results.xlsx
            lightning_logs/

    Args:
        dataset: Dataset identifier, e.g. ``"ETTh1"``.
        run_description: Unique run slug (typically includes architecture
            details and a timestamp).
        root: Override the experiments root; defaults to
            :func:`get_experiments_root`.
    """

    def __init__(self, dataset: str, run_description: str, root: Path | None = None) -> None:
        self._base = (root or get_experiments_root()) / dataset / "train" / run_description

    @property
    def run_dir(self) -> Path:
        """``experiments/{dataset}/train/{run_description}/``"""
        return self._base
