"""Analysis utilities for Optuna hyperparameter search studies."""

import argparse
import logging
import warnings

import numpy as np
import optuna
import scipy.stats as stats
from numpy._typing._array_like import NDArray
from optuna.trial._frozen import FrozenTrial
from optuna.visualization import plot_slice

warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)


def get_filtered_trials(study: optuna.study.Study) -> NDArray[np.float64]:
  """Filters trials to exclude outliers based on the interquartile range (IQR).

  Args:
    study (optuna.study.Study): The Optuna study containing the trials.

  Returns:
    NDArray[np.float64]: An array of filtered objective values.

  Raises:
    ValueError: If there are no completed trials in the study.
  """
  completed_trials: list[float] = [
    trial.value for trial in study.get_trials() if trial.value is not None
  ]
  if not completed_trials:
    raise ValueError("No completed trials found in the study.")

  # Calculate IQR for outlier filtering
  Q1, Q3 = np.percentile(completed_trials, [25, 75])
  IQR = Q3 - Q1
  filtered_trials: list[float] = [
    value for value in completed_trials if (value >= Q1 - 1.5 * IQR) and (value <= Q3 + 1.5 * IQR)
  ]
  return np.array(filtered_trials)


def calculate_confidence_interval(
  filtered_trials: NDArray[np.float64], confidence_level: float = 0.90
) -> tuple[np.float64, np.float64]:
  """Calculates the confidence interval for the filtered trials.

  Args:
    filtered_trials (NDArray[np.float64]): The array of filtered objective values.
    confidence_level (float, optional): The confidence level for the interval. Defaults to 0.90.

  Returns:
    Tuple[np.float64, np.float64]: The lower and upper bounds of the confidence interval.
  """
  mean: np.float64 = np.mean(filtered_trials)
  std_dev: np.float64 = np.std(filtered_trials)
  degrees_freedom: int = len(filtered_trials) - 1
  return stats.t.interval(
    confidence_level, degrees_freedom, mean, std_dev / np.sqrt(len(filtered_trials))
  )


def main() -> None:
  """Entry point — parse arguments and run the main workflow."""
  parser = argparse.ArgumentParser(description="Analyze Optuna study results.")
  parser.add_argument("--study_name", type=str, required=True, help="Name of the Optuna study")
  parser.add_argument(
    "--storage_folder", type=str, default="results", help="Folder where the study DB is stored"
  )
  parser.add_argument(
    "--show_plot", action="store_true", help="Flag to control whether to display the plot"
  )
  args: argparse.Namespace = parser.parse_args()

  # Load the study
  storage: str = f"sqlite:///{args.storage_folder}/{args.study_name}.db"
  study: optuna.Study = optuna.create_study(
    study_name=args.study_name, storage=storage, load_if_exists=True
  )

  # Display the best trial
  best_trial: FrozenTrial = study.best_trial
  logger.info("Best Trial is %s with the value %s", best_trial._trial_id, best_trial.value)
  logger.info("Params:")
  for param, value in best_trial.params.items():
    logger.info("  %s: %s", param, value)

  # Plot best trial visualization if requested
  if args.show_plot:
    figure = plot_slice(study)
    figure.show()

  # Filter trials and calculate confidence interval
  filtered_trials: NDArray[np.float64] = get_filtered_trials(study)
  confidence_interval: tuple[np.float64, np.float64] = calculate_confidence_interval(
    filtered_trials
  )

  logger.info("Mean of the filtered objective values: %s", np.mean(filtered_trials))
  logger.info(
    "90%% confidence interval for the objective values: %s, %s",
    float(confidence_interval[0]),
    float(confidence_interval[1]),
  )


if __name__ == "__main__":
  main()
