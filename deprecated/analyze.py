import argparse

import numpy as np
import optuna
import scipy.stats as stats
from optuna.visualization import plot_slice

# Set up argument parser
parser = argparse.ArgumentParser(description="Analyze Optuna study results.")
parser.add_argument("--study_name", type=str, required=True, help="Name of the Optuna study")
parser.add_argument("--storage_folder", type=str, default="results", help="Folder where the study DB is stored")
parser.add_argument("--show_plot", action="store_true", help="Flag to control whether to display the plot")

args = parser.parse_args()

# Use the command line params for study name and storage
study_name = args.study_name
storage_folder = args.storage_folder
storage = f"sqlite:///{storage_folder}/{study_name}.db"

# Load the study
study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)

# Visualization of the best trial
best_trial = study.best_trial
print(f"Best Trial is {best_trial._trial_id} with the value {best_trial.value}")
print("Params:")
print(best_trial.params)

# Plot slice visualization if the show_plot flag is set
if args.show_plot:
  figure = plot_slice(study)
  figure.show()

# Get all completed trials in the study
completed_trials = [trial.value for trial in study.get_trials() if trial.value is not None]

# Calculate the interquartile range (IQR) to identify outliers
Q1 = np.percentile(completed_trials, 25)
Q3 = np.percentile(completed_trials, 75)
IQR = Q3 - Q1

# Filter out outliers
filtered_trials = [value for value in completed_trials if (value >= Q1 - 1.5 * IQR) and (value <= Q3 + 1.5 * IQR)]

# Calculate the mean and the standard deviation for filtered trials
mean = np.mean(filtered_trials)
std_dev = np.std(filtered_trials)

# Assuming a 90% confidence interval
confidence_level = 0.90
degrees_freedom = len(filtered_trials) - 1
confidence_interval = stats.t.interval(confidence_level, degrees_freedom, mean, std_dev / np.sqrt(len(filtered_trials)))

# Print results for confidence interval analysis
print(f"Mean of the filtered objective values: {mean}")
print(f"90% confidence interval for the objective values: {confidence_interval}")
