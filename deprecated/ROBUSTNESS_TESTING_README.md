# Robustness Testing Modifications

## Overview

The `train_tune_gpu0.py` script has been modified to include robustness testing functionality. When a run achieves an MSE score that matches a target "best MSE" from previous experiments, the script automatically runs the same configuration with 5 different seeds to verify the robustness of the results.

## New Features

### 1. Command Line Arguments

A new argument has been added:
- `--best_mse`: The target MSE score from your database of 100 runs. When the current run's best MSE matches this value (within tolerance), robustness testing is triggered.

### 2. New Functions

#### `run_single_training(config, hyperparams, seed, run_suffix="")`
- Runs a single training session with specified hyperparameters and seed
- Used for both regular optimization and robustness testing
- Saves results to separate directories for each run

#### `run_robustness_test(config, best_hyperparams, tolerance=1e-6)`
- Runs the best configuration with 5 different seeds: [original_seed, 0, 2, 42, 2025]
- Returns results for statistical analysis
- Creates separate log directories for each seed

#### `check_mse_match(current_mse, best_mse, tolerance=1e-6)`
- Checks if the current MSE matches the target best MSE within the specified tolerance
- Default tolerance is 1e-6 (can be adjusted as needed)

### 3. Robustness Testing Workflow

When `--best_mse` is provided and a match is detected:

1. **Match Detection**: The script compares the best trial's MSE with the target MSE
2. **Robustness Testing**: If matched, runs the same configuration with 5 seeds
3. **Statistical Analysis**: Calculates mean, standard deviation, min, and max across the 5 runs
4. **Results Summary**: Prints comprehensive statistics and individual results

## Usage Examples

### Basic Robustness Testing
```bash
python forecasting/train_tune_gpu0.py \
    --best_mse 0.123456 \
    --num_trials 100 \
    --data_name ETTh1 \
    --seq_len 96 \
    --pred_len 96
```

### Without Robustness Testing (Original Behavior)
```bash
python forecasting/train_tune_gpu0.py \
    --num_trials 100 \
    --data_name ETTh1 \
    --seq_len 96 \
    --pred_len 96
```

## Output Structure

### Regular Output
- Study statistics
- Best trial information
- Cleanup of non-best trial directories

### Robustness Testing Output
When a match is detected, additional output includes:

```
==================================================
MATCH DETECTED!
Current best MSE: 0.123456
Target best MSE:  0.123456
Starting robustness test with multiple seeds...
==================================================

==================================================
ROBUSTNESS TEST: Running with 5 different seeds
==================================================

Run 1/5 with seed 56
========== ETTh1-SeqLen96-PredLen96-...-Seed56-RobustnessTest ===========
Test MSE for seed 56: 0.123456

Run 2/5 with seed 0
========== ETTh1-SeqLen96-PredLen96-...-Seed0-RobustnessTest ===========
Test MSE for seed 0: 0.123789

... (continues for all 5 seeds)

==================================================
ROBUSTNESS TEST RESULTS SUMMARY
==================================================
Seed   56: Test MSE = 0.123456, Val MSE = 0.098765
Seed    0: Test MSE = 0.123789, Val MSE = 0.098901
Seed    2: Test MSE = 0.123234, Val MSE = 0.098567
Seed   42: Test MSE = 0.123678, Val MSE = 0.098812
Seed 2025: Test MSE = 0.123445, Val MSE = 0.098734

Statistics:
  Mean Test MSE: 0.123520
  Std Test MSE:  0.000191
  Min Test MSE:  0.123234
  Max Test MSE:  0.123789
==================================================
```

## File Organization

### Log Directory Structure
```
logs/
├── {config}-Trial{N}/                    # Regular optimization trials
└── {config}-Seed{S}-RobustnessTest/     # Robustness test runs
    ├── train-epoch*.ckpt                # Model checkpoints
    ├── model.txt                        # Model info and config
    └── results.xlsx                     # MSE and MAE results
```

### Configuration Files
Each run saves:
- **model.txt**: Total parameters count and training configuration
- **results.xlsx**: MSE and MAE results for test and validation sets
- **Checkpoints**: Best model weights for each run

## Key Modifications Made

1. **Hyperparameter Handling**: Modified `PATCH_SIZES` to use string representation for Optuna compatibility
2. **Seed Management**: Enhanced seed handling for reproducibility across multiple runs
3. **Result Tracking**: Added comprehensive result tracking and statistical analysis
4. **Directory Management**: Organized log directories for easier result analysis
5. **Match Detection**: Implemented tolerance-based MSE matching for robustness trigger

## Technical Notes

- **Tolerance**: Default tolerance for MSE matching is 1e-6, adjustable via the `check_mse_match` function
- **Seeds**: Uses seeds [original_seed, 0, 2, 42, 2025] for robustness testing
- **Memory Management**: Each robustness run is independent to avoid memory issues
- **Reproducibility**: Ensures reproducible results through proper seed management

## Integration with Existing Workflow

The modifications are backward compatible:
- If `--best_mse` is not provided, the script behaves exactly as before
- All existing command-line arguments and functionality remain unchanged
- The robustness testing is an optional addition that doesn't interfere with normal operation

This allows you to:
1. Continue using the script for regular hyperparameter optimization
2. Enable robustness testing when you have a target MSE from your database
3. Verify the stability of your best results across different random seeds
