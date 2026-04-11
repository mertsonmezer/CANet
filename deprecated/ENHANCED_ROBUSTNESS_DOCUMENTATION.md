# Enhanced Robustness Testing Documentation

## Overview

The `train_tune_gpu0.py` script has been enhanced with advanced robustness testing capabilities that include:

1. **Automatic Database Detection**: Automatically finds and loads the best results from existing optuna databases
2. **Dual Metric Evaluation**: Tests robustness for both MSE and MAE metrics
3. **Excel Output**: Saves comprehensive results to Excel files with multiple sheets
4. **Statistical Analysis**: Provides detailed statistics for both metrics

## New Features

### 1. Automatic Database Loading

The script can now automatically find and load the best results from existing optuna databases without requiring manual specification of the best MSE value.

**Database Search Strategy:**
1. Searches in the main directory for `{study_name}.db`
2. Searches in `results/{dataset_name}/` directory
3. Performs recursive search in the entire `results/` directory
4. Falls back to similar configurations if exact match not found

### 2. Enhanced Robustness Testing

When a match is detected (either from automatic loading or manual specification), the script:
- Runs the best configuration with 5 different seeds: `[original_seed, 0, 2, 42, 2025]`
- Evaluates both MSE and MAE for each run
- Calculates comprehensive statistics for both metrics
- Saves all results to an Excel file with multiple sheets

### 3. Excel Output Structure

The Excel file contains 4 sheets:
- **MSE_Results**: Individual MSE results for each seed
- **MAE_Results**: Individual MAE results for each seed
- **MSE_Statistics**: Statistical summary for MSE (mean, std, min, max)
- **MAE_Statistics**: Statistical summary for MAE (mean, std, min, max)

## Usage Examples

### Automatic Database Detection (Recommended)
```bash
python forecasting/train_tune_gpu0.py \
    --data_name ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --first_norm_layer layer \
    --second_norm_layer adain \
    --num_trials 10
```

### Manual Best MSE Specification
```bash
python forecasting/train_tune_gpu0.py \
    --best_mse 0.355196 \
    --data_name ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --num_trials 10
```

### Quick Test Run
```bash
python forecasting/train_tune_gpu0.py \
    --data_name ETTh1 \
    --seq_len 96 \
    --pred_len 96 \
    --num_trials 5 \
    --num_epochs 3
```

## Sample Output

### Console Output
```
Study statistics:
  Number of finished trials:  10
  Number of pruned trials:  0
  Number of complete trials:  10

Best trial:
  Value:  0.355196
  Params:
    batch_size: 1024
    train_lr: 0.0017946915299967715
    dropout_rate: 0.3177346217694464
    patch_sizes: 8,32
    embed_dim: 128

No explicit best MSE provided. Searching for existing database...
✓ Found database: results/ETTh1/ETTh1-SeqLen96-PredLen96-FirstNormlayer-SecondNormadain-MPTrue-RankReductionTrue.db
Loaded target MSE: 0.355196 from database

==================================================
MATCH DETECTED!
Current best MSE: 0.355196
Target best MSE:  0.355196
Starting robustness test with multiple seeds...
==================================================

==================================================
ROBUSTNESS TEST: Running with 5 different seeds
==================================================

Run 1/5 with seed 56
========== ETTh1-SeqLen96-PredLen96-...-Seed56-RobustnessTest ===========
Test MSE for seed 56: 0.355196
Test MAE for seed 56: 0.284157

...

==================================================
ROBUSTNESS TEST RESULTS SUMMARY
==================================================
Seed   56: Test MSE = 0.355196, Val MSE = 0.298765
           Test MAE = 0.284157, Val MAE = 0.239012
Seed    0: Test MSE = 0.355789, Val MSE = 0.298901
           Test MAE = 0.284631, Val MAE = 0.239121
...

MSE Statistics:
  Mean Test MSE: 0.355520
  Std Test MSE:  0.000191
  Min Test MSE:  0.355196
  Max Test MSE:  0.355789

MAE Statistics:
  Mean Test MAE: 0.284416
  Std Test MAE:  0.000153
  Min Test MAE:  0.284157
  Max Test MAE:  0.284631

Results saved to: robustness_results/ETTh1-SeqLen96-PredLen96-...-RobustnessResults.xlsx
==================================================
```

## File Organization

### Excel Output Location
```
robustness_results/
└── {dataset}-SeqLen{seq_len}-PredLen{pred_len}-...-RobustnessResults.xlsx
```

### Log Directory Structure
```
logs/
├── {config}-Trial{N}/                           # Regular optimization trials
└── {config}-Seed{S}-RobustnessTest/            # Robustness test runs
    ├── train-epoch*.ckpt                       # Model checkpoints
    ├── model.txt                               # Model info and config
    └── results.xlsx                            # Individual run results
```

## Key Implementation Details

### Database File Naming Convention
Database files follow the pattern:
```
{data_name}-SeqLen{seq_len}-PredLen{pred_len}-FirstNorm{first_norm_layer}-SecondNorm{second_norm_layer}-MP{use_multi_patching}-RankReduction{use_rank_reduction}.db
```

### Tolerance for MSE Matching
- Default tolerance: `1e-5`
- Can be adjusted in the `check_mse_match()` function
- Allows for small numerical differences between runs

### Seeds for Robustness Testing
The script uses 5 predefined seeds to ensure reproducible robustness testing:
- **Original seed**: The seed specified in the configuration
- **0**: Standard baseline seed
- **2**: Alternative low-value seed
- **42**: Popular choice in computer science (Hitchhiker's Guide reference)
- **2025**: Current year for uniqueness

## Technical Notes

### Database Compatibility
- Works with any optuna SQLite database
- Automatically handles different study names
- Robust error handling for corrupted or missing databases

### Performance Considerations
- Each robustness run is independent (no memory accumulation)
- Excel writing is optimized with openpyxl engine
- Log directories are organized to prevent conflicts

### Error Handling
- Graceful fallback when databases are not found
- Continues with normal execution if robustness testing fails
- Comprehensive error messages for debugging

## Migration from Previous Version

The enhanced version is fully backward compatible:
- All existing command-line arguments work unchanged
- If `--best_mse` is not provided, automatic detection is attempted
- If no database is found, normal execution continues without robustness testing
- All original functionality remains intact

## Best Practices

1. **Use Automatic Detection**: Let the script find the database automatically for most use cases
2. **Manual Override**: Use `--best_mse` only when you want to test against a specific target
3. **Short Test Runs**: Use fewer trials and epochs for quick verification
4. **Monitor Logs**: Check the log directories for detailed training information
5. **Excel Analysis**: Use the Excel output for detailed statistical analysis and visualization
