for pred_len in 96 192 336
do
  python3 -m forecasting.train_tune \
    --data_name "custom" \
    --root_path "Exchange/" \
    --data_path "exchange_rate.csv" \
    --num_channels 8 \
    --embed_dims 32 64 128 \
    --depth 2 \
    --low_dropout 0.3 \
    --high_dropout 0.5 \
    --first_norm_layer layer \
    --second_norm_layer nsan \
    --patch_sizes "[[8, 32], [8, 48], [8, 64], [16, 48], [16, 64]]" \
    --num_trials 100 \
    --num_epochs 20 \
    --batch_sizes 16 32 64 128 256 \
    --seq_len 96 \
    --pred_len $pred_len \
    --low_lr 0.0001 \
    --high_lr 0.01 \
    --use_rank_reduction \
    --seed 42
done

# Note: When pred_len was 720, BATCH_SIZES was set to [16, 32] for the memory issues.
python3 -m forecasting.train_tune \
  --data_name "custom" \
  --root_path "Exchange/" \
  --data_path "exchange_rate.csv" \
  --num_channels 8 \
  --embed_dims 32 64 128 \
  --depth 2 \
  --low_dropout 0.3 \
  --high_dropout 0.5 \
  --first_norm_layer layer \
  --second_norm_layer nsan \
  --patch_sizes "[[8, 32], [8, 48], [8, 64], [16, 48], [16, 64]]" \
  --num_trials 100 \
  --num_epochs 20 \
  --batch_sizes 16 32 \
  --seq_len 96 \
  --pred_len 720 \
  --low_lr 0.0001 \
  --high_lr 0.01 \
  --use_rank_reduction \
  --seed 42