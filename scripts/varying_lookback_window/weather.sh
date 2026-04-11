for seq_len in 24 48
do
  python3 -m forecasting.train_tune \
    --data_name "custom" \
    --root_path "Weather/" \
    --data_path "weather.csv" \
    --num_channels 21 \
    --embed_dims 64 \
    --depth 2 \
    --low_dropout 0.3 \
    --high_dropout 0.5 \
    --first_norm_layer layer \
    --second_norm_layer nsan \
    --patch_sizes "[[4, 16]]" \
    --num_trials 50 \
    --num_epochs 20 \
    --batch_sizes 64 \
    --seq_len $seq_len \
    --pred_len 96 \
    --low_lr 0.0001 \
    --high_lr 0.01 \
    --style_blend_gate_affine \
    --use_rank_reduction \
    --seed 56
done

for seq_len in 96 192 336 720
do
  python3 -m forecasting.train_tune \
    --data_name "custom" \
    --root_path "Weather/" \
    --data_path "weather.csv" \
    --num_channels 21 \
    --embed_dims 64 \
    --depth 2 \
    --low_dropout 0.3 \
    --high_dropout 0.5 \
    --first_norm_layer layer \
    --second_norm_layer nsan \
    --patch_sizes "[[8, 64]]" \
    --num_trials 50 \
    --num_epochs 20 \
    --batch_sizes 64 \
    --seq_len $seq_len \
    --pred_len 96 \
    --low_lr 0.0001 \
    --high_lr 0.01 \
    --style_blend_gate_affine \
    --use_rank_reduction \
    --seed 56
done