for pred_len in 96 192 336 720
do
  python3 -m forecasting.train_tune \
    --data_name "ETTh1" \
    --root_path "ETT-small/" \
    --data_path "ETTh1.csv" \
    --num_channels 7 \
    --embed_dims 32 64 128 \
    --depth 2 \
    --low_dropout 0.3 \
    --high_dropout 0.5 \
    --first_norm_layer layer \
    --second_norm_layer nsan \
    --patch_sizes "[[8, 32], [8, 48], [8, 64], [16, 48], [16, 64]]" \
    --num_trials 100 \
    --num_epochs 20 \
    --batch_sizes 64 128 256 512 1024 \
    --seq_len 96 \
    --pred_len $pred_len \
    --low_lr 0.0001 \
    --high_lr 0.01 \
    --style_blend_gate_affine \
    --use_rank_reduction \
    --seed 56
done