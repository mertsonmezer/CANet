declare -a patch_sizes=("8, 16")

for patch_sizes in "${patch_sizes[@]}"
do
  python3 -m forecasting.train \
    --data_name "custom" \
    --root_path "ILI" \
    --data_path "illness.csv" \
    --seq_len 36 \
    --pred_len 24 \
    --num_epochs 50 \
    --batch_size 32 \
    --seed 42 \
    --emb_dim 128 \
    --depth 2 \
    --patch_sizes "$patch_sizes" \
    --num_channels 7 \
    --first_norm_layer layer \
    --second_norm_layer adain \
    --style_blend_gate_affine True \
    --use_multi_patching True \
    --use_rank_reduction True
done
