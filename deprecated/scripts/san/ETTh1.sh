for seq_len in 96
do
	for pred_len in 96 192 336 720
	do
    for norm in "layer"
    do
      python3 -m forecasting.train_tune_gpu0 \
      --data_name "ETTh1" \
      --root_path "../Baselines/Datasets/ETT-small/" \
      --data_path "ETTh1.csv" \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --num_trials 100 \
      --num_epochs 20 \
      --seed 56 \
      --num_channels 7 \
      --first_norm_layer layer \
      --second_norm_layer $norm \
      --style_blend_gate_affine True \
      --use_multi_patching True \
      --use_rank_reduction True
    done
	done
done
