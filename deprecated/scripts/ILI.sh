# Hyperparameter Search Space
# LOW_LR, HIGH_LR = 1e-4, 1e-2
# LOW_DROPOUT, HIGH_DROPOUT = 0.3, 0.5
# BATCH_SIZES: List[int] = [8, 16, 32]#, 64, 128, 256, 512, 1024]
# PATCH_SIZES: List[List[int]] = [[4, 12], [4, 16], [4, 24], [8, 16], [8, 24]]
# EMBED_DIMS: List[int] = [32, 64, 128]

for seq_len in 36
do
	for pred_len in 24 36 48 60
	do
		for norm in "adain"
		do
		  python3 -m forecasting.train_tune \
		  --data_name "custom" \
		  --root_path "ILI/" \
		  --data_path "illness.csv" \
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
