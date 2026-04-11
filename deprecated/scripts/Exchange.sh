# Hyperparameter Search Space
# LOW_LR, HIGH_LR = 1e-4, 1e-2
# LOW_DROPOUT, HIGH_DROPOUT = 0.3, 0.5
# BATCH_SIZES: List[int] = [16, 32, 64, 128, 256]
# PATCH_SIZES: List[List[int]] = [[8, 32], [8, 48], [8, 64], [16, 48], [16, 64]]
# EMBED_DIMS: List[int] = [32, 64, 128]
# Note: When pred_len was 720, BATCH_SIZES was set to [16, 32].

for seq_len in 96
do
	for pred_len in 96 192 336
	do
		for norm in "adain"
		do
			python3 -m forecasting.train_tune_gpu3 \
			--data_name "custom" \
			--root_path "../Baselines/Datasets/Exchange/" \
			--data_path "exchange_rate.csv" \
			--seq_len $seq_len \
			--pred_len $pred_len \
			--num_trials 25 \
			--num_epochs 20 \
			--seed 56 \
			--num_channels 8 \
			--first_norm_layer layer \
			--second_norm_layer $norm \
			--style_blend_gate_affine False \
			--use_multi_patching True \
			--use_rank_reduction True
		done
	done
done
