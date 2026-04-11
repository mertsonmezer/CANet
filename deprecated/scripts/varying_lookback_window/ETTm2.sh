# Hyperparameter Search Space
# LOW_LR, HIGH_LR = 1e-4, 1e-2
# LOW_DROPOUT, HIGH_DROPOUT = 0.3, 0.5
# BATCH_SIZES: List[int] = [128]
# PATCH_SIZES: List[List[int]] = [[16, 64]]
# EMBED_DIMS: List[int] = [64]

for seq_len in 24 48 96 192 336 720
do
	for pred_len in 96
	do
		for norm in "adain"
		do
			python3 -m forecasting.train_tune \
			--data_name "ETTm2" \
			--root_path "ETT-small/" \
			--data_path "ETTm2.csv" \
			--seq_len $seq_len \
			--pred_len $pred_len \
			--num_trials 50 \
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
