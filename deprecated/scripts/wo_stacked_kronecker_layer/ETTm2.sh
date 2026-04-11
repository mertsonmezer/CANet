for seq_len in 96
do
	for pred_len in 192
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
			--use_rank_reduction False
		done
	done
done
