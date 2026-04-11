PYTHONPATH=/home/pc-amax-2/Desktop/Mert/Mert-recovered/Mert/Net_final

for data_name in custom; do
  for pred_len in 24 36 48 60; do
    python forecasting/robustness_test.py \
      --data_name $data_name \
      --data_path illness.csv \
      --seq_len 36 \
      --num_channels 7 \
      --pred_len $pred_len \
      --custom_seeds 56,0,2,42,2025 \
      --root_path datasets/ILI \
      --database_path "results/ILI/custom-SeqLen36-PredLen$pred_len-FirstNormlayer-SecondNormadain-MPTrue-RankReductionTrue.db"
  done
done