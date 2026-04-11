PYTHONPATH=/home/pc-amax-2/Desktop/Mert/Mert-recovered/Mert/Net_final

for data_name in custom; do
  for pred_len in 96 192 336 720; do
    python forecasting/robustness_test.py \
      --data_name $data_name \
      --data_path electricity.csv \
      --seq_len 96 \
      --num_channels 321 \
      --pred_len $pred_len \
      --custom_seeds 56,0,2,42,2025 \
      --root_path datasets/Electricity \
      --database_path "results/Electricity/custom-SeqLen96-PredLen$pred_len-FirstNormlayer-SecondNormadain-MPTrue-RankReductionTrue.db"
  done
done