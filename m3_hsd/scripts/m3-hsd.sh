folder_name=m3-hsd

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$folder_name ]; then
    mkdir ./logs/$folder_name
fi

model_name=m3-hsd
seq_len=96

for model_name in m3-hsd
do
for pred_len in 96 192 336 720
do
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.001 \
    --ema_alpha_low 0.001 \
    --ema_alpha_mid 0.6 \
    --mid_kernel_size 7\
    --high_kernel_size 3\
    --lradj 'plateau' > logs/$folder_name/$model_name'_ETTh1_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/  \
    --data_path ETTh2.csv \
    --model_id ETTh2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTh2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --ema_alpha_low 0.001 \
    --ema_alpha_mid 0.6 \
    --mid_kernel_size 7\
    --high_kernel_size 3\
    --lradj 'plateau'> logs/$folder_name/$model_name'_ETTh2_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --model_id ETTm1_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTm1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --ema_alpha_low 0.005 \
    --ema_alpha_mid 0.6 \
    --mid_kernel_size 9\
    --high_kernel_size 3\
    --lradj 'plateau' > logs/$folder_name/$model_name'_ETTm1_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_$pred_len'_'$ma_type \
    --model $model_name \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --ema_alpha_low 0.001 \
    --ema_alpha_mid 0.6 \
    --mid_kernel_size 9\
    --high_kernel_size 3\
    --lradj 'plateau' > logs/$folder_name/$model_name'_ETTm2_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 21 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --ema_alpha_low 0.01 \
    --ema_alpha_mid 0.5 \
    --mid_kernel_size 7\
    --high_kernel_size 3\
    --lradj 'plateau'> logs/$folder_name/$model_name'_weather_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/traffic/ \
    --data_path traffic.csv \
    --model_id traffic_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 862 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 96 \
    --learning_rate 0.001 \
    --ema_alpha_low 0.001 \
    --ema_alpha_mid 0.5 \
    --mid_kernel_size 7\
    --high_kernel_size 3\
    --lradj 'plateau'> logs/$folder_name/$model_name'_traffic_'$seq_len'_'$pred_len.log 

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id electricity_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 256 \
    --learning_rate 0.001 \
    --ema_alpha_low 0.001 \
    --ema_alpha_mid 0.5 \
    --mid_kernel_size 7\
    --high_kernel_size 3\
    --lradj 'plateau'> logs/$folder_name/$model_name'_electricity_'$seq_len'_'$pred_len.log

  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id exchange_$pred_len'_'$ma_type \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0005 \
    --ema_alpha_low 0.01 \
    --ema_alpha_mid 0.5 \
    --mid_kernel_size 5\
    --high_kernel_size 3\
    --lradj 'plateau'> logs/$folder_name/$model_name'_exchange_'$seq_len'_'$pred_len.log
done
done

