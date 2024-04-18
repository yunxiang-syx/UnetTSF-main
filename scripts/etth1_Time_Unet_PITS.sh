seq_len=336
model_name=Time_Unet_PITS

root_path_name=../dataset/ETT/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

random_seed=2023
for pred_len in 96 192 336 720
do
    python ../run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --c_in 7 \
      --d_model 256 \
      --patch_len 24 \
      --stride 12 \
      --train_epochs 100 \
      --individual 1 \
      --shared_embedding 0 \
      --decomposition 1 \
      --d_model 128 \
      --head_dropout 0.3 \
      --padding_patch 'end' \
      --revin 1 \
      --affine 1 \
      --subtract_last 0 \
      --kernel_size 25 \
      --itr 1 --batch_size 256 --learning_rate 0.0001
done