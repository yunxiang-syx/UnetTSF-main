model_name=Time_Unet_TimeMixer
#export CUDA_VISIBLE_DEVICES=0,1,2,3
for pred_len in 96 192 336 720
do
seq_len=336

python -u ../run_longExp.py \
    --is_training 1 \
    --root_path ../dataset/ETT \
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 7 \
    --des 'Exp' \
    --stage_num 3 \
    --stage_pool_kernel 3 \
    --stage_pool_padding 0 \
    --c_in 7 \
    --c_out 7 \
    --patch_len 24 \
    --stride 12 \
    --train_epochs 100 \
    --individual 1 \
    --shared_embedding 0 \
    --d_model 128 \
    --head_dropout 0.3 \
    --padding_patch 'end' \
    --revin 1 \
    --bifpn_features 64 \
    --bifpn_numlayers 3 \
    --affine 1 \
    --subtract_last 0 \
    --kernel_size 25 \
    --equalizer 'transformer' \
    --prior 'mag' \
    --task_name 'long_term_forecast' \
    --d_ff 32 \
    --e_layers 2 \
    --down_sampling_layers 3 \
    --down_sampling_window 2 \
    --down_sampling_method 'avg' \
    --channel_independence 1 \
    --moving_avg 25 \
    --dropout 0.1 \
    --use_norm 1 \
    --decomp_method 'moving_avg' \
    --embed 'timeF' \
    --freq 'h' \
    --features 'M' \
    --itr 1 --batch_size 256 --learning_rate 0.01
done