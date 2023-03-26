#!/bin/bash
## author: xin luo
## create: 2022.5.11
## des:train the model based on the whole dataset

N=1
sign_begin='--------------------------- model training begin -------------------------'
sign_end='--------------------------- model training end -------------------------'
while [ $N -le 1 ]

do

  echo 'Iteration:' $N
  model_save=model_$N

  # echo $sign_begin
  # python scripts/trainer.py --model_type scales --model_name unet_scales_gate --dataset dset --s1_orbit as_des --model_save $model_save --num_epoch 500
  # echo $sign_end

  echo $sign_begin
  python scripts/trainer.py --model_type scales --model_name unet_scales_gate --dataset dset --s1_orbit as --model_save $model_save --num_epoch 500
  echo $sign_end

  # echo $sign_begin
  # python scripts/trainer.py --model_type scales --model_name unet_scales_gate --dataset dset --s1_orbit des --model_name $model_save --num_epoch 500
  # echo $sign_end


  N=$( echo $N + 1 | bc )

done


