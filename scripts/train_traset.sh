#!/bin/bash
## author: xin luo
## creat: 2023.2.3
## des: batch training for the structured models

N=0
sign_begin='--------------------------- model training begin -------------------------'
sign_end='--------------------------- model training end -------------------------'
while [ $N -le 9 ]
# while [ $N -le 0 ]

do

  echo 'Iteration:' $N
  model_save=train_$N

  echo $sign_begin
  python scripts/trainer.py --model_type scales --model_name unet_scales_gate --dataset traset --s1_orbit as_des --model_save $model_save --num_epoch 500
  echo $sign_end

  # echo $sign_begin
  # python scripts/trainer.py --model_type scales --model_name unet_scales_gate --dataset traset --s1_orbit as --model_save $model_save --num_epoch 300
  # echo $sign_end

  # echo $sign_begin
  # python scripts/trainer.py --model_type scales --model_name unet_scales_gate --dataset traset --s1_orbit des --model_save $model_save --num_epoch 300
  # echo $sign_end

  echo $sign_begin
  python scripts/trainer.py --model_type scales --model_name unet_scales --dataset traset --s1_orbit as_des --model_save $model_save --num_epoch 500
  echo $sign_end

  echo $sign_begin 
  python scripts/trainer.py --model_type single --model_name unet --dataset traset --s1_orbit as_des --model_save $model_save --num_epoch 500
  echo $sign_end

  # echo $sign_begin
  # python scripts/trainer.py --model_type single --model_name deeplabv3plus --dataset traset --s1_orbit as_des --model_save $model_save --num_epoch 300
  # echo $sign_end

  # echo $sign_begin
  # python scripts/trainer.py --model_type single --model_name deeplabv3plus_mobilev2 --dataset traset --s1_orbit as_des --model_save $model_save --num_epoch 300
  # echo $sign_end

  # echo $sign_begin
  # python scripts/trainer.py --model_type single --model_name hrnet --dataset traset --s1_orbit as_des --model_save $model_save --num_epoch 300
  # echo $sign_end


  N=$( echo $N + 1 | bc );

done