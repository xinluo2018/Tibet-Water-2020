#!/bin/bash
## author: xin luo
## create: 2022.5.10
## des: Surface water mapping for the tibet region by using the trained model.

cd /home/yons/Desktop/developer-luo/Tibet-Water-2020

## ----- 1. Models
## surface water mapping using deep learning models
model=model/trained_model/scales/unet_scales_gate/dset/as_des/model_1_weights.pth
model_as=model/trained_model/scales/unet_scales_gate/dset/as/model_1_weights.pth
model_des=model/trained_model/scales/unet_scales_gate/dset/des/model_1_weights.pth

dates='202001 202002 202003 202004 202005 202006 202007 202008 202009 202010 202011 202012'

for date in $dates
do
  echo 'Date:' $date
  # ----- 2. Data
  # --- ascending data
  as_path=/myDrive/tibet-water-data/tibet-$date/s1-ascend/*.tif
  des_path=/WD-myBook/tibet-water-data/tibet-$date/s1-descend/*.tif
  echo 'as_path:' $as_path
  echo 'des_path:' $des_path

  ## ----- 3. Run the model, !!should be modified
  python script/tibet_gmnet_infer.py -as $as_path -des $des_path -s 100 -m $model -m_as $model_as -m_des $model_des

done


