#!/bin/bash
## author: xin luo
## create: 2022.4.9; modify: 2023.2.3
## des: surface water inference and accurace evaluation for the test sites by using the trained model.


cd /home/xin/Developer-luo/Monthly-Surface-Water-in-Tibet

# IDS_MODEL='0 1 2 3 4 5 6 7 8 9'
IDS_MODEL='1'

for ID_MODEL in $IDS_MODEL

do 
  ## ----- 1. Configure models
  ## surface water mapping using deep learning models
  echo 'Model ID: ' $ID_MODEL

  VALSITES='01 02 03 04 05 06 07'

  for I_VAL in $VALSITES
  do
    echo 'number of validation site: ' $I_VAL
    path_valsite_as=data/dset/s1_ascend_clean/scene${I_VAL}_s1as.tif 
    path_valsite_des=data/dset/s1_descend_clean/scene${I_VAL}_s1des.tif

    # ## ---- unet_scales_gate with as_des input  
    # path_model_weights=model/trained_model/scales/unet_scales_gate/traset/as_des/train_${ID_MODEL}_weights.pth
    # dir_result_save=data/dset/valsite_wat_infer/unet_scales_gate/as_des/train_$ID_MODEL
    # python scripts/tibet_gmnet_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $path_model_weights -o $dir_result_save

    # # ## ---- unet_scales_gate with as input
    # path_model_weights=model/trained_model/scales/unet_scales_gate/traset/as/train_${ID_MODEL}_weights.pth
    # dir_result_save=data/dset/valsite_wat_infer/unet_scales_gate/as/train_$ID_MODEL
    # python scripts/gmnet_infer.py -as $path_valsite_as -s 1 -m_as $path_model_weights -o $dir_result_save

    # # ## ---- unet_gscales_gate with des input
    # path_model_weights=model/trained_model/scales/unet_scales_gate/traset/des/train_${ID_MODEL}_weights.pth
    # dir_result_save=data/dset/valsite_wat_infer/unet_scales_gate/des/train_$ID_MODEL
    # python scripts/gmnet_infer.py -des $path_valsite_des -s 1 -m_des $path_model_weights -o $dir_result_save

    ## ---- unet_scales with as_des input
    path_model_weights=model/trained_model/scales/unet_scales/traset/as_des/train_${ID_MODEL}_weights.pth
    dir_result_save=data/dset/valsite_wat_infer/unet_scales/as_des/train_$ID_MODEL
    python scripts/baseline_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $path_model_weights -o $dir_result_save

    # # ---- unet with as_des input
    # path_model_weights=model/trained_model/single/unet/traset/as_des/train_${ID_MODEL}_weights.pth
    # dir_result_save=data/dset/valsite_wat_infer/unet/as_des/train_$ID_MODEL
    # python scripts/baseline_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $path_model_weights -o $dir_result_save

    # ## ---- deeplabv3plus with as_des input
    # path_model_weights=model/trained_model/single/deeplabv3plus/traset/as_des/train_${ID_MODEL}_weights.pth
    # dir_result_save=data/dset/valsite_wat_infer/deeplabv3plus/as_des/train_$ID_MODEL
    # python scripts/baseline_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $path_model_weights -o $dir_result_save

    # ## ---- deeplabv3plus_mobilev2 with as_des input
    # path_model_weights=model/trained_model/single/deeplabv3plus_mobilev2/traset/as_des/train_${ID_MODEL}_weights.pth
    # dir_result_save=data/dset/valsite_wat_infer/deeplabv3plus_mobilev2/as_des/train_$ID_MODEL
    # python scripts/baseline_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $path_model_weights -o $dir_result_save

    # ## ---- hrnet with as_des input
    # path_model_weights=model/trained_model/single/hrnet/traset/as_des/train_${ID_MODEL}_weights.pth
    # dir_result_save=data/dset/valsite_wat_infer/hrnet/as_des/train_$ID_MODEL
    # python scripts/baseline_infer.py -as $path_valsite_as -des $path_valsite_des -s 1 -m $path_model_weights -o $dir_result_save

  done

done
