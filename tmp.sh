path_model_as_w=model/trained_model/scales/unet_scales_gate/dset/as/model_1_weights.pth
path_model_des_w=model/trained_model/scales/unet_scales_gate/dset/des/model_1_weights.pth
path_model_w=model/trained_model/scales/unet_scales_gate/dset/as_des/model_1_weights.pth

path_s1_as=data/test_demo/s1as.tif
path_s1_des=data/test_demo/s1des.tif
path_s1_stacked=data/test_demo/s1_stacked.tif
path_out_dir=data/test_demo

# python scripts/gmnet_infer.py -m $path_model_w -img $path_s1_stacked -orbit as_des -o $path_out_dir -s 1
python scripts/gmnet_infer.py -m $path_model_as_w -img $path_s1_as -orbit as -o $path_out_dir -s 1
python scripts/gmnet_infer.py -m $path_model_des_w -img $path_s1_des -orbit des -o $path_out_dir -s 1
