[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7768932.svg)](https://doi.org/10.5281/zenodo.7768932)

# Tibet-Water-2020
We produce monthly surface water maps in Tibet plateau in 2020 by using deep learning method and Sentinel-1 image.

## Highlights
- We proposed a new gated multiscale ConvNet (GMNet) for surface water mapping based on Sentinel-1 image.
- The monthly surface water dynamics are captured by using the new proposed GMNet.

## Study area and data
|<p align="center">Study area</p>|<p align="center">Sentinel-1 imagery</p>|
|:--|:--| 
|<img src='figure/study_area.png' width =400, height=250>|<img src='figure/sentinel-1_imagery.png' width =400, height=250>|

## Monthly surface water dynamics 
- Monthly surface water maps
<p align="center">
<img src="figure/monthly_surface_water.png" width="90%" height="50%">
</p>

- Monthly surface water trend
<p align="center">
<img src="figure/surface_water_monthly_trend.png"  width="85%" height="50%">
</p>

## GMNet structure
<p align="center">
<img src="figure/GMNet_structure.png"  width="85%" height="50%">
</p>

## How to use the GMNet for surface water mapping?
### -- Step 1
- Enter the following commands for downloading the code files, and then configure the python and deep learning environment. The deep learning software used in this repo is [Pytorch](https://pytorch.org/).

  ~~~console
  git clone  https://github.com/xinluo2018/Tibet-Water-2020.git
  ~~~

### -- Step 2
- Download Sentinel-1 images, acending image only, descending image only, or both the ascending and descending images. if both the ascending and descending images are used for surface water mapping, the ascending and descending images should be croped to the same size.

### -- Step 3
- Add the prepared sentinel-1 image to the **_data/test-demo_** directory, modify the data name in the **_notebooks/infer_demo.ipynb_** file, then running the code file: **_notebooks/infer_demo.ipynb_** and surface water map can be generated. The users can run the **_notebooks/infer_demo.ipynb_** without any modification to learn the surface water mapping processing.
- Users also can specify surface water mapping by using the gmnet_infer.py, specifically,  
- --- funtional API (**_notebook/infer_demo.ipynb_**):
  ~~~python
  from scripts.gmnet_infer import gmnet_infer   
  wat_pred_as = gmnet_infer(s1_as, path_model_as_w, orbit='as')  ### using s1 ascending image only
  wat_pred_des = gmnet_infer(s1_des, path_model_des_w, orbit='des')  ### using s1 descending image only
  wat_pred = gmnet_infer(s1_stacked, path_model_w, orbit='as_des') ### using both ascending and descending images
  ~~~
- --- command line API (**_scripts/infer_demo.sh_**):
  ~~~console
  ### using s1 ascending image only
  python scripts/gmnet_infer.py -m path/of/model_as -img path/of/s1as -orbit as -o path/of/output_dir -s 1

  ### using s1 descending image only
  python scripts/gmnet_infer.py -m path/of/model_des -img path/of/s1des -orbit des -o path/of/output_dir -s 1

  ### using both ascending and descending images
  python scripts/gmnet_infer.py -m path/of/model_des -img path/of/s1_stacked -orbit as_des -o path/of/output_dir -s 1
  ~~~

## -- Citation
- Xin Luo, Zhongwen Hu, Lin Liu. Investigating the seasonal dynamics of surface water over the Qinghai–Tibet Plateau using Sentinel-1 imagery and a novel gated multiscale ConvNet.