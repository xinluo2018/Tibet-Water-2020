/////////////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.10.20
// Description: This code export the selected image to the google drive.
/////////////////////////////////////////////////////////////////

var Bands_S1 = ['VV', 'VH']
var Bands_S2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
var date = '202011'

// region and source images
// training s1 image (scene39)
var scene_region = ee.Geometry.Rectangle(99.65, 36.99, 100.11, 37.31);
var scene_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201104T111907_20201104T111932_035098_04188D_F4DC').clip(scene_region);
var scene_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201109T232732_20201109T232757_035178_041B57_C9D8').clip(scene_region);

// if s1_water accessible
var s1_water = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water_'+date+'_debuf_mosaic').clip(scene_region)


////////////////////////////////////////////////////////////////////////////////
//// training regions loading
// var s2_img = ee.Image(scene01_s2_source).clip(region).select(Bands_S2)
var s1_as = ee.Image(scene_s1as).clip(scene_region).select(Bands_S1)
var s1_des = ee.Image(scene_s1des).clip(scene_region).select(Bands_S1)
print(s1_as, s1_des)

var visualization = {
    bands: ['b1'],
    min: 0, max: 1,
    palette: ['ffffff', 'ffbbbb', '0000ff']
};

// Map.addLayer(s2_img, {bands:['B4','B3','B2'], max:3000, min:0}, 'sen2 image')
Map.addLayer(s1_as, {bands:['VV','VH','VV'], max:0, min:-30}, 's1_ascend')
Map.addLayer(s1_des, {bands:['VV','VH','VV'], max:0, min:-30}, 's1_descend')
Map.addLayer(s1_water.eq(1).selfMask(), visualization, 's1_water_'+date);


// // Export to Google Drive
// //// Note: the dimensions should be specified to the same among the three images.
// Export.image.toDrive({
//   image: s2_img,
//   description: 'scene01_s2_img',
//   folder: 'Sar_WaterExt_Data',
//   scale: 10,
//   fileFormat: 'GeoTIFF',
//   // dimensions: '3732x3534',
//   region: region
//   });

// Export.image.toDrive({
//   image: s1_water,
//   description: 'scene39_s1_water',
//   folder: 'tibet_dset',
//   scale: 10,
//   fileFormat: 'GeoTIFF',
//   // dimensions: '3732x3534',
//   region: scene_region
//   });

// Export.image.toDrive({
//   image: s1_as,
//   description: 'scene39_s1as',
//   folder: 'tibet_dset',
//   scale: 10,
//   fileFormat: 'GeoTIFF',
//   // dimensions: '3732x3534',
//   region: scene_region
//   });

// Export.image.toDrive({
//   image: s1_des,
//   description: 'scene39_s1des',
//   folder: 'tibet_dset',
//   scale: 10,
//   fileFormat: 'GeoTIFF',
//   // dimensions: '3732x3534',
//   region: scene_region
//   });
