//////////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2022.3.10
// Description: This code help us to select the training scenes (s1 ascending and descending
//              pair image) with assistant of s2 image
//////////////////////////////////////////////////////////////


var start_time = '2020-11-01';
var end_time = '2020-11-30';

var scene_region = ee.Geometry.Rectangle(100.09, 36.97, 100.11, 36.98);
print('scene region area:', scene_region.area())

/// Sentinel-2 image for reference
var Bands_S2 = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
// Note: before 2019, sen2 image should be searched with the TOA data.
var s2_col = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(scene_region)
                  .filterDate(start_time, end_time)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40))
                  .sort('CLOUDY_PIXEL_PERCENTAGE');

var s2_img = s2_col.mosaic().clip(scene_region)



////////////////////////////////////////////////////////////////////
//// ----- Sentinel-1 images (ascending and descending obit).
////////////////////////////////////////////////////////////////////
//// Ensure the sen2 image fully contain the training region.
// var condition = function(image){
//     var footprint = ee.Geometry(image.get('system:footprint'))
//     var condition = ee.Geometry.Polygon(footprint.coordinates()).contains(scene_region)
//     return ee.Algorithms.If(condition, 
//                             image.set('data', 'true'), 
//                             ee.Image([]).set('data', 'false'))};

// var Bands_S1 = ['VV', 'VH']
// var ascendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
//     .filter(ee.Filter.eq('instrumentMode', 'IW'))
//     .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
//     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
//     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
//     .filterBounds(scene_region)
//     .filterDate(start_time, end_time);
// var ascendCol_Sel = ascendCol.map(condition).filterMetadata('data', 'equals','true')

// var descendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
//     .filter(ee.Filter.eq('instrumentMode', 'IW'))
//     .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
//     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
//     .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
//     .filterBounds(scene_region)
//     .filterDate(start_time, end_time);
// var descendCol_Sel = descendCol.map(condition).filterMetadata('data', 'equals','true')

// var scene_s1as = ascendCol_Sel.first().select(Bands_S1).float()
// var scene_s1des = descendCol_Sel.first().select(Bands_S1).float()

var scene_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201104T111907_20201104T111932_035098_04188D_F4DC')
var scene_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201109T232732_20201109T232757_035178_041B57_C9D8')

var empty = ee.Image().byte();
//// outline visualization of the study area.
var scene_outline = empty.paint({
    featureCollection: scene_region, color: 1, width: 3});

var visualization = {
    bands: ['b1'],
    min: 0, max: 1,
    palette: ['ffffff', 'ffbbbb', '0000ff']
};

Map.addLayer(scene_s1as, {bands:['VV','VH','VV'], max:10, min:-30}, 's1_ascend')
Map.addLayer(scene_s1des, {bands:['VV','VH','VV'], max:10, min:-30}, 's1_descend')
// Map.addLayer(s2_img, {bands: ['B4', 'B3', 'B2'], min:0, max:5000}, 's2 image')
Map.addLayer(scene_outline, {palette: 'FF0000'}, 'training region')


