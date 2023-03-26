//////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2022.2.26
// Des: check the monthly as/des image and surface water maps in tibet
///////////////////////////////////////////////////////
/// hardly water bodies: 202003_087; 202003_114; 202004_189; 202004_142


var bands_s1 = ['VV', 'VH']
var scale_DN = 100          // decrease the export size of image
var start_time = '2020-07-01'   // for s2/l8 images
var end_time = '2020-07-30'
var date='202007'     // for s1 images

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var tb_bound = area_tb.geometry().bounds();


// ********************************************************
///////////////////// tiles and image_id load /////////////
// ********************************************************
var tb_tiles = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tb_tiles');
var tb_tiles_buf = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tb_tiles_buf');
var s1_imgid_as = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date)
var s1_imgid_des = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date)
var s1_water = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+date+'_debuf_mosaic')
var s1_water_2 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+date+'_debuf_mosaic_2')

var tb_tiles_buf = tb_tiles_buf.sort('tile_id')
print('tb_tiles_buf:', tb_tiles_buf)
print('s1_imgid_as:', s1_imgid_as)
print('s1_imgid_des:', s1_imgid_des)
print('s1_water -- >', s1_water)

// ********************************************************
//////////////////////// Data collected //////////////////////
// ********************************************************
// --- 1. s1 images
function get_id(fea, ls){
    var id_ls = ee.List(ls);
    return id_ls.add(fea.get('img_id'))
    }
// collection s1 images: ascending and descending
var s1_id_as = s1_imgid_as.iterate(get_id, [])
var s1_imgs_as = ee.ImageCollection(s1_id_as.getInfo()).select(bands_s1)
var s1_id_des = s1_imgid_des.iterate(get_id, [])
var s1_imgs_des = ee.ImageCollection(s1_id_des.getInfo()).select(bands_s1)
print('s1_imgs_as -- >', s1_imgs_as)

// var s1_img_as = s1_imgs_as.mosaic().float()   // mosaic imgs into img
// var s1_img_des = s1_imgs_des.mosaic().float()   
// print('s1_img_as_mosaic -- >', s1_img_as)

//  --- 2. s2/l8 images
/// Landsat 8 images
var Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
var l8_col = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
  .filter(ee.Filter.lt('CLOUD_COVER_LAND', 20))
  .filterBounds(area_tb)
  .filterDate(start_time, end_time);
print('l8_collection --> ', l8_col)

/// Sentinel-2 image
var sen2_col = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(area_tb)
                  .filterDate(start_time, end_time)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',20));
print('sen2_collection --> ', sen2_col)


var empty = ee.Image().byte();
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 3});

var visual = {
    bands: ['b1'],
    min: 0, max: 1,
    palette: ['ffffff', 'ffbbbb', '0000ff']
};

var scene_region = ee.Geometry.Rectangle(88.88, 35.18, 89.30, 35.48);   
print('scene region area:', scene_region.area())
var empty = ee.Image().byte();
//// outline visualization of the study area.
var scene_outline = empty.paint({
    featureCollection: scene_region, color: 1, width: 3});



// Map.addLayer(sen2_col, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'sen2');
// Map.addLayer(l8_col, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'Landsat 8')
Map.addLayer(s1_imgs_as, {bands:['VV','VH','VV'], min: -30, max: 1}, 'as')
Map.addLayer(s1_imgs_des, {bands:['VV','VH','VV'], min: -30, max: 1}, 'des')
Map.addLayer(s1_water.eq(1).selfMask(), visual, 's1_water_'+date);
Map.addLayer(s1_water_2.eq(1).selfMask(), visual, 's1_water_2_'+date);
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline')
Map.addLayer(tb_tiles, {}, 'tb_tiles')
// Map.addLayer(scene_outline, {palette: 'FF0000'}, 'training region')




