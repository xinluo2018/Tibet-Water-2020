///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2022.2.10
// Des: temporal-spatial analysis for the monthly surface water in tibet
///////////////////////////////////////////////////////

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var tb_bound = area_tb.geometry().bounds();


// selected region 1
// var region = ee.Geometry.Rectangle(90.48, 31.18, 91.32, 31.81)  // region_1
// var region = ee.Geometry.Rectangle(93.07, 37.27, 94.15, 37.98)  // region_2
var region = ee.Geometry.Rectangle(102.24, 33.28, 102.61, 33.55) // region 3


// date of the s2/landsat8 data
var start_time_01 = '2020-01-01'; 
var end_time_01 = '2020-01-30';  // for optical l8 and s2 data
var date_01 = '202001'           // for the sentinel-1 data

var start_time_02 = '2020-02-01'; 
var end_time_02 = '2020-02-28';  // for optical l8 and s2 data
var date_02 = '202002'           // for the sentinel-1 data

var start_time_03 = '2020-03-01'; 
var end_time_03 = '2020-03-31';  // for optical l8 and s2 data
var date_03 = '202003'           // for the sentinel-1 data

var start_time_04 = '2020-04-01'; 
var end_time_04 = '2020-04-30';  // for optical l8 and s2 data
var date_04 = '202004'           // for the sentinel-1 data

var start_time_05 = '2020-05-01'; 
var end_time_05 = '2020-05-31';  // for optical l8 and s2 data
var date_05 = '202005'           // for the sentinel-1 data

var start_time_06 = '2020-06-01'; 
var end_time_06 = '2020-06-30';  // for optical l8 and s2 data
var date_06 = '202006'           // for the sentinel-1 data

var start_time_07 = '2020-07-01'; 
var end_time_07 = '2020-07-31';  // for optical l8 and s2 data
var date_07 = '202007'           // for the sentinel-1 data

var start_time_08 = '2020-08-01'; 
var end_time_08 = '2020-08-31';  // for optical l8 and s2 data
var date_08 = '202008'           // for the sentinel-1 data

var start_time_09 = '2020-09-01'; 
var end_time_09 = '2020-09-30';  // for optical l8 and s2 data
var date_09 = '202009'           // for the sentinel-1 data

var start_time_10 = '2020-10-01'; 
var end_time_10 = '2020-10-31';  // for optical l8 and s2 data
var date_10 = '202010'           // for the sentinel-1 data

var start_time_11 = '2020-11-01'; 
var end_time_11 = '2020-11-30';  // for optical l8 and s2 data
var date_11 = '202011'           // for the sentinel-1 data

var start_time_12 = '2020-12-01'; 
var end_time_12 = '2020-12-31';  // for optical l8 and s2 data
var date_12 = '202012'           // for the sentinel-1 data

// sentinel-1 image
var bands_s1 = ['VV', 'VH']
// --- ascending
var s1_imgid_as_01 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_01)
var s1_imgid_as_02 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_02)
var s1_imgid_as_03 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_03)
var s1_imgid_as_04 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_04)
var s1_imgid_as_05 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_05)
var s1_imgid_as_06 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_06)
var s1_imgid_as_07 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_07)
var s1_imgid_as_08 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_08)
var s1_imgid_as_09 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_09)
var s1_imgid_as_10 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_10)
var s1_imgid_as_11 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_11)
var s1_imgid_as_12 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_as_'+date_12)
// ---- descending
var s1_imgid_des_01 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_01)
var s1_imgid_des_02 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_02)
var s1_imgid_des_03 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_03)
var s1_imgid_des_04 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_04)
var s1_imgid_des_05 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_05)
var s1_imgid_des_06 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_06)
var s1_imgid_des_07 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_07)
var s1_imgid_des_08 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_08)
var s1_imgid_des_09 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_09)
var s1_imgid_des_10 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_10)
var s1_imgid_des_11 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_11)
var s1_imgid_des_12 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_data/tibet_des_'+date_12)



// collection s1 images: ascending and descending
// --- 1. obtain s1 images
function get_id(fea, ls){
    var id_ls = ee.List(ls);
    return id_ls.add(fea.get('img_id'))
    }

var s1_imgs_as_01 = ee.ImageCollection(s1_imgid_as_01.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_02 = ee.ImageCollection(s1_imgid_as_02.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_03 = ee.ImageCollection(s1_imgid_as_03.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_04 = ee.ImageCollection(s1_imgid_as_04.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_05 = ee.ImageCollection(s1_imgid_as_05.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_06 = ee.ImageCollection(s1_imgid_as_06.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_07 = ee.ImageCollection(s1_imgid_as_07.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_08 = ee.ImageCollection(s1_imgid_as_08.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_09 = ee.ImageCollection(s1_imgid_as_09.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_10 = ee.ImageCollection(s1_imgid_as_10.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_11 = ee.ImageCollection(s1_imgid_as_11.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_as_12 = ee.ImageCollection(s1_imgid_as_12.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)

var s1_imgs_des_01 = ee.ImageCollection(s1_imgid_des_01.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_02 = ee.ImageCollection(s1_imgid_des_02.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_03 = ee.ImageCollection(s1_imgid_des_03.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_04 = ee.ImageCollection(s1_imgid_des_04.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_05 = ee.ImageCollection(s1_imgid_des_05.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_06 = ee.ImageCollection(s1_imgid_des_06.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_07 = ee.ImageCollection(s1_imgid_des_07.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_08 = ee.ImageCollection(s1_imgid_des_08.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_09 = ee.ImageCollection(s1_imgid_des_09.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_10 = ee.ImageCollection(s1_imgid_des_10.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_11 = ee.ImageCollection(s1_imgid_des_11.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)
var s1_imgs_des_12 = ee.ImageCollection(s1_imgid_des_12.iterate(get_id, []).getInfo()).select(bands_s1).mosaic().clip(region)

print('s1_imgs_as_01 -- >', s1_imgs_as_01)


/// Landsat 8 images
var Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
var l8_col = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
      .filterBounds(region)
      .filterDate('2020-01-01', '2020-12-31')
      .filter(ee.Filter.lt('CLOUD_COVER_LAND', 30))
      
var l8_col_01 = l8_col.filterDate(start_time_01, end_time_01).mosaic().clip(region);
var l8_col_02 = l8_col.filterDate(start_time_02, end_time_02).mosaic().clip(region);
var l8_col_03 = l8_col.filterDate(start_time_03, end_time_03).mosaic().clip(region);
var l8_col_04 = l8_col.filterDate(start_time_04, end_time_04).mosaic().clip(region);
var l8_col_05 = l8_col.filterDate(start_time_05, end_time_05).mosaic().clip(region);
var l8_col_06 = l8_col.filterDate(start_time_06, end_time_06).mosaic().clip(region);
var l8_col_07 = l8_col.filterDate(start_time_07, end_time_07).mosaic().clip(region);
var l8_col_08 = l8_col.filterDate(start_time_08, end_time_08).mosaic().clip(region);
var l8_col_09 = l8_col.filterDate(start_time_09, end_time_09).mosaic().clip(region);
var l8_col_10 = l8_col.filterDate(start_time_10, end_time_10).mosaic().clip(region);
var l8_col_11 = l8_col.filterDate(start_time_11, end_time_11).mosaic().clip(region);
var l8_col_12 = l8_col.filterDate(start_time_12, end_time_12).mosaic().clip(region);
print('l8_collection_01 --> ', l8_col_01)

/// Sentinel-2 image
var sen2_col = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(region)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .filterDate('2020-01-01', '2020-12-31')


var sen2_col_01 = sen2_col.filterDate(start_time_01, end_time_01).mosaic().clip(region)
var sen2_col_02 = sen2_col.filterDate(start_time_02, end_time_02).mosaic().clip(region)
var sen2_col_03 = sen2_col.filterDate(start_time_03, end_time_03).mosaic().clip(region)
var sen2_col_04 = sen2_col.filterDate(start_time_04, end_time_04).mosaic().clip(region)
var sen2_col_05 = sen2_col.filterDate(start_time_05, end_time_05).mosaic().clip(region)
var sen2_col_06 = sen2_col.filterDate(start_time_06, end_time_06).mosaic().clip(region)
var sen2_col_07 = sen2_col.filterDate(start_time_07, end_time_07).mosaic().clip(region)
var sen2_col_08 = sen2_col.filterDate(start_time_08, end_time_08).mosaic().clip(region)
var sen2_col_09 = sen2_col.filterDate(start_time_09, end_time_09).mosaic().clip(region)
var sen2_col_10 = sen2_col.filterDate(start_time_10, end_time_10).mosaic().clip(region)
var sen2_col_11 = sen2_col.filterDate(start_time_11, end_time_11).mosaic().clip(region)
var sen2_col_12 = sen2_col.filterDate(start_time_12, end_time_12).mosaic().clip(region)
print('sen2_collection_01 --> ', sen2_col_01)


// Monthly surface water maps
var s1_wat_01 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_01 + '_debuf_mosaic').clip(region)
var s1_wat_02 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_02 + '_debuf_mosaic').clip(region)
var s1_wat_03 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_03 + '_debuf_mosaic').clip(region)
var s1_wat_04 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_04 + '_debuf_mosaic').clip(region)
var s1_wat_05 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_05 + '_debuf_mosaic').clip(region)
var s1_wat_06 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_06 + '_debuf_mosaic').clip(region)
var s1_wat_07 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_07 + '_debuf_mosaic').clip(region)
var s1_wat_08 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_08 + '_debuf_mosaic').clip(region)
var s1_wat_09 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_09 + '_debuf_mosaic').clip(region)
var s1_wat_10 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_10 + '_debuf_mosaic').clip(region)
var s1_wat_11 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_11 + '_debuf_mosaic').clip(region)
var s1_wat_12 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_'+ date_12 + '_debuf_mosaic').clip(region)

print('s1_water_01 -> ',s1_wat_01)


var empty = ee.Image().byte();
var tb_outline = empty.paint({
      featureCollection: area_tb, color: 1, width: 3});

var visual_wat = {
    bands: ['b1'],
    min: 0, max: 1,
    palette: ['ffffff', '0000ff']   // [white, light red ,blue]
};


// Map.setCenter(86.0, 32.0, 4);
// // 1. l8/s2 image
// Map.addLayer(sen2_col_01, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'sen2_01');
// Map.addLayer(sen2_col_02, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'sen2_02');
// Map.addLayer(sen2_col_03, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_03');
// Map.addLayer(sen2_col_04, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_04');
// Map.addLayer(sen2_col_05, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_05');
// Map.addLayer(sen2_col_06, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_06');
// Map.addLayer(sen2_col_07, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'sen2_07');
// Map.addLayer(sen2_col_08, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_08');
// Map.addLayer(sen2_col_09, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_09');
// Map.addLayer(sen2_col_10, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_10');
// Map.addLayer(sen2_col_11, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 'sen2_11');
// Map.addLayer(sen2_col_12, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'sen2_12');
// Map.addLayer(l8_col_01, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'landsat8_01');
Map.addLayer(l8_col_02, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'landsat8_02');
// Map.addLayer(l8_col_03, {bands: ['B5', 'B4', 'B3'], min:0, max:8000}, 'landsat8_03');
// Map.addLayer(l8_col_04, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_04');
// Map.addLayer(l8_col_05, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_05');
// Map.addLayer(l8_col_06, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_06');
Map.addLayer(l8_col_07, {bands: ['B4', 'B3', 'B2'], min:0, max:2000}, 'landsat8_07');
// Map.addLayer(l8_col_08, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_08');
// Map.addLayer(l8_col_09, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_09');
// Map.addLayer(l8_col_10, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_10');
// Map.addLayer(l8_col_11, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_11');
// Map.addLayer(l8_col_12, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'landsat8_12');


// 2. s1 image
Map.addLayer(s1_imgs_as_01, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_01');
Map.addLayer(s1_imgs_as_02, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_02');
Map.addLayer(s1_imgs_as_03, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_03');
Map.addLayer(s1_imgs_as_04, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_04');
Map.addLayer(s1_imgs_as_05, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_05');
Map.addLayer(s1_imgs_as_06, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_06');
Map.addLayer(s1_imgs_as_07, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_07');
Map.addLayer(s1_imgs_as_08, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_08');
Map.addLayer(s1_imgs_as_09, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_09');
Map.addLayer(s1_imgs_as_10, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_10');
Map.addLayer(s1_imgs_as_11, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_11');
Map.addLayer(s1_imgs_as_12, {bands: ['VV', 'VH', 'VV'], min: -50, max: 1}, 'ascendVV_12');

// Map.addLayer(s1_imgs_des_01, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_01');
// Map.addLayer(s1_imgs_des_02, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_02');
// Map.addLayer(s1_imgs_des_03, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_03');
// Map.addLayer(s1_imgs_des_04, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_04');
// Map.addLayer(s1_imgs_des_05, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_05');
// Map.addLayer(s1_imgs_des_06, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_06');
// Map.addLayer(s1_imgs_des_07, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_07');
// Map.addLayer(s1_imgs_des_08, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_08');
// Map.addLayer(s1_imgs_des_09, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_09');
// Map.addLayer(s1_imgs_des_10, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_10');
// Map.addLayer(s1_imgs_des_11, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_11');
// Map.addLayer(s1_imgs_des_12, {bands:['VV', 'VH', 'VV'], min: -50, max: 1}, 'descendVV_12');

// 3. surface water map
// Map.addLayer(s1_wat_01.eq(1).selfMask(), visual_wat, 's1_water_01');
// Map.addLayer(s1_wat_01, visual_wat, 's1_water_01');
// Map.addLayer(s1_wat_02, visual_wat, 's1_water_02');
// Map.addLayer(s1_wat_03, visual_wat, 's1_water_03');
// Map.addLayer(s1_wat_04, visual_wat, 's1_water_04');
// Map.addLayer(s1_wat_05, visual_wat, 's1_water_05');
// Map.addLayer(s1_wat_06, visual_wat, 's1_water_06');
// Map.addLayer(s1_wat_07, visual_wat, 's1_water_07');
// Map.addLayer(s1_wat_08, visual_wat, 's1_water_08');
// Map.addLayer(s1_wat_09, visual_wat, 's1_water_09');
// Map.addLayer(s1_wat_10, visual_wat, 's1_water_10');
// Map.addLayer(s1_wat_11, visual_wat, 's1_water_11');
// Map.addLayer(s1_wat_12, visual_wat, 's1_water_12');

Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline')




