///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2022.2.10
// Des: temporal-spatial analysis for the monthly surface water in tibet
///////////////////////////////////////////////////////

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var tb_bound = area_tb.geometry().bounds();

// selected regions
var region_1 = ee.Geometry.Rectangle(90.48, 31.18, 91.32, 31.81)
var region_2 = ee.Geometry.Rectangle(93.07, 37.27, 94.15, 37.98)
var region_3 = ee.Geometry.Rectangle(102.24, 33.28, 102.61, 33.55)


// Monthly surface water maps
var s1_wat_01 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202001_debuf_mosaic').clip(area_tb)
var s1_wat_02 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202002_debuf_mosaic').clip(area_tb)
var s1_wat_03 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202003_debuf_mosaic').clip(area_tb)
var s1_wat_04 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202004_debuf_mosaic').clip(area_tb)
var s1_wat_05 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202005_debuf_mosaic').clip(area_tb)
var s1_wat_06 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202006_debuf_mosaic').clip(area_tb)
var s1_wat_07 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202007_debuf_mosaic').clip(area_tb)
var s1_wat_08 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202008_debuf_mosaic').clip(area_tb)
var s1_wat_09 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202009_debuf_mosaic').clip(area_tb)
var s1_wat_10 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202010_debuf_mosaic').clip(area_tb)
var s1_wat_11 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202011_debuf_mosaic').clip(area_tb)
var s1_wat_12 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202012_debuf_mosaic').clip(area_tb)
print('s1_water_01 -> ',s1_wat_01)


var s1_water_occ = s1_wat_01.select('b1')
                        .add(s1_wat_02.select('b1'))
                        .add(s1_wat_03.select('b1'))
                        .add(s1_wat_04.select('b1'))
                        .add(s1_wat_05.select('b1'))
                        .add(s1_wat_06.select('b1'))
                        .add(s1_wat_07.select('b1'))
                        .add(s1_wat_08.select('b1'))
                        .add(s1_wat_09.select('b1'))
                        .add(s1_wat_10.select('b1'))
                        .add(s1_wat_11.select('b1'))
                        .add(s1_wat_12.select('b1'))
                        .divide(ee.Image(12))
                        .rename('wat_occ')
;

print('s1_water_occ -> ',s1_water_occ)

var s1_water_occ_ma = s1_water_occ.mask(s1_water_occ.gt(0)) // mask the non-water region


var empty = ee.Image().byte();
var tb_outline = empty.paint({
      featureCollection: area_tb, color: 1, width: 2});

var region1_line = empty.paint({
      featureCollection: region_1, color: 1, width: 2});
var region2_line = empty.paint({
      featureCollection: region_2, color: 1, width: 2});
var region3_line = empty.paint({
      featureCollection: region_3, color: 1, width: 2});


var visual_wat = {
    bands: ['b1'],
    min: 0, max: 1,
    palette: ['ffffff', '0000ff']   // [white, blue]
};


// var visual_occ = {
//     bands: ['wat_occ'],
//     min: 0, max: 1,
//     palette: ['ffffff', 'ffbbbb', '0000ff']   // [white, light red ,blue]
// };

var visual_occ = {
    bands: ['wat_occ'],
    min: 0, max: 1,
    palette: ['ffffff', '90EE90', '0000ff']   // [white, light green, blue]
};


// Export.image.toDrive({
//     image: s1_water_occ,
//     description: 's1_wat_occ_200m',
//     folder: 'tibet_s1_data',
//     scale: 200,
//     maxPixels: 1e9,
//     fileFormat: 'GeoTIFF',
//     region: area_tb.geometry(),
//     });


// Map.setCenter(86.0, 32.0, 4);
// Map.addLayer(s1_wat_01, visual_wat, 's1_water_01');
Map.addLayer(s1_water_occ, visual_occ, 's1_water_occ');
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline')
Map.addLayer(region1_line, {palette: 'FF00FF'}, 'region_1')
Map.addLayer(region2_line, {palette: 'FF00FF'}, 'region_2')
Map.addLayer(region3_line, {palette: 'FF00FF'}, 'region_3')





