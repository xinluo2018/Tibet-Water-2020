///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2022.4.21
// Des: variation analysis for the specific sub-region.
//      the sub-regions are visualized in the submitted paper.
///////////////////////////////////////////////////////

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var tb_bound = area_tb.geometry().bounds();
// hydro_basins  
var hydro_basins_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBound_HF_basins_Vorosmarty2010_subs');
print(hydro_basins_tb)

// tile-based statistic
var tb_tiles_wgs84 = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_tiles_wgs84');
print(tb_tiles_wgs84)


// selected region 1
var region_1 = ee.Geometry.Rectangle(87.4969, 29.9997, 91.4965, 31.9983)  // region_1: small variation
var region_2 = ee.Geometry.Rectangle(99.4962,  34.9973, 101.485, 37.998)  // region_2: large variation
var region_3 = ee.Geometry.Rectangle(97.518, 30.021, 100.485, 32.02) // region_3: large variation


// date of the s2/landsat8 data
var start_wet_season = '2020-07-01';
var end_wet_season = '2020-10-31';  // for optical l8 and s2 data
var start_dry_season = '2020-02-01';
var end_dry_season = '2020-04-29';  // for optical l8 and s2 data


/// Landsat 8 images
var Bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
var l8_col = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
      .filterDate('2020-01-01', '2020-12-31')
      // .filter(ee.Filter.lt('CLOUD_COVER_LAND', 10))
      
// region 1      
var l8_1_wet = l8_col.filterDate(start_wet_season, end_wet_season)
                      .filter(ee.Filter.lt('CLOUD_COVER_LAND', 20))
                      .mosaic().clip(region_1);
var l8_1_dry = l8_col.filterDate(start_dry_season, end_dry_season)
                      .filter(ee.Filter.lt('CLOUD_COVER_LAND', 20))
                      .mosaic().clip(region_1);
// region 2
var l8_2_dry = l8_col.filterDate(start_dry_season, end_dry_season)
                      .filter(ee.Filter.lt('CLOUD_COVER_LAND', 20))
                      .mosaic().clip(region_2);
// region 3
var l8_3_dry = l8_col.filterDate(start_dry_season, end_dry_season)
                      .filter(ee.Filter.lt('CLOUD_COVER_LAND', 50))
                      .mosaic().clip(region_3);

print('l8_2_wet --> ', l8_1_wet)

/// Sentinel-2 image
var sen2_col = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .filterDate('2020-01-01', '2020-12-31')

// region 2
var s2_2_wet = sen2_col.filterDate(start_wet_season, end_wet_season)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                  .mosaic().clip(region_2)

// region 3
var s2_3_wet = sen2_col.filterDate(start_wet_season, end_wet_season)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
                  .mosaic().clip(region_3)
var s2_3_dry = sen2_col.filterDate(start_dry_season, end_dry_season)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
                  .mosaic().clip(region_3)


print('sen2_collection_1 --> ', l8_1_wet)

var empty = ee.Image().byte();
var tb_outline = empty.paint({
      featureCollection: area_tb, color: 1, width: 3});
var tb_basins_outline = empty.paint({
      featureCollection: hydro_basins_tb, color: 1, width: 3});
var tiles_outline = empty.paint({
      featureCollection: tb_tiles_wgs84, color: 1, width: 3});

//// region 1
Map.addLayer(l8_1_wet, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'l8_1_wet');
Map.addLayer(l8_1_dry, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'l8_1_dry');

//// region 2
Map.addLayer(s2_2_wet, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 's2_2_wet');
Map.addLayer(l8_2_dry, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'l8_2_dry');

//// region 3
Map.addLayer(s2_3_wet, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 's2_3_wet');
Map.addLayer(l8_3_dry, {bands: ['B5', 'B4', 'B3'], min:0, max:3000}, 'l8_3_dry');
Map.addLayer(s2_3_dry, {bands: ['B8', 'B4', 'B3'], min: 0, max: 3000}, 's2_3_dry');


Map.setCenter(89.0, 32.0, 4);
// Map.addLayer(tiles_outline, {palette: '0000FF'}, 'Tiles_outline')
Map.addLayer(tb_basins_outline, {palette: '32CD32'}, 'Tibet_basins_outline')
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline')

