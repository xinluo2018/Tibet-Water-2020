///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2022.2.10
// Des: temporal-spatial analysis for the monthly surface water in tibet
///////////////////////////////////////////////////////


//// define projection
var wkt = 'PROJCS["Asia_North_Albers_Equal_Area_Conic",\
            GEOGCS["GCS_WGS_1984",DATUM["WGS_1984",\
                SPHEROID["WGS_1984",6378137,298.257223563]],\
              PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]],\
            PROJECTION["Albers_Conic_Equal_Area"],\
            PARAMETER["False_Easting",0],\
            PARAMETER["False_Northing",0],\
            PARAMETER["longitude_of_center",95],\
            PARAMETER["Standard_Parallel_1",15],\
            PARAMETER["Standard_Parallel_2",65],\
            PARAMETER["latitude_of_center",30],\
            UNIT["Meter",1],\
            AUTHORITY["EPSG","102025"]]'

var proj = ee.Projection(wkt)


// Study area
var tb_boundary = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var tb_bound = tb_boundary.geometry().bounds();
var tb_area = tb_boundary.geometry().area({'maxError':1, 'proj': proj}) // !should set the area-equal projection
var tb_area_km = ee.Number(tb_area).divide(1e6).round()
print('tibet area (km2):', tb_area_km)

// tiles file
var tb_tiles = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tibet_tiles_wgs84');
print('tb_tiles: ', tb_tiles)

// basin file
var tb_basins = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBound_HF_basins_Vorosmarty2010_subs_');
print('tb_basins: ', tb_basins)


// Layerstacking of monthly surface water maps
var tb_water_01 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202001_debuf_mosaic').clip(tb_boundary).rename('tb_water_01')
var tb_water_02 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202002_debuf_mosaic').clip(tb_boundary).rename('tb_water_02')
var tb_water_03 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202003_debuf_mosaic').clip(tb_boundary).rename('tb_water_03')
var tb_water_04 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202004_debuf_mosaic').clip(tb_boundary).rename('tb_water_04')
var tb_water_05 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202005_debuf_mosaic').clip(tb_boundary).rename('tb_water_05')
var tb_water_06 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202006_debuf_mosaic').clip(tb_boundary).rename('tb_water_06')
var tb_water_07 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202007_debuf_mosaic').clip(tb_boundary).rename('tb_water_07')
var tb_water_08 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202008_debuf_mosaic').clip(tb_boundary).rename('tb_water_08')
var tb_water_09 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202009_debuf_mosaic').clip(tb_boundary).rename('tb_water_09')
var tb_water_10 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202010_debuf_mosaic').clip(tb_boundary).rename('tb_water_10')
var tb_water_11 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202011_debuf_mosaic').clip(tb_boundary).rename('tb_water_11')
var tb_water_12 = ee.Image('users/xin_luo/SAR_Water_Extraction/tibet_water/tibet_water_202012_debuf_mosaic').clip(tb_boundary).rename('tb_water_12')
print('tb_water_01 -> ',tb_water_01)
var tb_water = ee.Image.cat([tb_water_01, tb_water_02, tb_water_03, tb_water_04, 
                                   tb_water_05, tb_water_06, tb_water_07, tb_water_08, 
                                   tb_water_09, tb_water_10, tb_water_11, tb_water_12]);

var tb_water_area  = tb_water.eq(1).multiply(ee.Image.pixelArea()) // convert m2 to km2
print('tb_water_area:', tb_water_area)


//// test reduceRegion (take tb_water_01 as example)
//// the surface water area vary with different scale parameter setting. 
var tile01_water_area = tb_water_area.select('tb_water_01').reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: tb_boundary.geometry(),
      scale: 50,        // test -> 10, 20, 50
      maxPixels: 1e11
      });
var tile01_water_area = ee.Number(tile01_water_area.get('tb_water_01')).divide(1e6)
print('tile01_water_area (test, km2):', tile01_water_area)


//// 1 calculate subregion(tile/basin) feature; 
////   !! the subregion file should be FeatureCollection file, and the properties contains area.
//// 1.1 calculate subregion area for each month
var tb_subregion_cal = function(fea) {
  // var previous_ = ee.List(previous);
  var water_area_subregion = tb_water_area.reduceRegion({
      reducer: ee.Reducer.sum(),
      geometry: ee.Feature(fea).geometry(),
      scale: 10,    
      maxPixels: 1e11
      });
  // !!! Property naming should make sure the non-number property not be the first order (for feature charting)
  var fea_subregion = ee.Feature(fea)
            .set('01_wat', ee.Number(water_area_subregion.get('tb_water_01')).divide(1e6))
            .set('02_wat', ee.Number(water_area_subregion.get('tb_water_02')).divide(1e6))
            .set('03_wat', ee.Number(water_area_subregion.get('tb_water_03')).divide(1e6))
            .set('04_wat', ee.Number(water_area_subregion.get('tb_water_04')).divide(1e6))
            .set('05_wat', ee.Number(water_area_subregion.get('tb_water_05')).divide(1e6))
            .set('06_wat', ee.Number(water_area_subregion.get('tb_water_06')).divide(1e6))
            .set('07_wat', ee.Number(water_area_subregion.get('tb_water_07')).divide(1e6))
            .set('08_wat', ee.Number(water_area_subregion.get('tb_water_08')).divide(1e6))
            .set('09_wat', ee.Number(water_area_subregion.get('tb_water_09')).divide(1e6))
            .set('10_wat', ee.Number(water_area_subregion.get('tb_water_10')).divide(1e6))
            .set('11_wat', ee.Number(water_area_subregion.get('tb_water_11')).divide(1e6))
            .set('12_wat', ee.Number(water_area_subregion.get('tb_water_12')).divide(1e6))
  // 1.2 calculate the subregion (tile/basin) surface water percentage
  var fea_subregions_ = ee.Feature(fea_subregion)
            .set('01_wat_per', ee.Number(fea_subregion.get('01_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('02_wat_per', ee.Number(fea_subregion.get('02_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('03_wat_per', ee.Number(fea_subregion.get('03_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('04_wat_per', ee.Number(fea_subregion.get('04_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('05_wat_per', ee.Number(fea_subregion.get('05_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('06_wat_per', ee.Number(fea_subregion.get('06_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('07_wat_per', ee.Number(fea_subregion.get('07_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('08_wat_per', ee.Number(fea_subregion.get('08_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('09_wat_per', ee.Number(fea_subregion.get('09_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('10_wat_per', ee.Number(fea_subregion.get('10_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('11_wat_per', ee.Number(fea_subregion.get('11_wat')).divide(fea_subregion.get('area')).multiply(100))
            .set('12_wat_per', ee.Number(fea_subregion.get('12_wat')).divide(fea_subregion.get('area')).multiply(100))
  return fea_subregions_;
  };

var tb_tiles_wat_stat = tb_tiles.map(tb_subregion_cal)
var tb_basins_wat_stat = tb_basins.map(tb_subregion_cal)


//// 1.2. calculate standard devation of 12-month surface water.
var subregion_std_cal = function(fea) {
  var fea_std = ee.Feature(fea)
                .toArray(['01_wat_per','02_wat_per','03_wat_per','04_wat_per','05_wat_per','06_wat_per',
                          '07_wat_per','08_wat_per','09_wat_per','10_wat_per','11_wat_per','12_wat_per'])
                .reduce(ee.Reducer.stdDev(), [0])
  return fea.set('99_std_per', fea_std.get([0]))
  }

var tb_tiles_wat_stat = tb_tiles_wat_stat.map(subregion_std_cal)
var tb_basins_wat_stat = tb_basins_wat_stat.map(subregion_std_cal)
print('tb_tiles_wat_stat:', tb_tiles_wat_stat)
// print('tb_basins_wat_stat:', tb_basins_wat_stat) // time comsuming

//// 2. calculate statiscs of overall tb region
var tb_wat_stat = tb_boundary.union().first();
var tb_wat_stat = tb_wat_stat.set('name','tibet_region')
              .set('01_wat', tb_tiles_wat_stat.aggregate_sum('01_wat'))
              .set('02_wat', tb_tiles_wat_stat.aggregate_sum('02_wat'))
              .set('03_wat', tb_tiles_wat_stat.aggregate_sum('03_wat'))
              .set('04_wat', tb_tiles_wat_stat.aggregate_sum('04_wat'))
              .set('05_wat', tb_tiles_wat_stat.aggregate_sum('05_wat'))
              .set('06_wat', tb_tiles_wat_stat.aggregate_sum('06_wat'))
              .set('07_wat', tb_tiles_wat_stat.aggregate_sum('07_wat'))
              .set('08_wat', tb_tiles_wat_stat.aggregate_sum('08_wat'))
              .set('09_wat', tb_tiles_wat_stat.aggregate_sum('09_wat'))
              .set('10_wat', tb_tiles_wat_stat.aggregate_sum('10_wat'))
              .set('11_wat', tb_tiles_wat_stat.aggregate_sum('11_wat'))
              .set('12_wat', tb_tiles_wat_stat.aggregate_sum('12_wat'))

var tb_wat_stat = ee.FeatureCollection([tb_wat_stat])
print('tb_stat', tb_wat_stat)

// Export.table.toAsset({
//     collection: tb_tiles_wat_stat,
//     description:'tb_tiles_wat_stat',
//     assetId: 'tb_tiles_wat_stat',
// });

// Export.table.toAsset({
//     collection: tb_basins_wat_stat,
//     description:'tb_basins_wat_stat',
//     assetId: 'tb_basins_wat_stat',
// });

// Export.table.toDrive({
//     collection: tb_tiles_wat_stat,
//     folder: 'tibet_s1_data',
//     description:'tb_tiles_wat_stat',
//     fileFormat: 'SHP'
// });

// Export.table.toDrive({
//     collection: tb_basins_wat_stat,
//     folder: 'tibet_s1_data',
//     description:'tb_basins_wat_stat',
//     fileFormat: 'SHP'
// });


// Export.table.toAsset({
//     collection: tb_wat_stat,
//     description:'tb_wat_stat',
//     assetId: 'tb_wat_stat',
// });

//// Visualization
var empty = ee.Image().byte();
var tb_outline = empty.paint({
      featureCollection: tb_boundary, 
      color: 1, width: 3});
var tiles_outline = empty.paint({
      featureCollection: tb_tiles,
      color: 1, width: 2});

var visual = {
    bands: ['tb_water_01'],
    min: 0, max: 1,
    palette: ['ffffff', 'ffbbbb', '0000ff']   // [white, light red ,blue]
};

Map.setCenter(86.0, 32.0, 4);
// Map.addLayer(s1_water.eq(1).selfMask(), visual, 's1_water_'+date);
Map.addLayer(tb_water_01, visual, 'tb_water_202001');
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline')
Map.addLayer(tiles_outline, {palette: '008000'}, 'tiles_outline');

