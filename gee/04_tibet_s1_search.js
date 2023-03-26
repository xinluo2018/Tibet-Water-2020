/////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.11.13, modify: 2022.03.01
// Description: select the sentinel-1 images that fully cover the tibetan plateau
//          1. search all the s1 data; 
//          2. anutomatic remove the repeat, small-area and invalid s1 images
//          3. automatic adjust overlab order(right cover left for as, left cover right for des)
/////////////////////////////////////////////////////////

// Study area
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var area_tb_bound = area_tb.geometry().bounds();

var s1_img_area = ee.Number(42000000000)

// date of the collected data
var start_time = '2020-07-01'
var end_time = '2020-07-31'

////**********************************************************////
// -------- data search and selection ------- ///
////**********************************************************////

////////////////////////////////////////////////////////////////////////
//// ---- 1. sentinel-1 images (ascending and descending obit).
////////////////////////////////////////////////////////////////////////
var ascendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filterBounds(area_tb)
    .filterDate(start_time, end_time);


var descendCol = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filterBounds(area_tb)
    .filterDate(start_time, end_time)
    .sort('system:time_start');

print('ascend images:', ascendCol.size(), ascendCol)
print('descend images:', descendCol.size(), descendCol)


/////////////////////////////////////////////////////////////////////////////////
//// 2. anutomatic remove the repeat, small-area and invalid s1 images
/////////////////////////////////////////////////////////////////////////////////
//// ---- 2.1. remove the small-area image.
var add_area = function(image){
    var area = image.geometry().area();
    return image.set('area', area);
    }
ascendCol = ascendCol.map(add_area)
            .filterMetadata('area', 'greater_than', s1_img_area.multiply(0.5))
descendCol = descendCol.map(add_area)
            .filterMetadata('area', 'greater_than', s1_img_area.multiply(0.5))

//// 2.2 remove invalid s1 images (which contains masked value: null data > 20%).
var valid_percent_func = function(image){
    var region_sel = image.geometry()
                          .centroid({'maxError': 1})
                          .buffer({'distance': 15000})
                          .bounds();
    var image_count = image.reduceRegion({
                           // count the non masked values
                          reducer: ee.Reducer.count(),  
                          maxPixels: 1e10, 
                          geometry: region_sel,
                          scale: 10,
                          })
    var img_count = image_count.getNumber('VV')
                          .min(image_count.getNumber('VH'))
                          .divide(ee.Number(7000000))
    return image.set('valid_value_per', img_count)
    }

var ascendCol = ascendCol.map(valid_percent_func)
                         .filterMetadata('valid_value_per', 'greater_than', 0.80)
var descendCol= descendCol.map(valid_percent_func)
                         .filterMetadata('valid_value_per','greater_than', 0.80)

////////////////////////////////////////////////////////////////////////
// ---- 2.3. remove the redundant image which at the same location
////////////////////////////////////////////////////////////////////////
var area_add_thre = s1_img_area.multiply(0.15);
var region_exclusive_func = function(image, imgCol_list){
    var imgCol_ls = ee.List(imgCol_list);
    var imgCol_ls_geo = ee.ImageCollection(imgCol_ls).union().geometry();
    var image_geo = image.geometry();
    var image_inter_area = imgCol_ls_geo.intersection(image_geo).area();
    var area_add = ee.Number(image_geo.area())
                            .subtract(ee.Number(image_inter_area)) 
    return imgCol_ls.add(image.set('area_add', area_add))
  }

var ascendCol_sel = ee.ImageCollection(
                      ee.List(ascendCol.iterate(region_exclusive_func, [])))
                      .filterMetadata('area_add', 'greater_than', area_add_thre)
                        
var descendCol_sel = ee.ImageCollection(
                      ee.List(descendCol.iterate(region_exclusive_func, [])))
                      .filterMetadata('area_add', 'greater_than', area_add_thre)
              

////////////////////////////////////////////////////////////////////////
////  ------- 3. automatic adjust overlab order
////  todo: combine the 3.1) and 3.2): clip the right edge region firstly, 
////        then let early image covers latter image.
////  3.1) right overlab left for ascending and left overlab right for descneding (!!selected).
////      because the right edge region of ascending image shows bad quality
////      and the left edge region of descending image shows bad quality.
////      actually, the right edge region corresponding flight direction suffers more noise.

// set longitute 
function set_center_lon(img){
    var point_center = ee.Geometry(img.get('system:footprint')).centroid()
    var lon_center = point_center.coordinates().get(0)
    return img.set('center_lon', lon_center);}
var ascendCol_sel = ascendCol_sel.map(set_center_lon).sort('center_lon', true)
var descendCol_sel = descendCol_sel.map(set_center_lon).sort('center_lon', false)

// //// 3.2) early acquired image covers the later acquired image.  
// ////      this way obtains better match-up ascending/decending pair images.
// var ascendCol_sel_2 = ascendCol_sel.sort('system:time_start', false)
// var descendCol_sel_2 = descendCol_sel.sort('system:time_start', false)

print('selected ascend images:', ascendCol_sel.size(), ascendCol_sel)
print('selected descend images:', descendCol_sel.size(), descendCol_sel)
////////////////////////////////////////////////////////////////////////



// //**********************************************************////
// -------- export image footprint and image id ------- ///
// //**********************************************************////
// 1) add image footprint and id
function generate_img_id(img){
    var img_geo = img.geometry();
    var img_id = ee.String('COPERNICUS/S1_GRD/').cat(img.id())
    var fea = ee.Feature(img_geo);
    return fea.set('img_id', img_id);}

var as_fp_id = ascendCol_sel.map(generate_img_id);
var des_fp_id = descendCol_sel.map(generate_img_id);
print('as_fp_id:', as_fp_id)


// // 2) Export to asset.
// Export.table.toAsset({
//   collection: as_fp_id,
//   description: 'tibet_as_202012',
//   assetId: 'tibet_as_202012',
// });

// Export.table.toAsset({
//   collection: des_fp_id,
//   description:'tibet_des_202012',
//   assetId: 'tibet_des_202012',
// });

// //// 3) export to drive
// Export.table.toDrive({
//     collection: as_fp,
//     description: 's1_as_202007_footprint',
//     folder: 'tibet_sar_data',
//     fileFormat: "KML"
//     })

// Export.table.toDrive({
//     collection: des_fp,
//     description: 's1_des_202007_footprint',
//     folder: 'tibet_sar_data',
//     fileFormat: "KML"
//   })


////**********************************************************////
// -------- visualization ------- ///
////**********************************************************////
var empty = ee.Image().byte();
//// outline visualization of the study area.
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 2});
//// outline visualization of the image footprint.
var as_fp_outline = empty.paint({
    featureCollection: as_fp_id, color: 1, width: 2});
var des_fp_outline = empty.paint({
    featureCollection: des_fp_id, color: 1, width: 2});


// Map.setCenter(86.0, 32.0, 4);
Map.addLayer(ascendCol_sel.select('VH', 'VV'), {bands:['VV','VH','VV'], min: -40, max: 1}, 'as')
Map.addLayer(descendCol_sel.select('VH', 'VV'), {bands:['VV','VH','VV'], min: -40, max: 1}, 'des')
Map.addLayer(as_fp_outline, {palette: 'FF00FF'}, 'as_footprint');
Map.addLayer(des_fp_outline, {palette: '000000'}, 'des_footprint');
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet_outline');

