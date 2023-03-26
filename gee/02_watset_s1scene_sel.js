//////////////////////////////////////////////////////////////
// Author: xin luo
// Create: 2020.10.20
// Description: This code checks the selected pair-wise sentinel-1 scenes, 
// and the matched sentinel-2 scenes.
//////////////////////////////////////////////////////////////

// tibet region
var area_tb = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/TPBoundary_HF');
var empty = ee.Image().byte();
var tb_outline = empty.paint({
    featureCollection: area_tb, color: 1, width: 3});

// satellite image (base map)
var sen2Coll = ee.ImageCollection('COPERNICUS/S2_SR')
                  .filterBounds(area_tb)
                  .filterDate('2020-07-15', '2020-10-15')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE',10))
                  .sort('CLOUDY_PIXEL_PERCENTAGE');

var sen2Img = sen2Coll.median()

// gsw dataset
var gsw = ee.Image('JRC/GSW1_0/GlobalSurfaceWater');
var occurrence = gsw.select('occurrence').clip(area_tb)
var water_mask = occurrence.updateMask(occurrence.gt(50));

// training regions loading
var scene01_region = ee.Geometry.Rectangle(82.26, 33.82, 82.68, 34.14)
var scene01_s2 = ee.Image('COPERNICUS/S2_SR/20190906T050659_20190906T051806_T44SPC').clip(scene01_region)
var scene01_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190908T123207_20190908T123232_028930_0347AE_1771').clip(scene01_region)
var scene01_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20190903T003331_20190903T003356_017866_0219F6_E5B3').clip(scene01_region)
// print(scene01_sen2,scene01_sen1_ascend,scene01_sen1_descend)

var scene02_region = ee.Geometry.Rectangle(97.93, 34.62, 98.35, 34.93) 
var scene02_s2 = ee.Image('COPERNICUS/S2_SR/20200902T040549_20200902T040615_T47SMU').clip(scene02_region)
var scene02_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200829T112638_20200829T112703_034121_03F666_B1BF').clip(scene02_region)
var scene02_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200829T232821_20200829T232846_034128_03F6AD_DCC0').clip(scene02_region)
// print(tra2_sen2,tra2_sen1_ascend,tra2_sen1_descend)

var scene03_region = ee.Geometry.Rectangle(90.61, 28.84, 91.01, 29.15)
var scene03_s2 = ee.Image('COPERNICUS/S2_SR/20201011T043719_20201011T043837_T46RBT').clip(scene03_region)
var scene03_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201012T115754_20201012T115819_034763_040CF1_EFA7').clip(scene03_region)
var scene03_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201007T235439_20201007T235504_034697_040A9F_19E9').clip(scene03_region)
// print(scene3_sen2,scene3_sen1_ascend,scene3_sen1_descend)

var scene04_region = ee.Geometry.Rectangle(86.67, 35.48, 87.09, 35.78) 
var scene04_s2 = ee.Image('COPERNICUS/S2_SR/20200624T045701_20200624T050509_T45SVV').clip(scene04_region)
var scene04_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200619T120745_20200619T120810_033086_03D533_F8DC').clip(scene04_region)
var scene04_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200619T001644_20200619T001709_022095_029EE2_204F').clip(scene04_region)

var scene05_region = ee.Geometry.Rectangle(89.78, 33.82, 90.14, 34.11)
var scene05_s2 = ee.Image('COPERNICUS/S2_SR/20200901T043709_20200901T043834_T45SYT').clip(scene05_region)
var scene05_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200825T115907_20200825T115932_034063_03F45D_8D93').clip(scene05_region)
var scene05_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200826T000130_20200826T000155_034070_03F49B_6CAB').clip(scene05_region)

var scene06_region = ee.Geometry.Rectangle(87.33, 31.77, 87.76, 32.09)
var scene06_s2 = ee.Image('COPERNICUS/S2_SR/20200914T044659_20200914T045805_T45SWR').clip(scene06_region)
var scene06_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200916T121507_20200916T121532_034384_03FF9D_86C7').clip(scene06_region)
var scene06_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200912T001020_20200912T001045_034318_03FD5A_3E94').clip(scene06_region)

var scene07_region = ee.Geometry.Rectangle(81.14,30.43, 81.51, 30.73)
var scene07_s2 = ee.Image('COPERNICUS/S2_SR/20200831T050659_20200831T051456_T44RNU').clip(scene07_region)
var scene07_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200826T123916_20200826T123941_034078_03F4DA_7766').clip(scene07_region)
var scene07_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200903T003512_20200903T003537_034187_03F8BE_B4F9').clip(scene07_region)

var scene08_region = ee.Geometry.Rectangle(97.43, 38.13, 97.82, 38.44)
var scene08_s2 = ee.Image('COPERNICUS/S2_SR/20201025T041849_20201025T041843_T47SLC').clip(scene08_region)
var scene08_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201016T112729_20201016T112754_034821_040F00_0DBC').clip(scene08_region)
var scene08_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201021T233542_20201021T233607_034901_0411BF_67A8').clip(scene08_region)

var scene09_region = ee.Geometry.Rectangle(86.35,33.77,86.77,34.10)
var scene09_s2 = ee.Image('COPERNICUS/S2_SR/20200818T045659_20200818T050858_T45SVT').clip(scene09_region)
var scene09_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200811T121530_20200811T121555_033859_03ED25_81E3').clip(scene09_region)
var scene09_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200818T001713_20200818T001738_022970_02B99E_0C3B').clip(scene09_region)

var scene10_region = ee.Geometry.Rectangle(90.95, 35.48, 91.35, 35.78)
var scene10_s2 = ee.Image('COPERNICUS/S2_SR/20190927T043659_20190927T044051_T46SCE').clip(scene10_region)
var scene10_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190924T115927_20190924T115952_029163_034FB0_6868').clip(scene10_region)
var scene10_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190925T000059_20190925T000124_029170_034FEC_BBB6').clip(scene10_region)

var scene11_region = ee.Geometry.Rectangle(79.47,33.36,79.86,33.66)
var scene11_s2 = ee.Image('COPERNICUS/S2_SR/20200923T051649_20200923T052705_T44SLC').clip(scene11_region)
var scene11_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200919T124007_20200919T124032_034428_040137_60A4').clip(scene11_region)
var scene11_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200920T004251_20200920T004320_034435_04017D_A03E').clip(scene11_region)

var scene12_region = ee.Geometry.Rectangle(98.70,29.38,99.08,29.68)
var scene12_s2 = ee.Image('COPERNICUS/S2_SR/20191114T040019_20191114T040017_T47RMN').clip(scene12_region)
var scene12_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20191115T112452_20191115T112517_029921_036A01_CC09').clip(scene12_region)
var scene12_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20191110T232122_20191110T232147_029855_0367B3_CB0E').clip(scene12_region)


var scene13_region = ee.Geometry.Rectangle(100.45,35.82,100.85,36.12)
var scene13_s2 = ee.Image('COPERNICUS/S2_SR/20190811T035541_20190811T040503_T47SPV').clip(scene13_region)
var scene13_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190818T111834_20190818T111859_028623_033D00_3E27').clip(scene13_region)
var scene13_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190818T231940_20190818T232005_028630_033D3D_FFD9').clip(scene13_region)

var scene14_region = ee.Geometry.Rectangle(91.27,31.82,91.67,32.13)
var scene14_s2 = ee.Image('COPERNICUS/S2_SR/20190813T043701_20190813T043704_T46SCA').clip(scene14_region)
var scene14_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190807T115834_20190807T115859_028463_033779_A571').clip(scene14_region)
var scene14_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190802T235339_20190802T235404_028397_03357D_E806').clip(scene14_region)

var scene15_region = ee.Geometry.Rectangle(102.07,33.65,102.46,33.95)
var scene15_s2 = ee.Image('COPERNICUS/S2_SR/20190816T035549_20190816T040759_T48STC').clip(scene15_region)
var scene15_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190813T110933_20190813T110958_028550_033A8E_47C3').clip(scene15_region)
var scene15_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190813T231155_20190813T231220_028557_033AC7_DD17').clip(scene15_region)


var scene16_region = ee.Geometry.Rectangle(96.08,32.95,96.47,33.26)
var scene16_s2 = ee.Image('COPERNICUS/S2_SR/20200831T041551_20200831T042007_T47SKS').clip(scene16_region)
var scene16_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200903T113413_20200903T113438_034194_03F8ED_E4E4').clip(scene16_region)
var scene16_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200903T233656_20200903T233721_034201_03F92F_8084').clip(scene16_region)


var scene17_region = ee.Geometry.Rectangle(94.42,29.29,94.80,29.58)
var scene17_s2 = ee.Image('COPERNICUS/S2_SR/20200213T041901_20200213T041911_T46RFT').clip(scene17_region)
var scene17_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200217T114124_20200217T114149_031292_039993_6FF2').clip(scene17_region)
var scene17_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200212T233803_20200212T233828_031226_039745_94C5').clip(scene17_region)


var scene18_region = ee.Geometry.Rectangle(74.69,38.55,75.09,38.84)
var scene18_s2 = ee.Image('COPERNICUS/S2_SR/20200907T054641_20200907T054642_T43SDC').clip(scene18_region)
var scene18_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200905T125805_20200905T125830_034224_03FA09_2DBA').clip(scene18_region)
var scene18_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200911T010600_20200911T010625_034304_03FCD1_0AA8').clip(scene18_region)


var scene19_region = ee.Geometry.Rectangle(83.85,31.29,84.22,31.58)
var scene19_s2 = ee.Image('COPERNICUS/S2_SR/20190926T050659_20190926T051645_T45RTQ').clip(scene19_region)
var scene19_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190927T122307_20190927T122332_029207_03512E_DCB2').clip(scene19_region)
var scene19_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20191004T002612_20191004T002641_018318_02281E_3F05').clip(scene19_region)


var scene20_region = ee.Geometry.Rectangle(88.94,37.39,89.36,37.69)
var scene20_s2 = ee.Image('COPERNICUS/S2_SR/20190707T044711_20190707T045327_T45SXB').clip(scene20_region)
var scene20_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190702T120012_20190702T120037_027938_032783_FE5F').clip(scene20_region)
var scene20_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20190708T000826_20190708T000855_028018_032A07_B240').clip(scene20_region)

var scene21_region = ee.Geometry.Rectangle(85.47, 35.00, 85.87, 35.30)
var scene21_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200811T121555_20200811T121620_033859_03ED25_7F02').clip(scene21_region)
var scene21_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200806T001647_20200806T001712_022795_02B434_20F2').clip(scene21_region)

var scene22_region = ee.Geometry.Rectangle(85.89, 36.39, 86.40, 36.75)
var scene22_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200811T121620_20200811T121645_033859_03ED25_1BB9').clip(scene22_region)
var scene22_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200806T001622_20200806T001647_022795_02B434_36D0').clip(scene22_region)

var scene23_region = ee.Geometry.Rectangle(93.2139, 35.3952, 93.6007, 35.6952)
var scene23_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200815T114312_20200815T114337_033917_03EF34_FD73').clip(scene23_region)
var scene23_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200815T234439_20200815T234504_033924_03EF6F_DDC7').clip(scene23_region)

var scene24_region = ee.Geometry.Rectangle(88.88, 35.18, 89.30, 35.48)
var scene24_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200103T120742_20200103T120807_030636_0382B3_30E3').clip(scene24_region)
var scene24_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200111T000057_20200111T000122_030745_038671_739C').clip(scene24_region)

var scene25_region = ee.Geometry.Rectangle(99.85, 35.37, 100.3, 35.70)
var scene25_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200109T111833_20200109T111858_030723_0385B4_0138').clip(scene25_region)
var scene25_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200109T231940_20200109T232005_030730_0385F1_E4AD').clip(scene25_region);

var scene26_region = ee.Geometry.Rectangle(94.59, 38.66, 95.02, 38.98);
var scene26_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200212T113545_20200212T113610_031219_03970B_D803').clip(scene26_region)
var scene26_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200205T234342_20200205T234407_031124_0393BF_AD5A').clip(scene26_region)

var scene27_region = ee.Geometry.Rectangle(98.76, 35.50, 99.18, 35.82);
var scene27_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200606T112633_20200606T112658_032896_03CF78_EE6E').clip(scene27_region);
var scene27_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200606T232751_20200606T232816_032903_03CFB0_0BFF').clip(scene27_region);

var scene28_region = ee.Geometry.Rectangle(80.86, 35.28, 81.30, 35.59); 
var scene28_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200704T123235_20200704T123300_033305_03DBD2_6440').clip(scene28_region);
var scene28_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200711T003308_20200711T003333_022416_02A8B8_B54D').clip(scene28_region);

var scene29_region = ee.Geometry.Rectangle(96.74, 31.71, 97.14, 32.0);   
var scene29_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200705T113345_20200705T113410_033319_03DC3C_4320').clip(scene29_region);
var scene29_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200712T232908_20200712T232933_033428_03DF9C_D763').clip(scene29_region);

var scene30_region = ee.Geometry.Rectangle(93.88, 36.40, 94.35, 36.75);
var scene30_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201002T114339_20201002T114404_034617_0407E4_1F38').clip(scene30_region);
var scene30_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201002T234416_20201002T234441_034624_040823_7CEA').clip(scene30_region);

var scene31_region = ee.Geometry.Rectangle(81.04, 33.45, 81.45, 33.76); 
var scene31_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200907T124007_20200907T124032_034253_03FB02_FF7B').clip(scene31_region);
var scene31_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20200909T003336_20200909T003409_023291_02C3BC_85E6').clip(scene31_region);

var scene32_region = ee.Geometry.Rectangle(92.58, 33.85, 92.97, 34.14);   
var scene32_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201112T115101_20201112T115126_035215_041C99_58CD').clip(scene32_region);
var scene32_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201112T235324_20201112T235349_035222_041CD7_7F5C').clip(scene32_region);

var scene33_region = ee.Geometry.Rectangle(94.29, 33.88, 94.69, 34.18); 
var scene33_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201201T114248_20201201T114313_035492_04263C_D644').clip(scene33_region);
var scene33_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201201T234506_20201201T234531_035499_042676_C8ED').clip(scene33_region);

var scene34_region = ee.Geometry.Rectangle(83.89, 33.22, 84.28, 33.52);
var scene34_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201202T122338_20201202T122403_035507_0426B9_3B33').clip(scene34_region);
var scene34_s1des = ee.Image('COPERNICUS/S1_GRD/S1B_IW_GRDH_1SDV_20201209T002531_20201209T002556_024618_02ED65_5A6A').clip(scene34_region);

var scene35_region = ee.Geometry.Rectangle(90.474, 36.85, 90.93, 37.20);   
var scene35_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200403T115949_20200403T120014_031963_03B0F0_B693').clip(scene35_region);
var scene35_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200404T000032_20200404T000057_031970_03B12F_529E').clip(scene35_region);

var scene36_region = ee.Geometry.Rectangle(90.38, 31.84, 90.73, 32.12);
var scene36_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200509T115836_20200509T115901_032488_03C32A_8BD2').clip(scene36_region);
var scene36_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200504T235341_20200504T235406_032422_03C11C_0373').clip(scene36_region);

var scene37_region = ee.Geometry.Rectangle(82.66, 33.13, 83.14, 33.49);
var scene37_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200306T123140_20200306T123205_031555_03A2AC_816A').clip(scene37_region);
var scene37_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200302T002620_20200302T002649_031489_03A058_5216').clip(scene37_region);

var scene38_region = ee.Geometry.Rectangle(90.71, 30.68, 91.11, 31.00);
var scene38_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200602T115812_20200602T115837_032838_03CDB6_CA27').clip(scene38_region);
var scene38_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20200609T235408_20200609T235433_032947_03D0F8_64D4').clip(scene38_region);

var scene39_region = ee.Geometry.Rectangle(99.65, 36.99, 100.11, 37.31);
var scene39_s1as = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201104T111907_20201104T111932_035098_04188D_F4DC').clip(scene39_region);
var scene39_s1des = ee.Image('COPERNICUS/S1_GRD/S1A_IW_GRDH_1SDV_20201109T232732_20201109T232757_035178_041B57_C9D8').clip(scene39_region);



// // Map.centerObject(area_tb, 5);
// // satellite image and tibet region
// Map.addLayer(sen2Img, {bands: ['B4', 'B3', 'B2'], min:0, max:3000}, 'sen2 image')
Map.addLayer(tb_outline, {palette: 'FF0000'}, 'Tibet region');
// // scene region 01
Map.addLayer(scene01_region, {color:'green'}, 'scene01 region');
// Map.addLayer(scene01_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene01 s2 image')
// Map.addLayer(scene01_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene01 s1 ascend');
// Map.addLayer(scene01_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene01 s1 descend')

// // scene region 02
Map.addLayer(scene02_region, {color:'green'}, 'scene02 region');
// Map.addLayer(scene02_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene02 s2 image')
// Map.addLayer(scene02_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene02 s1 ascend');
// Map.addLayer(scene02_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene02 s1 descend')

// // scene region 03
Map.addLayer(scene03_region, {color:'green'}, 'scene03 region');
// Map.addLayer(scene03_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene03 s2 image')
// Map.addLayer(scene03_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene03 s1 ascend');
// Map.addLayer(scene03_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, ‘scene03 s1 descend')

// scene region 04
Map.addLayer(scene04_region, {color:'green'}, 'scene04 region');
// Map.addLayer(scene04_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene04 s2 image')
// Map.addLayer(scene04_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene04 s1 ascend');
// Map.addLayer(scene04_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene04 s1 descend')

// scene region 05
Map.addLayer(scene05_region, {color:'green'}, 'scene05 region');
// Map.addLayer(scene05_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene05 s2 image')
// Map.addLayer(scene05_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene05 s1 ascend');
// Map.addLayer(scene05_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene05 s1 descend')

// // scene region 06
Map.addLayer(scene06_region, {color:'green'}, 'scene06 region');
// Map.addLayer(scene06_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene06 s2 image')
// Map.addLayer(scene06_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene06 s1 ascend');
// Map.addLayer(scene06_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene06 s1 descend')

// scene region 07
Map.addLayer(scene07_region, {color:'green'}, 'scene07 region');
// Map.addLayer(scene07_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene07 s2 image')
// Map.addLayer(scene07_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene07 s1 ascend');
// Map.addLayer(scene07_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene07 s1 descend')

// //scene region 08
Map.addLayer(scene08_region, {color:'green'}, 'scene08 region');
// Map.addLayer(scene08_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene08 s2 image'); 
// Map.addLayer(scene08_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene08 s1 ascend');
// Map.addLayer(scene08_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene08 s1 descend')

// // scene region 09
Map.addLayer(scene09_region, {color:'green'}, 'scene09 region');
// Map.addLayer(scene09_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene09 s2 image');
// Map.addLayer(scene09_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene09 s1 ascend');
// Map.addLayer(scene09_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene09 s1 descend')


// // scene region 10
Map.addLayer(scene10_region, {color:'green'}, 'scene10 region');
// Map.addLayer(scene10_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene10 s2 image');
// Map.addLayer(scene10_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene10 s1 ascend');
// Map.addLayer(scene10_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene10 s1 descend')


// // scene region 11
Map.addLayer(scene11_region, {color:'green'}, 'scene11 region');
// Map.addLayer(scene11_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene11 s2 image');
// Map.addLayer(scene11_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene11 s1 ascend');
// Map.addLayer(scene11_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene11 s1 descend')


// // scene region 12
Map.addLayer(scene12_region, {color:'green'}, 'scene12 region');
// Map.addLayer(scene12_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene12 s2 image');
// Map.addLayer(scene12_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene12 s1 ascend');
// Map.addLayer(scene12_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene12 s1 descend')

// // scene region 13
Map.addLayer(scene13_region, {color:'green'}, 'scene13 region');
// Map.addLayer(scene13_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene13 s2 image');
// Map.addLayer(scene13_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene13 s1 ascend');
// Map.addLayer(scene13_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene13 s1 descend')

// // scene region 14
Map.addLayer(scene14_region, {color:'green'}, 'scene14 region');
// Map.addLayer(scene14_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene14 s2 image');
// Map.addLayer(scene14_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene14 s1 ascend');
// Map.addLayer(scene14_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene14 s1 descend')

// // scene region 15
Map.addLayer(scene15_region, {color:'green'}, 'scene15 region');
// Map.addLayer(scene15_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene15 s2 image');
// Map.addLayer(scene15_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene15 s1 ascend');
// Map.addLayer(scene15_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene15 s1 descend')

// // scene region 16
Map.addLayer(scene16_region, {color:'green'}, 'scene16 region');
// Map.addLayer(scene16_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene16 s2 image');
// Map.addLayer(scene16_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene16 s1 ascend');
// Map.addLayer(scene16_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene16 s1 descend')

// // scene region 17
Map.addLayer(scene17_region, {color:'green'}, 'scene17 region');
// Map.addLayer(scene17_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene17 s2 image');
// Map.addLayer(scene17_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene17 s1 ascend');
// Map.addLayer(scene17_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene17 s1 descend')

// // scene region 18
Map.addLayer(scene18_region, {color:'green'}, 'scene18 region');
// Map.addLayer(scene18_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene18 s2 image');
// Map.addLayer(scene18_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene18 s1 ascend');
// Map.addLayer(scene18_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene18 s1 descend')

// // scene region 19
Map.addLayer(scene19_region, {color:'green'}, 'scene19 region');
// Map.addLayer(scene19_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene19 sen2 image');
// Map.addLayer(scene19_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene19 sen1 ascend');
// Map.addLayer(scene19_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene19 sen1 descend')

// // scene region 20
Map.addLayer(scene20_region, {color:'green'}, 'scene20 region');
// Map.addLayer(scene20_s2,{bands:['B4','B3','B2'], max:3000, min:0}, 'scene20 s2 image');
// Map.addLayer(scene20_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene20 s1 ascend');
// Map.addLayer(scene20_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene20 s1 descend')

// // scene region 21
Map.addLayer(scene21_region, {color:'green'}, 'scene21 region');
// Map.addLayer(scene21_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene21 s1 ascend');
// Map.addLayer(scene21_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene21 s1 descend');

// // scene region 22
Map.addLayer(scene22_region, {color:'green'}, 'scene22 region');
// Map.addLayer(scene22_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene22 s1 ascend');
// Map.addLayer(scene22_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene22 s1 descend');

// // scene region 23
Map.addLayer(scene23_region, {color:'green'}, 'scene23 region');
// Map.addLayer(scene23_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene23 s1 ascend');
// Map.addLayer(scene23_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene23 s1 descend');

// // scene region 24
Map.addLayer(scene24_region, {color:'green'}, 'scene24 region');
// Map.addLayer(scene24_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene24 s1 ascend');
// Map.addLayer(scene24_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene24 s1 descend');

// // scene region 25
Map.addLayer(scene25_region, {color:'green'}, 'scene25 region');
// Map.addLayer(scene25_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene25 s1 ascend');
// Map.addLayer(scene25_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene25 s1 descend');

// // scene region 26
Map.addLayer(scene26_region, {color:'green'}, 'scene26 region');
// Map.addLayer(scene26_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene26 s1 ascend');
// Map.addLayer(scene26_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene26 s1 descend');


// // scene region 27
Map.addLayer(scene27_region, {color:'green'}, 'scene27 region');
// Map.addLayer(scene27_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene27 s1 ascend');
// Map.addLayer(scene27_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene27 s1 descend');

// // scene region 28
Map.addLayer(scene28_region, {color:'green'}, 'scene28 region');
// Map.addLayer(scene28_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene28 s1 ascend');
// Map.addLayer(scene28_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene28 s1 descend');

// // scene region 29
Map.addLayer(scene29_region, {color:'green'}, 'scene29 region');
// Map.addLayer(scene29_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene29 s1 ascend');
// Map.addLayer(scene29_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene29 s1 descend');

// // scene region 30
Map.addLayer(scene30_region, {color:'green'}, 'scene30 region');
// Map.addLayer(scene30_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene30 s1 ascend');
// Map.addLayer(scene30_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene30 s1 descend');

// // scene region 31
Map.addLayer(scene31_region, {color:'green'}, 'scene31 region');
// Map.addLayer(scene31_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene31 s1 ascend');
// Map.addLayer(scene31_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene31 s1 descend');

// // scene region 32
Map.addLayer(scene32_region, {color:'green'}, 'scene32 region');
// Map.addLayer(scene32_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene32 s1 ascend');
// Map.addLayer(scene32_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene32 s1 descend');

// // scene region 33
Map.addLayer(scene33_region, {color:'green'}, 'scene33 region');
// Map.addLayer(scene33_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene33 s1 ascend');
// Map.addLayer(scene33_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene33 s1 descend');

// // scene region 34
Map.addLayer(scene34_region, {color:'green'}, 'scene34 region');
// Map.addLayer(scene34_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene34 s1 ascend');
// Map.addLayer(scene34_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene34 s1 descend');

// // scene region 35
Map.addLayer(scene35_region, {color:'green'}, 'scene35 region');
// Map.addLayer(scene35_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene35 s1 ascend');
// Map.addLayer(scene35_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene35 s1 descend');

// // scene region 36
Map.addLayer(scene36_region, {color:'green'}, 'scene36 region');
// Map.addLayer(scene36_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene36 s1 ascend');
// Map.addLayer(scene36_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene36 s1 descend');

// // scene region 37
Map.addLayer(scene37_region, {color:'green'}, 'scene37 region');
// Map.addLayer(scene37_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene37 s1 ascend');
// Map.addLayer(scene37_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene37 s1 descend');

// // scene region 38
Map.addLayer(scene38_region, {color:'green'}, 'scene38 region');
// Map.addLayer(scene38_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene38 s1 ascend');
// Map.addLayer(scene38_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene38 s1 descend');

// // scene region 39
Map.addLayer(scene39_region, {color:'green'}, 'scene39 region');
// Map.addLayer(scene39_s1as, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene39 s1 ascend');
// Map.addLayer(scene39_s1des, {bands:['VV','VH','VV'], max:0, min:-30}, 'scene39 s1 descend');


// Map.addLayer(water_mask, {palette: ['blue']}, '50% occurrence water mask');





