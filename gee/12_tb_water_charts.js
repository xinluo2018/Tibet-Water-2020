///////////////////////////////////////////////////////
// Author: xin luo
// Create: 2022.2.16
// Des: generate statistic charts of tb water features
///////////////////////////////////////////////////////

var tb_stat = ee.FeatureCollection('users/xin_luo/SAR_Water_Extraction/tb_wat_stat');
print(tb_stat)


// Define a dictionary that associates property names with values and labels.
var tb_Info = {
  '01_wat': {v: 1, f: 'Jan'},
  '02_wat': {v: 2, f: 'Feb'},
  '03_wat': {v: 3, f: 'Mar'},
  '04_wat': {v: 4, f: 'Apr'},
  '05_wat': {v: 5, f: 'May'},
  '06_wat': {v: 6, f: 'Jun'},
  '07_wat': {v: 7, f: 'Jul'},
  '08_wat': {v: 8, f: 'Aug'},
  '09_wat': {v: 9, f: 'Sep'},
  '10_wat': {v: 10, f: 'Oct'},
  '11_wat': {v: 11, f: 'Nov'},
  '12_wat': {v: 12, f: 'Dec'}
  };

// Organize property information into objects for defining x properties and
// their tick labels.
var xProp = {};  // Dictionary to codify x-axis property names as values.
var xLabels = [];   // Holds dictionaries that label codified x-axis values.
for (var key in tb_Info) {
    xProp[key] = tb_Info[key].v;
    xLabels.push(tb_Info[key]);
    }

print('xProp:', xProp)
print('xLabels:', xLabels)

// Define the chart and print it to the console.
var chart = ui.Chart.feature
                .byProperty({features: tb_stat,
                             xProperties: xProp,
                             seriesProperty: 'label'
                              })
                .setChartType('ColumnChart')
                .setOptions({
                  title: 'Surface water area in Tibet region by Month',
                  hAxis: {
                    title: 'Month',
                    titleTextStyle: {italic: false, bold: true},
                    ticks: xLabels
                          },
                  vAxis: {
                    title: 'Area (km2)',
                    titleTextStyle: {italic: false, bold: true}
                          },
                  colors: ['000080'],
                            });
print(chart);

