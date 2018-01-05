Quicklinks: [Data Preparation](EXAMPLE_DATA.md) --- [Project Setup](EXAMPLE_SETUP.md) --- [Running METRIC](EXAMPLE_METRIC.md)

# pyMETRIC Data Preparation Example

This example will step through acquiring and prepping all of the Landsat images and ancillary data needed to run the pyMETRIC code for a single Landsat path/row.  The target study area for this example is the Harney basin in central Oregon, located in Landsat path 43 row 30.

## Project Folder

All of the example script calls listed below assume that the pyMETRIC repository was installed on a Windows computer to "D:\pyMETRIC" and that the scripts are being called from this folder.

After cloning the repository, the first step is to create a project folder if it doesn't exists.  This can be done from the command line (using the following command) or in the file explorer.
```
D:\pyMETRIC>mkdir harney
```

## Script Parameters

Most of the setup/prep scripts listed below will need to have command line arguments passed to them.  For the most part, the arguments are standaradized between scripts, but the user is strongly encouraged to look at the possible arguments by first passing the help "-h" argument to the script.
For example, adding an "-h" or "--help" argument to the script call:
```
D:\pyMETRIC>python code\download\download_ned.py -h
```

will return the following description of the script, the possible command line arguments, as well as the argument type and default value:
```
D:\pyMETRIC>python code\download\download_ned.py -h
usage: download_ned.py [-h] --extent FILE [--output FOLDER] [-o] [--debug]

Download NED

optional arguments:
  -h, --help       show this help message and exit
  --extent FILE    Study area shapefile (default: None)
  --output FOLDER  Output folder (default: D:\METRIC\dem\tiles)
  -o, --overwrite  Force overwrite of existing files (default: None)
  --debug          Debug level logging (default: 20)
```

Almost all of the scripts will have the "-h", "--overwrite" (or "-o"), and "--debug" command line arguments.  The overwrite flag is used to indicate to the script that existing files should be overwritten.  If this is not set, the scripts will typically operations if the output file is present.  The debug flag is used to turn on debug level logging which will output additional text to the console and may be helpful if the scripts aren't working.

## Study Area

The first step in setting up the pyMETRIC codes is identifying or constructing a study area shapefile.  The study area shapefile path can then be passed to many of the following prep scripts using the "--extent" command line argument, in order to limit the spatial extent of the rasters.

A shapefile has been provided within this distribution to be used with the example workflow of pyMETRIC.  This example study area shapefile is located in \pyMETRIC\harney\study_area , and encompasses the area of the Harney Basin, Oregon.  This study area was derived from the USGS National Hydrography Dataset (WBDHU8).

## Landsat skip/keep lists

Before running pyMETRIC, it is important to identify Landsat images that should not be processed at all due to excessive clouds, smoke, haze, snow, shadows, or general bad data in the study area.  Many of the pyMETRIC tools are expecting or will honor a text file of Landsat scene IDs that should be skipped.  This file is typically refered to as a "skip list" in the documentation and INI files.

One approach for generating this skip list is to the use the [Cloud Free Scene Counts tools](https://github.com/Open-ET/cloud-free-scene-counts).  The Landsat path/row used in the example for those tools is also 43/30.

For the purpose of this example, we will directly use the list of "cloudy" scenes in 2015 identified at the end of the [Cloud Free Scene Counts example](https://github.com/Open-ET/cloud-free-scene-counts/blob/master/EXAMPLE.md).  The following list of 28 Landsat scene IDs should be pasted into a file called "cloudy_scenes.txt" and saved in "D:\pyMETRIC\harney\landsat":

```
LE07_043030_20150101
LC08_043030_20150109
LE07_043030_20150117
LE07_043030_20150202
LC08_043030_20150226
LC08_043030_20150314
LE07_043030_20150322
LC08_043030_20150330
LE07_043030_20150407
LC08_043030_20150517
LE07_043030_20150525
LC08_043030_20150602
LC08_043030_20150618
LC08_043030_20150704
LE07_043030_20150712
LC08_043030_20150805
LE07_043030_20150829
LE07_043030_20150914
LC08_043030_20151008
LC08_043030_20151024
LE07_043030_20151101
LC08_043030_20151109
LE07_043030_20151117
LC08_043030_20151125
LE07_043030_20151203
LC08_043030_20151211
LE07_043030_20151219
LC08_043030_20151227
```

## Landsat Images

`Note: Landsat tar.gz files will need to be stored in nested separate folders by path, row, and year`

The Landsat images can be downloaded using the [Landsat578 tool](https://github.com/dgketchum/Landsat578).  This tool will need to be installed with pip (see the [pyMETRIC README](README)) and a credentials file will need to be generated before using (see the [Landsat 578 README](https://github.com/dgketchum/Landsat578/blob/master/README.md)).

The Landsat 7 and 8 images from 2015 for the study area can be downloaded using the following commands.  The Landsat images are being downloaded to the non-project landsat folder so that they can be used by other projects, but they could be downloaded directly to the project folder instead.
```
D:\pyMETRIC>landsat --satellite LE7 --start 2015-01-01 --end 2015-12-31 --path 43 --row 30 --output .\landsat --credentials .\landsat\usgs.txt --zipped
D:\pyMETRIC>landsat --satellite LC8 --start 2015-01-01 --end 2015-12-31 --path 43 --row 30 --output .\landsat --credentials .\landsat\usgs.txt --zipped
```

After downloading, you will need to run the following script to rename and move the Landsat tar.gz files into the correct folder structure.  Eventually, the Landsat578 download tool may support writing directly to the target folders.
```
D:\pyMETRIC>python landsat\landsat_image_organize.py
```

## Ancillary Data

The ancillary data files should be downloaded once and saved in a common folder or network drive to avoid needing to repeatedly download data.

### Landsat WRS2 Descending Footprints

The following command will download the global Landsat WRS2 descending footprint shapefile.  By default, the shapefile will be saved to the folder ".\landsat\footprints".
```
D:\pyMETRIC>python code\download\download_footprints.py
```

### National Elevation Dataset (NED)

The following command will download the 1x1 degree 1-arcsecond (~30m) resolution NED tiles that intersect the study area.  By default, the NED tiles will be saved to the folder ".\dem\tiles".
```
D:\pyMETRIC>python code\download\download_ned.py --extent harney\study_area\harney_wgs84z11.shp
```

The NED tiles are being downloaded from the [USGS FTP server](ftp://rockyftp.cr.usgs.gov/vdelivery/Datasets/Staged/Elevation/1/IMG) and can be downloaded manually also.

### National Land Cover Database (NLCD)

The CONUS-wide NLCD image can be downloaded using the following command.  This script can only download the 2006 or 2011 NLCD images.  By default, the NLCD image will be saved to the folder ".\nlcd".
```
D:\pyMETRIC>python code\download\download_nlcd.py -y 2011
```

### Cropland Data Layer (CDL) (optional)

The CDL data is updated annually and can give a slightly better representation of crop area in the study area.  CDL data can also be used in the pyMETRIC workflow to generated quasi field boundaries if a field boundary dataset is not available.  CDL data will not be used for this example, but the following command will downloaded the CONUS-wide CDL image.  By default, the CDL image will be saved to the folder '.\cdl'.

```
D:\pyMETRIC>python code\download\download_cdl.py -y 2015
```

### LANDFIRE (optional)

LANDFIRE data will not be used for this example, but the following command will downloaded the CONUS-wide LANDFIRE image.  By default, the LANDFIRE image will be saved to the folder ".\landfire".

```
D:\pyMETRIC>python code\download\download_landfire.py -v 140
```

### Available Water Capacity (AWC)

CONUS-wide AWC rasters can be downloaded to the appropriate directory using the following script:
```
D:\pyMETRIC>python code\download\download_soils.py
```
CONUS-wide AWC rasters can be manually downloaded from the following URLs:
* STATSGO - [https://storage.googleapis.com/openet/statsgo/AWC_WTA_0to10cm_statsgo.tif](https://storage.googleapis.com/openet/statsgo/AWC_WTA_0to10cm_statsgo.tif)
* SSURGO - [https://storage.googleapis.com/openet/ssurgo/AWC_WTA_0to10cm_composite.tif](https://storage.googleapis.com/openet/ssurgo/AWC_WTA_0to10cm_composite.tif)

## Daily Weather Data

Weather data are stored in multi-band rasters with a separate band for each day of year (DOY).  This was primarily done to reduce the total number of files generated but also helps simplify the data extraction within the Python code.

### GRIDMET

Generate elevation, latitude, and longitude rasters.
```
D:\pyMETRIC>python code\gridmet\gridmet_ancillary.py
```

The following command will download the precipitation (PPT) and reference ET (ETr) components variable NetCDF files.  Make sure to always download and prep a few extra months of data before the target date range in order have enough extra to spin-up the soil water balance.
```
D:\pyMETRIC>python code\gridmet\gridmet_download.py --start 2014-10-01 --end 2015-12-31
```

The following commands will generate daily reference ET (from the components variables) and precipitation IMG rasters.
```
D:\pyMETRIC>python code\gridmet\gridmet_daily_refet.py --start 2014-10-01 --end 2015-12-31 --extent harney\study_area\harney_wgs84z11.shp
D:\pyMETRIC>python code\gridmet\gridmet_daily_ppt.py --start 2014-10-01 --end 2015-12-31 --extent harney\study_area\harney_wgs84z11.shp
```

### Spatial CIMIS

Generate elevation, latitude, and longitude rasters.
```
D:\pyMETRIC>python code\cimis\cimis_ancillary.py
```

The following command will download the reference ET (ETr) components variable GZ files.  Make sure to always download and prep a few extra months of data before the target date range in order to spin-up the soil water balance.
```
D:\pyMETRIC>python code\cimis\cimis_download.py --start 2014-10-01 --end 2015-12-31
```

The ASCII rasters then need to be extracted from the GZ files and converted to IMG.
```
D:\pyMETRIC>python code\cimis\cimis_extract_convert.py --start 2014-10-01 --end 2015-12-31
```

The following commands will generate daily reference ET (from the components variables)
```
D:\pyMETRIC>python code\cimis\cimis_daily_refet.py --start 2014-10-01 --end 2015-12-31 --extent harney\study_area\harney_wgs84z11.shp
```

GRIDMET (or anothere data set) must still be used for the precipitation, since it is not provided with Spatial CIMIS.
```
D:\pyMETRIC>python code\gridmet\gridmet_download.py --start 2014-10-01 --end 2015-12-31 --vars pr
D:\pyMETRIC>python code\gridmet\gridmet_daily_ppt.py --start 2014-10-01 --end 2015-12-31 --extent harney\study_area\harney_wgs84z11.shp
```

## Hourly Weather Data

### NLDAS

```
D:\pyMETRIC>python code\nldas\nldas_ancillary.py
```

In order to download the NLDAS hourly data, you will need to create an [Earthdata login](https://urs.earthdata.nasa.gov/)

Begin downloading the NLDAS hourly GRB files.  All of the NLDAS variables for a single hour are stored in a single GRB file.  The "--landsat" parameter is set in order to limit the download to only those dates and times that are needed for the Landsat images in the study area and time period.  If you don't specify the "--landsat" parameter, the script will attempt to download all hourly data within the "--start" and "--end" range.
```
D:\pyMETRIC>python code\nldas\nldas_download.py <USERNAME> <PASSWORD> --start 2015-01-01 --end 2015-12-31  --landsat .\landsat
```

#### Reference ET (ETr)

The "--landsat" argument is optional at this point, since GRB files were only downloaded for Landsat dates in the previous step.  This flag can be useful for other projects if you have downloaded a more complete set of NLDAS data.  This code also supports the processing of both houlry ETo (Grass reference evapotranspiration) and ETr (Alfalfa reference evapotranspiration).  For the purposes of pyMETRIC, only ETr is needed

```
D:\pyMETRIC>python code\nldas\nldas_hourly_refet.py --start 2015-01-01 --end 2015-12-31 --extent harney\study_area\harney_wgs84z11.shp --landsat .\landsat
```

#### Vapor Pressure

```
D:\pyMETRIC>python code\nldas\nldas_hourly_ea.py --start 2015-01-01 --end 2015-12-31 --extent harney\study_area\harney_wgs84z11.shp --landsat .\landsat
```

#### Wind Speed

```
D:\pyMETRIC>python code\nldas\nldas_hourly_wind.py --start 2015-01-01 --end 2015-12-31 --extent harney\study_area\harney_wgs84z11.shp --landsat .\landsat
```

#### Additional Parameters

#### Optimization
:red_circle: Explain some of the NLDAS command line arguments

+ stats: Compute raster statistics
+ times: To minimize the amount of data that needs to be downloaded and stored in each daily file, the following three scripts can all be run with a "--times" argument to specify which hours to process.
+ te:  To minimize the amount of data that needs to be downloaded and stored in each daily file, a custom extent can be manually entered.  This argument requires the input of a western limit, southern limit, eastern limit, and northern limit (x-min, y-min, x-max, and y-max) in units of decimal degrees.


# Command Summary

Below is an example work flow for downloading all ancillary data needed to run pyMETRIC for the Harney Basin study area example.  Be aware that this is only an example and that variations in your installation directory or general setup may render these commands inoperable.  Provided that pyMETRIC has been installed in the D: directory, these commands or formatted so that they may be pasted into the Windows Command Prompt *(accessible by pressing 'Windows Key'+R on your keyboard and typing "CMD" in the 'Run' window)*.

#### Download gridMET data

This section will perform the downloading and processing of daily meteorological data necessary for calculating and interpolating evapotranspiration estimates, and must be downloaded for the entire period of interest.  For the purposes of this example, data is acquired for the entire year of 2015.

```
python D:\pyMETRIC\code\gridmet\gridmet_ancillary.py  
python D:\pyMETRIC\code\gridmet\gridmet_download.py --start 2015-01-01 --end 2015-12-31  
python D:\pyMETRIC\code\gridmet\gridmet_daily_refet.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf D:\pyMETRIC\gridmet\netcdf  
python D:\pyMETRIC\code\gridmet\gridmet_daily_temp.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf D:\pyMETRIC\gridmet\netcdf  
python D:\pyMETRIC\code\gridmet\gridmet_daily_ppt.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf D:\pyMETRIC\gridmet\netcdf  
python D:\pyMETRIC\code\gridmet\gridmet_daily_ea.py --start 2015-01-01 --end 2015-12-31 -o -te -120.1 42.4 -118.3 44.3 --netcdf D:\pyMETRIC\gridmet\netcdf  
```

#### Download NLDAS data

This section will perform the downloading and processing of hourly meteorological data necessary for calculating and interpolating evapotranspiration estimates, and must be downloaded for the entire period of interest.  For the purposes of this example, data is acquired for the entire year of 2015.

__Please note that that an [Earthdata username and password](https://urs.earthdata.nasa.gov/) must be acquired in order to download NLDAS data.__

```
python D:\pyMETRIC\code\nldas\nldas_ancillary.py
python D:\pyMETRIC\code\nldas\nldas_download.py <Earthdata USERNAME> <Earthdata PASSWORD> --start 2015-01-01 --end 2015-12-31  
python D:\pyMETRIC\code\nldas\nldas_hourly_ea.py --start 2015-01-01 --end 2015-12-31 -te -120.1 42.4 -118.3 44.3 --grb D:\pyMETRIC\nldas\grb  
python D:\pyMETRIC\code\nldas\nldas_hourly_refet.py --start 2015-01-01 --end 2015-12-31 -te -120.1 42.4 -118.3 44.3 --grb D:\pyMETRIC\nldas\grb  
python D:\pyMETRIC\code\nldas\nldas_hourly_wind.py --start 2015-01-01 --end 2015-12-31 -te -120.1 42.4 -118.3 44.3 --grb D:\pyMETRIC\nldas\grb  
```
#### Download Landcover/Land surface data

This section downloads land surface data. This data includes information on elevation, agricultural land delineation, land cover, and Landsat footprints.

```
python D:\pyMETRIC\code\download\download_footprints.py
python D:\pyMETRIC\code\download\download_ned.py --extent D:\pyMETRIC\harney\study_area\harney_wgs84z11.shp  
python D:\pyMETRIC\code\download\download_cdl.py --year 2015  
python D:\pyMETRIC\code\download\download_landfire.py  
python D:\pyMETRIC\code\download\download_nlcd.py  
```

#### Landsat data download and prep

pyMETRIC uses the [Landsat578](https://github.com/dgketchum/Landsat578) package for downloading Landsat imagery products.

__Credentials must be established from the [USGS Earth Resources Observation and Science (EROS)](https://ers.cr.usgs.gov/register/) site and stored in .txt file prior to running Landsat data download script__

```
python landsat --satellite LC8 --start 2015-01-01 --end 2015-12-31 --path 43 --row 30 -o D:\pyMETRIC\harney\landsat --credentials *Your individual credential file* --zipped
python landsat --satellite LE7 --start 2015-01-01 --end 2015-12-31 --path 43 --row 30 -o D:\pyMETRIC\harney\landsat --credentials *Your individual credential file* --zipped
python D:\pyMETRIC\landsat\landsat_image_organize.py
```