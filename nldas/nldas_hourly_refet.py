#--------------------------------
# Name:         nldas_hourly_refet.py
# Purpose:      Calculate NLDAS ETr/ETo
# Python:       2.7
#--------------------------------

import argparse
import datetime as dt
import logging
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
from osgeo import gdal

from support import et_common
from support import gdal_common as gdc
from support import parse_int_set, valid_date

np.seterr(invalid='ignore')


def main(grb_ws=os.getcwd(), ancillary_ws=os.getcwd(), output_ws=os.getcwd(),
         etr_flag=False, eto_flag=False, landsat_ws=None,
         start_date=None, end_date=None, times_str='',
         extent_path=None, output_extent=None, daily_flag=True,
         stats_flag=True, overwrite_flag=False):
    """Compute hourly ETr/ETo from NLDAS data

    Args:
        grb_ws (str): folder of NLDAS GRB files
        ancillary_ws (str): folder of ancillary rasters
        output_ws (str): folder of output rasters
        etr_flag (bool): if True, compute alfalfa reference ET (ETr)
        eto_flag (bool): if True, compute grass reference ET (ETo)
        landsat_ws (str): folder of Landsat scenes or tar.gz files
        start_date (str): ISO format date (YYYY-MM-DD)
        end_date (str): ISO format date (YYYY-MM-DD)
        times (str): comma separated values and/or ranges of UTC hours
            (i.e. "1, 2, 5-8")
            Parsed with python_common.parse_int_set()
        extent_path (str): file path defining the output extent
        output_extent (list): decimal degrees values defining output extent
        daily_flag (bool): if True, save daily ETr/ETo sum raster.
            Default is True
        stats_flag (bool): if True, compute raster statistics.
            Default is True.
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nComputing NLDAS hourly ETr/ETo')
    np.seterr(invalid='ignore')

    # Compute ETr and/or ETo
    if not etr_flag and not eto_flag:
        logging.info('  ETo/ETr flag(s) not set, defaulting to ETr')
        etr_flag = True

    # If a date is not set, process 2017
    try:
        start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
        logging.debug('  Start date: {}'.format(start_dt))
    except:
        start_dt = dt.datetime(2017, 1, 1)
        logging.info('  Start date: {}'.format(start_dt))
    try:
        end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
        logging.debug('  End date:   {}'.format(end_dt))
    except:
        end_dt = dt.datetime(2017, 12, 31)
        logging.info('  End date:   {}'.format(end_dt))

    # Only process a specific hours
    if not times_str:
        time_list = range(0, 24, 1)
    else:
        time_list = list(parse_int_set(times_str))
    time_list = ['{:02d}00'.format(t) for t in time_list]

    etr_folder = 'etr'
    eto_folder = 'eto'
    hour_fmt = '{}_{:04d}{:02d}{:02d}_hourly_nldas.img'
    # hour_fmt = '{}_{:04d}{:02d}{:02d}_{4:04d}_nldas.img'
    day_fmt = '{}_{:04d}{:02d}{:02d}_nldas.img'
    # input_fmt = 'NLDAS_FORA0125_H.A{:04d}{:02d}{:02d}.{}.002.grb'
    input_re = re.compile(
        'NLDAS_FORA0125_H.A(?P<YEAR>\d{4})(?P<MONTH>\d{2})' +
        '(?P<DAY>\d{2}).(?P<TIME>\d{4}).002.grb$')

    # Assume NLDAS is NAD83
    # input_epsg = 'EPSG:4269'

    # Ancillary raster paths
    mask_path = os.path.join(ancillary_ws, 'nldas_mask.img')
    elev_path = os.path.join(ancillary_ws, 'nldas_elev.img')
    lat_path = os.path.join(ancillary_ws, 'nldas_lat.img')
    lon_path = os.path.join(ancillary_ws, 'nldas_lon.img')

    # Build a date list from landsat_ws scene folders or tar.gz files
    date_list = []
    if landsat_ws is not None and os.path.isdir(landsat_ws):
        logging.info('\nReading dates from Landsat IDs')
        logging.info('  {}'.format(landsat_ws))
        landsat_re = re.compile(
            '^(?:LT04|LT05|LE07|LC08)_(?:\d{3})(?:\d{3})_' +
            '(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})')
        for root, dirs, files in os.walk(landsat_ws, topdown=True):
            # If root matches, don't explore subfolders
            try:
                landsat_match = landsat_re.match(os.path.basename(root))
                date_list.append(dt.datetime.strptime(
                    '_'.join(landsat_match.groups()), '%Y_%m_%d').date().isoformat())
                dirs[:] = []
            except:
                pass

            for file in files:
                try:
                    landsat_match = landsat_re.match(file)
                    date_list.append(dt.datetime.strptime(
                        '_'.join(landsat_match.groups()), '%Y_%m_%d').date().isoformat())
                except:
                    pass
        date_list = sorted(list(set(date_list)))
    # elif landsat_ws is not None and os.path.isfile(landsat_ws):
    #     with open(landsat_ws) as landsat_f:

    # This allows GDAL to throw Python Exceptions
    # gdal.UseExceptions()
    # mem_driver = gdal.GetDriverByName('MEM')

    # Get the NLDAS spatial reference from the mask raster
    nldas_ds = gdal.Open(mask_path)
    nldas_osr = gdc.raster_ds_osr(nldas_ds)
    nldas_proj = gdc.osr_proj(nldas_osr)
    nldas_cs = gdc.raster_ds_cellsize(nldas_ds, x_only=True)
    nldas_extent = gdc.raster_ds_extent(nldas_ds)
    nldas_geo = nldas_extent.geo(nldas_cs)
    nldas_x, nldas_y = nldas_extent.origin()
    nldas_ds = None
    logging.debug('  Projection: {}'.format(nldas_proj))
    logging.debug('  Cellsize: {}'.format(nldas_cs))
    logging.debug('  Geo: {}'.format(nldas_geo))
    logging.debug('  Extent: {}'.format(nldas_extent))

    # Subset data to a smaller extent
    if output_extent is not None:
        logging.info('\nComputing subset extent & geo')
        logging.debug('  Extent: {}'.format(output_extent))
        nldas_extent = gdc.Extent(output_extent)
        nldas_extent.adjust_to_snap('EXPAND', nldas_x, nldas_y, nldas_cs)
        nldas_geo = nldas_extent.geo(nldas_cs)
        logging.debug('  Geo: {}'.format(nldas_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    elif extent_path is not None:
        logging.info('\nComputing subset extent & geo')
        if extent_path.lower().endswith('.shp'):
            nldas_extent = gdc.feature_path_extent(extent_path)
            extent_osr = gdc.feature_path_osr(extent_path)
            extent_cs = None
        else:
            nldas_extent = gdc.raster_path_extent(extent_path)
            extent_osr = gdc.raster_path_osr(extent_path)
            extent_cs = gdc.raster_path_cellsize(extent_path, x_only=True)
        nldas_extent = gdc.project_extent(
            nldas_extent, extent_osr, nldas_osr, extent_cs)
        nldas_extent.adjust_to_snap('EXPAND', nldas_x, nldas_y, nldas_cs)
        nldas_geo = nldas_extent.geo(nldas_cs)
        logging.debug('  Geo: {}'.format(nldas_geo))
        logging.debug('  Extent: {}'.format(nldas_extent))
    logging.debug('')

    # Read the NLDAS mask array if present
    if mask_path and os.path.isfile(mask_path):
        mask_array, mask_nodata = gdc.raster_to_array(
            mask_path, mask_extent=nldas_extent, fill_value=0,
            return_nodata=True)
        mask_array = mask_array != mask_nodata
    else:
        mask_array = None

    # Read ancillary arrays (or subsets?)
    elev_array = gdc.raster_to_array(
        elev_path, mask_extent=nldas_extent, return_nodata=False)
    # pair_array = et_common.air_pressure_func(elev_array)
    lat_array = gdc.raster_to_array(
        lat_path, mask_extent=nldas_extent, return_nodata=False)
    lon_array = gdc.raster_to_array(
        lon_path, mask_extent=nldas_extent, return_nodata=False)

    # Hourly RefET functions expects lat/lon in radians
    lat_array *= (math.pi / 180)
    lon_array *= (math.pi / 180)

    # Build output folder
    etr_ws = os.path.join(output_ws, etr_folder)
    eto_ws = os.path.join(output_ws, eto_folder)
    if etr_flag and not os.path.isdir(etr_ws):
        os.makedirs(etr_ws)
    if eto_flag and not os.path.isdir(eto_ws):
        os.makedirs(eto_ws)

    # DEADBEEF - Instead of processing all available files, the following
    #   tools will process files for target dates
    # for input_dt in date_range(start_dt, end_dt + dt.timedelta(1)):
    #     logging.info(input_dt.date())

    # Iterate all available files and check dates if necessary
    # Each sub folder in the main folder has all imagery for 1 day
    #   (in UTC time)
    # The path for each subfolder is the /YYYY/DOY
    errors = defaultdict(list)
    for root, folders, files in os.walk(grb_ws):
        root_split = os.path.normpath(root).split(os.sep)

        # If the year/doy is outside the range, skip
        if (re.match('\d{4}', root_split[-2]) and
                re.match('\d{3}', root_split[-1])):
            root_dt = dt.datetime.strptime('{}_{}'.format(
                root_split[-2], root_split[-1]), '%Y_%j')
            logging.info('{}'.format(root_dt.date()))
            if ((start_dt is not None and root_dt < start_dt) or
                    (end_dt is not None and root_dt > end_dt)):
                continue
            elif date_list and root_dt.date().isoformat() not in date_list:
                continue
        # If the year is outside the range, don't search subfolders
        elif re.match('\d{4}', root_split[-1]):
            root_year = int(root_split[-1])
            logging.info('Year: {}'.format(root_year))
            if ((start_dt is not None and root_year < start_dt.year) or
                    (end_dt is not None and root_year > end_dt.year)):
                folders[:] = []
            else:
                folders[:] = sorted(folders)
            continue
        else:
            continue
        logging.debug('  {}'.format(root))

        # Start off assuming every file needs to be processed
        day_skip_flag = False

        # Build output folders if necessary
        etr_year_ws = os.path.join(etr_ws, str(root_dt.year))
        eto_year_ws = os.path.join(eto_ws, str(root_dt.year))
        if etr_flag and not os.path.isdir(etr_year_ws):
            os.makedirs(etr_year_ws)
        if eto_flag and not os.path.isdir(eto_year_ws):
            os.makedirs(eto_year_ws)

        # Build daily total paths
        etr_day_path = os.path.join(etr_year_ws, day_fmt.format(
            'etr', root_dt.year, root_dt.month, root_dt.day))
        eto_day_path = os.path.join(eto_year_ws, day_fmt.format(
            'eto', root_dt.year, root_dt.month, root_dt.day))
        etr_hour_path = os.path.join(etr_year_ws, hour_fmt.format(
            'etr', root_dt.year, root_dt.month, root_dt.day))
        eto_hour_path = os.path.join(eto_year_ws, hour_fmt.format(
            'eto', root_dt.year, root_dt.month, root_dt.day))
        # logging.debug('  {}'.format(etr_hour_path))

        # If daily ETr/ETo files are present, day can be skipped
        if not overwrite_flag and daily_flag:
            if etr_flag and not os.path.isfile(etr_day_path):
                pass
            elif eto_flag and not os.path.isfile(eto_day_path):
                pass
            else:
                day_skip_flag = True

        # If the hour and daily files don't need to be made, skip the day
        if not overwrite_flag:
            if etr_flag and not os.path.isfile(etr_hour_path):
                pass
            elif eto_flag and not os.path.isfile(eto_hour_path):
                pass
            elif day_skip_flag:
                logging.debug('  File(s) already exist, skipping')
                continue

        # Create a single raster for each day with 24 bands
        # Each time step will be stored in a separate band
        if etr_flag:
            logging.debug('  {}'.format(etr_day_path))
            gdc.build_empty_raster(
                etr_hour_path, band_cnt=24, output_dtype=np.float32,
                output_proj=nldas_proj, output_cs=nldas_cs,
                output_extent=nldas_extent, output_fill_flag=True)
        if eto_flag:
            logging.debug('  {}'.format(eto_day_path))
            gdc.build_empty_raster(
                eto_hour_path, band_cnt=24, output_dtype=np.float32,
                output_proj=nldas_proj, output_cs=nldas_cs,
                output_extent=nldas_extent, output_fill_flag=True)

        # Sum all ETr/ETo images in each folder to generate a UTC day total
        etr_day_array = 0
        eto_day_array = 0

        # Process each hour file
        for input_name in sorted(files):
            logging.info('  {}'.format(input_name))
            input_match = input_re.match(input_name)
            if input_match is None:
                logging.debug('    Regular expression didn\'t match, skipping')
                continue
            input_dt = dt.datetime(
                int(input_match.group('YEAR')),
                int(input_match.group('MONTH')),
                int(input_match.group('DAY')))
            input_doy = int(input_dt.strftime('%j'))
            time_str = input_match.group('TIME')
            band_num = int(time_str[:2]) + 1
            # if start_dt is not None and input_dt < start_dt:
            #     continue
            # elif end_dt is not None and input_dt > end_dt:
            #     continue
            # elif date_list and input_dt.date().isoformat() not in date_list:
            #     continue
            if not daily_flag and time_str not in time_list:
                logging.debug('    Time not in list and not daily, skipping')
                continue

            input_path = os.path.join(root, input_name)
            logging.debug('    Time: {} {}'.format(input_dt.date(), time_str))
            logging.debug('    Band: {}'.format(band_num))

            # Determine band numbering/naming
            try:
                input_band_dict = grib_band_names(input_path)
            except RuntimeError as e:
                errors[input_path].append(e)
                logging.error(
                    ' RuntimeError: {} Skipping: {}'.format(e, input_path))
                continue

            # Read input bands
            input_ds = gdal.Open(input_path)

            # Temperature should be in C for et_common.refet_hourly_func()
            if 'Temperature [K]' in input_band_dict.keys():
                temp_band_units = 'K'
                temp_array = gdc.raster_ds_to_array(
                    input_ds, band=input_band_dict['Temperature [K]'],
                    mask_extent=nldas_extent, return_nodata=False)
            elif 'Temperature [C]' in input_band_dict.keys():
                temp_band_units = 'C'
                temp_array = gdc.raster_ds_to_array(
                    input_ds, band=input_band_dict['Temperature [C]'],
                    mask_extent=nldas_extent, return_nodata=False)
            else:
                logging.error('Unknown Temperature units, skipping')
                logging.error('  {}'.format(input_band_dict.keys()))
                continue

            # DEADBEEF - Having issue with T appearing to be C but labeled as K
            # Try to determine temperature units from values
            temp_mean = float(np.nanmean(temp_array))
            temp_units_dict = {20: 'C', 293: 'K'}
            temp_array_units = temp_units_dict[
                min(temp_units_dict, key=lambda x:abs(x - temp_mean))]
            if temp_array_units == 'K' and temp_band_units == 'K':
                logging.debug('  Converting temperature from K to C')
                temp_array -= 273.15
            elif temp_array_units == 'C' and temp_band_units == 'C':
                pass
            elif temp_array_units == 'C' and temp_band_units == 'K':
                logging.debug(
                    ('  Temperature units are K in the GRB band name, ' +
                     'but values appear to be C\n    Mean temperature: {:.2f}\n' +
                     '  Values will NOT be adjusted').format(temp_mean))
            elif temp_array_units == 'K' and temp_band_units == 'C':
                logging.debug(
                    ('  Temperature units are C in the GRB band name, ' +
                     'but values appear to be K\n    Mean temperature: {:.2f}\n' +
                     '  Values will be adjusted from K to C').format(temp_mean))
                temp_array -= 273.15
            try:
                sph_array = gdc.raster_ds_to_array(
                    input_ds,
                    band=input_band_dict['Specific humidity [kg/kg]'],
                    mask_extent=nldas_extent, return_nodata=False)
                rs_array = gdc.raster_ds_to_array(
                    input_ds,
                    band=input_band_dict['Downward shortwave radiation flux [W/m^2]'],
                    mask_extent=nldas_extent, return_nodata=False)
                wind_u_array = gdc.raster_ds_to_array(
                    input_ds,
                    band=input_band_dict['u-component of wind [m/s]'],
                    mask_extent=nldas_extent, return_nodata=False)
                wind_v_array = gdc.raster_ds_to_array(
                    input_ds,
                    band=input_band_dict['v-component of wind [m/s]'],
                    mask_extent=nldas_extent, return_nodata=False)
                input_ds = None
            except KeyError as e:
                errors[input_path].append(e)
                logging.error(' KeyError: {} Skipping: {}'.format(
                    e, input_ds.GetDescription()))
                continue

            rs_array *= 0.0036  # W m-2 to MJ m-2 hr-1
            wind_array = np.sqrt(wind_u_array ** 2 + wind_v_array ** 2)
            del wind_u_array, wind_v_array

            # ETr
            if etr_flag:
                etr_array = et_common.refet_hourly_func(
                    temp_array, sph_array, rs_array, wind_array,
                    zw=10, elev=elev_array, lat=lat_array, lon=lon_array,
                    doy=input_doy, time=int(time_str) / 100, ref_type='ETR')
                if daily_flag:
                    etr_day_array += etr_array
                if time_str in time_list:
                    gdc.array_to_comp_raster(
                        etr_array.astype(np.float32), etr_hour_path,
                        band=band_num, stats_flag=False)
                    del etr_array

            # ETo
            if eto_flag:
                eto_array = et_common.refet_hourly_func(
                    temp_array, sph_array, rs_array, wind_array,
                    zw=10, elev=elev_array, lat=lat_array, lon=lon_array,
                    doy=input_doy, time=int(time_str) / 100, ref_type='ETO')
                if eto_flag and daily_flag:
                    eto_day_array += eto_array
                if eto_flag and time_str in time_list:
                    gdc.array_to_comp_raster(
                        eto_array.astype(np.float32), eto_hour_path,
                        band=band_num, stats_flag=False)
                    del eto_array

            del temp_array, sph_array, rs_array, wind_array

        if stats_flag and etr_flag:
            gdc.raster_statistics(etr_hour_path)
        if stats_flag and eto_flag:
            gdc.raster_statistics(eto_hour_path)

        # Save the projected ETr/ETo as 32-bit floats
        if not day_skip_flag and daily_flag:
            if etr_flag:
                try:
                    gdc.array_to_raster(
                        etr_day_array.astype(np.float32), etr_day_path,
                        output_geo=nldas_geo, output_proj=nldas_proj,
                        stats_flag=stats_flag)
                except AttributeError:
                    pass
            if eto_flag:
                try:

                    gdc.array_to_raster(
                        eto_day_array.astype(np.float32), eto_day_path,
                        output_geo=nldas_geo, output_proj=nldas_proj,
                        stats_flag=stats_flag)
                except AttributeError:
                    pass

        del etr_day_array, eto_day_array

    if len(errors) > 0:
        logging.info('\nThe following errors were encountered:')
        for key, value in errors.items():
            logging.error(' Filepath: {}, error: {}'.format(key, value))

    logging.debug('\nScript Complete')


def grib_band_names(grib_path):
    """Return a dictionary of all GRIB_ELEMENT and band number items"""
    band_dict = dict()
    grib_ds = gdal.Open(grib_path)
    for i in range(grib_ds.RasterCount):
        band_meta = [
            x for x in grib_ds.GetRasterBand(i + 1).GetMetadata_List()
            if 'GRIB_COMMENT' in x][0]
            # if 'GRIB_ELEMENT' in x][0]
        band_dict[band_meta.split('=')[1]] = i + 1
    grib_ds = None
    return band_dict


def arg_parse():
    """

    Base all default folders from script location
        scripts: ./pyMETRIC/tools/nldas
        tools:    ./pyMETRIC/tools
        output:  ./pyMETRIC/nldas
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    nldas_folder = os.path.join(project_folder, 'nldas')

    parser = argparse.ArgumentParser(
        description='NLDAS hourly reference ETr/ETo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--grb', default=os.path.join(nldas_folder, 'grb'),
        metavar='PATH', help='Input GRB folder path')
    parser.add_argument(
        '--ancillary', default=os.path.join(nldas_folder, 'ancillary'),
        metavar='PATH', help='Ancillary raster folder path')
    parser.add_argument(
        '--output', default=nldas_folder,
        metavar='PATH', help='Output raster folder path')
    parser.add_argument(
        '--etr', default=False, action="store_true",
        help='Compute alfalfa reference ET (ETr)')
    parser.add_argument(
        '--eto', default=False, action="store_true",
        help='Compute grass reference ET (ETo)')
    parser.add_argument(
        '--landsat', default=None, metavar='PATH',
        help='Folder of Landsat scenes or tar.gz files')
    parser.add_argument(
        '--start', default='2017-01-01', type=valid_date,
        help='Start date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--end', default='2017-12-31', type=valid_date,
        help='End date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--times', default='0-23', type=str,
        help='Time list and/or range (0-23 for all times)')
    parser.add_argument(
        '--extent', default=None, metavar='PATH',
        help='Subset extent path')
    parser.add_argument(
        '-te', default=None, type=float, nargs=4,
        metavar=('xmin', 'ymin', 'xmax', 'ymax'),
        help='Subset extent in decimal degrees')
    parser.add_argument(
        '--no_daily', default=False, action="store_true",
        help='Don\'t save daily ETr/ETo sum rasters')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Default is to build daily ETr/ETo sum rasters
    #     (opposite of --no_daily default=False)
    args.daily = not args.no_daily

    # Convert relative paths to absolute paths
    if args.grb and os.path.isdir(os.path.abspath(args.grb)):
        args.grb = os.path.abspath(args.grb)
    if args.ancillary and os.path.isdir(os.path.abspath(args.ancillary)):
        args.ancillary = os.path.abspath(args.ancillary)
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
    if args.landsat and os.path.isdir(os.path.abspath(args.landsat)):
        args.landsat = os.path.abspath(args.landsat)
    if args.extent and os.path.isfile(os.path.abspath(args.extent)):
        args.extent = os.path.abspath(args.extent)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{:<20s} {}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{:<20s} {}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(grb_ws=args.grb, ancillary_ws=args.ancillary, output_ws=args.output,
         eto_flag=args.eto, etr_flag=args.etr, landsat_ws=args.landsat,
         start_date=args.start, end_date=args.end, times_str=args.times,
         extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite,
         daily_flag=args.daily)
