#--------------------------------
# Name:         daymet_daily_ea.py
# Purpose:      Extract DAYMET vapor pressure
# Python:       2.7
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import sys

import netCDF4
import numpy as np
from osgeo import gdal

import gdal_common as gdc
import et_common
from python_common import date_range, valid_date

np.seterr(invalid='ignore')


def main(netcdf_ws=os.getcwd(), ancillary_ws=os.getcwd(),
         output_ws=os.getcwd(), start_date=None, end_date=None,
         extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Extract DAYMET temperature

    Args:
        netcdf_ws (str): folder of DAYMET netcdf files
        ancillary_ws (str): folder of ancillary rasters
        output_ws (str): folder of output rasters
        start_date (str): ISO format date (YYYY-MM-DD)
        end_date (str): ISO format date (YYYY-MM-DD)
        extent_path (str): file path defining the output extent
        output_extent (list): decimal degrees values defining output extent
        stats_flag (bool): if True, compute raster statistics.
            Default is True.
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nExtracting DAYMET vapor pressure')

    # If a date is not set, process 2015
    try:
        start_dt = dt.datetime.strptime(start_date, '%Y-%m-%d')
        logging.debug('  Start date: {}'.format(start_dt))
    except:
        start_dt = dt.datetime(2015, 1, 1)
        logging.info('  Start date: {}'.format(start_dt))
    try:
        end_dt = dt.datetime.strptime(end_date, '%Y-%m-%d')
        logging.debug('  End date:   {}'.format(end_dt))
    except:
        end_dt = dt.datetime(2015, 12, 31)
        logging.info('  End date:   {}'.format(end_dt))

    # Get DAYMET spatial reference from an ancillary raster
    mask_raster = os.path.join(ancillary_ws, 'daymet_mask.img')
    elev_raster = os.path.join(ancillary_ws, 'daymet_elev.img')

    daymet_re = re.compile('daymet_v3_(?P<VAR>\w+)_(?P<YEAR>\d{4})_na.nc4$')

    # DAYMET band name dictionary
    # daymet_band_dict = dict()
    # daymet_band_dict['prcp'] = 'precipitation_amount'
    # daymet_band_dict['srad'] = 'surface_downwelling_shortwave_flux_in_air'
    # daymet_band_dict['sph'] = 'specific_humidity'
    # daymet_band_dict['tmin'] = 'air_temperature'
    # daymet_band_dict['tmax'] = 'air_temperature'

    # Get extent/geo from mask raster
    daymet_ds = gdal.Open(mask_raster)
    daymet_osr = gdc.raster_ds_osr(daymet_ds)
    daymet_proj = gdc.osr_proj(daymet_osr)
    daymet_cs = gdc.raster_ds_cellsize(daymet_ds, x_only=True)
    daymet_extent = gdc.raster_ds_extent(daymet_ds)
    daymet_geo = daymet_extent.geo(daymet_cs)
    daymet_x, daymet_y = daymet_extent.origin()
    daymet_ds = None
    logging.debug('  Projection: {}'.format(daymet_proj))
    logging.debug('  Cellsize: {}'.format(daymet_cs))
    logging.debug('  Geo: {}'.format(daymet_geo))
    logging.debug('  Extent: {}'.format(daymet_extent))
    logging.debug('  Origin: {} {}'.format(daymet_x, daymet_y))

    # Subset data to a smaller extent
    if output_extent is not None:
        logging.info('\nComputing subset extent & geo')
        logging.debug('  Extent: {}'.format(output_extent))
        # Assume input extent is in decimal degrees
        output_extent = gdc.project_extent(
            gdc.Extent(output_extent), gdc.epsg_osr(4326), daymet_osr, 0.001)
        output_extent = gdc.intersect_extents([daymet_extent, output_extent])
        output_extent.adjust_to_snap('EXPAND', daymet_x, daymet_y, daymet_cs)
        output_geo = output_extent.geo(daymet_cs)
        logging.debug('  Geo: {}'.format(output_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    elif extent_path is not None:
        logging.info('\nComputing subset extent & geo')
        if extent_path.lower().endswith('.shp'):
            output_extent = gdc.feature_path_extent(extent_path)
            extent_osr = gdc.feature_path_osr(extent_path)
            extent_cs = None
        else:
            output_extent = gdc.raster_path_extent(extent_path)
            extent_osr = gdc.raster_path_osr(extent_path)
            extent_cs = gdc.raster_path_cellsize(extent_path, x_only=True)
        output_extent = gdc.project_extent(
            output_extent, extent_osr, daymet_osr, extent_cs)
        output_extent = gdc.intersect_extents([daymet_extent, output_extent])
        output_extent.adjust_to_snap('EXPAND', daymet_x, daymet_y, daymet_cs)
        output_geo = output_extent.geo(daymet_cs)
        logging.debug('  Geo: {}'.format(output_geo))
        logging.debug('  Extent: {}'.format(output_extent))
    else:
        output_extent = daymet_extent.copy()
        output_geo = daymet_geo[:]
    # output_shape = output_extent.shape(cs=daymet_cs)
    xi, yi = gdc.array_geo_offsets(daymet_geo, output_geo, daymet_cs)
    output_rows, output_cols = output_extent.shape(daymet_cs)
    logging.debug('  Shape: {} {}'.format(output_rows, output_cols))
    logging.debug('  Offsets: {} {} (x y)'.format(xi, yi))

    # Read the elevation array
    elev_array = gdc.raster_to_array(
        elev_raster, mask_extent=output_extent, return_nodata=False)
    pair_array = et_common.air_pressure_func(elev_array)
    del elev_array

    # Process each variable
    input_var = 'vp'
    output_var = 'ea'
    logging.info("\nVariable: {}".format(input_var))

    # Build output folder
    var_ws = os.path.join(output_ws, output_var)
    if not os.path.isdir(var_ws):
        os.makedirs(var_ws)

    # Process each file in the input workspace
    for input_name in sorted(os.listdir(netcdf_ws)):
        logging.debug("{}".format(input_name))
        input_match = daymet_re.match(input_name)
        if not input_match:
            logging.debug('  Regular expression didn\'t match, skipping')
            continue
        elif input_match.group('VAR') != input_var:
            logging.debug('  Variable didn\'t match, skipping')
            continue
        year_str = input_match.group('YEAR')
        logging.info("  Year: {}".format(year_str))
        year_int = int(year_str)
        year_days = int(dt.datetime(year_int, 12, 31).strftime('%j'))
        if start_dt is not None and year_int < start_dt.year:
            logging.debug('    Before start date, skipping')
            continue
        elif end_dt is not None and year_int > end_dt.year:
            logging.debug('    After end date, skipping')
            continue

        # Build input file path
        input_raster = os.path.join(netcdf_ws, input_name)
        # if not os.path.isfile(input_raster):
        #     logging.debug(
        #         '    Input raster doesn\'t exist, skipping    {}'.format(
        #             input_raster))
        #     continue

        # Build output folder
        output_year_ws = os.path.join(var_ws, year_str)
        if not os.path.isdir(output_year_ws):
            os.makedirs(output_year_ws)

        # Read in the DAYMET NetCDF file
        input_nc_f = netCDF4.Dataset(input_raster, 'r')
        # logging.debug(input_nc_f.variables)

        # Check all valid dates in the year
        year_dates = date_range(
            dt.datetime(year_int, 1, 1), dt.datetime(year_int + 1, 1, 1))
        for date_dt in year_dates:
            if start_dt is not None and date_dt < start_dt:
                logging.debug('  {} - before start date, skipping'.format(
                    date_dt.date()))
                continue
            elif end_dt is not None and date_dt > end_dt:
                logging.debug('  {} - after end date, skipping'.format(
                    date_dt.date()))
                continue
            else:
                logging.info('  {}'.format(date_dt.date()))

            output_path = os.path.join(
                output_year_ws, '{}_{}_daymet.img'.format(
                    output_var, date_dt.strftime('%Y%m%d')))
            if os.path.isfile(output_path):
                logging.debug('    {}'.format(output_path))
                if not overwrite_flag:
                    logging.debug('    File already exists, skipping')
                    continue
                else:
                    logging.debug('    File already exists, removing existing')
                    os.remove(output_path)

            doy = int(date_dt.strftime('%j'))
            doy_i = range(1, year_days + 1).index(doy)

            # Arrays are being read as masked array with a fill value of -9999
            # Convert to basic numpy array arrays with nan values
            try:
                input_ma = input_nc_f.variables[input_var][
                    doy_i, yi: yi + output_rows, xi: xi + output_cols]
            except IndexError:
                logging.info('    date not in netcdf, skipping')
                continue
            input_nodata = float(input_ma.fill_value)
            sph_array = input_ma.data.astype(np.float32)
            sph_array[sph_array == input_nodata] = np.nan

            # Compute ea [kPa] from specific humidity [kg/kg]
            ea_array = (sph_array * pair_array) / (0.622 + 0.378 * sph_array)

            # Save the array as 32-bit floats
            gdc.array_to_raster(
                ea_array.astype(np.float32), output_path,
                output_geo=output_geo, output_proj=daymet_proj,
                stats_flag=stats_flag)

            del input_ma, ea_array, sph_array
        input_nc_f.close()
        del input_nc_f

    logging.debug('\nScript Complete')


def arg_parse():
    """

    Base all default folders from script location
        scripts: ./pyMETRIC/tools/daymet
        tools:    ./pyMETRIC/tools
        output:  ./pyMETRIC/daymet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    daymet_folder = os.path.join(project_folder, 'daymet')

    parser = argparse.ArgumentParser(
        description='DAYMET daily vapor pressure',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--netcdf', default=os.path.join(daymet_folder, 'netcdf'),
        metavar='PATH', help='Input netCDF folder path')
    parser.add_argument(
        '--ancillary', default=os.path.join(daymet_folder, 'ancillary'),
        metavar='PATH', help='Ancillary raster folder path')
    parser.add_argument(
        '--output', default=daymet_folder,
        metavar='PATH', help='Output raster folder path')
    parser.add_argument(
        '--start', default='2015-01-01', type=valid_date,
        help='Start date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--end', default='2015-12-31', type=valid_date,
        help='End date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--extent', default=None, metavar='PATH',
        help='Subset extent path')
    parser.add_argument(
        '-te', default=None, type=float, nargs=4,
        metavar=('xmin', 'ymin', 'xmax', 'ymax'),
        help='Subset extent in decimal degrees')
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

    # Convert relative paths to absolute paths
    if args.netcdf and os.path.isdir(os.path.abspath(args.netcdf)):
        args.netcdf = os.path.abspath(args.netcdf)
    if args.ancillary and os.path.isdir(os.path.abspath(args.ancillary)):
        args.ancillary = os.path.abspath(args.ancillary)
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
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

    main(netcdf_ws=args.netcdf, ancillary_ws=args.ancillary,
         output_ws=args.output, start_date=args.start, end_date=args.end,
         extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite)
