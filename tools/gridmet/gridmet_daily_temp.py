#--------------------------------
# Name:         gridmet_daily_temp.py
# Purpose:      Extract GRIDMET temperature
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
from python_common import date_range, valid_date


def main(netcdf_ws=os.getcwd(), ancillary_ws=os.getcwd(),
         output_ws=os.getcwd(), start_date=None, end_date=None,
         extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Extract GRIDMET temperature

    Args:
        netcdf_ws (str): folder of GRIDMET netcdf files
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
    logging.info('\nExtracting GRIDMET temperature')

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

    # Save GRIDMET lat, lon, and elevation arrays
    elev_raster = os.path.join(ancillary_ws, 'gridmet_elev.img')

    output_fmt = '{}_{}_daily_gridmet.img'
    gridmet_re = re.compile('(?P<VAR>\w+)_(?P<YEAR>\d{4}).nc$')

    # GRIDMET band name dictionary
    gridmet_band_dict = dict()
    gridmet_band_dict['pr'] = 'precipitation_amount'
    gridmet_band_dict['srad'] = 'surface_downwelling_shortwave_flux_in_air'
    gridmet_band_dict['sph'] = 'specific_humidity'
    gridmet_band_dict['tmmn'] = 'air_temperature'
    gridmet_band_dict['tmmx'] = 'air_temperature'
    gridmet_band_dict['vs'] = 'wind_speed'

    # Get extent/geo from elevation raster
    gridmet_ds = gdal.Open(elev_raster)
    gridmet_osr = gdc.raster_ds_osr(gridmet_ds)
    gridmet_proj = gdc.osr_proj(gridmet_osr)
    gridmet_cs = gdc.raster_ds_cellsize(gridmet_ds, x_only=True)
    gridmet_extent = gdc.raster_ds_extent(gridmet_ds)
    gridmet_full_geo = gridmet_extent.geo(gridmet_cs)
    gridmet_x, gridmet_y = gridmet_extent.origin()
    gridmet_ds = None
    logging.debug('  Projection: {}'.format(gridmet_proj))
    logging.debug('  Cellsize: {}'.format(gridmet_cs))
    logging.debug('  Geo: {}'.format(gridmet_full_geo))
    logging.debug('  Extent: {}'.format(gridmet_extent))

    # Subset data to a smaller extent
    if output_extent is not None:
        logging.info('\nComputing subset extent & geo')
        logging.debug('  Extent: {}'.format(output_extent))
        gridmet_extent = gdc.Extent(output_extent)
        gridmet_extent.adjust_to_snap(
            'EXPAND', gridmet_x, gridmet_y, gridmet_cs)
        gridmet_geo = gridmet_extent.geo(gridmet_cs)
        logging.debug('  Geo: {}'.format(gridmet_geo))
        logging.debug('  Extent: {}'.format(gridmet_extent))
    elif extent_path is not None:
        logging.info('\nComputing subset extent & geo')
        if extent_path.lower().endswith('.shp'):
            gridmet_extent = gdc.feature_path_extent(extent_path)
            extent_osr = gdc.feature_path_osr(extent_path)
            extent_cs = None
        else:
            gridmet_extent = gdc.raster_path_extent(extent_path)
            extent_osr = gdc.raster_path_osr(extent_path)
            extent_cs = gdc.raster_path_cellsize(extent_path, x_only=True)
        gridmet_extent = gdc.project_extent(
            gridmet_extent, extent_osr, gridmet_osr, extent_cs)
        gridmet_extent.adjust_to_snap(
            'EXPAND', gridmet_x, gridmet_y, gridmet_cs)
        gridmet_geo = gridmet_extent.geo(gridmet_cs)
        logging.debug('  Geo: {}'.format(gridmet_geo))
        logging.debug('  Extent: {}'.format(gridmet_extent))
    else:
        gridmet_geo = gridmet_full_geo

    # Get indices for slicing/clipping input arrays
    g_i, g_j = gdc.array_geo_offsets(
        gridmet_full_geo, gridmet_geo, cs=gridmet_cs)
    g_rows, g_cols = gridmet_extent.shape(cs=gridmet_cs)

    # Process each variable
    logging.info("")
    variables = ['tmmn', 'tmmx']
    for input_var in variables:
        logging.info("\nVariable: {}".format(input_var))
        output_var = input_var

        # Build output folder
        var_ws = os.path.join(output_ws, output_var)
        if not os.path.isdir(var_ws):
            os.makedirs(var_ws)

        # Process each file in the input workspace
        for input_name in sorted(os.listdir(netcdf_ws)):
            input_match = gridmet_re.match(input_name)
            if not input_match:
                logging.debug("{}".format(input_name))
                logging.debug('  Regular expression didn\'t match, skipping')
                continue
            elif input_match.group('VAR') != input_var:
                logging.debug("{}".format(input_name))
                logging.debug('  Variable didn\'t match, skipping')
                continue
            else:
                logging.info("{}".format(input_name))

            year_str = input_match.group('YEAR')
            logging.info("  {}".format(year_str))
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
            #         '  Input NetCDF doesn\'t exist, skipping    {}'.format(
            #             input_raster))
            #     continue

            # Create a single raster for each year with 365 bands
            # Each day will be stored in a separate band
            output_path = os.path.join(
                var_ws, output_fmt.format(output_var, year_str))
            logging.debug('  {}'.format(output_path))
            if os.path.isfile(output_path):
                logging.debug('    {}'.format(output_path))
                if not overwrite_flag:
                    logging.debug('    File already exists, skipping')
                    continue
                else:
                    logging.debug('    File already exists, removing existing')
                    os.remove(output_path)
            gdc.build_empty_raster(
                output_path, band_cnt=366, output_dtype=np.float32,
                output_proj=gridmet_proj, output_cs=gridmet_cs,
                output_extent=gridmet_extent, output_fill_flag=True)

            # Read in the GRIDMET NetCDF file
            # Immediatly clip input array to save memory
            input_nc_f = netCDF4.Dataset(input_raster, 'r')
            input_nc = input_nc_f.variables[gridmet_band_dict[input_var]][
                :, g_i:g_i + g_cols, g_j:g_j + g_rows].copy()
            input_nc = np.transpose(input_nc, (0, 2, 1))

            # A numpy array is returned when slicing a masked array
            #   if there are no masked pixels
            # This is a hack to force the numpy array back to a masked array
            if type(input_nc) != np.ma.core.MaskedArray:
                input_nc = np.ma.core.MaskedArray(
                    input_nc, np.zeros(input_nc.shape, dtype=bool))

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

                doy = int(date_dt.strftime('%j'))
                doy_i = range(1, year_days + 1).index(doy)

                # Arrays are read as masked array with a fill value of -9999
                # Convert to basic numpy array arrays with nan values
                try:
                    input_full_ma = input_nc[doy_i, :, :]
                except IndexError:
                    logging.info('    date not in netcdf, skipping')
                    continue
                input_full_array = input_full_ma.data.astype(np.float32)
                input_full_nodata = float(input_full_ma.fill_value)
                input_full_array[input_full_array == input_full_nodata] = np.nan

                # Since inputs are netcdf, need to create GDAL raster
                #   datasets in order to use gdal_common functions
                # Create an in memory dataset of the full ETo array
                input_full_ds = gdc.array_to_mem_ds(
                    input_full_array, output_geo=gridmet_full_geo,
                    output_proj=gridmet_proj)

                # Then extract the subset from the in memory dataset
                output_array = gdc.raster_ds_to_array(
                    input_full_ds, 1, mask_extent=gridmet_extent,
                    return_nodata=False)

                # Convert Kelvin to Celsius
                if input_var in ['tmmx', 'tmmn']:
                    output_array -= 273.15

                # Save the projected array as 32-bit floats
                gdc.array_to_comp_raster(
                    output_array.astype(np.float32), output_path,
                    band=doy, stats_flag=False)
                # gdc.array_to_raster(
                #     output_array.astype(np.float32), output_path,
                #     output_geo=gridmet_geo, output_proj=gridmet_proj,
                #     stats_flag=False)
                del output_array
            input_nc_f.close()
            del input_nc_f

        if stats_flag:
            gdc.raster_statistics(output_path)

    logging.debug('\nScript Complete')


def arg_parse():
    """

    Base all default folders from script location
        scripts: ./pyMETRIC/tools/gridmet
        tools:    ./pyMETRIC/tools
        output:  ./pyMETRIC/gridmet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    gridmet_folder = os.path.join(project_folder, 'gridmet')

    parser = argparse.ArgumentParser(
        description='GRIDMET daily temperature',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--netcdf', default=os.path.join(gridmet_folder, 'netcdf'),
        metavar='PATH', help='Input netCDF folder path')
    parser.add_argument(
        '--ancillary', default=os.path.join(gridmet_folder, 'ancillary'),
        metavar='PATH', help='Ancillary raster folder path')
    parser.add_argument(
        '--output', default=gridmet_folder,
        metavar='PATH', help='Output raster folder path')
    parser.add_argument(
        '--start', default='2017-01-01', type=valid_date,
        help='Start date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--end', default='2017-12-31', type=valid_date,
        help='End date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--extent', default=None, metavar='PATH',
        help='Subset extent raster path')
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
