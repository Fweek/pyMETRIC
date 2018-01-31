#--------------------------------
# Name:         gridmet_daily_refet.py
# Purpose:      Calculate GRIDMET ETr/ETo
# Python:       2.7
#--------------------------------

import argparse
import datetime as dt
import logging
import math
import os
import sys

import netCDF4
import numpy as np
from osgeo import gdal

from tools.support import gdal_common as gdc
from tools.support import et_common
from tools.support.python_common import date_range, valid_date


def main(netcdf_ws=os.getcwd(), ancillary_ws=os.getcwd(),
         output_ws=os.getcwd(), etr_flag=False, eto_flag=False,
         start_date=None, end_date=None,
         extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Compute daily ETr/ETo from GRIDMET data

    Args:
        netcdf_ws (str): folder of GRIDMET netcdf files
        ancillary_ws (str): folder of ancillary rasters
        output_ws (str): folder of output rasters
        etr_flag (bool): if True, compute alfalfa reference ET (ETr)
        eto_flag (bool): if True, compute grass reference ET (ETo)
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
    logging.info('\nComputing GRIDMET ETo/ETr')
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

    # Save GRIDMET lat, lon, and elevation arrays
    elev_raster = os.path.join(ancillary_ws, 'gridmet_elev.img')
    lat_raster = os.path.join(ancillary_ws, 'gridmet_lat.img')

    # Wind speed is measured at 2m
    zw = 10

    etr_fmt = 'etr_{}_daily_gridmet.img'
    eto_fmt = 'eto_{}_daily_gridmet.img'
    # gridmet_re = re.compile('(?P<VAR>\w+)_(?P<YEAR>\d{4}).nc')

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
        logging.debug('  Extent: {}'.format(output_extent))
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

    # Read the elevation and latitude arrays
    elev_array = gdc.raster_to_array(
        elev_raster, mask_extent=gridmet_extent, return_nodata=False)
    lat_array = gdc.raster_to_array(
        lat_raster, mask_extent=gridmet_extent, return_nodata=False)
    lat_array *= math.pi / 180

    # Check elevation and latitude arrays
    if np.all(np.isnan(elev_array)):
        logging.error('\nERROR: The elevation array is all nodata, exiting\n')
        sys.exit()
    elif np.all(np.isnan(lat_array)):
        logging.error('\nERROR: The latitude array is all nodata, exiting\n')
        sys.exit()

    # Build output folder
    etr_ws = os.path.join(output_ws, 'etr')
    eto_ws = os.path.join(output_ws, 'eto')
    if etr_flag and not os.path.isdir(etr_ws):
        os.makedirs(etr_ws)
    if eto_flag and not os.path.isdir(eto_ws):
        os.makedirs(eto_ws)

    # By default, try to process all possible years
    if start_dt.year == end_dt.year:
        year_list = [str(start_dt.year)]
    year_list = sorted(map(str, range((start_dt.year), (end_dt.year + 1))))

    # Process each year separately
    for year_str in year_list:
        logging.info("\nYear: {}".format(year_str))
        year_int = int(year_str)
        year_days = int(dt.datetime(year_int, 12, 31).strftime('%j'))
        if start_dt is not None and year_int < start_dt.year:
            logging.debug('  Before start date, skipping')
            continue
        elif end_dt is not None and year_int > end_dt.year:
            logging.debug('  After end date, skipping')
            continue

        # Build input file path
        tmin_path = os.path.join(netcdf_ws, 'tmmn_{}.nc'.format(year_str))
        tmax_path = os.path.join(netcdf_ws, 'tmmx_{}.nc'.format(year_str))
        sph_path = os.path.join(netcdf_ws, 'sph_{}.nc'.format(year_str))
        rs_path = os.path.join(netcdf_ws, 'srad_{}.nc'.format(year_str))
        wind_path = os.path.join(netcdf_ws, 'vs_{}.nc'.format(year_str))
        # Check that all input files are present
        missing_flag = False
        for input_path in [tmin_path, tmax_path, sph_path,
                           rs_path, wind_path]:
            if not os.path.isfile(input_path):
                logging.debug('  Input NetCDF doesn\'t exist\n    {}'.format(
                    input_path))
                missing_flag = True
        if missing_flag:
            logging.debug('  skipping')
            continue
        logging.debug("  {}".format(tmin_path))
        logging.debug("  {}".format(tmax_path))
        logging.debug("  {}".format(sph_path))
        logging.debug("  {}".format(rs_path))
        logging.debug("  {}".format(wind_path))

        # Create a single raster for each year with 365 bands
        # Each day will be stored in a separate band
        etr_raster = os.path.join(etr_ws, etr_fmt.format(year_str))
        eto_raster = os.path.join(eto_ws, eto_fmt.format(year_str))
        if etr_flag and (overwrite_flag or not os.path.isfile(etr_raster)):
            logging.debug('  {}'.format(etr_raster))
            gdc.build_empty_raster(
                etr_raster, band_cnt=366, output_dtype=np.float32,
                output_proj=gridmet_proj, output_cs=gridmet_cs,
                output_extent=gridmet_extent, output_fill_flag=True)
        if eto_flag and (overwrite_flag or not os.path.isfile(eto_raster)):
            logging.debug('  {}'.format(eto_raster))
            gdc.build_empty_raster(
                eto_raster, band_cnt=366, output_dtype=np.float32,
                output_proj=gridmet_proj, output_cs=gridmet_cs,
                output_extent=gridmet_extent, output_fill_flag=True)
        # DEADBEEF - Need to find a way to test if both of these conditionals
        #   did not pass and pass logging debug message to user

        # Read in the GRIDMET NetCDF file
        tmin_nc_f = netCDF4.Dataset(tmin_path, 'r')
        tmax_nc_f = netCDF4.Dataset(tmax_path, 'r')
        sph_nc_f = netCDF4.Dataset(sph_path, 'r')
        rs_nc_f = netCDF4.Dataset(rs_path, 'r')
        wind_nc_f = netCDF4.Dataset(wind_path, 'r')

        logging.info('  Reading NetCDFs into memory')
        # Immediatly clip input arrays to save memory
        tmin_nc = tmin_nc_f.variables[gridmet_band_dict['tmmn']][
            :, g_i:g_i + g_cols, g_j:g_j + g_rows].copy()
        tmax_nc = tmax_nc_f.variables[gridmet_band_dict['tmmx']][
            :, g_i:g_i + g_cols, g_j:g_j + g_rows].copy()
        sph_nc = sph_nc_f.variables[gridmet_band_dict['sph']][
            :, g_i:g_i + g_cols, g_j:g_j + g_rows].copy()
        rs_nc = rs_nc_f.variables[gridmet_band_dict['srad']][
            :, g_i:g_i + g_cols, g_j:g_j + g_rows].copy()
        wind_nc = wind_nc_f.variables[gridmet_band_dict['vs']][
            :, g_i:g_i + g_cols, g_j:g_j + g_rows].copy()
        # tmin_nc = tmin_nc_f.variables[gridmet_band_dict['tmmn']][:]
        # tmax_nc = tmax_nc_f.variables[gridmet_band_dict['tmmx']][:]
        # sph_nc = sph_nc_f.variables[gridmet_band_dict['sph']][:]
        # rs_nc = rs_nc_f.variables[gridmet_band_dict['srad']][:]
        # wind_nc = wind_nc_f.variables[gridmet_band_dict['vs']][:]

        # Transpose arrays back to row x col
        tmin_nc = np.transpose(tmin_nc, (0, 2, 1))
        tmax_nc = np.transpose(tmax_nc, (0, 2, 1))
        sph_nc = np.transpose(sph_nc, (0, 2, 1))
        rs_nc = np.transpose(rs_nc, (0, 2, 1))
        wind_nc = np.transpose(wind_nc, (0, 2, 1))

        # A numpy array is returned when slicing a masked array
        #   if there are no masked pixels
        # This is a hack to force the numpy array back to a masked array
        # For now assume all arrays need to be converted
        if type(tmin_nc) != np.ma.core.MaskedArray:
            tmin_nc = np.ma.core.MaskedArray(
                tmin_nc, np.zeros(tmin_nc.shape, dtype=bool))
        if type(tmax_nc) != np.ma.core.MaskedArray:
            tmax_nc = np.ma.core.MaskedArray(
                tmax_nc, np.zeros(tmax_nc.shape, dtype=bool))
        if type(sph_nc) != np.ma.core.MaskedArray:
            sph_nc = np.ma.core.MaskedArray(
                sph_nc, np.zeros(sph_nc.shape, dtype=bool))
        if type(rs_nc) != np.ma.core.MaskedArray:
            rs_nc = np.ma.core.MaskedArray(
                rs_nc, np.zeros(rs_nc.shape, dtype=bool))
        if type(wind_nc) != np.ma.core.MaskedArray:
            wind_nc = np.ma.core.MaskedArray(
                wind_nc, np.zeros(wind_nc.shape, dtype=bool))

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

            # Arrays are being read as masked array with a fill value of -9999
            # Convert to basic numpy array arrays with nan values
            try:
                tmin_ma = tmin_nc[doy_i, :, :]
            except IndexError:
                logging.info('    date not in netcdf, skipping')
                continue
            tmin_array = tmin_ma.data.astype(np.float32)
            tmin_nodata = float(tmin_ma.fill_value)
            tmin_array[tmin_array == tmin_nodata] = np.nan

            try:
                tmax_ma = tmax_nc[doy_i, :, :]
            except IndexError:
                logging.info('    date not in netcdf, skipping')
                continue
            tmax_array = tmax_ma.data.astype(np.float32)
            tmax_nodata = float(tmax_ma.fill_value)
            tmax_array[tmax_array == tmax_nodata] = np.nan

            try:
                sph_ma = sph_nc[doy_i, :, :]
            except IndexError:
                logging.info('    date not in netcdf, skipping')
                continue
            sph_array = sph_ma.data.astype(np.float32)
            sph_nodata = float(sph_ma.fill_value)
            sph_array[sph_array == sph_nodata] = np.nan

            try:
                rs_ma = rs_nc[doy_i, :, :]
            except IndexError:
                logging.info('    date not in netcdf, skipping')
                continue
            rs_array = rs_ma.data.astype(np.float32)
            rs_nodata = float(rs_ma.fill_value)
            rs_array[rs_array == rs_nodata] = np.nan

            try:
                wind_ma = wind_nc[doy_i, :, :]
            except IndexError:
                logging.info('    date not in netcdf, skipping')
                continue
            wind_array = wind_ma.data.astype(np.float32)
            wind_nodata = float(wind_ma.fill_value)
            wind_array[wind_array == wind_nodata] = np.nan
            del tmin_ma, tmax_ma, sph_ma, rs_ma, wind_ma

            # Since inputs are netcdf, need to create GDAL raster
            #   datasets in order to use gdal_common functions
            # Create an in memory dataset of the full ETo array
            tmin_ds = gdc.array_to_mem_ds(
                tmin_array, output_geo=gridmet_geo,
                # tmin_array, output_geo=gridmet_full_geo,
                output_proj=gridmet_proj)
            tmax_ds = gdc.array_to_mem_ds(
                tmax_array, output_geo=gridmet_geo,
                # tmax_array, output_geo=gridmet_full_geo,
                output_proj=gridmet_proj)
            sph_ds = gdc.array_to_mem_ds(
                sph_array, output_geo=gridmet_geo,
                # sph_array, output_geo=gridmet_full_geo,
                output_proj=gridmet_proj)
            rs_ds = gdc.array_to_mem_ds(
                rs_array, output_geo=gridmet_geo,
                # rs_array, output_geo=gridmet_full_geo,
                output_proj=gridmet_proj)
            wind_ds = gdc.array_to_mem_ds(
                wind_array, output_geo=gridmet_geo,
                # wind_array, output_geo=gridmet_full_geo,
                output_proj=gridmet_proj)

            # Then extract the subset from the in memory dataset
            tmin_array = gdc.raster_ds_to_array(
                tmin_ds, 1, mask_extent=gridmet_extent, return_nodata=False)
            tmax_array = gdc.raster_ds_to_array(
                tmax_ds, 1, mask_extent=gridmet_extent, return_nodata=False)
            sph_array = gdc.raster_ds_to_array(
                sph_ds, 1, mask_extent=gridmet_extent, return_nodata=False)
            rs_array = gdc.raster_ds_to_array(
                rs_ds, 1, mask_extent=gridmet_extent, return_nodata=False)
            wind_array = gdc.raster_ds_to_array(
                wind_ds, 1, mask_extent=gridmet_extent, return_nodata=False)
            del tmin_ds, tmax_ds, sph_ds, rs_ds, wind_ds

            # Adjust units
            tmin_array -= 273.15
            tmax_array -= 273.15
            rs_array *= 0.0864

            # ETr/ETo
            if etr_flag:
                etr_array = et_common.refet_daily_func(
                    tmin_array, tmax_array, sph_array, rs_array,
                    wind_array, zw, elev_array, lat_array, doy, 'ETR')
            if eto_flag:
                eto_array = et_common.refet_daily_func(
                    tmin_array, tmax_array, sph_array, rs_array,
                    wind_array, zw, elev_array, lat_array, doy, 'ETO')
            # del tmin_array, tmax_array, sph_array, rs_array, wind_array

            # Save the projected array as 32-bit floats
            if etr_flag:
                gdc.array_to_comp_raster(
                    etr_array.astype(np.float32), etr_raster,
                    band=doy, stats_flag=False)
                # gdc.array_to_raster(
                #     etr_array.astype(np.float32), etr_raster,
                #     output_geo=gridmet_geo, output_proj=gridmet_proj,
                #     stats_flag=stats_flag)
                del etr_array
            if eto_flag:
                gdc.array_to_comp_raster(
                    eto_array.astype(np.float32), eto_raster,
                    band=doy, stats_flag=False)
                # gdc.array_to_raster(
                #     eto_array.astype(np.float32), eto_raster,
                #     output_geo=gridmet_geo, output_proj=gridmet_proj,
                #     stats_flag=stats_flag)
                del eto_array

        del tmin_nc
        del tmax_nc
        del sph_nc
        del rs_nc
        del wind_nc

        tmin_nc_f.close()
        tmax_nc_f.close()
        sph_nc_f.close()
        rs_nc_f.close()
        wind_nc_f.close()
        del tmin_nc_f, tmax_nc_f, sph_nc_f, rs_nc_f, wind_nc_f

        if stats_flag and etr_flag:
            gdc.raster_statistics(etr_raster)
        if stats_flag and eto_flag:
            gdc.raster_statistics(eto_raster)

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
        description='GRIDMET daily reference ETr/ETo',
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
        '--etr', default=False, action="store_true",
        help='Compute alfalfa reference ET (ETr)')
    parser.add_argument(
        '--eto', default=False, action="store_true",
        help='Compute grass reference ET (ETo)')
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
         output_ws=args.output, eto_flag=args.eto, etr_flag=args.etr,
         start_date=args.start, end_date=args.end,
         extent_path=args.extent, output_extent=args.te,
         stats_flag=args.stats, overwrite_flag=args.overwrite)
