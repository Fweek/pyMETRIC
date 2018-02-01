#--------------------------------
# Name:         cimis_gridmet_fill.py
# Purpose:      Get GRIDMET ETo/ETr values if missing from spatial CIMIS data
# Python:       2.7
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import re
import sys

import numpy as np
from osgeo import gdal

from support import date_range, valid_date
from support import gdal_common as gdc


def main(cimis_ws=os.getcwd(), gridmet_ws=None, ancillary_ws=os.getcwd(),
         etr_flag=False, eto_flag=False, start_date=None, end_date=None,
         stats_flag=True, overwrite_flag=False):
    """Fill missing CIMIS days with projected data from GRIDMET

    Currently missing (CGM 2014-08-15)
    2010-11-16 -> 2010-11-23

    Args:
        cimis_ws (str): root folder path of CIMIS data
        gridmet_ws (str): root folder path of GRIDMET data
        ancillary_ws (str): folder of ancillary rasters
        etr_flag (bool): if True, compute alfalfa reference ET (ETr)
        eto_flag (bool): if True, compute grass reference ET (ETo)
        start_date (str): ISO format date (YYYY-MM-DD)
        end_date (str): ISO format date (YYYY-MM-DD)
        stats_flag (bool): if True, compute raster statistics.
            Default is True.
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nFilling CIMIS with GRIDMET')
    cimis_re = re.compile(
        '(?P<VAR>etr)_(?P<YYYY>\d{4})_daily_(?P<GRID>\w+).img$')
    # gridmet_re = re.compile(
    #     '(?P<VAR>ppt)_(?P<YYY>\d{4})_daily_(?P<GRID>\w+).img$')
    gridmet_fmt = 'etr_{}_daily_gridmet.img'

    # Compute ETr and/or ETo
    if not etr_flag and not eto_flag:
        logging.info('  ETo/ETr flag(s) not set, defaulting to ETr')
        etr_flag = True

    logging.debug('  CIMIS: {}'.format(cimis_ws))
    logging.debug('  GRIDMET: {}'.format(gridmet_ws))

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

    # Get GRIDMET spatial reference and cellsize from elevation raster
    # gridmet_elev_raster = os.path.join(ancillary_ws, 'gridmet_elev.img')

    # Get CIMIS spatial reference and cellsize from mask raster
    cimis_mask_raster = os.path.join(ancillary_ws, 'cimis_mask.img')

    # Resample type
    # 0 = GRA_NearestNeighbour, Nearest neighbour (select on one input pixel)
    # 1 = GRA_Bilinear,Bilinear (2x2 kernel)
    # 2 = GRA_Cubic, Cubic Convolution Approximation (4x4 kernel)
    # 3 = GRA_CubicSpline, Cubic B-Spline Approximation (4x4 kernel)
    # 4 = GRA_Lanczos, Lanczos windowed sinc interpolation (6x6 kernel)
    # 5 = GRA_Average, Average (computes the average of all non-NODATA contributing pixels)
    # 6 = GRA_Mode, Mode (selects the value which appears most often of all the sampled points)
    resample_type = gdal.GRA_Bilinear

    # ETo/ETr workspaces
    cimis_eto_ws = os.path.join(cimis_ws, 'eto')
    cimis_etr_ws = os.path.join(cimis_ws, 'etr')
    gridmet_eto_ws = os.path.join(gridmet_ws, 'eto')
    gridmet_etr_ws = os.path.join(gridmet_ws, 'etr')

    # This allows GDAL to throw Python Exceptions
    # gdal.UseExceptions()
    # mem_driver = gdal.GetDriverByName('MEM')

    # Get CIMIS grid properties from mask
    logging.info('\nCIMIS Properties')
    cimis_mask_ds = gdal.Open(cimis_mask_raster)
    cimis_osr = gdc.raster_ds_osr(cimis_mask_ds)
    cimis_proj = gdc.osr_proj(cimis_osr)
    cimis_cs = gdc.raster_ds_cellsize(cimis_mask_ds, x_only=True)
    cimis_extent = gdc.raster_ds_extent(cimis_mask_ds)
    cimis_geo = cimis_extent.geo(cimis_cs)
    cimis_mask_ds = None
    logging.debug('  Projection: {}'.format(cimis_proj))
    logging.debug('  Cellsize: {}'.format(cimis_cs))
    logging.debug('  Geo: {}'.format(cimis_geo))
    logging.debug('  Extent: {}'.format(cimis_extent))

    # Read the CIMIS mask array if present
    cimis_mask, cimis_mask_nodata = gdc.raster_to_array(
        cimis_mask_raster)
    cimis_mask = cimis_mask != cimis_mask_nodata

    # # Get extent/geo from elevation raster
    # logging.info('\nGRIDMET Properties')
    # gridmet_ds = gdal.Open(gridmet_elev_raster)
    # gridmet_osr = gdc.raster_ds_osr(gridmet_ds)
    # gridmet_proj = gdc.osr_proj(gridmet_osr)
    # gridmet_cs = gdc.raster_ds_cellsize(gridmet_ds, x_only=True)
    # gridmet_full_extent = gdc.raster_ds_extent(gridmet_ds)
    # gridmet_full_geo = gridmet_full_extent.geo(gridmet_cs)
    # gridmet_x, gridmet_y = gridmet_full_extent.origin()
    # gridmet_ds = None
    # logging.debug('  Projection: {}'.format(gridmet_proj))
    # logging.debug('  Cellsize: {}'.format(gridmet_cs))
    # logging.debug('  Geo: {}'.format(gridmet_full_geo))
    # logging.debug('  Extent: {}'.format(gridmet_full_extent))

    # # Project CIMIS extent to the GRIDMET spatial reference
    # logging.info('\nGet CIMIS extent in GRIDMET spat. ref.')
    # gridmet_sub_extent = gdc.project_extent(
    #     cimis_extent, cimis_osr, gridmet_osr, cimis_cs)
    # gridmet_sub_extent.buffer_extent(4 * gridmet_cs)
    # gridmet_sub_extent.adjust_to_snap(
    #     'EXPAND', gridmet_x, gridmet_y, gridmet_cs)
    # gridmet_sub_geo = gridmet_sub_extent.geo(gridmet_cs)
    # logging.debug('  Geo: {}'.format(gridmet_sub_geo))
    # logging.debug('  Extent: {}'.format(gridmet_sub_extent))

    # Process Missing ETo
    if eto_flag:
        logging.info('\nETo')
        for cimis_name in sorted(os.listdir(cimis_eto_ws)):
            logging.debug("\n{}".format(cimis_name))
            cimis_match = cimis_re.match(cimis_name)
            if not cimis_match:
                logging.debug('  Regular expression didn\'t match, skipping')
                continue
            year = int(cimis_match.group('YYYY'))
            logging.info("  {}".format(str(year)))
            if start_dt is not None and year < start_dt.year:
                logging.debug('  Before start date, skipping')
                continue
            elif end_dt is not None and year > end_dt.year:
                logging.debug('  After end date, skipping')
                continue

            cimis_path = os.path.join(cimis_eto_ws, cimis_name)
            gridmet_path = os.path.join(
                gridmet_eto_ws, gridmet_fmt.format(str(year)))
            if not os.path.isfile(gridmet_path):
                logging.debug('  GRIDMET raster does not exist, skipping')
                continue
            if not os.path.isfile(cimis_path):
                logging.error('  CIMIS raster does not exist, skipping')
                continue

            # Check all valid dates in the year
            year_dates = date_range(
                dt.datetime(year, 1, 1), dt.datetime(year + 1, 1, 1))
            for date_dt in year_dates:
                if start_dt is not None and date_dt < start_dt:
                    continue
                elif end_dt is not None and date_dt > end_dt:
                    continue
                doy = int(date_dt.strftime('%j'))

                # Look for arrays that don't have data
                eto_array = gdc.raster_to_array(
                    cimis_path, band=doy, return_nodata=False)
                if np.any(np.isfinite(eto_array)):
                    logging.debug('  {} - no missing data, skipping'.format(
                        date_dt.strftime('%Y-%m-%d')))
                    continue
                else:
                    logging.info('  {}'.format(date_dt.strftime('%Y-%m-%d')))

                # # This is much faster but doesn't apply the CIMIS mask
                # # Create an in memory dataset of the full ETo array
                # eto_full_rows, eto_full_cols = eto_full_array[:,:,doy_i].shape
                # eto_full_type, eto_full_nodata = numpy_to_gdal_type(np.float32)
                # eto_full_ds = mem_driver.Create(
                #     '', eto_full_cols, eto_full_rows, 1, eto_full_type)
                # eto_full_ds.SetProjection(gridmet_proj)
                # eto_full_ds.SetGeoTransform(gridmet_full_geo)
                # eto_full_band = eto_full_ds.GetRasterBand(1)
                # # eto_full_band.Fill(eto_full_nodata)
                # eto_full_band.SetNoDataValue(eto_full_nodata)
                # eto_full_band.WriteArray(eto_full_array[:,:,doy_i], 0, 0)
                #
                # # Extract the subset
                # eto_sub_array, eto_sub_nodata = gdc.raster_ds_to_array(
                #     eto_full_ds, 1, gridmet_sub_extent)
                # eto_sub_rows, eto_sub_cols = eto_sub_array.shape
                # eto_full_ds = None
                #
                # # Create projected raster
                # eto_sub_ds = mem_driver.Create(
                #     '', eto_sub_cols, eto_sub_rows, 1, eto_full_type)
                # eto_sub_ds.SetProjection(gridmet_proj)
                # eto_sub_ds.SetGeoTransform(gridmet_sub_geo)
                # eto_sub_band = eto_sub_ds.GetRasterBand(1)
                # eto_sub_band.Fill(eto_sub_nodata)
                # eto_sub_band.SetNoDataValue(eto_sub_nodata)
                # eto_sub_band.WriteArray(eto_sub_array, 0, 0)
                # eto_sub_ds.FlushCache()
                #
                # # Project input DEM to CIMIS spat. ref.
                # gdc.project_raster_ds(
                #     eto_sub_ds, gridmet_path, resample_type,
                #     env.snap_proj, env.cellsize, cimis_extent)
                # eto_sub_ds = None

                # Extract the subset
                gridmet_ds = gdal.Open(gridmet_path)
                gridmet_extent = gdc.raster_ds_extent(gridmet_ds)
                gridmet_cs = gdc.raster_ds_cellsize(gridmet_ds, x_only=True)
                gridmet_osr = gdc.raster_ds_osr(gridmet_ds)
                eto_full_array = gdc.raster_ds_to_array(
                    gridmet_ds, band=doy, return_nodata=False)
                gridmet_ds = None

                # Get the projected subset of the full ETo array
                # This is slower than projecting the subset above
                eto_sub_array = gdc.project_array(
                    eto_full_array, resample_type,
                    gridmet_osr, gridmet_cs, gridmet_extent,
                    cimis_osr, cimis_cs, cimis_extent)

                # Save the projected array
                gdc.array_to_comp_raster(
                    eto_sub_array, cimis_path, band=doy, stats_flag=False)
                # gdc.array_to_raster(
                #     eto_sub_array, output_path, output_geo=cimis_geo,
                #     output_proj=cimis_proj, stats_flag=False)
                # gdc.array_to_raster(
                #     eto_sub_array, output_path,
                #     output_geo=cimis_geo, output_proj=cimis_proj,
                #     mask_array=cimis_mask, stats_flag=False)

                del eto_sub_array, eto_full_array

            if stats_flag:
                gdc.raster_statistics(cimis_path)

    # Process Missing ETr
    if etr_flag:
        logging.info('\nETr')
        for cimis_name in sorted(os.listdir(cimis_etr_ws)):
            cimis_match = cimis_re.match(cimis_name)
            if not cimis_match:
                continue
            year = int(cimis_match.group('YYYY'))
            if start_dt is not None and year < start_dt.year:
                continue
            elif end_dt is not None and year > end_dt.year:
                continue
            logging.info("{}".format(str(year)))

            cimis_path = os.path.join(cimis_etr_ws, cimis_name)
            gridmet_path = os.path.join(
                gridmet_etr_ws, gridmet_fmt.format(str(year)))
            if not os.path.isfile(gridmet_path):
                continue
            if not os.path.isfile(cimis_path):
                logging.error('  CIMIS raster does not exist')
                continue

            # Check all valid dates in the year
            year_dates = date_range(
                dt.datetime(year, 1, 1), dt.datetime(year + 1, 1, 1))
            for date_dt in year_dates:
                if start_dt is not None and date_dt < start_dt:
                    continue
                elif end_dt is not None and date_dt > end_dt:
                    continue
                doy = int(date_dt.strftime('%j'))

                # Look for arrays that don't have data
                etr_array = gdc.raster_to_array(
                    cimis_path, band=doy, return_nodata=False)
                if np.any(np.isfinite(etr_array)):
                    logging.debug('  {} - skipping'.format(
                        date_dt.strftime('%Y-%m-%d')))
                    continue
                else:
                    logging.info('  {}'.format(date_dt.strftime('%Y-%m-%d')))

                # Extract the subset
                gridmet_ds = gdal.Open(gridmet_path)
                gridmet_extent = gdc.raster_ds_extent(gridmet_ds)
                gridmet_cs = gdc.raster_ds_cellsize(gridmet_ds, x_only=True)
                gridmet_osr = gdc.raster_ds_osr(gridmet_ds)
                etr_full_array = gdc.raster_ds_to_array(
                    gridmet_ds, band=doy, return_nodata=False)
                gridmet_ds = None

                # Get the projected subset of the full ETr array
                # This is slower than projecting the subset
                etr_sub_array = gdc.project_array(
                    etr_full_array, resample_type,
                    gridmet_osr, gridmet_cs, gridmet_extent,
                    cimis_osr, cimis_cs, cimis_extent)

                # # Save the projected array
                gdc.array_to_comp_raster(
                    etr_sub_array, cimis_path, band=doy, stats_flag=False)
                # gdc.array_to_raster(
                #     etr_sub_array, output_path,
                #     output_geo=cimis_geo, output_proj=cimis_proj,
                #     mask_array=cimis_mask, stats_flag=False)

                del etr_sub_array, etr_full_array

            if stats_flag:
                gdc.raster_statistics(cimis_path)

    logging.debug('\nScript Complete')


def arg_parse():
    """

    Base all default folders from script location
        scripts: ./pyMETRIC/tools/cimis
        tools:    ./pyMETRIC/tools
        output:  ./pyMETRIC/cimis
        gridmet: ./pyMETRIC/gridmet
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    cimis_folder = os.path.join(project_folder, 'cimis')
    gridmet_folder = os.path.join(project_folder, 'gridmet')

    parser = argparse.ArgumentParser(
        description='Fill CIMIS with GRIDMET',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--cimis', default=cimis_folder,
        metavar='PATH', help='Input CIMIS root folder path')
    parser.add_argument(
        '--gridmet', default=gridmet_folder,
        metavar='PATH', help='Input GRIDMET root folder path')
    parser.add_argument(
        '--ancillary', default=os.path.join(cimis_folder, 'ancillary'),
        metavar='PATH', help='Ancillary raster folder path')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    parser.add_argument(
        '--start', default='2017-01-01', type=valid_date,
        help='Start date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--end', default='2017-12-31', type=valid_date,
        help='End date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--eto', default=False, action="store_true",
        help='Compute grass reference ET (ETo)')
    parser.add_argument(
        '--etr', default=False, action="store_true",
        help='Compute alfalfa reference ET (ETr)')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '--stats', default=False, action="store_true",
        help='Compute raster statistics')
    args = parser.parse_args()

    # Convert relative paths to absolute paths
    if args.cimis and os.path.isdir(os.path.abspath(args.cimis)):
        args.cimis = os.path.abspath(args.cimis)
    if args.gridmet and os.path.isdir(os.path.abspath(args.gridmet)):
        args.gridmet = os.path.abspath(args.gridmet)
    if args.ancillary and os.path.isdir(os.path.abspath(args.ancillary)):
        args.ancillary = os.path.abspath(args.ancillary)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{:<20s} {}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{:<20s} {}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(cimis_ws=args.cimis, gridmet_ws=args.gridmet,
         ancillary_ws=args.ancillary, etr_flag=args.etr, eto_flag=args.eto,
         start_date=args.start, end_date=args.end, stats_flag=not args.stats,
         overwrite_flag=args.overwrite)
