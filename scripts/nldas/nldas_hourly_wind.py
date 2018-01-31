#--------------------------------
# Name:         nldas_hourly_wind.py
# Purpose:      Extract NLDAS wind rasters
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


from scripts.support import et_common
from scripts.support import gdal_common as gdc
from scripts.support.python_common import parse_int_set, valid_date

# np.seterr(invalid='ignore')


def main(grb_ws=os.getcwd(), ancillary_ws=os.getcwd(), output_ws=os.getcwd(),
         landsat_ws=None, start_date=None, end_date=None,
         times_str='', extent_path=None, output_extent=None,
         stats_flag=True, overwrite_flag=False):
    """Extract hourly NLDAS wind rasters

    Args:
        grb_ws (str): folder of NLDAS GRB files
        ancillary_ws (str): folder of ancillary rasters
        output_ws (str): folder of output rasters
        landsat_ws (str): folder of Landsat scenes or tar.gz files
        start_date (str): ISO format date (YYYY-MM-DD)
        end_date (str): ISO format date (YYYY-MM-DD)
        times (str): comma separated values and/or ranges of UTC hours
            (i.e. "1, 2, 5-8")
            Parsed with python_common.parse_int_set()
        extent_path (str): file path defining the output extent
        output_extent (list): decimal degrees values defining output extent
        stats_flag (bool): if True, compute raster statistics.
            Default is True.
        overwrite_flag (bool): if True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nExtracting NLDAS wind rasters')

    # input_fmt = 'NLDAS_FORA0125_H.A{:04d}{:02d}{:02d}.{}.002.grb'
    input_re = re.compile(
        'NLDAS_FORA0125_H.A(?P<YEAR>\d{4})(?P<MONTH>\d{2})' +
        '(?P<DAY>\d{2}).(?P<TIME>\d{4}).002.grb$')

    output_folder = 'wind'
    output_fmt = 'wind_{:04d}{:02d}{:02d}_hourly_nldas.img'
    # output_fmt = 'wind_{:04d}{:02d}{:02d}_{:04d}_nldas.img'

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

    # Assume NLDAS is NAD83
    # input_epsg = 'EPSG:4269'

    # Ancillary raster paths
    mask_path = os.path.join(ancillary_ws, 'nldas_mask.img')

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

    # Build output folder
    var_ws = os.path.join(output_ws, output_folder)
    if not os.path.isdir(var_ws):
        os.makedirs(var_ws)

    # Each sub folder in the main folde has all imagery for 1 day
    # The path for each subfolder is the /YYYY/DOY

    # This approach will process files for target dates
    # for input_dt in date_range(start_dt, end_dt + dt.timedelta(1)):
    #     logging.info(input_dt.date())

    # Iterate all available files and check dates if necessary
    logging.info('\nReading NLDAS GRIBs')
    for root, folders, files in os.walk(grb_ws, topdown=True):
        root_split = os.path.normpath(root).split(os.sep)

        # If the year/doy is outside the range, skip
        if (re.match('\d{4}', root_split[-2]) and
                re.match('\d{3}', root_split[-1])):
            root_dt = dt.datetime.strptime('{}_{}'.format(
                root_split[-2], root_split[-1]), '%Y_%j')
            logging.info('{}-{:02d}-{:02d}'.format(
                root_dt.year, root_dt.month, root_dt.day))
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

        # Create a single raster for each day with 24 bands
        # Each time step will be stored in a separate band
        output_name = output_fmt.format(
            root_dt.year, root_dt.month, root_dt.day)
        output_path = os.path.join(
            var_ws, str(root_dt.year), output_name)
        logging.debug('  {}'.format(output_path))
        if os.path.isfile(output_path):
            if not overwrite_flag:
                logging.debug('    File already exists, skipping')
                continue
            else:
                logging.debug('    File already exists, removing existing')
                os.remove(output_path)
        logging.debug('  {}'.format(root))
        if not os.path.isdir(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        gdc.build_empty_raster(
            output_path, band_cnt=24, output_dtype=np.float32,
            output_proj=nldas_proj, output_cs=nldas_cs,
            output_extent=nldas_extent, output_fill_flag=True)

        # Iterate through hourly files
        for input_name in sorted(files):
            logging.info('  {}'.format(input_name))
            input_path = os.path.join(root, input_name)
            input_match = input_re.match(input_name)
            if input_match is None:
                logging.debug('  Regular expression didn\'t match, skipping')
                continue
            input_dt = dt.datetime(
                int(input_match.group('YEAR')),
                int(input_match.group('MONTH')),
                int(input_match.group('DAY')))
            time_str = input_match.group('TIME')
            band_num = int(time_str[:2]) + 1
            # if start_dt is not None and input_dt < start_dt:
            #     continue
            # elif end_dt is not None and input_dt > end_dt:
            #     continue
            # elif date_list and input_dt.date().isoformat() not in date_list:
            #     continue
            if time_str not in time_list:
                logging.debug('    Time not in list, skipping')
                continue
            logging.debug('    Time: {} {}'.format(input_dt.date(), time_str))
            logging.debug('    Band: {}'.format(band_num))

            # Determine band numbering/naming
            input_band_dict = grib_band_names(input_path)

            # Compute magnitude of wind from components
            input_ds = gdal.Open(input_path)
            wind_u_array = gdc.raster_ds_to_array(
                input_ds, band=input_band_dict['u-component of wind [m/s]'],
                mask_extent=nldas_extent, return_nodata=False)
            wind_v_array = gdc.raster_ds_to_array(
                input_ds, band=input_band_dict['v-component of wind [m/s]'],
                mask_extent=nldas_extent, return_nodata=False)
            wind_array = np.sqrt(wind_u_array ** 2 + wind_v_array ** 2)

            # Save the projected array as 32-bit floats
            gdc.array_to_comp_raster(
                wind_array.astype(np.float32), output_path, band=band_num)
            # gdc.block_to_raster(
            #     ea_array.astype(np.float32), output_path, band=band_num)
            # gdc.array_to_raster(
            #     wind_array.astype(np.float32), output_path,
            #     output_geo=nldas_geo, output_proj=nldas_proj,
            #     stats_flag=stats_flag)

            del wind_array, wind_u_array, wind_v_array
            input_ds = None

        if stats_flag:
            gdc.raster_statistics(output_path)

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
        scripts: ./pyMETRIC/scripts/nldas
        scripts:    ./pyMETRIC/scripts
        output:  ./pyMETRIC/nldas
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    nldas_folder = os.path.join(project_folder, 'nldas')

    parser = argparse.ArgumentParser(
        description='NLDAS hourly wind speed',
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

    home = os.path.expanduser('~')
    lolo = os.path.join(home, 'images', 'lolo')
    grb = os.path.join(lolo, 'nldas', '2016')
    anc = os.path.join(lolo, 'nldas')
    landsat = os.path.join(lolo, 'landsat')
    shapefile = os.path.join(lolo, 'extent', 'Missoula_County.shp')
    start = '2016-01-01'
    end = '2016-12-31'

    main(grb_ws=grb, ancillary_ws=anc, output_ws=grb,
         extent_path=shapefile, start_date=start,
         end_date=end, overwrite_flag=True)

