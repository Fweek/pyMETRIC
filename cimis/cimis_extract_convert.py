#--------------------------------
# Name:         cimis_extract_convert.py
# Purpose:      Extract CIMIS ASCII files from .gz files
# Python:       2.7
#--------------------------------

import argparse
import datetime as dt
import gzip
import logging
import os
import re
import sys

import numpy as np

from support import gdal_common as gdc
from support import valid_date


def main(input_ws=os.getcwd(), output_ws=os.getcwd(),
         start_date=None, end_date=None,
         remove_gz_flag=False, remove_ascii_flag=True,
         stats_flag=False, overwrite_flag=False):
    """Extract CIMIS data from tar.gz files

    Args:
        input_ws (str): folder path of the input tar.gz files
        output_ws (str): folder path of the output IMG rasters
        start_date (str): ISO format date string (YYYY-MM-DD)
        end_date (str): ISO format date string (YYYY-MM-DD)
        remove_gz_flag (bool): If True, remove downloaded .gz files
        remove_ascii_flag (bool): If True, remove extracted ascii files
        stats_flag (bool): if True, compute raster statistics.
            Default is False.
        overwrite_flag (bool): If True, overwrite existing files

    Returns:
        None
    """
    logging.info('\nExtracting CIMIS data')

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

    # CIMIS rasters to extract
    data_list = ['ETo', 'Rso', 'Rs', 'Tdew', 'Tn', 'Tx', 'U2']

    # Spatial reference parameters
    cimis_proj4 = (
        "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 " +
        "+x_0=0 +y_0=-4000000 +ellps=GRS80 +datum=NAD83 +units=m +no_defs")
    cimis_osr = gdc.proj4_osr(cimis_proj4)
    # cimis_epsg = 3310  # NAD_1983_California_Teale_Albers
    # cimis_osr = gdc.epsg_osr(cimis_epsg)
    cimis_osr.MorphToESRI()
    cimis_proj = cimis_osr.ExportToWkt()

    # Set data types to upper case for comparison
    data_list = map(lambda x: x.lower(), data_list)

    # Look for .asc.gz files
    for year_str in sorted(os.listdir(input_ws)):
        logging.info('{}'.format(year_str))
        if not re.match('^\d{4}$', year_str):
            logging.debug('  Not a 4 digit year folder, skipping')
            continue
        year_ws = os.path.join(input_ws, year_str)
        if start_dt is not None and int(year_str) < start_dt.year:
            logging.debug('  Before start date, skipping')
            continue
        elif end_dt is not None and int(year_str) > end_dt.year:
            logging.debug('  After end date, skipping')
            continue

        for date_str in sorted(os.listdir(year_ws)):
            date_ws = os.path.join(year_ws, date_str)
            try:
                date_dt = dt.datetime.strptime(date_str, '%Y_%m_%d')
            except ValueError:
                logging.debug(
                    '  Invalid folder date format (YYYY_MM_DD), skipping')
                continue
            if start_dt is not None and date_dt < start_dt:
                logging.debug('  Before start date, skipping')
                continue
            elif end_dt is not None and date_dt > end_dt:
                logging.debug('  After end date, skipping')
                continue
            logging.info(date_str)

            for file_item in sorted(os.listdir(date_ws)):
                logging.debug('  {}'.format(file_item))
                if not file_item.endswith('.asc.gz'):
                    logging.debug(
                        '  Invalid file type (not .asc.gz), skipping')
                    continue
                gz_path = os.path.join(date_ws, file_item)
                asc_path = gz_path.replace(
                    input_ws, output_ws).replace('.gz', '')
                raster_path = gz_path.replace(
                    input_ws, output_ws).replace('.asc.gz', '.img')

                # Only process selected raster types
                if file_item.replace('.asc.gz', '').lower() not in data_list:
                    logging.debug('  Unused file/variable, skipping')
                    continue

                if os.path.isfile(raster_path):
                    logging.debug('    {}'.format(raster_path))
                    if not overwrite_flag:
                        logging.debug('    File already exists, skipping')
                        continue
                    else:
                        logging.debug(
                            '    File already exists, removing existing')
                        os.remove(raster_path)

                # Build the output folder if necessary
                if not os.path.isdir(os.path.dirname(raster_path)):
                    os.makedirs(os.path.dirname(raster_path))

                # Uncompress '.gz' file to a new file
                # DEADBEEF - This needs to catch specific exceptions!
                try:
                    input_f = gzip.open(gz_path, 'rb')
                    input_data = input_f.read()
                    input_f.close()
                except:
                    logging.error("  ERROR EXTRACTING FILE")
                try:
                    with open(asc_path, 'wb') as output_f:
                        output_f.write(input_data)
                    if remove_gz_flag:
                        os.remove(gz_path)
                except:
                    logging.error("  ERROR WRITING FILE")

                # # Set spatial reference of the ASCII files
                # if build_prj_flag:
                #     output_osr.MorphToESRI()
                #     cimis_proj = output_osr.ExportToWkt()
                #     prj_file = open(asc_path.replace('.asc','.prj'), 'w')
                #     prj_file.write(cimis_proj)
                #     prj_file.close()

                # Convert the ASCII raster to a IMG raster
                gdc.ascii_to_raster(
                    asc_path, raster_path, input_type=np.float32,
                    input_proj=cimis_proj, stats_flag=stats_flag)
                if remove_ascii_flag:
                    os.remove(asc_path)

                # Cleanup
                del gz_path, asc_path, raster_path

    logging.debug('\nScript Complete')


def arg_parse():
    """

    Base all default folders from script location
        scripts: ./pyMETRIC/tools/cimis
        tools:    ./pyMETRIC/tools
        output:  ./pyMETRIC/cimis
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    cimis_folder = os.path.join(project_folder, 'cimis')

    parser = argparse.ArgumentParser(
        description='CIMIS extract/convert',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--gz', default=os.path.join(cimis_folder, 'input_gz'),
        metavar='PATH', help='Input tar.gz root folder path')
    parser.add_argument(
        '--img', default=os.path.join(cimis_folder, 'input_img'),
        metavar='PATH', help='Output IMG raster folder path')
    parser.add_argument(
        '--start', default='2017-01-01', type=valid_date,
        help='Start date (format YYYY-MM-DD)', metavar='DATE')
    parser.add_argument(
        '--end', default='2017-12-31', type=valid_date,
        help='End date (format YYYY-MM-DD)', metavar='DATE')
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
    if args.gz and os.path.isdir(os.path.abspath(args.gz)):
        args.gz = os.path.abspath(args.gz)
    if args.img and os.path.isdir(os.path.abspath(args.img)):
        args.img = os.path.abspath(args.img)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    logging.info('{:<20s} {}'.format(
        'Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info('{:<20s} {}'.format(
        'Script:', os.path.basename(sys.argv[0])))

    main(input_ws=args.gz, output_ws=args.img,
         start_date=args.start, end_date=args.end,
         stats_flag=args.stats, overwrite_flag=args.overwrite)
