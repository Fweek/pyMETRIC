#--------------------------------
# Name:         download_nlcd.py
# Purpose:      Download NLCD raster
# Python:       2.7, 3.5, 3.6
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys
import zipfile

from python_common import url_download


def main(output_folder, year=2011, overwrite_flag=False):
    """Download NLCD raster

    Args:
        output_folder (str): folder path where files will be saved
        year (str, int): NLCD year (2006 or 2011)
        overwrite_flag (bool): If True, overwrite existing files

    Returns:
        None
    """
    download_url = (
        'http://gisdata.usgs.gov/TDDS/DownloadFile.php?'
        'TYPE=nlcd{0}&FNAME=nlcd_{0}_landcover_2011_edition_2014_10_10.zip').format(year)

    zip_name = 'nlcd_{}_landcover_2011_edition_2014_10_10.zip'.format(year)
    zip_path = os.path.join(output_folder, zip_name)

    output_name = zip_name.replace('.zip', '.img')
    # output_path = os.path.join(output_folder, output_name)
    output_path = os.path.join(
        output_folder, os.path.splitext(zip_name)[0], output_name)

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if ((not os.path.isfile(zip_path) and not os.path.isfile(output_path)) or
            overwrite_flag):
        logging.info('\nDownloading NLCD')
        logging.info('  {}'.format(download_url))
        logging.info('  {}'.format(zip_path))
        url_download(download_url, zip_path)
    else:
        logging.debug('\nNLCD raster already downloaded')

    if ((overwrite_flag or not os.path.isfile(output_path)) and
            os.path.isfile(zip_path)):
        logging.info('\nExtracting NLCD files')
        logging.debug('  {}'.format(output_path))
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_folder)
    else:
        logging.debug('\nNLCD raster already extracted')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pyMETRIC/tools/download
        tools:    ./pyMETRIC/tools
        output:  ./pyMETRIC/nlcd
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'nlcd')

    parser = argparse.ArgumentParser(
        description='Download NLCD',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output', help='Output folder', metavar='FOLDER',
        default=os.path.join(project_folder, 'nlcd'))
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-y', '--year', metavar='YEAR', default='2011',
        choices=['2006', '2011'], help='NLCD Year (2006 or 2011)')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Convert output folder to an absolute path
    if args.output and os.path.isdir(os.path.abspath(args.output)):
        args.output = os.path.abspath(args.output)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')

    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Run Time Stamp:', dt.datetime.now().isoformat(' ')))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(output_folder=args.output, overwrite_flag=args.overwrite)
