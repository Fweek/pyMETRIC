#--------------------------------
# Name:         download_soils.py
# Purpose:      Download soil AWC raster
# Python:       2.7, 3.5, 3.6
#--------------------------------

import argparse
import datetime as dt
import logging
import os
import sys
import zipfile

from python_common import url_download


def main(output_folder, overwrite_flag=False):
    """Download soil Available Water Capacity (AWC) raster

    Args:
        output_folder (str): folder path where files will be saved
        overwrite_flag (bool): If True, overwrite existing files

    Returns:
        None
    """
    # Composite SSURGO/STATSGO
    download_url = 'https://storage.googleapis.com/openet/ssurgo/AWC_WTA_0to10cm_composite.tif'

    # STATSGO Only
    # download_url = 'https://storage.googleapis.com/openet/statsgo/AWC_WTA_0to10cm_statsgo.tif'

    output_name = download_url.split('/')[-1]
    output_path = os.path.join(output_folder, output_name)
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    if not os.path.isfile(output_path) or overwrite_flag:
        logging.info('\nDownloading AWC')
        logging.info('  {}'.format(download_url))
        logging.info('  {}'.format(output_path))
        url_download(download_url, output_path)
    else:
        logging.debug('\nAWC raster already downloaded')


def arg_parse():
    """Base all default folders from script location
        scripts: ./pyMETRIC/code/download
        code:    ./pyMETRIC/code
        output:  ./pyMETRIC/soils
    """
    script_folder = sys.path[0]
    code_folder = os.path.dirname(script_folder)
    project_folder = os.path.dirname(code_folder)
    output_folder = os.path.join(project_folder, 'soils')

    parser = argparse.ArgumentParser(
        description='Download Soil Available Water Capcity (AWC)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output', help='Output folder', metavar='FOLDER',
        default=os.path.join(project_folder, 'soils'))
    parser.add_argument(
        '-o', '--overwrite', default=None, action="store_true",
        help='Force overwrite of existing files')
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
