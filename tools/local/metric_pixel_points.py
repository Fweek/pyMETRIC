#!/usr/bin/env python
#--------------------------------
# Name:         metric_pixel_points.py
# Purpose:      Run METRIC pixel points function for all images
# Python:       2.7, 3.5, 3.6
#--------------------------------

import argparse
from datetime import datetime
import logging
import multiprocessing as mp
import os
import re
import subprocess
import sys

from tools.support.python_common import open_ini, read_param, call_mp


def main(ini_path, tile_list=None, groupsize=1, blocksize=2046,
         shapefile_flag=True, geojson_flag=False, overwrite_flag=False,
         mp_procs=1, delay=0, debug_flag=False, new_window_flag=False):
    """Run pixel points for all images

    Args:
        ini_path (str): file path of the input parameters file
        tile_list (list): list of Landsat path/row (i.e. [p45r43, p45r33])
            This will override the tile list in the INI file
        groupsize (int):
            Script will try to place calibration point randomly into a
            labeled group of clustered values with at least n pixels.
            -1 = In the largest group
             0 = Anywhere in the image (not currently implemented)
             1 >= In any group with a pixel count greater or equal to n
        blocksize (int):
        shapefile_flag (bool): if True, save calibration points to shapefile
            (default False)
        geojson_flag (bool): if True, save calibration points to GeoJSON
        overwrite_flag (bool): if True, overwrite existing files
        mp_procs (int): number of cores to use
        delay (float): max random delay starting function in seconds
        debug_flag (bool): if True, enable debug level logging
        new_window_flag (bool): if True, open each process in a new terminal.
            Microsoft Windows only.

    Returns:
        None
    """
    logging.info('\nRunning METRIC Pixel Points for all images')
    log_fmt = '  {:<18s} {}'

    # Open config file
    config = open_ini(ini_path)

    # Get input parameters
    logging.debug('  Reading Input File')
    year = config.getint('INPUTS', 'year')
    if tile_list is None:
        tile_list = read_param('tile_list', [], config, 'INPUTS')
    project_ws = config.get('INPUTS', 'project_folder')
    logging.debug('  Year: {}'.format(year))
    logging.debug('  Path/rows: {}'.format(', '.join(tile_list)))
    logging.debug('  Project: {}'.format(project_ws))

    func_path = config.get('INPUTS', 'pixel_points_func')

    # DEADBEEF - seems like this is passed in at the command line
    # groupsize = config.getint('INPUTS', 'groupsize')

    # Only allow new terminal windows on Windows
    if os.name is not 'nt':
        new_window_flag = False

    # Regular expressions
    tile_re = re.compile('p\d{3}r\d{3}', re.IGNORECASE)
    image_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})')

    # Check inputs folders/paths
    if not os.path.isdir(project_ws):
        logging.error('\n Folder {} does not exist'.format(project_ws))
        sys.exit()

    # Setup command line argument
    call_args = [sys.executable, func_path]
    call_args.extend(['--groupsize', str(groupsize)])
    if blocksize:
        call_args.extend(['--blocksize', str(blocksize)])
    if shapefile_flag:
        call_args.append('--shapefile')
    if geojson_flag:
        call_args.append('--geojson')
    if overwrite_flag:
        call_args.append('--overwrite')
    if debug_flag:
        call_args.append('--debug')

    mp_list = []
    for tile_name in sorted(tile_list):
        tile_ws = os.path.join(project_ws, str(year), tile_name)
        if not os.path.isdir(tile_ws) and not tile_re.match(tile_name):
            continue

        # Check that there are scene folders
        scene_id_list = [
            scene_id for scene_id in sorted(os.listdir(tile_ws))
            if (os.path.isdir(os.path.join(tile_ws, scene_id)) or
                image_re.match(scene_id))]
        if not scene_id_list:
            continue
        logging.debug('  {} {}'.format(year, tile_name))

        # Run METRIC Pixel Points
        for scene_id in scene_id_list:
            logging.debug('  {}'.format(scene_id))
            scene_ws = os.path.join(tile_ws, scene_id)
            pixel_ws = os.path.join(scene_ws, 'PIXELS')
            # Since the GeoJSON will be appended, delete it in the wrapper
            #  script if the overwrite_flag=True
            if geojson_flag and os.path.isdir(pixel_ws):
                for pixel_file in os.listdir(pixel_ws):
                    if re.match('\w+.geojson$', pixel_file):
                        os.remove(os.path.join(pixel_ws, pixel_file))
            if mp_procs > 1:
                mp_list.append(
                    [call_args, scene_ws, delay, new_window_flag])
            else:
                subprocess.call(call_args, cwd=scene_ws)

    if mp_list:
        pool = mp.Pool(mp_procs)
        results = pool.map(call_mp, mp_list, chunksize=1)
        pool.close()
        pool.join()
        del results, pool


def arg_parse():
    """"""
    parser = argparse.ArgumentParser(
        description='Batch METRIC Pixel Points',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-i', '--ini', required=True,
        help='Landsat project input file', metavar='PATH')
    parser.add_argument(
        '-gs', '--groupsize', default=1, type=int,
        help='Minimum group size for placing calibration points')
    parser.add_argument(
        '-bs', '--blocksize', default=2048, type=int,
        help='Block size')
    # The "no_shapefile" parameter is negated below to become "shapefile".
    # By default, pixel_points_func will NOT save calibration point shapefiles.
    # If a user runs this "local" script, they probably want shapefiles.
    # If not, user can "turn off" saving shapefiles.
    parser.add_argument(
        '--no_shapefile', default=False, action="store_true",
        help='Don\'t save calibration points to shapefile')
    parser.add_argument(
        '-j', '--geojson', default=False, action="store_true",
        help='Save calibration points to GeoJSON')
    parser.add_argument(
        '-o', '--overwrite', default=False, action="store_true",
        help='Force overwrite of existing files')
    parser.add_argument(
        '-mp', '--multiprocessing', default=1, type=int,
        metavar='N', nargs='?', const=mp.cpu_count(),
        help='Number of processers to use')
    parser.add_argument(
        '--window', default=False, action="store_true",
        help='Open each process in a new terminal (windows only)')
    parser.add_argument(
        '--delay', default=0, type=int, metavar='N',
        help='Max random delay starting job in seconds')
    parser.add_argument(
        '-d', '--debug', default=logging.INFO, const=logging.DEBUG,
        help='Debug level logging', action="store_const", dest="loglevel")
    args = parser.parse_args()

    # Default is to save point shapfiles (opposite of --no_stats default=False)
    args.shapefile = not args.no_shapefile

    # Convert relative paths to absolute paths
    if args.ini and os.path.isfile(os.path.abspath(args.ini)):
        args.ini = os.path.abspath(args.ini)
    return args


if __name__ == '__main__':
    args = arg_parse()

    logging.basicConfig(level=args.loglevel, format='%(message)s')
    logging.info('\n{}'.format('#' * 80))
    log_f = '{:<20s} {}'
    logging.info(log_f.format('Run Time Stamp:', datetime.now().isoformat(' ')))
    logging.info(log_f.format('Current Directory:', os.getcwd()))
    logging.info(log_f.format('Script:', os.path.basename(sys.argv[0])))

    main(ini_path=args.ini, groupsize=args.groupsize,
         blocksize=args.blocksize, shapefile_flag=args.shapefile,
         geojson_flag=args.geojson, overwrite_flag=args.overwrite,
         mp_procs=args.multiprocessing, delay=args.delay,
         debug_flag=args.loglevel==logging.DEBUG, new_window_flag=args.window)
