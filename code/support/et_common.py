#--------------------------------
# Name:         et_common.py
# Purpose:      Common ET support functions
# Python:       2.7, 3.5, 3.6
#--------------------------------

from calendar import monthrange
from collections import defaultdict
import datetime as dt
import logging
import math
import os
import re
import sys

import numpy as np
# Used by soil water balance point_swb_func
from osgeo import gdal, ogr, osr

import gdal_common as gdc
import python_common


def landsat_folder_split(landsat_ws):
    """"""
    return landsat_id_split(os.path.basename(landsat_ws))

# def landsat_name_split(folder_name):
#     """Split Landsat folder name into components (Landsat, path, row, year, month, year)
#
#     """
#     # DEADBEEF - Scenes will not be merged, so it is unnecessary to
#     #   to return a row_start and row_end
#     landsat_pre_re = re.compile('^(LT4|LT5|LE7|LC8)\d{3}\d{3}\d{7}')
#     landsat_c1_re = re.compile('^(LT04|LT05|LE07|LC08)\d{3}_\d{6}_\d{8}')
#     if landsat_pre_re.match(folder_name):
#         landsat = folder_name[0: 3]
#         path = folder_name[3: 6]
#         row_start = folder_name[7: 10]
#         row_end = folder_name[10: 13]
#         year = folder_name[14: 18]
#         doy = folder_name[18: 21]
#     elif landsat_c1_re.match(folder_name):
#         landsat = folder_name[0: 3]
#         path = folder_name[3: 6]
#         row_start = folder_name[6: 9]
#         row_end = folder_name[6: 9]
#         year = folder_name[9:13]
#         month = folder_name[13:16]
#         day = folder_name[13:16]
#     # elif landsat_cloud_re.match(folder_name):
#     #     landsat = folder_name[0: 3]
#     #     path = folder_name[3: 6]
#     #     row_start = folder_name[7: 10]
#     #     row_end = folder_name[10: 13]
#     #     year = folder_name[14: 18]
#     #     doy = folder_name[18: 21]
#     else:
#         logging.error(
#             'ERROR: Could not parse landsat folder {}'.format(folder_name))
#     return landsat, path, row_start, row_end, year, month, day

def landsat_c1_id_func(pre_id):
    """Return a new style Landsat ID from an old style one"""
    landsat_pre_re = re.compile(
        '^(LT4|LT5|LE7|LC8)(\d{3})(\d{3})(\d{4})(\d{3})(\w{3})?(\d{2})?')
    m_groups = landsat_pre_re.match(pre_id).groups()
    satellite = m_groups[0][:2] + '0' + m_groups[0][2]
    path, row, year, doy = map(int, m_groups[1:5])
    id_dt = dt.datetime.strptime('{}_{}'.format(year, doy), '%Y_%j')
    return '{}_{:03d}{:03d}_{}'.format(
        satellite, path, row, id_dt.strftime('%Y%m%d'))

def landsat_pre_id_func(c1_id):
    """Return an old style Landsat ID from a new style one"""
    landsat_c1_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})')
    m_groups = landsat_c1_re.match(c1_id).groups()
    satellite = m_groups[0][:2] + m_groups[0][3]
    path, row, year, month, day = map(int, m_groups[1:6])
    id_dt = dt.datetime.strptime('{}{}{}'.format(year, month, day), '%Y%m%d')
    return '{}{:03d}{:03d}{:04d}{:03d}'.format(
        satellite, path, row, year, id_dt.strftime('%j'))

def landsat_id_split(landsat_id):
    """Split Landsat ID into components (Landsat, path, row, year, DOY)

    Will work on old or new style IDs

    """
    landsat_c1_re = re.compile(
        '^(LT04|LT05|LE07|LC08)_(\d{3})(\d{3})_(\d{4})(\d{2})(\d{2})')
    landsat_pre_re = re.compile(
        '^(LT4|LT5|LE7|LC8)(\d{3})(\d{3})(\d{4})(\d{3})(\w{3})?(\d{2})?')
    if landsat_c1_re.match(landsat_id):
        m_groups = landsat_c1_re.match(landsat_id).groups()
        satellite, path, row, year, month, day = m_groups[0:6]
    elif landsat_pre_re.match(landsat_id):
        m_groups = landsat_pre_re.match(landsat_id).groups()
        satellite = m_groups[0][:2] + '0' + m_groups[0][2]
        path, row, row_end, year, doy = m_groups[1:6]
        id_dt = dt.datetime.strptime('{}_{}'.format(year, doy), '%Y_%j')
        month = id_dt.month
        day = id_dt.day
    else:
        logging.error(
            'ERROR: Could not parse landsat folder {}'.format(landsat_id))
        landsat, path, row, year, month, day = None, None, None, None, None, None
    return satellite, path, row, year, month, day


def band_dict_to_array(data_dict, band_dict):
    """"""
    return np.array(
        [v for k, v in sorted(data_dict.items())
         if k in band_dict.keys()]).astype(np.float32)


def landsat_band_image_dict(ws, landsat_re):
    """Return a dictionary of Landsat images and band strings

    Copied from landsat_prep_scene_func.py
    Consider moving into et_common.py or making part of image class

    """
    if not os.path.isdir(ws):
        return dict()
    output_dict = dict()
    for item in os.listdir(ws):
        if not os.path.isfile(os.path.join(ws, item)):
            continue
        landsat_match = landsat_re.match(item)
        if not landsat_match:
            continue
        band = landsat_match.group('band').replace('B', '')
        # Only keep first thermal band from Landsat 7: "B6_VCID_1" -> "6"
        band = band.replace('_VCID_1', '')
        output_dict[band] = os.path.join(ws, item)
    return output_dict


def doy_range_func(landsat_doy_list, year, min_days_in_month=10):
    """Calculate DOY Range

    Args:
        landsat_doy_list ():
        year ():
        min_days_in_month ():

    Returns:
        list
    """
    # logging.info('\nDOY List: {}'.format(landsat_doy_list))
    year = int(year)
    doy_start = int(landsat_doy_list[0])
    doy_end = int(landsat_doy_list[-1])
    doy_zero_dt = dt.datetime(year, 1, 1) + dt.timedelta(-1)
    doy_start_dt = doy_zero_dt + dt.timedelta(doy_start)
    doy_end_dt = doy_zero_dt + dt.timedelta(doy_end)
    # First day of current start month and last day of current end month
    month_start_dt = dt.datetime(
        year, python_common.doy2month(year, doy_start), 1)
    month_end_dt = dt.datetime(
        year, python_common.doy2month(year, doy_end),
        monthrange(year, python_common.doy2month(year, doy_end))[-1])
    # Won't work for December because datetime doesn't accept month 13
    # month_end_dt = dt.datetime(year, month_end + 1, 1) + dt.timedelta(-1)
    # First day of next start month and last day of prior end month
    month_start_next_dt = dt.datetime(
        year, python_common.doy2month(year, doy_start)+1, 1)
    month_end_prev_dt = dt.datetime(
        year, python_common.doy2month(year, doy_end), 1) + dt.timedelta(-1)
    # Count of number of days between doy and inner month endpoints
    month_start_day_count = (month_start_next_dt - doy_start_dt).days
    month_end_day_count = (doy_end_dt - month_end_prev_dt).days
    # Check that there are enough days in start/end months
    if month_start_day_count < min_days_in_month:
        doy_start = (month_start_next_dt - (doy_zero_dt)).days
        doy_start_dt = doy_zero_dt + dt.timedelta(doy_start)
        logging.info(
            ('\nFirst day set to DOY: {:>3d}  ({})\n since there are '
             'only {} days of data in the previous month').format(
                 doy_start, doy_start_dt, month_start_day_count))
    else:
        doy_start = (month_start_dt - (doy_zero_dt)).days
        doy_start_dt = doy_zero_dt + dt.timedelta(doy_start)
        logging.info(('\nFirst day set to DOY: {:>3d}  ({})').format(
            doy_start, doy_start_dt))
    if month_end_day_count < min_days_in_month:
        doy_end = (month_end_prev_dt - (doy_zero_dt)).days
        doy_end_dt = doy_zero_dt + dt.timedelta(doy_end)
        logging.info(
            ('Last day set to DOY:  {:>3d}  ({})\n  There are '
             'only {} days of data in the next month').format(
                 doy_end, doy_end_dt, month_end_day_count))
    else:
        doy_end = (month_end_dt - (doy_zero_dt)).days
        doy_end_dt = doy_zero_dt + dt.timedelta(doy_end)
        logging.info(('Last day set to DOY:  {:>3d}  ({})').format(
            doy_end, doy_end_dt))
    return range(doy_start, doy_end+1)


def read_refet_instantaneous_func(refet_file, year, doy, localtime=None,
                                  ref_type='ETR'):
    """Read in instantaneous RefET data

    Args:
        refet_file (str):
        year (int):
        doy (int): day of year
        localtime ():
        ref_type (str): 'ETO' or 'ETR'

    Returns:
        tuple of floats: dew_point, wind_speed, ea, etr, & etr_24hr
    """
    logging.debug('  RefET: {}'.format(refet_file))

    # Field names
    year_field = 'Yr'
    month_field = 'Mo'
    day_field = 'Day'
    doy_field = 'DoY'
    hrmn_field = 'HrMn'
    tmax_field = 'Tmax'
    tmin_field = 'Tmin'
    rs_field = 'Rs'
    wind_field = 'Wind'
    dewp_field = 'DewP'
    if ref_type.upper() == 'ETO':
        etr_field = 'ASCE_stPM_ETo'
    else:
        etr_field = 'ASCE_stPM_ETr'

    # Field properties
    field_dict = dict()
    field_dict[month_field] = ('i8', '{:>2d}', '{:>2s}')
    field_dict[day_field] = ('i8', '{:>3d}', '{:>3s}')
    field_dict[year_field] = ('i8', '{:>4d}', '{:>4s}')
    field_dict[doy_field] = ('i8', '{:>3d}', '{:>3s}')
    field_dict[hrmn_field] = ('i8', '{:>4d}', '{:>4s}')
    field_dict[tmax_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[tmin_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[rs_field] = ('f8', '{:>4.0f}', '{:>4s}')
    field_dict[wind_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[dewp_field] = ('f8', '{:>5.2f}', '{:>5s}')
    field_dict[etr_field] = ('f8', '{:>5.2f}', '{:>5s}')

    # If localtime is not set, return daily means
    if localtime is None:
        daily_flag = True
    # If localtime is set, check that localtime value is valid
    elif not (0 <= localtime <= 24):
        logging.error((
            '\nERROR: localtime must be between 0 and 24.\n'
            'ERROR: value {} is invalid').format(localtime))
        sys.exit()
    else:
        daily_flag = False

    # Read in RefET file
    with open(refet_file, 'r') as refet_f:
        refet_list = refet_f.readlines()
    refet_f.close

    # Get line number where data starts
    header_split_line = 'RESULTS (SI Units):'
    refet_strip_list = [line.strip() for line in refet_list]
    try:
        header_line = refet_strip_list.index(header_split_line.strip())
        data_line = header_line + 6
    except IndexError:
        logging.error(
            '\nERROR: The line "RESULTS (SI Units):" could not be found in the RefET file'
            '\nERROR: This line is used to determine where to read data from the RefET file'
            '\nERROR: The units may not be metric or the file may be empty\n')
        sys.exit()
    # Split RefET file into header and data sections
    refet_header_list = refet_strip_list[header_line+2:data_line]
    refet_data_list = refet_list[data_line:]
    del refet_list, refet_strip_list

    # Filter spaces and newline characters at beginning and end
    # refet_list = [line.strip() for line in refet_list]
    refet_header_list = [line.strip() for line in refet_header_list]

    # This splits on whitespace
    # refet_list = [re.findall(r'[^\s]+', line) for line in refet_list]
    refet_header_list = [
        re.findall(r'[^\s]+', line) for line in refet_header_list]

    # Get field names
    refet_field_list = map(list, zip(*refet_header_list))

    # join with spaces, remove '-', remove leading/trailing whitespace
    # Last, to match genfromtxt, replace ' ' with '_'
    refet_field_name_list = [
        ' '.join(l[:3]).replace('-', '').strip().replace(' ', '_')
        for l in refet_field_list]
    refet_field_unit_list = [
        l[3].replace('-', '') for l in refet_field_list]
    refet_field_count = len(refet_field_list)
    logging.debug(
        '  Field name list:\n    {}'.format(refet_field_name_list))
    logging.debug(
        '  Field unit list:\n    {}'.format(refet_field_unit_list))

    # Check that date fields exist
    if year_field not in refet_field_name_list:
        logging.error(
            ('\nERROR: Year field {} was not found in the '
             'RefET file\n').format(year_field))
        sys.exit()
    if (month_field in refet_field_name_list and
        day_field in refet_field_name_list):
        doy_flag = False
    elif doy_field in refet_field_name_list:
        doy_flag = True
    else:
        logging.error((
            '\nERROR: Month field {} and Day field {} or DOY field '
            '{} were not found in the RefET file\n').format(
                month_field, day_field, doy_field))
        sys.exit()
    refet_field_name_list = [
        f for f in refet_field_name_list if f in field_dict.keys()]
    dtype_list = ','.join([field_dict[f][0] for f in refet_field_name_list])

    # Read data as record array
    refet_data = np.genfromtxt(
        refet_data_list, names=refet_field_name_list,
        dtype=dtype_list)

    # Build doy_array if necessary
    year_array = refet_data[year_field].astype(np.int)
    if not doy_flag:
        month_array = refet_data[month_field].astype(np.int)
        day_array = refet_data[day_field].astype(np.int)
        dt_array = np.array([
            dt.datetime(int(year), int(month), int(day))
            for year, month, day in zip(year_array, month_array, day_array)])
        doy_array = np.array([d.timetuple().tm_yday for d in dt_array])
        del month_array, day_array
        del dt_array
    else:
        doy_array = refet_data[doy_field].astype(np.int)
    doy_mask = (doy_array == doy) & (year_array == year)

    # Check that there is data for year/doy
    if not np.any(doy_mask):
        logging.error(
            '\nERROR: No data for Year {} and DOY {}\n'.format(
                year, doy))
        sys.exit()

    # Print daily data
    refet_data_subset = refet_data[doy_mask]
    del refet_data, doy_mask
    logging.debug('  ' + ' '.join(
        field_dict[f][2].format(f) for f in refet_field_name_list))
    for row in refet_data_subset:
        # DEADBEEF - In a try/except since it crashes for NumPy 1.6.1
        # The default for ArcGIS 10.1 is NumPy 1.6.1
        try:
            logging.debug('  ' + ' '.join(
                field_dict[f][1].format(value)
                for f, value in zip(refet_field_name_list, row)))
        except:
            pass

    # Extract sub arrays for interpolating
    hrmn_array = refet_data_subset[hrmn_field].astype(np.float32)

    # If times go from 1,2,...22,23,0 in a day, interpolation will break
    if hrmn_array[-1] == 0:
        hrmn_array[-1] = 2400

    # Convert HHMM to a float HH.MM to match localtime
    hrmn_array *= 0.01
    tmin_array = refet_data_subset[tmin_field].astype(np.float32)
    tmax_array = refet_data_subset[tmax_field].astype(np.float32)
    # rs_array = refet_data_subset[rs_field].astype(np.float32)
    wind_array = refet_data_subset[wind_field].astype(np.float32)
    dewp_array = refet_data_subset[dewp_field].astype(np.float32)
    etr_array = refet_data_subset[etr_field].astype(np.float32)

    # Calculate vapor pressure
    ea_array = saturation_vapor_pressure_func(dewp_array)
    # Interpolate dewpoint from RefET data
    # Add 0.5 hours because RefET data is an average of
    #   the previous hour

    # This may need to be set by the user or adjusted
    tair_inst = float(np.interp(
        [localtime + 0.5], hrmn_array, tmax_array)[0])
    dew_point = float(np.interp(
        [localtime + 0.5], hrmn_array, dewp_array)[0])
    ea_inst = float(np.interp(
        [localtime + 0.5], hrmn_array, ea_array)[0])
    wind_speed = float(np.interp(
        [localtime + 0.5], hrmn_array, wind_array)[0])
    etr_inst = float(np.interp(
        [localtime + 0.5], hrmn_array, etr_array)[0])

    # ETr 24hr (mm)
    etr_24hr = float(np.sum(etr_array))

    return dew_point, wind_speed, ea_inst, etr_inst, etr_24hr


def read_refet_daily_func(refet_list, year, doy_range, ref_type='ETR'):
    """Read in daily RefET data

    Args:
        refet_list ():
        year (int):
        doy_range (list):
        ref_type (str): 'ETO' or 'ETR'

    Returns:
        dict of DOY,ETr key/values
    """
    # Initialize ETr dictionary
    # doy_etr_dict = dict([(doy, 0) for doy in range(1,367)])
    doy_etr_dict = defaultdict(float)

    if ref_type.upper() == 'ETO':
        etr_field = 'ETo'
    else:
        etr_field = 'ETr'

    # Remove header information, everything above RESULTS
    # This re checks that any whitespace character can separate the words
    refet_results_re = re.compile('RESULTS\s+\(SI\s+Units\):')
    for i, refet_line in enumerate(refet_list):
        if refet_results_re.match(refet_line):
            refet_list[0:i+2] = []
            refet_header_list = refet_list[0:4]
            logging.debug('\n  RefET Data:')
            for refet_header_line in refet_header_list[0:4]:
                # logging.debug('    {}'.format(refet_header_line))
                refet_split_line = re.findall(r'[^\s]+', refet_header_line)
                logging.debug('    ' + ' '.join(
                    ['{:<5}'.format(i) for i in refet_split_line]))
            break
    try:
        len(refet_header_list)
    except NameError:
        logging.error(
            '\nERROR: The line "RESULTS (SI Units):" could not be found in the RefET file'
            '\nERROR: This line is used to determine where to read data from the RefET file'
            '\nERROR: The units may not be metric or the file may be empty\n')
        sys.exit()

    # From header rows, determine index for necessary fields
    for refet_header_line in refet_header_list:
        refet_split_line = re.findall(r'[^\s]+', refet_header_line)
        try:
            refet_yr_col = refet_split_line.index('Yr')
            refet_header_col_count = len(refet_split_line)
        except ValueError:
            pass
        try:
            refet_doy_col = refet_split_line.index('DoY')
        except ValueError:
            pass
        try:
            refet_etr_col = refet_split_line.index(etr_field)
        except ValueError:
            pass
    if (not refet_yr_col or not refet_doy_col or not refet_etr_col):
        logging.error('\nERROR: One of the necessary fields was not '
                      'found in the RefET file\n')
        sys.exit()

    # Calculate daily refet
    for refet_line in refet_list:
        # re finds every character that is not a whitespace character
        #   and splits on the whitespace
        refet_split_line = re.findall(r'[^\s]+', refet_line)
        if refet_split_line[refet_yr_col] == str(year):
            if refet_header_col_count != len(refet_split_line):
                logging.info('    {}'.format(refet_line))
                logging.error('\nERROR: The RefET file may be missing data\n'
                              'ERROR: The # of columns in the header '
                              'does not equal the # of columns of data')
                sys.exit()
            doy = int(refet_split_line[refet_doy_col])
            doy_etr_dict[doy] += float(refet_split_line[refet_etr_col])

    if not set(doy_range).issubset(doy_etr_dict.keys()):
        logging.error(
            ('\nERROR: The RefET file does not have ' +
             'data for year {}').format(year))
        sys.exit()
    return doy_etr_dict


def read_nvet_daily_func(nvet_list, year, doy_range):
    """Read in daily NVET data

    Args:
        nvet_list ():
        year (int):
        doy_range ():

    Returns:
        dict of DOY,ETr key/values
    """
    # Initialize RefET dictionary
    doy_etr_dict = dict([(doy, 0) for doy in range(1, 367)])

    # Remove header information, everything above RESULTS
    # This re checks that any whitespace character can separate the words
    nvet_header_list = nvet_list[0:5]
    nvet_list[0:5] = []
    logging.info('  NVET Header:')
    for nvet_header_line in nvet_header_list[0:5]:
        logging.info('    {}'.format(nvet_header_line))
    for nvet_line in nvet_list[0:3]:
        logging.info('    {}'.format(nvet_line))
    nvet_list[0:5] = []

    # Column numbers are hardcoded here
    logging.warning('\n  NVET columns are hardcoded and need to be checked')
    nvet_yr_col = 2
    nvet_doy_col = 3
    nvet_etr_col = 14
    logging.warning('    Year Column:  {:2d}'.format(nvet_yr_col+1))
    logging.warning('    DOY Column:   {:2d}'.format(nvet_doy_col+1))
    logging.warning('    RefET Column: {:2d}'.format(nvet_etr_col+1))

    # Calculate daily refet
    for nvet_line in nvet_list:
        # re finds every character that is not a whitespace character
        #   and splits on the whitespace
        nvet_split_line = re.findall(r'[^\s]+', nvet_line)
        if nvet_split_line[nvet_yr_col] == year:
            doy = int(nvet_split_line[nvet_doy_col])
            etr = float(nvet_split_line[nvet_etr_col])
            doy_etr_dict[doy] = etr

    # ETr must be greater than 0 to be valid?
    doy_valid_etr_list = [doy for doy in doy_range if doy_etr_dict[doy] > 0]

    # Check that there is ETr data for each DOY in doy_range
    if len(doy_valid_etr_list) == 0:
        logging.error(('\nERROR: The CSV ETr file does not contain data '
                       'for the year {}\n').format(year))
        sys.exit()
    elif set(doy_range) - set(doy_valid_etr_list):
        logging.error(
            ('\nERROR: The CSV ETr appears to have missing data' +
             '\n  The following days are missing:\n  {}').format(
                sorted(map(int, list(set(doy_range)-set(doy_valid_etr_list))))))
        sys.exit()
    return doy_etr_dict


def read_csv_etr_daily_func(csv_etr_list, year, doy_range):
    """Read in daily ETr from a CSV file

    Args:
        csv_etr_list ():
        year (int):
        doy_range ():

    Returns:
        dict
    """
    # Initialize RefET dictionary
    doy_etr_dict = dict([(doy, 0) for doy in range(1, 367)])

    # Remove header information, everything above RESULTS
    # This re checks that any whitespace character can separate the words
    header_line = csv_etr_list[0]
    data_list = csv_etr_list[1:]
    logging.info('  CSV ETr data:')
    logging.info('    {}'.format(header_line))

    # Print the first few lines as a check
    for data_line in data_list[0:3]:
        logging.info('    {}'.format(data_line))

    # Column names are hardcoded here
    year_field = 'YEAR'
    doy_field = 'DOY'
    etr_field = 'ETR'
    field_i_dict = dict()

    # Figure out column index for each field name
    split_line = [s.upper() for s in header_line.split(',')]
    for field in [year_field, doy_field, etr_field]:
        try:
            field_i_dict[field] = split_line.index(field.upper())
            logging.info('    {} Column:  {:>2d}'.format(
                field, field_i_dict[field]+1))
        except ValueError:
            logging.error(
                ('\nERROR: {} field does not exist in '
                 'CSV ETr file').format(field))
            sys.exit()

    # Calculate daily refet
    for data_line in data_list:
        # re finds every character that is not a whitespace character
        #   and splits on the whitespace
        split_line = data_line.split(',')
        if int(split_line[field_i_dict[year_field]]) == int(year):
            doy = int(split_line[field_i_dict[doy_field]])
            etr = float(split_line[field_i_dict[etr_field]])
            doy_etr_dict[doy] = etr

    # ETr must be greater than 0 to be valid?
    doy_valid_etr_list = [doy for doy in doy_range if doy_etr_dict[doy] > 0]

    # Check that there is ETr data for each DOY in doy_range
    if len(doy_valid_etr_list) == 0:
        logging.error(('\nERROR: The CSV ETr file does not contain data '
                       'for the year {}\n').format(year))
        sys.exit()
    elif set(doy_range) - set(doy_valid_etr_list):
        logging.error(
            ('\nERROR: The CSV ETr appears to have missing data' +
             '\n  The following days are missing:\n  {}').format(sorted(
                map(int, list(set(doy_range) - set(doy_valid_etr_list))))))
        sys.exit()
    return doy_etr_dict


def fixed_etr_data_func(etr, year, doy_range):
    """Assign a fixed ETr value to all doys in doy_range"""
    return dict([(doy, etr) for doy in range(1, 367) if doy in doy_range])


def u_star_station_func(wind_speed_height_flt, station_roughness_flt,
                        wind_speed_mod_flt):
    """U* at the station [m/s]"""
    return ((wind_speed_mod_flt * 0.41) /
            math.log(wind_speed_height_flt / station_roughness_flt))


def u3_func(u_star_station, z3, station_roughness):
    """U at blending height (200m) [m/s]"""
    return (u_star_station * math.log(z3 / station_roughness)) / 0.41


def refet_hourly_func(t, q, rs, uz, zw, elev, lat, lon, doy, time,
                      ref_type='ETR'):
    """ASCE Standardized Reference Evapotranspiration

    Multiply W m-2 by 0.0036 to get MJ m-2 hr-1

    Args:
        t (array: :class:`numpy.array`): average hourly temperature [C]
        q (array: :class:`numpy.array`): specific humidity [kg/kg]
        rs (array: :class:`numpy.array`): shortwave solar radiation
            [MJ m-2 day-1]
        uz (array: :class:`numpy.array`): windspeed [m/s]
        zw (float): windspeed measurement/estimated height [m]
        elev (array: :class:`numpy.array`): the workspace (path)
            of the landsat scene folder
        lat (array: :class:`numpy.array`): latitude [radians]
        lon (array: :class:`numpy.array`): longitude [radians]
        doy (array: :class:`numpy.array`): day of year
        time (array: :class:`numpy.array`): UTC hour
        ref_type (str): 'ETO' or 'ETR'

    Returns:
        array: :class:`numpy.array`
    """

    # Convert all inputs to NumPy arrays
    t = np.array(t, copy=True, ndmin=1)
    q = np.array(q, copy=True, ndmin=1)
    rs = np.array(rs, copy=True, ndmin=1)
    uz = np.array(uz, copy=True, ndmin=1)
    elev = np.array(elev, copy=True, ndmin=1)
    lat = np.array(lat, copy=True, ndmin=1)
    lon = np.array(lon, copy=True, ndmin=1)
    doy = np.array(doy, copy=True, ndmin=1)
    time = np.array(time, copy=True, ndmin=1)

    # ETo (tall reference) ASCE Penman-Monteith parameters
    if ref_type.upper() == 'ETO':
        cn_day = 37.0
        cd_day = 0.24
        g_rn_day = 0.1
        cn_night = 37.0
        cd_night = 0.96
        g_rn_night = 0.5
    # ETr (tall reference) ASCE Penman-Monteith parameters
    else:
        cn_day = 66.0
        cd_day = 0.25
        g_rn_day = 0.04
        cn_night = 66.0
        cd_night = 1.7
        g_rn_night = 0.2

    # To match standardized form, psy is calculated from elevation based pair
    pair = air_pressure_func(elev)
    # This matches the air pressure calculation in older versions of RefET
    # pair = 101.3 * np.power((285 - 0.0065 * elev) / 285, 5.26)
    psy = 0.000665 * pair
    es = saturation_vapor_pressure_func(t)
    es_slope = 4098 * es / np.power((t + 237.3), 2)

    # Vapor pressure from specific humidity
    # To match standardized form, ea is calculated from elevation based pair
    ea = actual_vapor_pressure_func(q, pair)

    # Extraterrestrial radiation
    ra = ra_hourly_func(lat, lon, doy, time)

    # Simplified clear sky solar formulation (Eqn 45)
    # rso = ra * (elev * 0.00002 + 0.75)

    # This is the full clear sky solar formulation
    sc = seasonal_correction_func(doy)
    omega = omega_func(solar_time_rad_func(lon, time, sc))
    # To match IN2, compute beta at start of period (700 or 7)
    #   even though times are listed at middle of period (730 or 7.5)
    # If times are at start omega_shift should be identical to omega
    omega_shift = omega_func(
        solar_time_rad_func(lon, np.round(time), sc))

    # sin of the angle of the sun above the horizon (D.6 and Eqn 62)
    delta = delta_func(doy)
    sin_beta = (
        np.sin(lat) * np.sin(delta) +
        np.cos(lat) * np.cos(delta) * np.cos(omega))
    sin_beta_shift = (
        np.sin(lat) * np.sin(delta) +
        np.cos(lat) * np.cos(delta) * np.cos(omega_shift))

    # Precipitable water (Eqn D.3)
    w = precipitable_water_func(pair, ea)

    # Clearness index for direct beam radiation (Eqn D.2)
    # Limit sin_beta >= 0.01 so that KB does not go undefined
    kt = 1.0
    kb = 0.98 * np.exp(
        (-0.00146 * pair) / (kt * np.maximum(sin_beta, 0.01)) -
        0.075 * np.power((w / np.maximum(sin_beta, 0.01)), 0.4))
    # Transmissivity index for diffuse radiation (Eqn D.4)
    kd = np.minimum(-0.36 * kb + 0.35, 0.82 * kb + 0.18)
    # (Eqn D.1)
    rso = ra * (kb + kd)

    # Cloudiness fraction (Eqn 45)
    # beta = np.arcsin(np.maximum(sin_beta, 0))
    beta = np.arcsin(np.maximum(sin_beta_shift, 0))
    fcd = np.ones(beta.shape)
    fcd[rso > 0] = 1.35 * np.clip(rs[rso > 0] / rso[rso > 0], 0.3, 1) - 0.35

    # For now just set fcd to 1 for low sun angles
    # DEADBEEF - Still need to get daytime value of fcd when beta > 0.3
    # Get closest value in time (array space) when beta > 0.3
    fcd[beta < 0.3] = 1

    # Net long-wave radiation (Eqn 44)
    rnl = rnl_hourly_func(t, ea, fcd)

    # Net radiation (Eqns 42 and 43)
    rn = rs * 0.77 - rnl

    # Adjust coefficients for daytime/nighttime
    # Nighttime is defined as when Rn < 0 (pg 44)
    cn = np.zeros(rn.shape)
    cd = np.zeros(rn.shape)
    g_rn = np.zeros(rn.shape)
    cn[:] = cn_day
    cd[:] = cd_day
    g_rn[:] = g_rn_day
    rn_mask = rn < 0
    cn[rn_mask] = cn_night
    cd[rn_mask] = cd_night
    g_rn[rn_mask] = g_rn_night

    # Soil heat flux (Eqns 65 and 66)
    g = rn * g_rn

    # Wind speed (Eqn 67)
    u2 = wind_height_adjust_func(uz, zw)

    # Hourly reference ET (Eqn 1)
    pet = (
        (0.408 * es_slope * (rn - g) + (psy * cn * u2 * (es - ea) / (t + 273))) /
        (es_slope + psy * (cd * u2 + 1)))
    return pet


def refet_daily_func(tmin, tmax, q, rs, uz, zw, elev, lat, doy,
                     ref_type='ETR', rso_type='FULL', rso=None):
    """ASCE Standardized Reference Evapotranspiration

    cn: 900 for ETo, 1600 for ETr
    cd: 0.34 for ETo, 0.38 for ETr
    Multiply W m-2 by 0.0864 to get MJ m-2 day-1

    Args:
        tmin (array: :class:`numpy.array`): minimum daily temperature [C]
        tmax (array: :class:`numpy.array`): maximum daily temperature [C]
        q (array: :class:`numpy.array`): specific humidity [kg/kg]
        rs (array: :class:`numpy.array`): incoming shortwave solar radiation
            [MJ m-2 day-1]
        uz (array: :class:`numpy.array`): windspeed [m/s]
        zw (array: :class:`numpy.array`): windspeed height [m]
        elev (array: :class:`numpy.array`): elevation [m]
        lat (array: :class:`numpy.array`): latitude [radians]
        doy (integer): day of year
        ref_type (str): 'ETO' or 'ETR'
        rso_type (str): 'SIMPLE', 'FULL', or 'ARRAY'
            ARRAY type will read rso function argument

    Returns:
        array: :class:`numpy.array`
    """

    # Convert all inputs to NumPy arrays
    # tmin = np.array(tmin, copy=True, ndmin=1)
    # tmax = np.array(tmax, copy=True, ndmin=1)
    # q = np.array(q, copy=True, ndmin=1)
    # rs = np.array(rs, copy=True, ndmin=1)
    # uz = np.array(uz, copy=True, ndmin=1)
    # elev = np.array(elev, copy=True, ndmin=1)
    # lat = np.array(lat, copy=True, ndmin=1)

    # Default to ETr if a type is not specified
    if ref_type.upper() == 'ETO':
        cn, cd = 900, 0.34
    else:
        cn, cd = 1600, 0.38
    # To match standardized form, psy is calculated from elevation based pair
    pair = air_pressure_func(elev)

    # This matches the air pressure calculation in older versions of RefET
    # pair = 101.3 * np.power((285 - 0.0065 * elev) / 285, 5.26)

    psy = 0.000665 * pair

    # Vapor pressure
    es_tmax = saturation_vapor_pressure_func(tmax)
    es_tmin = saturation_vapor_pressure_func(tmin)
    tmean = 0.5 * (tmax + tmin)
    es_tmean = saturation_vapor_pressure_func(tmean)
    es_slope = 4098.0 * es_tmean / (np.power((tmean + 237.3), 2))
    es = 0.5 * (es_tmax + es_tmin)

    # Vapor pressure from RHmax and RHmin
    # ea = 0.5 * (es_tmin * rhmax + es_tmax * rhmin)
    # Vapor pressure from specific humidity
    # To match standardized form, ea is calculated from elevation based pair
    ea = actual_vapor_pressure_func(q, pair)

    # Extraterrestrial radiation
    ra = ra_daily_func(lat, doy)

    if rso_type.upper() == 'FULL':
        # This is the full clear sky solar formulation
        # sin of the angle of the sun above the horizon (D.5 and Eqn 62)
        sin_beta_24 = np.sin(
            0.85 + 0.3 * lat * np.sin(doy_fraction_func(doy) - 1.39435) -
            0.42 * np.power(lat, 2))
        sin_beta_24 = np.maximum(sin_beta_24, 0.1)
        # Precipitable water (Eqn D.3)
        w = precipitable_water_func(pair, ea)
        # Clearness index for direct beam radiation (Eqn D.2)
        # Limit sin_beta >= 0.01 so that KB does not go undefined
        kb = (0.98 * np.exp((-0.00146 * pair) / sin_beta_24 -
                            0.075 * np.power((w / sin_beta_24), 0.4)))
        # Transmissivity index for diffuse radiation (Eqn D.4)
        kd = np.minimum(-0.36 * kb + 0.35, 0.82 * kb + 0.18)
        # Clear sky solar radiation (Eqn D.1)
        rso = ra * (kb + kd)
    elif rso_type.upper() == 'SIMPLE':
        # Simplified clear sky solar formulation (Eqn 19)
        rso = (0.75 + 2E-5 * elev) * ra
    elif rso_type.upper() == 'ARRAY':
        pass
    else:
        raise ValueError('rso_type must be "SIMPLE", "FULL", or "ARRAY')

    # Cloudiness fraction (Eqn 18)
    fcd = 1.35 * np.clip(rs / rso, 0.3, 1.0) - 0.35

    # Net long-wave radiation (Eqn 17)
    rnl = rnl_daily_func(tmax, tmin, ea, fcd)

    # Net radiation (Eqns 15 and 16)
    rn = 0.77 * rs - rnl

    # Wind speed (Eqn 33)
    u2 = wind_height_adjust_func(uz, zw)

    # Daily ETo (Eqn 1)
    pet = (
        (0.408 * es_slope * rn + (psy * cn * u2 * (es - ea) / (tmean + 273))) /
        (es_slope + psy * (cd * u2 + 1)))
    return pet


def doy_fraction_func(doy):
    """Fraction of the DOY in the year [radians]"""
    return doy * (2 * math.pi / 365.)


def delta_func(doy):
    """Earth declination [radians]"""
    return 0.40928 * np.sin(doy_fraction_func(doy) - 1.39435)
    # return 0.409 * np.sin(doy_fraction_func(doy) - 1.39)


def saturation_vapor_pressure_func(temperature):
    """Saturation vapor pressure [kPa] from temperature

    Args:
        temperature (array: :class:`numpy.array`): air temperature [C]

    Returns:
        array: :class:`numpy.array`
    """
    e = np.array(temperature, copy=True, ndmin=1).astype(np.float64)
    e += 237.3
    np.reciprocal(e, out=e)
    e *= temperature
    e *= 17.27
    np.exp(e, out=e)
    e *= 0.6108
    return e.astype(np.float32)
    # return 0.6108 * np.exp(17.27 * temperature / (temperature + 237.3))


def actual_vapor_pressure_func(q, pair):
    """"Actual vapor pressure [kPa] from specific humidity

    Args:
        q (array: :class:`numpy.array`): specific humidity [kg/kg]
        pair (array: :class:`numpy.array`): air pressure [kPa]

    Returns:
        array: :class:`numpy.array`
    """
    ea = np.array(q, copy=True, ndmin=1).astype(np.float64)
    ea *= 0.378
    ea += 0.622
    np.reciprocal(ea, out=ea)
    ea *= pair
    ea *= q
    return ea
    # return q * pair / (0.622 + 0.378 * q)


def specific_humidity_func(ea, pair):
    """"Specific humidity [kg/kg] from actual vapor pressure

    Args:
        ea (array: :class:`numpy.array`): specific humidity [kPa]
        pair (array: :class:`numpy.array`): air pressure [kPa]

    Returns:
        array: :class:`numpy.array`
    """
    q = np.array(ea, copy=True, ndmin=1).astype(np.float64)
    q *= -0.378
    q += pair
    np.reciprocal(q, out=q)
    q *= ea
    q *= 0.622
    return q
    # return 0.622 * ea / (pair - 0.378 * ea)


def air_pressure_func(elevation):
    """Air pressure [kPa]

    Args:
        elevation (array: :class:`numpy.array`): elevation [m]

    Returns:
        array: :class:`numpy.array`
    """
    pair = np.array(elevation, copy=True, ndmin=1).astype(np.float64)
    pair *= -0.0065
    pair += 293.15
    pair /= 293.15
    np.power(pair, 5.26, out=pair)
    pair *= 101.3
    return pair.astype(np.float32)
    # return 101.3 * np.power(((293.15 - 0.0065 * elev) / 293.15), 5.26)


def precipitable_water_func(pair, ea):
    """Precipitable water

    Args:
        pair: float/NumPy array of the air pressure [kPa]
        ea: float/NumPy array of the vapor pressure [kPa?]
    Returns:
        float/NumPy array
    """
    return pair * 0.14 * ea + 2.1


def dr_func(doy):
    """Inverse square of the Earth-Sun Distance

    This function returns 1 / d^2, not d, for direct use in radiance to
      TOA reflectance calculation
    pi * L * d^2 / (ESUN * cos(theta)) -> pi * L / (ESUN * cos(theta) * d)

    Args:
        doy: integer of the day of year
    Returns:
        float
    """
    return 1.0 + 0.033 * np.cos(doy_fraction_func(doy))


def ee_dr_func(doy):
    """Earth-Sun Distance values used by Earth Engine"""
    return 0.033 * np.cos(doy_fraction_func(doy)) + 1.0


def seasonal_correction_func(doy):
    """Seasonal correction for solar time [hour]"""
    b = 2 * math.pi * (doy - 81.) / 364.
    return 0.1645 * np.sin(2 * b) - 0.1255 * np.cos(b) - 0.0250 * np.sin(b)


def solar_time_rad_func(lon, time, sc):
    """Solar time [hours]

    Args:
        lon (array: :class:`numpy.array`): UTC hour [radians]
        time (array: :class:`numpy.array`): UTC hour [hours]
        sc (array: :class:`numpy.array`): seasonal correction [hours]

    Returns:
        array: :class:`numpy.array`

    """
    return time + (lon * 24 / (2 * math.pi)) + sc - 12

# def solar_time_func(lon, time, sc):
#     """Solar time (seconds) with longitude in degrees"""
#     return time + (lon / 15.) + sc

# def solar_time_deg_func(lon, time, sc):
#     """Solar time (seconds) with longitude in degrees"""
#     return time + (lon / 15.) + sc


def omega_func(solar_time):
    """Hour angle [radians]

   Args:
        solar_time (array: :class:`numpy.array`): UTC hour

    Returns:
        array: :class:`numpy.array`
    """
    omega = (2 * math.pi / 24.0) * solar_time

    # Need to adjust omega so that the values go from -pi to pi
    # Values outside this range are wrapped (i.e. -3*pi/2 -> pi/2)
    omega = wrap_func(omega, -math.pi, math.pi)

    return omega


def wrap_func(x, x_min, x_max):
    """Wrap floating point values into range

    Args:
        x (array: :class:`numpy.array`): array of values to wrap
        x_min (float): minimum value in output range
        x_max (float): maximum value in output range

    Returns:
        array: :class:`numpy.array`
    """
    return np.mod((x - x_min), (x_max - x_min)) + x_min


def omega_sunset_func(lat, delta):
    """Sunset hour angle [radians] (Eqn 59)

    Args:
        lat (array: :class:`numpy.array`): latitude [radians]
        delta (array: :class:`numpy.array`): earth declination [radians]

    Returns:
        array: :class:`numpy.array`
    """
    return np.arccos(-np.tan(lat) * np.tan(delta))


def ra_daily_func(lat, doy):
    """Daily extraterrestrial radiation [MJ m-2 d-1]

    Args:
        lat (array: :class:`numpy.array`): latitude [radians]
        doy (array: :class:`numpy.array`): day of year

    Returns:
        array: :class:`numpy.array`
    """
    delta = delta_func(doy)
    omegas = omega_sunset_func(lat, delta)
    theta = (omegas * np.sin(lat) * np.sin(delta) +
             np.cos(lat) * np.cos(delta) * np.sin(omegas))
    return (24. / math.pi) * 4.92 * dr_func(doy) * theta


def ra_hourly_func(lat, lon, doy, time):
    """Hourly extraterrestrial radiation [MJ m-2 h-1]

    Args:
        lat (array: :class:`numpy.array`): latitude [radians]
        lon (array: :class:`numpy.array`): longitude [radians]
        doy (array: :class:`numpy.array`): day of year
        time (array: :class:`numpy.array`): UTC hour

    Returns:
        array: :class:`numpy.array`
    """
    omega = omega_func(solar_time_rad_func(
        lon, time, seasonal_correction_func(doy)))
    delta = delta_func(doy)
    omegas = omega_sunset_func(lat, delta)

    # Solar time as start and end of period (Eqns 53 & 54)
    # Modify omega1 and omega2 at sunrise and sunset (Eqn 56)
    omega1 = np.clip(omega - (math.pi / 24), -omegas, omegas)
    omega2 = np.clip(omega + (math.pi / 24), -omegas, omegas)
    omega1 = np.minimum(omega1, omega2)

    # Extraterrestrial radiation (Eqn 48)
    theta = (
        ((omega2 - omega1) * np.sin(lat) * np.sin(delta)) +
        (np.cos(lat) * np.cos(delta) * (np.sin(omega2) - np.sin(omega1))))
    return (12. / math.pi) * 4.92 * dr_func(doy) * theta


def rnl_hourly_func(t, ea, fcd):
    """Hourly net longwave radiation [MJ m-2 h-1]

    Args:
        t (array: :class:`numpy.array`): mean hourly air temperature [C]
        ea (array: :class:`numpy.array`): actual vapor pressure [kPa]
        fcd (array: :class:`numpy.array`): cloudiness fraction

    Returns:
        array: :class:`numpy.array`
    """
    return (
         2.042E-10 * fcd * (0.34 - 0.14 * np.sqrt(ea)) *
         np.power((t + 273.16), 4))


def rnl_daily_func(tmax, tmin, ea, fcd):
    """Daily net longwave radiation [MJ m-2 d-1]

    Args:
        tmax (array: :class:`numpy.array`): daily maximum air temperature [C]
        tmin (array: :class:`numpy.array`): daily minimum air temperature [C]
        ea (array: :class:`numpy.array`): actual vapor pressure [kPa]
        fcd (array: :class:`numpy.array`): cloudiness fraction

    Returns:
        array: :class:`numpy.array`
    """
    return (
        4.901E-9 * fcd * (0.34 - 0.14 * np.sqrt(ea)) *
        0.5 * (np.power(tmax + 273.15, 4) + np.power(tmin + 273.15, 4)))


def wind_height_adjust_func(uz, zw):
    """Adjust wind speed to new height based on full logarithmic profile

    Args:
        uz (array: :class:`numpy.array`): wind speed [m/s]
        zw (array: :class:`numpy.array`): wind measurement height [m]

    Returns:
        array: :class:`numpy.array`
    """
    return uz * 4.87 / np.log(67.8 * zw - 5.42)


def cos_theta_solar_func(sun_elevation):
    """Cosine of theta at a point given sun elevation angle"""
    return math.sin(sun_elevation * math.pi / 180.)


def cos_theta_centroid_func(t, doy, dr, lon_center, lat_center):
    """Cosine of theta at a point

    Args:
        t ():
        doy ():
        dr ():
        lon_center ():
        lat_center ():

    Returns:
        float
    """
    # Solar time seasonal correction
    sc = seasonal_correction_func(doy)
    # Earth declination
    delta = delta_func(doy)
    # Latitude in radians
    solar_time = solar_time_rad_func(lon_center, t, sc)
    omega = omega_func(solar_time)
    # Cosine of theta for a flat earth
    cos_theta = ((math.sin(delta) * math.sin(lat_center)) +
                 (math.cos(delta) * math.cos(lat_center) * math.cos(omega)))
    log_f = '  {:<18s} {}'
    logging.debug('\n' + log_f.format(
        'Latitude Center:', (lat_center * math.pi / 180)))
    logging.debug(log_f.format(
        'Longitude Center:', (lon_center * math.pi / 180)))
    logging.debug(log_f.format('Delta:', delta))
    logging.debug(log_f.format('Sc [hour]:', sc))
    logging.debug(log_f.format('Sc [min]:', sc*60))
    logging.debug(log_f.format('Phi:', lat_center))
    logging.debug(log_f.format('SolarTime [hour]:', solar_time))
    logging.debug(log_f.format('SolarTime [min]:', solar_time*60))
    logging.debug(log_f.format('Omega: ', omega))
    logging.debug(log_f.format('cos_theta:', cos_theta))
    # return (env.mask_array * cos_theta).astype(np.float32)
    return cos_theta


def array_swb_func(awc, etr, ppt):
    """Daily soil water balance using arrays

    Calculate the spatial soil evaporation coefficient (Ke) through time.
    Script will assume the 0th axis of the input arrays is time.
    Spinup days are assumed to be in the data.

    Args:
        awc (array: :class:`numpy.array`): available water content [mm?]
        etr (array: :class:`numpy.array`): reference ET [mm]
        ppt (array: :class:`numpy.array`): precipitaiton [mm]

    Returns:
        array: :class:`numpy.array`: soil evaporation coeficient (Ke)

    """
    # logging.debug('Daily Soil Water Balance')
    ke = np.full(etr.shape, np.nan, np.float32)

    # Initialize Soil Water Balance parameters
    # Readily evaporable water (mm)
    rew = 54.4 * awc + 0.8
    # rew = (54.4 * awc / 1000) + 0.8

    # Total evaporable water (mm)
    tew = 166.0 * awc - 3.7
    # tew = (166.0 * awc / 1000) - 3.7

    # Total evaporable water (mm)
    # tew = (fc - 0.5 * wp) * (0.1 * 1000)

    # Difference of TEW and REW
    # tew_rew = tew - rew

    # Half Initial Depletion
    # de = 0.5 * tew
    # d_rew = 0.5 * rew

    # Dry initial Depletion
    de = np.copy(tew)
    d_rew = np.copy(rew)

    # Wet Initial Depletion
    # de = 0
    # d_rew = 0

    # Coefficients for Skin Layer Retention Efficiency, Allen (2011)
    c0 = 0.8
    c1 = 2 * (1 - c0)

    # ETr ke max
    ke_max = 1.0

    for i in xrange(etr.shape[0]):
        ke[i], de, d_rew = daily_swb_func(
            etr[i], ppt[i], de, d_rew, rew, tew, c0, c1, ke_max)
    return ke


def raster_swb_func(output_dt, output_osr, output_cs, output_extent,
                    awc_path, etr_ws, etr_re, ppt_ws, ppt_re,
                    spinup_days=30, min_spinup_days=5):
    """Calculate daily soil water balance for a raster for a single date

    Calculations will be done in AWC spatial reference & cellsize
    Final Ke will be projected to output spatial reference & cellsize
    Spinup SWB model for N spinup dates

    Args:
        output_dt (:class:`datetime.datetime`): datetime object representing the
            day for which the calculation is to be run
        output_osr (:class:`osr.SpatialReference): spatial reference
        output_cs (float): cellsize
        output_extent (): extent
        awc_path (str): filepath of the available water content raster
        etr_ws (str): directory path of the ETr workspace, which
            contains separate rasters for each year
        etr_re (:class:`re`): compiled regular expression object from the
            Python native 're' module that will match ETr rasters
        ppt_ws (str): directory path of the precipitation workspace, which
            contains separate rasters for each year
        ppt_re (:class:`re`): compiled regular expression object from the
            native Python re module that will match precipitaiton rasters
        spinup_days (int): number of days that should be used in the spinup
            of the model
        min_spinup_days (int): minimum number of days that are needed for
            spinup of the model

    Returns:
        array: :class:`numpy.array`: soil evaporation coeficient (Ke)

    """
    # DEADBEEF - There is probably a better way to handle the daterange input.
    #  Perhaps something like setting a minimum spinup and maximum spinup
    #  days and allowing the code to take however many etr and ppt rasters
    #  it can find within that range is good. Also, we should probably
    #  add in a flag for dry vs wet starting point (when it comes to
    #  total evaporative water [tew])
    # logging.debug('Daily Soil Water Balance')

    # Compute list of dates needed for spinup
    # date_range function doesn't return end date so add 1 day to end
    dt_list = sorted(python_common.date_range(
        output_dt - dt.timedelta(days=spinup_days),
        output_dt + dt.timedelta(days=1)))
    year_list = sorted(list(set([d.year for d in dt_list])))

    # Get all available ETr and PPT paths in date range
    if not os.path.isdir(etr_ws):
        logging.error('  ETr folder does not exist\n    {}'.format(
            etr_ws))
        sys.exit()
    if not os.path.isdir(ppt_ws):
        logging.info('  PPT folder does not exist\n    {}'.format(
            ppt_ws))
        sys.exit()

    # DEADBEEF - What specific errors should be caught here?
    etr_path_dict = dict()
    ppt_path_dict = dict()
    for etr_name in os.listdir(etr_ws):
        try:
            test_year = etr_re.match(etr_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            etr_path_dict[str(test_year)] = os.path.join(etr_ws, etr_name)
    for ppt_name in os.listdir(ppt_ws):
        try:
            test_year = ppt_re.match(ppt_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            ppt_path_dict[str(test_year)] = os.path.join(ppt_ws, ppt_name)
    if not etr_path_dict:
        logging.error('  No ETr rasters were found for the point SWB\n')
        sys.exit()
    elif not ppt_path_dict:
        logging.error('  No PPT rasters were found for the point SWB\n')
        sys.exit()

    # Get raster properties from one of the rasters
    # Project Landsat scene extent to ETr/PPT rasters
    logging.debug('  ETr: {}'.format(etr_path_dict[str(output_dt.year)]))
    etr_ds = gdal.Open(etr_path_dict[str(output_dt.year)], 0)
    etr_osr = gdc.raster_ds_osr(etr_ds)
    etr_cs = gdc.raster_ds_cellsize(etr_ds, x_only=True)
    etr_x, etr_y = gdc.raster_ds_origin(etr_ds)
    etr_extent = gdc.project_extent(
        output_extent, output_osr, etr_osr, cellsize=output_cs)
    etr_extent.buffer_extent(etr_cs * 2)
    etr_extent.adjust_to_snap('EXPAND', etr_x, etr_y, etr_cs)
    etr_ds = None

    logging.debug('  PPT: {}'.format(ppt_path_dict[str(output_dt.year)]))
    ppt_ds = gdal.Open(ppt_path_dict[str(output_dt.year)], 0)
    ppt_osr = gdc.raster_ds_osr(ppt_ds)
    ppt_cs = gdc.raster_ds_cellsize(ppt_ds, x_only=True)
    ppt_x, ppt_y = gdc.raster_ds_origin(ppt_ds)
    ppt_extent = gdc.project_extent(
        output_extent, output_osr, ppt_osr, cellsize=output_cs)
    ppt_extent.buffer_extent(ppt_cs * 2)
    ppt_extent.adjust_to_snap('EXPAND', ppt_x, ppt_y, ppt_cs)
    ppt_ds = None

    # Get AWC raster properties
    # Project Landsat scene extent to AWC raster
    logging.debug('  AWC: {}'.format(awc_path))
    awc_ds = gdal.Open(awc_path, 0)
    awc_osr = gdc.raster_ds_osr(awc_ds)
    awc_cs = gdc.raster_ds_cellsize(awc_ds, x_only=True)
    awc_x, awc_y = gdc.raster_ds_origin(awc_ds)
    awc_extent = gdc.project_extent(
        output_extent, output_osr, awc_osr, cellsize=output_cs)
    awc_extent.buffer_extent(awc_cs * 4)
    awc_extent.adjust_to_snap('EXPAND', awc_x, awc_y, awc_cs)
    awc_ds = None

    # SWB computations will be done in the AWC OSR, cellsize, and extent
    awc = gdc.raster_to_array(
        awc_path, band=1, mask_extent=awc_extent,
        return_nodata=False).astype(np.float32)
    # Clip/project AWC to Landsat scene
    # awc = clip_project_raster_func(
    #     awc_path, 1, gdal.GRA_NearestNeighbour,
    #     awc_osr, awc_cs, awc_extent,
    #     output_osr, output_cs, output_extent)

    # Convert units from cm/cm to mm/m
    # awc *= 1000
    # Scale available water capacity by 1000
    # Scale field capacity and wilting point from percentage to decimal
    # fc *= 0.01
    # wp *= 0.01

    # Initialize Soil Water Balance parameters
    # Readily evaporable water (mm)
    rew = 54.4 * awc + 0.8
    # rew = (54.4 * awc / 1000) + 0.8
    # Total evaporable water (mm)
    tew = 166.0 * awc - 3.7
    # tew = (166.0 * awc / 1000) - 3.7
    # Total evaporable water (mm)
    # tew = (fc - 0.5 * wp) * (0.1 * 1000)
    # Difference of TEW and REW
    # tew_rew = tew - rew

    # Dry initial Depletion
    de = np.copy(tew)
    d_rew = np.copy(rew)
    # de = np.copy(tew)
    # d_rew = np.copy(rew)
    # Half Initial Depletion
    # de = 0.5 * tew
    # d_rew = 0.5 * rew
    # Wet Initial Depletion
    # de = 0
    # d_rew = 0

    # Coefficients for Skin Layer Retention Efficiency, Allen (2011)
    c0 = 0.8
    c1 = 2 * (1 - c0)
    # ETr ke max
    ke_max = 1.0
    # ETo Ke max
    # ke_max = 1.2

    # Spinup model up to test date, iteratively calculating Ke
    # Pass doy as band number to raster_value_at_xy
    for spinup_dt in dt_list:
        logging.debug('    {}'.format(spinup_dt.date().isoformat()))
        etr = clip_project_raster_func(
            etr_path_dict[str(spinup_dt.year)],
            # int(spinup_dt.strftime('%j')), gdal.GRA_NearestNeighbour,
            int(spinup_dt.strftime('%j')), gdal.GRA_Bilinear,
            etr_osr, etr_cs, etr_extent, awc_osr, awc_cs, awc_extent)
        ppt = clip_project_raster_func(
            ppt_path_dict[str(spinup_dt.year)],
            # int(spinup_dt.strftime('%j')), gdal.GRA_NearestNeighbour,
            int(spinup_dt.strftime('%j')), gdal.GRA_Bilinear,
            ppt_osr, ppt_cs, ppt_extent, awc_osr, awc_cs, awc_extent)
        ke, de, d_rew = daily_swb_func(
            etr, ppt, de, d_rew, rew, tew, c0, c1, ke_max)

    # Project to output spatial reference, cellsize, and extent
    ke = gdc.project_array(
        # ke, gdal.GRA_NearestNeighbour,
        ke, gdal.GRA_Bilinear,
        awc_osr, awc_cs, awc_extent,
        output_osr, output_cs, output_extent,
        output_nodata=None)
    return ke


def clip_project_raster_func(input_raster, band, resampling_type,
                             input_osr, input_cs, input_extent,
                             ouput_osr, output_cs, output_extent):
    """Clip and then project an input raster"""
    # Read array from input raster using input extent
    input_array = gdc.raster_to_array(
        input_raster, band=band, mask_extent=input_extent,
        return_nodata=False).astype(np.float32)
    # Convert nan to a nodata value so a copy isn't made in project_array
    input_array[np.isnan(input_array)] = gdc.numpy_type_nodata(
        input_array.dtype)
    # Project and clip array to block
    output_array = gdc.project_array(
        input_array, resampling_type,
        input_osr, input_cs, input_extent,
        ouput_osr, output_cs, output_extent)
    return output_array


def point_swb_func(test_dt, test_xy, test_osr, awc_path,
                   etr_ws, etr_re, ppt_ws, ppt_re,
                   spinup_days=30, min_spinup_days=5):
    """Calculate daily soil water balance at a point for a single date

    Spinup SWB model for N spinup dates and calculate the Ke (soil evaporation
    coefficient) for the desired x/y coordinates.

    Args:
        test_dt (:class:`datetime.datetime`): datetime object representing the
            day for which the calculation is to be run
        test_xy (tuple): tuple of the x and y coordinates for which the soil
            water balance is to be calculated. Must be in the same projection
            as the test_osr
        test_osr (:class:`osr.SpatialReference): spatial reference of the
            text_xy point coordinates
        awc_path (str): filepath of the available water content raster
        etr_ws (str): directory path of the ETr workspace, which
            contains separate rasters for each year
        etr_re (:class:`re`): compiled regular expression object from the
            Python native 're' module that will match ETr rasters
        ppt_ws (str): directory path of the precipitation workspace, which
            contains separate rasters for each year
        ppt_re (:class:`re`): compiled regular expression object from the
            native Python re module that will match precipitaiton rasters
        spinup_days (int): number of days that should be used in the spinup
            of the model
        min_spinup_days (int): minimum number of days that are needed for
            spinup of the model

    Returns:
        float: soil evaporation coeficient (Ke)

    """
    # DEADBEEF - There is probably a better way to handle the daterange input.
    #  Perhaps something like setting a minimum spinup and maximum spinup
    #  days and allowing the code to take however many etr and ppt rasters
    #  it can find within that range is good. Also, we should probably
    #  add in a flag for dry vs wet starting point (when it comes to
    #  total evaporative water [tew])
    logging.debug('Daily Soil Water Balance')
    logging.debug('  Test Point: {} {}'.format(*test_xy))

    # Compute list of dates needed for spinup
    # date_range function doesn't return end date so add 1 day to end
    dt_list = sorted(python_common.date_range(
        test_dt - dt.timedelta(days=spinup_days),
        test_dt + dt.timedelta(days=1)))
    year_list = sorted(list(set([d.year for d in dt_list])))

    # Get all available ETr and PPT paths in date range
    etr_path_dict = dict()
    ppt_path_dict = dict()
    if not os.path.isdir(etr_ws):
        logging.error('  ETr folder does not exist\n    {}'.format(
            etr_ws))
        sys.exit()
    if not os.path.isdir(ppt_ws):
        logging.info('  PPT folder does not exist\n    {}'.format(
            ppt_ws))
        sys.exit()

    # DEADBEEF - What specific errors should be caught here?
    for etr_name in os.listdir(etr_ws):
        try:
            test_year = etr_re.match(etr_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            etr_path_dict[str(test_year)] = os.path.join(etr_ws, etr_name)
    for ppt_name in os.listdir(ppt_ws):
        try:
            test_year = ppt_re.match(ppt_name).group('YYYY')
        except:
            continue
        if int(test_year) in year_list:
            ppt_path_dict[str(test_year)] = os.path.join(ppt_ws, ppt_name)
    if not etr_path_dict:
        logging.error('  No ETr rasters were found for the point SWB\n')
        sys.exit()
    elif not ppt_path_dict:
        logging.error('  No PPT rasters were found for the point SWB\n')
        sys.exit()

    # for year in year_list:
    #     etr_year_ws = os.path.join(etr_ws, str(year))
    #     if os.path.isdir(etr_year_ws):
    #         for etr_name in os.listdir(etr_year_ws):
    #             try:
    #                 test_dt = dt.datetime.strptime(
    #                     etr_re.match(etr_name).group('YYYYMMDD'), '%Y%m%d')
    #             except:
    #                 continue
    #             if test_dt in dt_list:
    #                 etr_path_dict[test_dt.date().isoformat()] = os.path.join(
    #                     etr_year_ws, etr_name)
    #     else:
    #         logging.info('  ETr year folder does not exist\n    {}'.format(
    #             etr_year_ws))

    #     ppt_year_ws = os.path.join(ppt_ws, str(year))
    #     if os.path.isdir(ppt_year_ws):
    #         for ppt_name in os.listdir(ppt_year_ws):
    #             try:
    #                 test_dt = dt.datetime.strptime(
    #                     ppt_re.match(ppt_name).group('YYYYMMDD'), '%Y%m%d')
    #             except:
    #                 continue
    #             if test_dt in dt_list:
    #                 ppt_path_dict[test_dt.date().isoformat()] = os.path.join(
    #                     ppt_year_ws, ppt_name)
    #     else:
    #         logging.info('  PPT year folder does not exist\n    {}'.format(
    #             ppt_year_ws))

    # DEADBEEF - Need a different way to check for spin up dates
    # # Check the number of available ETr/PPT images
    # etr_spinup_days = len(etr_path_dict.keys()) - 1
    # ppt_spinup_days = len(ppt_path_dict.keys()) - 1
    # if etr_spinup_days < spinup_days:
    #     logging.warning('  Only {}/{} ETr spinup days available'.format(
    #         etr_spinup_days, spinup_days))
    #     if etr_spinup_days <= min_spinup_days:
    #         logging.error('    Exiting')
    #         exit()
    # if ppt_spinup_days < spinup_days:
    #     logging.warning('  Only {}/{} PPT spinup days available'.format(
    #         ppt_spinup_days, spinup_days))
    #     if ppt_spinup_days <= min_spinup_days:
    #         logging.error('    Exiting')
    #         sys.exit()

    # Project input point to AWC coordinate system
    awc_pnt = ogr.Geometry(ogr.wkbPoint)
    awc_pnt.AddPoint(test_xy[0], test_xy[1])
    awc_pnt.Transform(osr.CoordinateTransformation(
        test_osr, gdc.raster_path_osr(awc_path)))
    logging.debug('  AWC Point: {} {}'.format(
        awc_pnt.GetX(), awc_pnt.GetY()))

    # Project input point to ETr coordinate system
    etr_pnt = ogr.Geometry(ogr.wkbPoint)
    etr_pnt.AddPoint(test_xy[0], test_xy[1])
    etr_pnt.Transform(osr.CoordinateTransformation(
        test_osr, gdc.raster_path_osr(list(etr_path_dict.values())[0])))
    logging.debug('  ETr Point: {} {}'.format(
        etr_pnt.GetX(), etr_pnt.GetY()))

    # Project input point to PPT coordinate system
    ppt_pnt = ogr.Geometry(ogr.wkbPoint)
    ppt_pnt.AddPoint(test_xy[0], test_xy[1])
    ppt_pnt.Transform(osr.CoordinateTransformation(
        test_osr, gdc.raster_path_osr(list(ppt_path_dict.values())[0])))
    logging.debug('  PPT Point: {} {}'.format(
        ppt_pnt.GetX(), ppt_pnt.GetY()))

    # Read in soil properties
    awc = gdc.raster_value_at_point(awc_path, awc_pnt)
    # Convert units from cm/cm to mm/m
    # awc *= 1000
    # fc = gdc.raster_value_at_point(fc_path, test_pnt)
    # wp = gdc.raster_value_at_point(wp_path, test_pnt)
    # Scale available water capacity by 1000
    # Scale field capacity and wilting point from percentage to decimal
    # fc *= 0.01
    # wp *= 0.01

    # Initialize Soil Water Balance parameters
    # Readily evaporable water (mm)
    rew = 54.4 * awc + 0.8
    # rew = (54.4 * awc / 1000) + 0.8
    # Total evaporable water (mm)
    tew = 166.0 * awc - 3.7
    # tew = (166.0 * awc / 1000) - 3.7
    # Total evaporable water (mm)
    # tew = (fc - 0.5 * wp) * (0.1 * 1000)
    # Difference of TEW and REW
    # tew_rew = tew - rew

    # Dry initial Depletion
    de = float(tew)
    d_rew = float(rew)
    # de = np.copy(tew)
    # d_rew = np.copy(rew)
    # Half Initial Depletion
    # de = 0.5 * tew
    # d_rew = 0.5 * rew
    # Wet Initial Depletion
    # de = 0
    # d_rew = 0

    # Coefficients for Skin Layer Retention Efficiency, Allen (2011)
    c0 = 0.8
    c1 = 2 * (1 - c0)
    # ETr ke max
    ke_max = 1.0
    # ETo Ke max
    # ke_max = 1.2

    logging.debug('  AWC: {}'.format(awc))
    # logging.debug('  FC:  {}'.format(fc))
    # logging.debug('  WP:  {}'.format(wp))
    logging.debug('  REW: {}'.format(rew))
    logging.debug('  TEW: {}'.format(tew))
    logging.debug('  de:  {}'.format(de))
    logging.debug(
        '\n  {:>10s} {:>5s} {:>5s} {:>5s} {:>5s} {:>5s}'.format(
            *'DATE,ETR,PPT,KE,DE,D_REW'.split(',')))

    # Spinup model up to test date, iteratively calculating Ke
    # Pass doy as band number to raster_value_at_point
    for spinup_dt in dt_list:
        etr, ppt = 0., 0.
        try:
            etr = gdc.raster_value_at_point(
                etr_path_dict[str(spinup_dt.year)], etr_pnt,
                band=int(spinup_dt.strftime('%j')))
        except KeyError:
            logging.debug(
                '  ETr raster for date {} does not exist'.format(
                    spinup_dt.date().isoformat()))
        try:
            ppt = gdc.raster_value_at_point(
                ppt_path_dict[str(spinup_dt.year)], ppt_pnt,
                band=int(spinup_dt.strftime('%j')))
        except KeyError:
            logging.debug(
                '  PPT raster for date {} does not exist'.format(
                    spinup_dt.date().isoformat()))

        ke, de, d_rew = map(float, daily_swb_func(
            etr, ppt, de, d_rew, rew, tew, c0, c1, ke_max))
        logging.debug((
            '  {:>10s} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}').format(
                spinup_dt.date().isoformat(), etr, ppt, ke, de, d_rew))
    return ke


def daily_swb_func(etr, ppt, de_prev, d_rew_prev, rew, tew,
                   c0=0.8, c1=0.4, ke_max=1.0):
    """Daily soil water balance function

    Args:
        etr ():
        ppt ():
        de_prev ():
        d_rew_prev ():
        rew ():
        tew ():
        c0 ():
        c1 ():
        ke_max ():

    Returns:
        tuple: numpy.arrays (ke, de, d_rew)
    """
    # Stage 1 evaporation (Eqn 1)
    e1 = np.array(etr, copy=True, ndmin=1)
    e1 *= ke_max

    # Evaporation reduction coefficient (Eqn 5b)
    kr = np.clip((tew - de_prev) / (tew - rew), 0, 1)

    # Fraction of time interval residing in stage 1 (Eqn 10b)
    # Don't calculate ".min(1)" here, wait until in Es calc
    ft = np.clip(np.nan_to_num((rew - d_rew_prev) / e1), 0, 1)

    # Total evaporation from the soil (Eqn 11)
    es = np.clip((1 - ft) * kr * e1 - d_rew_prev + rew, 0, e1)

    # Infiltration efficiency factor (Eqn 13)
    ceff = np.clip(c1 * ((tew - de_prev) / tew) + c0, 0, 1)

    # Depletion of the skin layer
    # With skin evap calculation (Eqn 12)
    # Modified to remove fb adjustment
    d_rew = np.copy(np.clip((d_rew_prev - (ceff * ppt) + es), 0, rew))
    # d_rew = np.clip((d_rew_prev - (ceff * ppt) + es), 0, rew)

    # Without skin evap (Eqn )
    # var d_rew = rew
    # Depth of evaporation of the TEW surface soil layer (Eqn 9)
    # Modified to remove fb adjustment
    de = np.copy(np.clip(de_prev - ppt + es, 0, tew))
    # de = np.clip(de_prev - ppt + es, 0, tew)

    # # Save current as previous for next iteration
    # de_prev = de
    # d_rew_prev = d_rew

    # Evaporation coefficient (Eqn )
    ke = np.clip(np.nan_to_num(es / etr), 0, 1)
    return ke, de, d_rew


def cell_value_set(test_raster, test_name, cold_xy, hot_xy, log_level='INFO'):
    """Extract the raster values at the cold and hot calibration points

    X and Y coordinates need to be in the same spatial reference as the raster

    Args:
        test_raster (str): filepath of the raster to be sampled
        test_name (str): display name of the raster (for logging)
        cold_xy (tuple): x and y coordinate of the cold calibration point
        hot_xy (tuple): x and y coordinate of the cold calibration point
        log_level (str): logging level (INFO, DEBUG)

    Returns:
        tuple of the values at the calibration points
    """
    cold_flt = gdc.raster_value_at_xy(test_raster, cold_xy)
    hot_flt = gdc.raster_value_at_xy(test_raster, hot_xy)
    log_str = '    {:<14s}  {:14.8f}  {:14.8f}'.format(
        test_name+':', cold_flt, hot_flt)
    if log_level == 'DEBUG':
        logging.debug(log_str)
    else:
        logging.info(log_str)
    return cold_flt, hot_flt
