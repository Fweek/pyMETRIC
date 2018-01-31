#--------------------------------
# Name:         et_numpy.py
# Purpose:      NumPy ET functions
# Python:       2.7
#--------------------------------

import logging
import math

import numpy as np

import et_common


def cos_theta_spatial_func(time, doy, dr, lon, lat):
    """"""
    sc = et_common.seasonal_correction_func(doy)
    delta = et_common.delta_func(doy)
    omega = et_common.omega_func(et_common.solar_time_rad_func(time, lon, sc))
    cos_theta = ((math.sin(delta) * np.sin(lat)) +
                 (math.cos(delta) * np.cos(lat) * np.cos(omega)))
    return cos_theta


def cos_theta_mountain_func(time, doy, dr, lon, lat, slope, aspect):
    """"""
    sc = et_common.seasonal_correction_func(doy)
    delta = et_common.delta_func(doy)
    omega = et_common.omega_func(et_common.solar_time_rad_func(time, lon, sc))
    sin_omega = np.sin(omega)
    cos_omega = np.cos(omega)
    del omega
    sin_slope = np.sin(slope)
    cos_slope = np.cos(slope)
    # Aspect is 0 as north, function is expecting 0 as south
    sin_aspect = np.sin(aspect - math.pi)
    cos_aspect = np.cos(aspect - math.pi)
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    cos_theta_unadjust_array = (
        (math.sin(delta) * sin_lat * cos_slope) -
        (math.sin(delta) * cos_lat * sin_slope * cos_aspect) +
        (math.cos(delta) * cos_lat * cos_slope * cos_omega) +
        (math.cos(delta) * sin_lat * sin_slope * cos_aspect * cos_omega) +
        (math.cos(delta) * sin_aspect * sin_slope * sin_omega))
    del sin_lat, cos_lat, sin_slope
    del sin_aspect, cos_aspect, sin_omega, cos_omega
    cos_theta_array = np.maximum(
        (cos_theta_unadjust_array / cos_slope), 0.1)
    del cos_slope
    return cos_theta_array

# def cos_theta_mountain_func(time, doy, dr, lon, lat, slope, aspect):
#     """"""
#     cos_theta_array = 0
#     # Term 1 (sin(Delta)*sin(Latitude)*cos(Slope))
#     temp_array = math.sin(delta)
#     temp_array *= np.sin(lat)
#     temp_array *= np.cos(slope)
#     temp_array *= np.cos(aspect)
#     temp_array *= np.cos(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Term 2 (-sin(Delta)*cos(Latitude)*sin(Slope)*cos(Aspect))
#     temp_array = math.sin(delta)
#     temp_array *= np.cos(lat)
#     temp_array *= np.sin(slope)
#     temp_array *= np.cos(aspect
#     cos_theta_array -= temp_array
#     del temp_array
#     # Term 3 (+cos(Delta)*cos(Latitude)*cos(Slope)*cos(Omega))
#     temp_array = math.cos(delta)
#     temp_array *= np.cos(lat)
#     temp_array *= np.cos(slope)
#     temp_array *= np.cos(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Term 4 (+cos(Delta)*sin(Latitude)*sin(Slope)*cos(Aspect)*cos(Omega))
#     temp_array = math.cos(delta)
#     temp_array *= np.sin(lat)
#     temp_array *= np.sin(slope)
#     temp_array *= np.cos(aspect)
#     temp_array *= np.cos(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Term 5 (+cos(Delta)*sin(Slope)*sin(Aspect)*sin(Omega))
#     temp_array = math.cos(delta)
#     temp_array *= np.sin(slope)
#     temp_array *= np.sin(aspect)
#     temp_array *= np.sin(omega)
#     cos_theta_array += temp_array
#     del temp_array
#     # Adjust
#     cos_theta_array /= np.cos(slope)
#     cos_theta_array = np.maximum(
#         cos_theta_array, 0.1, dtype=np.float32)
#     #  ((sin(Delta)*sin(Latitude)*cos(Slope))
#     #  -(sin(Delta)*cos(Latitude)*sin(Slope)*cos(Aspect))
#     #  +(cos(Delta)*cos(Latitude)*cos(Slope)*cos(Omega))
#     #  +(cos(Delta)*sin(Latitude)*sin(Slope)*cos(Aspect)*cos(Omega))
#     #  +(cos(Delta)*sin(Slope)*sin(Aspect)*sin(Omega)))
#     # cos_theta_array = (
#     #     (sin_delta * sin_lat * cos_slope) -
#     #     (sin_delta * cos_lat * sin_slope * cos_aspect) +
#     #     (cos_delta * cos_lat * cos_slope * cos_omega) +
#     #     (cos_delta * sin_lat * sin_slope * cos_aspect * cos_omega) +
#     #     (cos_delta * sin_slope * sin_aspect * sin_omega))
#     # del sin_lat, cos_lat, sin_slope
#     # del sin_aspect, cos_aspect, sin_omega, cos_omega
#     # cos_theta_array /= cos_slope
#     # del cos_slope
#     # cos_theta_array = np.maximum(
#     #     cos_theta_array, 0.1, dtype=np.float32)
#     return cos_theta_array

def l457_refl_toa_func(dn, cos_theta, dr, esun,
                       lmin, lmax, qcalmin, qcalmax,
                       # refl_mult, refl_add,
                       band_toa_sur_mask):
    """Calculate Landsat 4, 5, or 7 TOA reflectance for all bands"""
    refl_toa = np.copy(dn).astype(np.float64)
    refl_toa -= qcalmin
    refl_toa *= ((lmax - lmin) / (qcalmax - qcalmin))
    refl_toa += lmin
    # refl_toa *= refl_mult
    # refl_toa += refl_add
    refl_toa /= esun
    refl_toa[:, :, band_toa_sur_mask] /= cos_theta[
        :, :, np.newaxis].repeat(band_toa_sur_mask.size, 2)
    refl_toa[:, :, band_toa_sur_mask] *= (math.pi / dr)
    # Don't clip thermal band since it is not scaled from 0-1
    refl_toa[:, :, band_toa_sur_mask] = np.clip(
        refl_toa[:, :, band_toa_sur_mask], 0.0001, 1)
    return refl_toa.astype(np.float32)


def l457_refl_toa_band_func(dn, cos_theta, dr, esun,
                            lmin, lmax, qcalmin, qcalmax):
                            # refl_mult, refl_add):
    """Landsat 4, 5, or 7 DN -> TOA reflectance (single band)"""
    refl_toa = np.copy(dn).astype(np.float64)
    refl_toa -= qcalmin
    refl_toa *= ((lmax - lmin) / (qcalmax - qcalmin))
    refl_toa += lmin
    # refl_toa *= refl_mult
    # refl_toa += refl_add
    refl_toa /= cos_theta
    refl_toa *= (math.pi / (dr * esun))
    np.clip(refl_toa, 0.0001, 1, out=refl_toa)
    return refl_toa.astype(np.float32)


def l457_ts_bt_band_func(dn, lmin, lmax, qcalmin, qcalmax,
                         # rad_mult, rad_add,
                         k1, k2):
    """Landsat 4, 5, or 7 DN -> brightness temperature (single band)"""
    ts_bt = np.copy(dn).astype(np.float64)
    ts_bt -= qcalmin
    ts_bt *= ((lmax - lmin) / (qcalmax - qcalmin))
    ts_bt += lmin
    # ts_bt *= rad_mult
    # ts_bt += rad_add
    return ts_bt_func(ts_bt, k1, k2).astype(np.float32)


def l8_refl_toa_band_func(dn, cos_theta, refl_mult, refl_add):
    """Landsat 8 DN -> TOA reflectance (single band)"""
    refl_toa = np.copy(dn).astype(np.float64)
    refl_toa *= refl_mult
    refl_toa += refl_add
    refl_toa /= cos_theta
    np.clip(refl_toa, 0.0001, 1, out=refl_toa)
    return refl_toa


def l8_ts_bt_band_func(dn, rad_mult, rad_add, k1, k2):
    """Landsat 8 -> brightness temperature (single band)"""
    ts_bt = np.copy(dn).astype(np.float64)
    ts_bt *= rad_mult
    ts_bt += rad_add
    return ts_bt_func(ts_bt, k1, k2).astype(np.float32)


def bqa_fmask_func(qa):
    """Construct Fmask array from Landsat Collection 1 TOA QA array

    https://landsat.usgs.gov/collectionqualityband
    https://tools.earthengine.google.com/356a3580096cca315785d0859459abbd

    Confidence values
    00 = "Not Determined" = Algorithm did not determine the status of this condition
    01 = "No" = Algorithm has low to no confidence that this condition exists
        (0-33 percent confidence)
    10 = "Maybe" = Algorithm has medium confidence that this condition exists
        (34-66 percent confidence)
    11 = "Yes" = Algorithm has high confidence that this condition exists
        (67-100 percent confidence
    """

    # Extracting cloud masks from BQA using np.right_shift() and np.bitwise_and()
    # Cloud (med & high confidence), then snow, then shadow, then fill
    # Low confidence clouds tend to be the FMask buffer
    fill_mask = np.bitwise_and(np.right_shift(qa, 0), 1)
    cloud_mask = np.bitwise_and(np.right_shift(qa, 4), 1)  # cloud bit
    cloud_mask &= np.bitwise_and(np.right_shift(qa, 5), 3) >= 2  # cloud conf.
    cloud_mask |= np.bitwise_and(np.right_shift(qa, 11), 3) >= 3  # cirrus
    shadow_mask = np.bitwise_and(np.right_shift(qa, 7), 3) >= 3
    snow_mask = np.bitwise_and(np.right_shift(qa, 9), 3) >= 3

    fmask = (fill_mask != 1).astype(np.uint8)
    fmask[shadow_mask] = 2
    fmask[snow_mask] = 3
    fmask[cloud_mask] = 4

    return fmask


# def refl_toa_comp_func(dn, cos_theta, dr, esun, lmin, lmax, qcalmin, qcalmax,
#                        band_toa_sur_mask):
#     """"""
#     if np.all(np.isnan(dn)):
#         return dn
#     refl_toa = np.copy(dn).astype(np.float32)
#     refl_toa -= qcalmin
#     refl_toa *= ((lmax - lmin) / (qcalmax - qcalmin))
#     refl_toa += lmin
#     refl_toa /= esun
#     refl_toa[:,:,band_toa_sur_mask] /= cos_theta[
#         :,:,np.newaxis].repeat(band_toa_sur_mask.size,2)
#     refl_toa[:,:,band_toa_sur_mask] *= (math.pi / dr)
#     # Don't clip thermal band since it is not scaled from 0-1
#     refl_toa[:,:,band_toa_sur_mask] = np.clip(
#         refl_toa[:,:,band_toa_sur_mask], 0.0001, 1)
#     return refl_toa
# def refl_toa_band_func(dn, cos_theta, dr, esun, lmin, lmax, qcalmin, qcalmax,
#                        thermal_band_flag=False):
#     """"""
#     if np.all(np.isnan(dn)):
#         return dn
#     refl_toa = np.copy(dn).astype(np.float32)
#     refl_toa -= qcalmin
#     refl_toa *= ((lmax - lmin) / (qcalmax - qcalmin))
#     refl_toa += lmin
#     refl_toa /= esun
#     if not thermal_band_flag:
#         refl_toa /= cos_theta
#         refl_toa *= (math.pi / dr)
#         # Don't clip thermal band since it is not scaled from 0-1
#         np.clip(refl_toa, 0.0001, 1, out=refl_toa)
#     return refl_toa


def tau_broadband_func(pair, w, cos_theta, kt=1):
    """Broadband transmittance"""
    def tau_direct_func(pair, w, cos_theta, kt=1):
        """"""
        t1 = np.copy(pair).astype(np.float64)
        t1 /= kt
        t1 *= -0.00146
        t1 /= cos_theta
        t2 = np.copy(w).astype(np.float64)
        t2 /= cos_theta
        np.power(t2, 0.4, out=t2)
        t2 *= 0.075
        t1 -= t2
        del t2
        np.exp(t1, out=t1)
        t1 *= 0.98
        return t1
        # return 0.98 * np.exp((-0.00146 * pair / kt) - (0.075 * np.power(w, 0.4)))

    def tau_diffuse_func(tau_direct):
        """"""
        tau = np.copy(tau_direct).astype(np.float64)
        tau *= -0.36
        tau += 0.35
        return tau
        # Model differs from formulas in manual
        # Eqn is not aplied, per Rick Allen it is not needed
        # return np.where(tau_direct_array >= 0.15),
        #                 (0.35-0.36*tau_direct_array),
        #                 (0.18-0.82*tau_direct_array))
    tau_broadband = tau_direct_func(pair, w, cos_theta, kt)
    tau_broadband += tau_diffuse_func(tau_broadband)
    return tau_broadband.astype(np.float32)


def tau_narrowband_func(pair, w, cos_theta, kt, c1, c2, c3, c4, c5):
    """Narrowband transmittance"""
    # Incoming narrowband transmittance
    # Outgoing narrowband transmittance
    # IN  (C1*exp(((C2*Pair)/(Kt*cos_theta))-((C3*W+C4)/cos_theta))+C5)
    # OUT (C1*exp(((C2*Pair)/(Kt*1.0))-((C3*W+C4)/1.0))+C5)
    t1 = np.copy(pair).astype(np.float64)
    t1 /= kt
    t1 *= c2
    t2 = np.copy(w)
    t2 *= c3
    t2 += c4
    t1 -= t2
    del t2
    t1 /= cos_theta
    np.exp(t1, out=t1)
    t1 *= c1
    t1 += c5
    return t1.astype(np.float32)
    # return (c1 * np.exp((c2 * pair / kt) - (c3 * w + c4)) + c5)


def refl_sur_tasumi_func(refl_toa, pair, w, cos_theta, kt,
                         c1, c2, c3, c4, c5, cb, band_cnt):
    """Tasumi at-surface reflectance"""
    if np.all(np.isnan(refl_toa)):
        return refl_toa
    # Reshape arrays to match the surface reflectance arrays
    pair_mod = pair[:, :, np.newaxis].repeat(band_cnt, 2).astype(np.float64)
    w_mod = w[:, :, np.newaxis].repeat(band_cnt, 2).astype(np.float64)
    cos_theta_mod = cos_theta[:, :, np.newaxis].repeat(band_cnt, 2).astype(np.float64)
    # (((refl_toa-Cb*(1.0-tau_in))/(tau_in*tau_out))>0.0)*
    #  ((refl_toa-Cb*(1.0-tau_in))/(tau_in*tau_out))
    tau_in = tau_narrowband_func(
        pair_mod, w_mod, cos_theta_mod, kt, c1, c2, c3, c4, c5)
    tau_out = tau_narrowband_func(
        pair_mod, w_mod, 1, kt, c1, c2, c3, c4, c5)
    del cos_theta_mod, pair_mod, w_mod
    refl_sur = np.copy(tau_in)
    refl_sur *= -1
    refl_sur += 1
    refl_sur *= -cb
    refl_sur += refl_toa
    refl_sur /= tau_in
    refl_sur /= tau_out
    np.clip(refl_sur, 0.0001, 1, out=refl_sur)
    return refl_sur.astype(np.float32)
    # return (refl_toa - cb * (1 - tau_in)) / (tau_in * tau_out)


def albedo_sur_func(refl_sur, wb):
    """Tasumi at-surface albedo"""
    return np.sum(refl_sur * wb, axis=2)


def albedo_ts_corrected_func(albedo_sur, ndvi_toa, ts_array, hot_px_temp,
                             cold_px_temp, k_value, dense_veg_min_albedo):
    """Updated Ts based on METRIC Manual - Eqn. 16-1"""
    masked = (albedo_sur < dense_veg_min_albedo) & (ndvi_toa > 0.45)
    ts_array[masked] = ts_array[masked] + ((dense_veg_min_albedo  - albedo_sur[masked]) * k_value * ((hot_px_temp-cold_px_temp) * (0.95)))

    """Updated albedo based on METRIC Manual - Eqn. 16-1"""
    masked = (albedo_sur < dense_veg_min_albedo) & (ndvi_toa > 0.6)
    albedo_sur[masked] = dense_veg_min_albedo
    masked = (albedo_sur < dense_veg_min_albedo) & ((ndvi_toa > 0.4) & (ndvi_toa < 0.6))
    albedo_sur[masked] = dense_veg_min_albedo - (dense_veg_min_albedo - albedo_sur[masked]) * (1 - ((ndvi_toa[masked] - 0.4) / (0.6 - 0.4)))
    return ts_array, albedo_sur


# Vegetation Indices
def ndi_func(a, b, l=0.):
    """Normalized difference index function

    Can be used to calculate SAVI by setting l != 0
    """
    ndi = ((1. + l) * (a - b) / (l + a + b))
    # Manually set output value when a and b are zero
    # ndi[((l+a+b) != 0)] = 0
    return ndi


def savi_lai_func(savi):
    """"""
    return np.clip((11. * np.power(savi, 3)), 0, 6)


def ndvi_lai_func(ndvi):
    """"""
    return np.clip((7. * np.power(ndvi, 3)), 0, 6)


def ratio_func(a, b):
    """"""
    return a / b


def evi_func(b1, b3, b4):
    """"""
    return ((2.5 * (b4 - b3)) / (b4 + 6 * b3 - 7.5 * b1 + 1))


def tc_bright_func(refl_toa, image_type='TOA'):
    """Tasseled cap brightness"""
    if image_type == 'SUR':
        tc_bright = np.array([
            0.2043, 0.4158, 0.5524, 0.5741,
            0.3124, 0, 0.2303]).astype(np.float32)
    elif image_type == 'TOA':
        tc_bright = np.array([
            0.3561, 0.3972, 0.3904, 0.6966,
            0.2286, 0, 0.1596]).astype(np.float32)
    return np.sum(refl_toa * tc_bright, axis=2)


def tc_green_func(refl_toa, image_type='TOA'):
    """Tasseled cap greenness"""
    if image_type == 'SUR':
        tc_green = np.array([
            -0.1063, -0.2819, -0.4934, 0.7940,
            -0.0002, 0, -0.1446]).astype(np.float32)
    elif image_type == 'TOA':
        tc_green = np.array([
            -0.3344, -0.3544, -0.4556, 0.6966,
            -0.0242, 0, -0.2630]).astype(np.float32)
    return np.sum(refl_toa * tc_green, axis=2)


def tc_wet_func(refl_toa, image_type='TOA'):
    """Tasseled cap wetness"""
    if image_type == 'SUR':
        tc_wet = np.array([
            0.0315, 0.2021, 0.3102, 0.1594,
            -0.6806, 0, -0.6109]).astype(np.float32)
    elif image_type == 'TOA':
        tc_wet = np.array([
            0.2626, 0.2141, 0.0926, 0.06564,
            -0.7629, 0, -0.5388]).astype(np.float32)
    return np.sum(refl_toa * tc_wet, axis=2)


def etstar_func(evi, etstar_type='mean'):
    """ET star from Jordan Beamer's thesis"""
    c_dict = dict()
    c_dict['mean'] = np.array([-0.1955, 2.9042, -1.5916]).astype(np.float32)
    c_dict['lpi'] = np.array([-0.2871, 2.9192, -1.6263]).astype(np.float32)
    c_dict['upi'] = np.array([-0.1039, 2.8893, -1.5569]).astype(np.float32)
    c_dict['lci'] = np.array([-0.2142, 2.9175, -1.6554]).astype(np.float32)
    c_dict['uci'] = np.array([-0.1768, 2.891, -1.5278]).astype(np.float32)
    try:
        c = c_dict[etstar_type]
    except KeyError:
        raise SystemExit()
    # ET* calculation
    etstar = np.copy(evi)
    etstar *= c[2]
    etstar += c[1]
    etstar *= evi
    etstar += c[0]
    np.maximum(etstar, 0., out=etstar)
    return etstar


def etstar_etg_func(etstar, eto, ppt):
    """"""
    return (np.copy(etstar) * (eto - ppt))


def etstar_et_func(etstar, eto, ppt):
    """"""
    return (np.copy(etstar) * (eto - ppt)) + ppt


def em_nb_func(lai, water_index, water_threshold=0):
    """Narrowband emissivity"""
    em_nb = np.copy(lai).astype(np.float32)
    em_nb /= 300.
    em_nb += 0.97
    em_nb[(water_index > water_threshold) & (lai > 3)] = 0.98
    em_nb[water_index < water_threshold] = 0.99
    return em_nb
    # return np.where(
    #     ndvi_filter,
    #     np.where(lai_filter, (0.97 + (lai_toa / 300.)), 0.98),
    #     np.where(albedo_filter, 0.99, 0.99))


def em_0_func(lai, water_index, water_threshold=0):
    """Broadband emissivity"""
    em_0 = np.copy(lai).astype(np.float32)
    em_0 /= 100.
    em_0 += 0.95
    em_0[(water_index > water_threshold) & (lai > 3)] = 0.98
    em_0[water_index <= water_threshold] = 0.985
    return em_0
    # return np.where(
    #     ndvi_filter,
    #     np.where(lai_filter, (0.95 + (lai / 100.)), 0.98), 0.985)


def rc_func(thermal_rad_toa, em_nb, rp, tnb, rsky):
    """Corrected Radiance"""
    rc = np.array(thermal_rad_toa, copy=True, ndmin=1).astype(np.float64)
    # rc = np.copy(thermal_rad_toa).astype(np.float32)
    rc -= rp
    rc /= tnb
    rc -= rsky
    rc += (em_nb * rsky)
    return rc.astype(np.float32)
    # return ((thermal_rad_toa - rp) / tnb) - ((1. - em_nb) * rsky)


def ts_func(em_nb, rc, k1, k2):
    """Surface Temperature"""
    ts = np.copy(em_nb).astype(np.float64)
    ts *= k1
    ts /= rc
    ts += 1.
    np.log(ts, out=ts)
    np.reciprocal(ts, out=ts)
    ts *= k2
    return ts.astype(np.float32)
    # return (k['2'] / np.log(((em_nb * k['1']) / rc) + 1.))


def ts_bt_func(thermal_rad_toa, k1, k2):
    """Calculate brightness temperature from thermal radiance"""
    ts_bt = np.copy(thermal_rad_toa).astype(np.float64)
    ts_bt[ts_bt <= 0] = np.nan
    np.reciprocal(ts_bt, out=ts_bt)
    ts_bt *= k1
    ts_bt += 1.
    np.log(ts_bt, out=ts_bt)
    np.reciprocal(ts_bt, out=ts_bt)
    ts_bt *= k2
    return ts_bt.astype(np.float32)
    # return (k['2'] / np.log(((k['1']) / rc) + 1.))


def thermal_rad_func(ts_bt, k1, k2):
    """Back calculate thermal radiance from brightness temperature"""
    thermal_rad = np.copy(ts_bt).astype(np.float64)
    np.reciprocal(thermal_rad, out=thermal_rad)
    thermal_rad *= k2
    np.exp(thermal_rad, out=thermal_rad)
    thermal_rad -= 1.
    np.reciprocal(thermal_rad, out=thermal_rad)
    thermal_rad *= k1
    return thermal_rad.astype(np.float32)
    # return k1 / (np.exp(k2 / ts_bt) - 1.)


def ts_lapsed_func(ts, elevation, datum, lapse_elev, lapse_flat, lapse_mtn):
    """Lapse surface temperature based on elevation"""
    ts_a = np.copy(elevation).astype(np.float64)
    ts_a -= datum
    ts_a *= (lapse_flat * -0.001)
    ts_a += ts
    ts_b = np.copy(elevation).astype(np.float64)
    ts_b -= lapse_elev
    ts_b *= (lapse_mtn * -0.001)
    ts_b -= ((lapse_elev - datum) * lapse_flat * 0.001)
    ts_b += ts
    return np.where(elevation < lapse_elev, ts_a, ts_b).astype(np.float32)
    # return np.where(elevation < lapse_elev,
    #                 (ts - ((elevation - datum) * lapse_flat * 0.001)),
    #                 (ts - ((lapse_elev - datum) * lapse_flat * 0.001) -
    #                  ((elevation - lapse_elev) * lapse_mtn * 0.001)))


def ts_delapsed_func(ts, elevation, datum, lapse_elev, lapse_flat, lapse_mtn):
    """Delapse surface temperature based on elevation"""
    ts_a = np.copy(elevation).astype(np.float64)
    ts_a -= datum
    ts_a *= (lapse_flat * 0.001)
    ts_a += ts
    ts_b = np.copy(elevation).astype(np.float64)
    ts_b -= lapse_elev
    ts_b *= (lapse_mtn * 0.001)
    ts_b += ((lapse_elev - datum) * lapse_flat * 0.001)
    ts_b += ts
    return np.where(elevation < lapse_elev, ts_a, ts_b).astype(np.float32)
    # return np.where(elevation < lapse_elev,
    #                 (ts + ((elevation - datum) * lapse_flat * 0.001)),
    #                 (ts + ((lapse_elev - datum) * lapse_flat * 0.001) +
    #                  ((elevation - lapse_elev) * lapse_mtn * 0.001)))


def ts_dem_dry_func(ts_dem_cold, ts_dem_hot, kc_cold, kc_hot):
    """"""
    return (ts_dem_hot + (kc_hot * ((ts_dem_hot - ts_dem_cold) /
                                    (kc_cold - kc_hot))))


def rl_in_func(tau, ts_cold_lapsed, rl_in_coef1, rl_in_coef2):
    """Incoming Longwave Radiation"""
    rl_in = np.copy(tau).astype(np.float64)
    np.log(rl_in, out=rl_in)
    np.negative(rl_in, out=rl_in)
    np.power(rl_in, rl_in_coef2, out=rl_in)
    rl_in *= (rl_in_coef1 * 5.67E-8)
    rl_in *= np.power(ts_cold_lapsed, 4)
    return rl_in.astype(np.float32)


def rl_out_func(rl_in, ts, em_0):
    """Outgoing Longwave Radiation (Emitted + Reflected)"""
    rl_out = np.copy(ts).astype(np.float64)
    np.power(rl_out, 4, out=rl_out)
    rl_out *= em_0
    rl_out *= 5.67E-8
    rl_out += rl_in
    rl_out -= em_0 * rl_in
    return rl_out.astype(np.float32)


def rs_in_func(cos_theta, tau, dr):
    """Incoming Shortwave Radiation"""
    rs_in = np.copy(cos_theta).astype(np.float64)
    rs_in *= tau
    rs_in *= (1367. * dr)
    return rs_in.astype(np.float32)


def rs_out_func(rs_in, albedo_sur):
    """Outgoing Shortwave Radiation"""
    rs_out = np.copy(rs_in).astype(np.float64)
    rs_out *= albedo_sur
    return rs_out.astype(np.float32)


def rn_func(rs_in, rs_out, rl_in, rl_out):
    """Net Radiation"""
    rn = np.copy(rs_in)
    rn -= rs_out
    rn += rl_in
    rn -= rl_out
    return rn


def rn_24_slob_func(lat, ts, ts_cold, ts_hot, albedo_sur, rs_in, doy):
    """Daily Net Radiation - Slob method"""
    ra = et_common.ra_daily_func(lat, doy)
    rnl_cold_24 = 140 * (rs_in / ra)
    rnl_hot_24 = 110 * (rs_in / ra)
    del ra
    rnl_24_pixel = (
        ((ts - ts_cold) / (ts_hot - ts_cold)) *
        (rnl_hot_24 - rnl_cold_24) + rnl_cold_24)
    del rnl_cold_24, rnl_hot_24
    rn_24_array = 1 - albedo_sur
    rn_24_array *= rs_in
    rn_24_array -= rnl_24_pixel
    return rn_24_array


def g_ag_func(lai_toa, ts, rn, coef1=1.8, coef2=0.084):
    """METRIC Ag G function"""
    a = np.copy(lai_toa).astype(np.float64)
    a *= -0.521
    np.exp(a, out=a)
    a *= 0.18
    a += 0.05
    a *= rn
    b = ts - 273.16
    b *= coef1
    b /= rn
    b += coef2
    b *= rn
    return np.where(lai_toa >= 0.5, a, b).astype(np.float32)
    # return np.where(
    #     lai_toa >= 0.5,
    #     ((0.05 + (0.18 * np.exp(-0.521 * lai_toa))) * rn),
    #     (coef1 * (ts - 273.16) + (coef2 * rn)))


def g_wim_func(ts, albedo_sur, ndvi_toa):
    """Wim Bastiaanssen's G function"""
    g_wim = np.copy(ndvi_toa).astype(np.float64)
    np.power(g_wim, 4, out=g_wim)
    g_wim *= -0.98
    g_wim += 1
    g_wim *= ts
    g_wim *= (albedo_sur * 0.0074 + 0.0038)
    return g_wim


def g_water_func(rn, acq_doy):
    """Adjust water heat storage based on day of year"""
    return rn * (-1.25 + (0.0217 * acq_doy) - (5.87E-5 * (acq_doy ** 2)))


def excess_res_func(u3):
    """

    Excess res. needs to be recalculated if additional wind is applied
    """
    if u3 < 2.6:
        return 0.0
    elif u3 > 15.0:
        return 2.0
    else:
        return (
            (0.01303 * u3 ** 3) - (0.43508 * u3 ** 2) +
            (4.27477 * u3) - 8.283524)


def perrier_zom_func(lai_toa):
    """
    Perrier 1982 method, see eg Hydrology Handbook p 197.
    Minimum zom is 0.005 m equal to bare soil. Dec 28 09, JK
    The use of the function is applicable for tall vegetation (forests)
    The canopy distribution coefficient, a, is assumed to be a=0.6,
      i.e. slightly top heavy canopy
    The vegetation height is estimated as h=2.5LAI (LAImax=6 -> 2.5*6=15 m),
      compared to h=0.15LAI for agric crops
    """
    perrier = -1.2 * lai_toa
    perrier /= 2.
    np.exp(perrier, out=perrier)
    perrier = ((1 - perrier) * perrier) * (2.5 * lai_toa)
    return np.maximum(perrier, 0.005, dtype=np.float32)


def zom_nlcd_func(nlcd, minimum_lai, perrier_zom):
    """Generate NLCD landuse dependent Zom (roughness) values

    Updated zom values from METRIC Appendex 2.0.8 Table A2-1
    Function needs to not be dependent on the mask
    So that it can be used on calibration points
    """
    # zom = np.ones(env.mask_array.shape).astype(np.float32)
    # zom[env.mask_array == 0] = np.nan
    zom = np.ones(minimum_lai.shape).astype(np.float32)
    zom[np.isnan(minimum_lai)] = np.nan
    zom *= 0.005
    zom[nlcd == 11] = 0.0005
    zom[nlcd == 12] = 0.005
    zom[nlcd == 21] = 0.05
    zom[nlcd == 22] = 0.08
    zom[nlcd == 23] = 0.1
    zom[nlcd == 24] = 0.2
    zom[(nlcd == 31) | (nlcd == 32)] = 0.005
    perrier_filter = ((nlcd >= 41) & (nlcd <= 43))
    zom[perrier_filter] = perrier_zom[perrier_filter]
    del perrier_filter
    zom[(nlcd == 51) | (nlcd == 52)] = 0.2
    zom[nlcd == 71] = 0.05
    zom[nlcd == 72] = 0.03
    min_lai_filter = ((nlcd == 81) | (nlcd == 82))
    zom[min_lai_filter] = minimum_lai[min_lai_filter] * 0.018
    del min_lai_filter
    zom[nlcd == 90] = 0.4
    zom[nlcd == 94] = 0.2
    zom[nlcd == 95] = 0.1
    return zom


def zom_cdl_func(cdl, minimum_lai, perrier_zom):
    """Generate CDL landuse dependent Zom (roughness) values

    Function needs to not be dependent on the mask
    So that it can be used on calibration points
    """
    # zom = np.ones(env.mask_array.shape).astype(np.float32)
    # zom[env.mask_array == 0] = np.nan
    zom = np.ones(minimum_lai.shape).astype(np.float32)
    zom[np.isnan(minimum_lai)] = np.nan
    zom *= 0.005
    return zom


def le_calibration_func(etr, kc, ts):
    """LE at the calibration points

    1000000/3600 was simplified to 2500/9 in following eqn
    """
    return etr * kc * (2.501 - 2.361E-3 * (ts - 273)) * 2500 / 9


# Following eqns are float specific, separate from eqns below
def dt_calibration_func(h, rah, density):
    """"""
    return (h * rah) / (density * 1004.)


def l_calibration_func(h, density, u_star, ts):
    """"""
    return np.where(
        h != 0,
        ((-1004. * density * (u_star ** 3.0) * ts) / (0.41 * 9.81 * h)),
        -1000)


def h_func(air_density, dt, rah):
    """Sensible Heat Flux [W/m^2]"""
    h = np.array(air_density, copy=True, ndmin=1)
    h *= 1004.
    h *= dt
    h /= rah
    return h


def u_star_func(u3, z3, zom, psi_z3, wind_coef=1):
    """"""
    u_star = np.array(zom, copy=True, ndmin=1)
    np.reciprocal(u_star, out=u_star)
    u_star *= z3
    oldsettings = np.geterr()
    np.seterr(invalid='ignore')
    np.log(u_star, out=u_star)
    np.seterr(invalid=oldsettings['invalid'])
    u_star -= psi_z3
    np.reciprocal(u_star, out=u_star)
    u_star *= (u3 * wind_coef * 0.41)
    # u_star = ((u3 * wind_coef * 0.41) / (np.log(z3 / zom) - psi_z3))
    return u_star


def rah_func(z_flt_dict, psi_z2, psi_z1, u_star, excess_res=0):
    """"""
    rah = np.array(psi_z1, copy=True, ndmin=1)
    rah -= psi_z2
    rah += math.log(z_flt_dict[2] / z_flt_dict[1])
    rah /= 0.41
    rah /= u_star
    rah += excess_res
    return rah
    # return ((math.log(z_flt_dict[2] / z_flt_dict[1]) -
    #          psi_z2 + psi_z1) / (0.41 * u_star)) + excess_res


def density_func(elev, ts, dt):
    """"""
    density = np.array(elev, copy=True, ndmin=1).astype(np.float64)
    density *= -0.0065
    density += 293.15
    density /= 293.15
    np.power(density, 5.26, out=density)
    density *= ((1000 * 101.3) / (1.01 * 287))
    density /= (ts - dt)
    return density.astype(np.float32)
    # return (1000. * 101.3 *
    #         np.power(((293.15 - 0.0065 * elev) / 293.15), 5.26) /
    #         (1.01 * (ts - dt) * 287))


def x_func(l, z):
    """"""
    x = np.array(l, copy=True, ndmin=1)
    l_mask = (x > 0)
    np.reciprocal(x, out=x)
    x *= (-16 * z)
    x += 1
    np.power(x, 0.25, out=x)
    x[l_mask] = 0
    del l_mask
    return x
    # return np.where(l < 0, np.power((1 - 16 * z / l), 0.25), 0)


def psi_func(l, z_index, z):
    """"""
    # Begin calculation of Psi unstable
    x = x_func(l, z)
    psi = np.array(x, copy=True, ndmin=1)
    np.power(x, 2, out=psi)
    psi += 1
    psi /= 2.
    oldsettings = np.geterr()
    np.seterr(invalid='ignore')
    np.log(psi, out=psi)
    np.seterr(invalid=oldsettings['invalid'])
    # Adjust Psi unstable calc based on height
    if z_index == 3:
        psi_temp = np.copy(x)
        psi_temp += 1
        psi_temp /= 2.
        oldsettings = np.geterr()
        np.seterr(invalid='ignore')
        np.log(psi_temp, out=psi_temp)
        np.seterr(invalid=oldsettings['invalid'])
        psi_temp *= 2.
        psi += psi_temp
        del psi_temp
        psi_temp = np.copy(x)
        np.arctan(x, out=psi_temp)
        psi_temp *= 2.
        psi -= psi_temp
        del psi_temp
        psi += (0.5 * math.pi)
        # DEADBEEF
        # return np.where(l > 0, (-5 * 2 / l),
        #     ((2 * np.log((1 + x) / 2.)) +
        #      np.log((1 + np.power(x, 2)) / 2.) -
        #      (2 * np.arctan(x)) + (0.5 * math.pi)))
    else:
        psi *= 2.
        # return np.where(l > 0, (-5 * z / l),
        #     (2 * np.log((1 + np.power(x, 2)) / 2.)))
    del x
    # Calculate Psi stable for all pixels
    psi_stable = np.array(l, copy=True, ndmin=1)
    np.reciprocal(psi_stable, out=psi_stable)
    if z_index == 3:
        psi_stable *= (-5 * 2)
    else:
        psi_stable *= (-5 * z)
    # Only keep Psi stable for pixels with l > 0
    l_mask = np.array(l, copy=True, ndmin=1) > 0
    psi[l_mask] = psi_stable[l_mask]
    return psi
    # return np.where((l > 0), psi_stable, psi_unstable)


# Following eqns are array specific, separate from eqns above
def dt_func(dt_adjust_flag, ts_dem, a, b, ts_threshold, dt_slope_factor=4):
    """"""
    dt = np.copy(ts_dem)
    dt *= a
    dt += b
    if dt_adjust_flag:
        dt_adjust = ts_dem - ts_threshold
        dt_adjust *= (a / dt_slope_factor)
        dt_adjust += (a * ts_threshold + b)
        np.where(ts_dem < ts_threshold, dt_adjust, dt)
    return dt


def l_func(dt, u_star, ts, rah):
    """"""
    # Change zeros to -1000 to avoid divide by zero
    dt[dt == 0] = -1000
    l = np.power(u_star, 3)
    l *= ts
    l *= rah
    l /= -(0.41 * 9.81)
    l /= dt
    return l
    # dt_mod = np.where((np.absolute(dt)==0.), -1000., dt)
    # return -(np.power(u_star, 3) * ts * rah) / (0.41 * 9.81 * dt_mod)


def le_func(rn, g, h):
    """Latent Heat Flux [W/m^2]"""
    le = np.copy(rn)
    le -= g
    le -= h
    return le
    # le = rn - g - h


def ef_func(le, rn, g):
    """Evaporative fraction instantaneous"""
    ef = np.copy(le)
    ef /= (rn - g)
    # ef = np.copy(rn)
    # ef -= g
    # np.reciprocal(ef, out=ef)
    # ef *= le
    return ef


def heat_vaporization_func(ts):
    """ET instantaneous [mm/hr]"""
    heat_vaporization = np.copy(ts).astype(np.float64)
    heat_vaporization -= 273.15
    heat_vaporization *= -0.00236
    heat_vaporization += 2.501
    heat_vaporization *= 1E6
    return heat_vaporization.astype(np.float32)


def et_inst_func(le, ts):
    """"""
    et_inst = np.copy(le).astype(np.float64)
    et_inst *= 3600
    et_inst /= heat_vaporization_func(ts)
    # heat_vaporization = ((2.501 - 0.00236 * (ts - 273.15)) * 1E6)
    # et_inst = 3600 * le / heat_vaporization
    return et_inst.astype(np.float32)


def etrf_func(et_inst, etr):
    """ET Reference Fraction - ETrF"""
    return et_inst / etr


def et_24_func(etr_24hr, etrf):
    """ET 24hr [mm/day]"""
    return etr_24hr * etrf


# def lat_lon_array_func(lat_sub_raster, lon_sub_raster):
#     """"""
#     cs = 0.005
#     hcs = 0.5 * cs
#     print env.mask_extent
#     input_extent = project_extent(
#         env.mask_extent, env.snap_proj, env.snap_geog_proj)
#     print input_extent
#     ## Increase extent to nearest tenth
#     input_extent.xmin = np.around(input_extent.xmin-0.05, 1)
#     input_extent.xmax = np.around(input_extent.xmax+0.05, 1)
#     input_extent.ymin = np.around(input_extent.ymin-0.05, 1)
#     input_extent.ymax = np.around(input_extent.ymax+0.05, 1)
#     print input_extent
#     input_rows, input_cols = extent_shape(input_extent, cs)
#     input_geo = extent_geo(input_extent, cs)
#     # Cell lat/lon values are measured half a cell in from extent
#     # Note that y increments go from max to min
#     lon_array, lat_array = np.meshgrid(
#         np.linspace(
#             input_extent.xmin+hcs, input_extent.xmax-hcs, input_cols),
#         np.linspace(
#             input_extent.ymax-hcs, input_extent.ymin+hcs, input_rows))
#     # lat/lon arrays are float64, could have cast as float32
#     # Instead, modified gdal_type function to return float32 for float64
#     project_array(lon_array, env.snap_geog_proj, input_geo,
#                   lon_sub_raster, env.snap_proj, gdal.GRA_Bilinear)
#     project_array(lat_array, env.snap_geog_proj, input_geo,
#                   lat_sub_raster, env.snap_proj, gdal.GRA_Bilinear)
#     return True

