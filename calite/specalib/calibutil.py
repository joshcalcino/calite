from numba import jit
import numpy as np
from scipy.interpolate import interp1d
import scipy.fftpack as fft
from scipy.signal import savgol_filter
import pyckles
import calite.specstruct as st
import matplotlib.pyplot as plt


def get_best_spectra_template(photo, filters):
    """
    Finds the best matching spectra to photo based off differences in color.

    Assumes photo and bands have the same length, and are in the same order.

    """

    spec_list = pyckles.SpectralLibrary("pickles")

    min = np.inf

    photo_diffs = np.zeros(len(filters.bands)-1)
    spec_diffs = np.zeros(len(filters.bands)-1)
    spec_photo = np.zeros(len(filters.bands))

    for i in range(len(filters.bands)-1):
        photo_diffs[i] = photo[filters.bands[i]] - photo[filters.bands[i+1]]

    for spec_name in spec_list.available_spectra:
        spec = spec_list[spec_name]
        flux = np.array(spec.data['flux'], dtype=np.float)
        wave = np.array(spec.data['wavelength'], dtype=np.float)

        var = np.zeros(len(wave))
        for i, band in enumerate(filters.bands):
            spec_photo[i], _ = computeABmag(filters[band], wave, flux, var)

        for i in range(len(filters.bands)-1):
            spec_diffs[i] = spec_photo[i] - spec_photo[i+1]

        diffs = abs(photo_diffs - spec_diffs)
        diff = np.sqrt(np.sum(diffs**2))

        if diff < min :
            min = diff
            selectedTemplate = spec

    spectra = st.SpectraLite(wavelength=selectedTemplate.data['wavelength'], flux=selectedTemplate.data['flux'])

    return spectra


def computeABmag(filter, tmp_wave, tmp_flux, tmp_var):
    """
    Computes the AB magnitude for given transmission functions and spectrum
    `f_lambda`. Returns the magnitude and variance.

    """
    # Takes and returns variance
    # trans_ : transmission function data
    # tmp_ : spectral data

    # trans/tmp not necessarily defined over the same wavelength range
    # first determine the wavelength range over which both are defined
    minV = min(filter.wavelength)
    if minV < min(tmp_wave):
        minV = min(tmp_wave)
    maxV = max(filter.wavelength)
    if maxV > max(filter.wavelength):
        maxV = max(filter.wavelength)

    # Make new vectors for the flux just using that range (assuming spectral binning)
    interp_wave, tmp_flux2, tmp_var2 = make_masked_arrays_var(tmp_wave, tmp_flux, tmp_var, minV, maxV)

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_interp = interp1d(filter.wavelength, filter.transmission)(interp_wave)

    # And now calculate the magnitude and uncertainty
    c = 2.992792e18  # Angstrom/s
    Num, Num_var, Den = get_num_var(interp_wave, tmp_flux2, tmp_var2, trans_interp)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
            magABvar = 1.17882 * Num_var / (Num ** 2)
        except FloatingPointError:
            magAB = 99.
            magABvar = 99.

    return magAB, magABvar



def compute_mag_var_fast(filter, tmp_wave, tmp_flux):
    """
    Computes the AB magnitude for given transmission functions and spectrum
    `f_lambda`. Returns the magnitude and variance.

    """
    # Takes and returns variance
    # trans_ : transmission function data
    # tmp_ : spectral data

    # trans/tmp not necessarily defined over the same wavelength range
    # first determine the wavelength range over which both are defined
    minV = min(filter.wavelength)
    if minV < min(tmp_wave):
        minV = min(tmp_wave)
    maxV = max(filter.wavelength)
    if maxV > max(filter.wavelength):
        maxV = max(filter.wavelength)

    # Make new vectors for the flux just using that range (assuming spectral binning)
    interp_wave, tmp_flux2 = make_masked_arrays(tmp_wave, tmp_flux, minV, maxV)

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_interp = interp1d(filter.wavelength, filter.transmission)(interp_wave)

    # And now calculate the magnitude and uncertainty
    c = 2.992792e18  # Angstrom/s
    Num, Num_var, Den = get_num(interp_wave, tmp_flux2, trans_interp)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
        except FloatingPointError:
            magAB = 99.

    return magAB, magABvar


def compute_mag_fast(filter, tmp_wave, tmp_flux):
    """
    Computes the AB magnitude for given transmission functions and spectrum
    `f_lambda`. Returns the magnitude and variance.

    """
    # Takes and returns variance
    # trans_ : transmission function data
    # tmp_ : spectral data

    # trans/tmp not necessarily defined over the same wavelength range
    # first determine the wavelength range over which both are defined
    minV = min(filter.wavelength)
    if minV < min(tmp_wave):
        minV = min(tmp_wave)

    maxV = max(filter.wavelength)
    if maxV > max(filter.wavelength):
        maxV = max(filter.wavelength)

    # Make new vectors for the flux just using that range (assuming spectral binning)
    interp_wave, tmp_flux2 = make_masked_arrays(tmp_wave, tmp_flux, minV, maxV)

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_interp = interp1d(filter.wavelength, filter.transmission)(interp_wave)

    # And now calculate the magnitude and uncertainty
    c = 2.992792e18  # Angstrom/s
    Num, Den = get_num(interp_wave, tmp_flux2, trans_interp)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
        except FloatingPointError:
            magAB = 99.

    return magAB


@jit(nopython=True)
def make_masked_arrays(tmp_wave, tmp_flux, minV, maxV):
    interp_wave = np.zeros(tmp_wave.shape[0])
    tmp_flux2 = np.zeros(tmp_wave.shape[0])

    j=0

    for i in range(tmp_wave.shape[0]):
        if minV < tmp_wave[i] < maxV:
            interp_wave[j] = tmp_wave[i]
            tmp_flux2[j] = tmp_flux[i]
            j+=1

    return interp_wave[:j], tmp_flux2[:j]


@jit(nopython=True)
def make_masked_arrays_var(tmp_wave, tmp_flux, tmp_var, minV, maxV):
    interp_wave = np.zeros(tmp_wave.shape[0])
    tmp_flux2 = np.zeros(tmp_wave.shape[0])
    tmp_var2 = np.zeros(tmp_wave.shape[0])

    j=0

    for i in range(tmp_wave.shape[0]):
        if minV < tmp_wave[i] < maxV:
            interp_wave[j] = tmp_wave[i]
            tmp_flux2[j] = tmp_flux[i]
            tmp_var2[j] = tmp_var[i]
            j+=1

    return interp_wave[:j], tmp_flux2[:j], tmp_var2[:j]


@jit(nopython=True)
def get_num(interp_wave, tmp_flux2, trans_interp):
    Num = np.nansum(tmp_flux2 * trans_interp * interp_wave)
    Den = np.nansum(trans_interp / interp_wave)
    return Num, Den


@jit(nopython=True)
def get_num_var(interp_wave, tmp_flux2, tmp_var2, trans_interp):
    Num = np.nansum(tmp_flux2 * trans_interp * interp_wave)
    Num_var = np.nansum(tmp_var2 * (trans_interp * interp_wave) ** 2)
    Den = np.nansum(trans_interp / interp_wave)
    return Num, Num_var, Den


def smooth_spectra_savitsky_golay(flux, window=21, order=2):
    smoothed = savgol_filter(flux, window, order)
    return smoothed


def filter_nans(*args):
    """ reutrns the arrays nan elements from first
        array removed from all arrays.
    """

    nan_mask = np.array([np.isnan(f) for f in args[0]])

    if len(args) > 1:
        return tuple(arg[~nan_mask] for arg in args)
    else:
        return args[0][~nan_mask]
