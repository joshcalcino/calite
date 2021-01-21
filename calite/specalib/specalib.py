import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import matplotlib.pyplot as plt
import sys
import os
import emcee
from chainconsumer import ChainConsumer
import calite.utils as ut
from .. import specio
from ..specstruct import SpectrumCoadd, Spectrumv18, SingleSpec


def calibSpec(obj_name, spectra, photo, outBase, filters, plotFlag, coaddFlag,
              interpFlag, redshift):
    """
    This function will determine extensions which can be calibrated,
    calculate the scale factors, warping function, and output a new fits file
    with the scaled spectra.

    Parameters
    ----------
    obj_name : str
        The name of the object.

    Returns
    -------

    """

    if plotFlag != False:
        plotName = os.path.join(plotFlag, obj_name)

        # Build the path to the plot directory if it does not exist
        filepath = os.path.dirname(plotName)
        if not os.path.exists(filepath):
            ut.build_path(filepath)

    else:
        plotName = False

    # First we decide which extensions are worth scaling
    extensions, noPhotometry, badQC = prevent_Excess(spectra, photo, filters.bands, interpFlag)

    # Then we calculate the scale factors
    badData, scaling = scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, filters, interpFlag,
                                        plotName)

    # Remove last minute trouble makers
    extensions = [e for e in extensions if e not in badData]

    badQC = badQC + badData

    # And finally warp the data
    for s in extensions:
        if plotFlag != False:
            plotName = os.path.join(plotFlag, obj_name + "_" + str(s))
            # plotName = plotFlag + obj_name + "_" + str(s)
        else:
            plotName = False

        title = '{} {}'.format(spectra.data[s*3].header['RA'], spectra.data[s*3].header['DEC'])

        # scale the spectra
        spectra.flux[:, s], spectra.variance[:, s] =\
                    warp_spectra(scaling[0:3, s], scaling[3:6, s],
                                 spectra.flux[:, s], spectra.variance[:, s],
                                 spectra.wavelength, filters.centers, plotName, plotTitle=title)

    if coaddFlag == False:
        specio.create_output_single(obj_name, extensions, scaling, spectra,
                                    noPhotometry, badQC, photo.name,
                                    outBase, redshift)

    elif coaddFlag in ['Run', 'Date', 'All']:
        coadd_output(obj_name, extensions, scaling, spectra,
                            noPhotometry, badQC, photo.name,
                            outBase, plotFlag, coaddFlag, redshift)
    else:
        print("What do you want me to do with this data? Please specify output type.")

    return


def calibSpec_from_fit(obj_name, spectra, coadd_spectra, photo, outBase, filters, plotFlag, coaddFlag, redshift, **kwargs):
    """
    This function will determine extensions which can be calibrated,
    fit the scale factors to a previously coadded spectra, warping function, and output a new fits file
    with the scaled spectra.

    Parameters
    ----------
    obj_name : str
        The name of the object.

    Returns
    -------

    """

    if plotFlag != False:
        plotName = os.path.join(plotFlag, obj_name)

        # Build the path to the plot directory if it does not exist
        filepath = os.path.dirname(plotName)
        if not os.path.exists(filepath):
            ut.build_path(filepath)

    else:
        plotName = False

    interpFlag = None

    # First we decide which extensions are worth scaling
    extensions, badQC = get_badQC(spectra)


    # Then we calculate the scale factors
    badData, scaling = scaling_matrix_from_fit(spectra, coadd_spectra, extensions, badQC, photo, filters, plotFlag, **kwargs)

    # Remove last minute trouble makers
    extensions = [e for e in extensions if e not in badData]

    badQC = badQC + badData

    # And finally warp the data
    for s in extensions:
        if plotFlag != False:
            plotName = os.path.join(plotFlag, obj_name + "_" + str(s))
            # plotName = plotFlag + obj_name + "_" + str(s)
        else:
            plotName = False

        title = '{} {}'.format(spectra.data[s*3].header['RA'], spectra.data[s*3].header['DEC'])

        # scale the spectra
        spectra.flux[:, s], spectra.variance[:, s] =\
                    warp_spectra(scaling[0:3, s], scaling[3:6, s],
                                 spectra.flux[:, s], spectra.variance[:, s],
                                 spectra.wavelength, filters.centers, plotName, plotTitle=title)

    if coaddFlag == False:
        specio.create_output_single(obj_name, extensions, scaling, spectra,
                                    noPhotometry, badQC, photo.name,
                                    outBase, redshift)

    elif coaddFlag in ['Run', 'Date', 'All']:
        coadd_output(obj_name, extensions, scaling, spectra,
                            noPhotometry, badQC, photo.name,
                            outBase, plotFlag, coaddFlag, redshift)
    else:
        print("What do you want me to do with this data? Please specify output type.")

    return



def prevent_Excess(spectra, photo, bands, interpFlag):
    """
    This function removes extensions from the list to calibrate because of
    insufficient photometric data or bad quality flags.

    """
    # First, find the min/max date for which we have photometry taken on each side of the spectroscopic observation
    # This will be done by finding the highest date for which we have photometry in each band
    # and taking the max/min of those values
    # This is done because we perform a linear interpolation between photometric data points to estimate the magnitudes
    # observed at the specific time of the spectroscopic observation
    # If you want to use the Gaussian process fitting you can forecast into the future/past by the number of days
    # set by the delay term.

    maxPhot = np.zeros(len(bands))
    minPhot = np.array([100000]*len(bands))

    # If using Gaussian process fitting you can forecast up to 28 days.  You probably want to make some plots to check
    # this isn't crazy though!
    delay = 0
    if interpFlag == 'BBK':
        delay = 28

    if interpFlag == 'first':
        # Ensure we do not remove any spectra based off missing photometry
        maxPhot = np.array([np.inf]*len(bands))
        minPhot = np.array([-np.inf]*len(bands))

    else:
        for e in range(len(photo['Date'][:])):
            for b, band in enumerate(bands):
                if photo['Band'][e] == band:
                    if photo['Date'][e] > maxPhot[b]:
                        maxPhot[b] = photo['Date'][e]
                    if photo['Date'][e] < minPhot[b]:
                        minPhot[b] = photo['Date'][e]


    photLim = min(maxPhot) + delay
    photLimMin = max(minPhot) - delay
    noPhotometry = []
    badQC = []

    allowedQC = ['ok', 'backup']

    extensions = []

    for s in range(spectra.numEpochs):
        if interpFlag not in ['average', 'first']:
            # Remove data with insufficient photometry
            if spectra.dates[s] > photLim:
                noPhotometry.append(s)
            if spectra.dates[s] < photLimMin:
                noPhotometry.append(s)
        # Only allow spectra with quality flags 'ok' and 'backup'
        if spectra.qc[s] not in allowedQC:
            badQC.append(s)

    # Make a list of extensions which need to be analyzed
    for s in range(spectra.numEpochs):
        if s not in noPhotometry and s not in badQC:
            extensions.append(s)

    return extensions, noPhotometry, badQC


def get_badQC(spectra, allowedQC=['ok', 'backup']):
    """
    This function removes extensions from the list to calibrate because of
    insufficient photometric data or bad quality flags.

    """

    badQC = []
    extensions = []

    for s in range(spectra.numEpochs):
        if spectra.qc[s] not in allowedQC:
            badQC.append(s)
        else:
            extensions.append(s)

    return extensions, badQC


def scaling_Matrix(spectra, extensions, badQC, noPhotometry, photo, filters, interpFlag, plotFlag):
    """
    Finds the nearest photometry and interpolates mags to find values at the
    time of the spectroscopic observations. Calculates the mag that would be
    observed from the spectra and calculates the scale factors to bring
    them into agreement. Saves the data in the scaling matrix.

    """
    # scale factors for each extension saved in the following form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagError = scaling[9,:] (interpolated from neighbouring observations)
    # rMag = scaling[10,:]   rMagError = scaling[11,:]
    # iMag = scaling[12,:]   iMagError = scaling[13,:]

    scaling = np.zeros((14, spectra.numEpochs))

    # Judge goodness of spectra
    for e in range(spectra.numEpochs):
        if e in badQC:
            scaling[6, e] = False
        else:
            scaling[6, e] = True
        if e in noPhotometry:
            scaling[7, e] = False
        else:
            scaling[7, e] = True

    ozdesPhoto = np.zeros((3, spectra.numEpochs))
    desPhoto = np.zeros((3, spectra.numEpochs))

    ozdesPhotoU = np.zeros((3, spectra.numEpochs))
    desPhotoU = np.zeros((3, spectra.numEpochs))

    if interpFlag == 'BBK':
        desPhoto, desPhotoU = des_photo_BBK(photo, spectra.dates, filters.bands, spectra.numEpochs, plotFlag)

        scaling[8, :] = desPhoto[0, :]
        scaling[10, :] = desPhoto[1, :]
        scaling[12, :] = desPhoto[2, :]

        scaling[9, :] = desPhotoU[0, :]
        scaling[11, :] = desPhotoU[1, :]
        scaling[13, :] = desPhotoU[2, :]

    if interpFlag == 'average':
        desPhoto, desPhotoU = des_photo_avg(photo, filters.bands, spectra.numEpochs)

        scaling[8, :] = desPhoto[0, :]
        scaling[10, :] = desPhoto[1, :]
        scaling[12, :] = desPhoto[2, :]

        scaling[9, :] = desPhotoU[0, :]
        scaling[11, :] = desPhotoU[1, :]
        scaling[13, :] = desPhotoU[2, :]

    if interpFlag == 'flats':
        desPhoto, desPhotoU = photo_standard(photo, spectra.dates, bands, spectra.numEpochs, plotFlag)

        scaling[8, :] = desPhoto[0, :]
        scaling[10, :] = desPhoto[1, :]
        scaling[12, :] = desPhoto[2, :]

        scaling[9, :] = desPhotoU[0, :]
        scaling[11, :] = desPhotoU[1, :]
        scaling[13, :] = desPhotoU[2, :]


    badData = []

    for e in extensions:
        # Find OzDES photometry

        ozdesPhoto[0, e], ozdesPhotoU[0, e] = computeABmag(filters['g'],
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[1, e], ozdesPhotoU[1, e] = computeABmag(filters['r'],
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[2, e], ozdesPhotoU[2, e] = computeABmag(filters['i'],
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])

        # Sometimes the total flux in the band goes zero and this obviously creates issues further down the line and
        # is most noticeable when the calculated magnitude is nan.  Sometimes it is because the data is very noisy
        # or the occasional negative spectrum is a known artifact of the data, more common in early OzDES runs.  In the
        # case where the observation doesn't get cut based on quality flag it will start getting ignored here.  The runs
        # ignored will eventually be saved with the badQC extensions.

        if np.isnan(ozdesPhoto[:, e]).any() == True:
            badData.append(e)

        # Find DES photometry
        if interpFlag == 'linear':
            desPhoto[:, e], desPhotoU[:, e] = des_photo(photo, spectra.dates[e], filters.bands)

            scaling[8, e] = desPhoto[0, e]
            scaling[10, e] = desPhoto[1, e]
            scaling[12, e] = desPhoto[2, e]

            scaling[9, e] = desPhotoU[0, e]
            scaling[11, e] = desPhotoU[1, e]
            scaling[13, e] = desPhotoU[2, e]

        # Find Scale Factor
        scaling[0, e], scaling[3, e] = scale_factors(desPhoto[0, e] - ozdesPhoto[0, e],
                                                     desPhotoU[0, e] + ozdesPhotoU[0, e])
        scaling[1, e], scaling[4, e] = scale_factors(desPhoto[1, e] - ozdesPhoto[1, e],
                                                     desPhotoU[1, e] + ozdesPhotoU[1, e])
        scaling[2, e], scaling[5, e] = scale_factors(desPhoto[2, e] - ozdesPhoto[2, e],
                                                     desPhotoU[2, e] + ozdesPhotoU[2, e])


    return badData, scaling


def scaling_matrix_from_fit(spectra, coadd_spectra, extensions, badQC, photo, filters, plotFlag, **kwargs):
    """
    Finds the nearest photometry and interpolates mags to find values at the
    time of the spectroscopic observations. Calculates the mag that would be
    observed from the spectra and calculates the scale factors to bring
    them into agreement. Saves the data in the scaling matrix.

    """
    # scale factors for each extension saved in the following form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagError = scaling[9,:] (interpolated from neighbouring observations)
    # rMag = scaling[10,:]   rMagError = scaling[11,:]
    # iMag = scaling[12,:]   iMagError = scaling[13,:]

    scaling = np.zeros((14, spectra.numEpochs))

    # Judge goodness of spectra
    for e in range(spectra.numEpochs):
        if e in badQC:
            scaling[6, e] = False
        else:
            scaling[6, e] = True

        scaling[7, e] = False


    ozdesPhoto = np.zeros((3, spectra.numEpochs))
    desPhoto = np.zeros((3, spectra.numEpochs))

    ozdesPhotoU = np.zeros((3, spectra.numEpochs))
    desPhotoU = np.zeros((3, spectra.numEpochs))


    badData = []
    #
    # for e in extensions:
    #     print(spectra.flux[:, e])
    #     plt.plot(spectra.wavelength, spectra.flux[:, e])
    #     plt.show()

    for e in extensions:
        # Find OzDES photometry

        ozdesPhoto[0, e], ozdesPhotoU[0, e] = computeABmag(filters['g'],
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[1, e], ozdesPhotoU[1, e] = computeABmag(filters['r'],
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])
        ozdesPhoto[2, e], ozdesPhotoU[2, e] = computeABmag(filters['i'],
                                                           spectra.wavelength, spectra.flux[:, e],
                                                           spectra.variance[:, e])

        if np.isnan(ozdesPhoto[:, e]).any() == True:
            badData.append(e)

        scaling[:6, e] = fit_spectra_to_coadd(spectra, coadd_spectra, filters.centers, fit_method='emcee', index=e, **kwargs)

        mock_photo, mock_photo_var = mock_photo_from_fit(scaling[:3, e], scaling[3:6], ozdesPhoto)

        scaling[8, e] = mock_photo[0, e]
        scaling[10, e] = mock_photo[1, e]
        scaling[12, e] = mock_photo[2, e]

        scaling[9, e] = mock_photo_var[0, e]
        scaling[11, e] = mock_photo_var[1, e]
        scaling[13, e] = mock_photo_var[2, e]

    return badData, scaling


def mock_photo_from_fit(scale_factor, scale_factor_sigma, ozdes_photo, ozdes_photo_err):
    flux_ratio = 1/scale_factor
    mag_diff = np.log10(flux_ratio)/0.4
    mag_mock = mag_diff + ozdes_photo
    mag_mock_var = scale_factor_sigma/(scale_factor*0.4*2.3)
    return mag_mock, mag_mock_var

def scale_factors(mag_diff, mag_diff_var):
    """
    Calculates the scale factor and variance needed to change spectrscopically
    derived magnitude to the observed photometry.

    """

    flux_ratio = np.power(10., 0.4 * mag_diff)  # f_synthetic/f_photometry
    scale_factor = (1. / flux_ratio)
    scale_factor_sigma = mag_diff_var * (scale_factor * 0.4 * 2.3) ** 2   # ln(10) ~ 2.3

    return scale_factor, scale_factor_sigma


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

    interp_wave = []
    tmp_flux2 = []
    tmp_var2 = []

    # Make new vectors for the flux just using that range (assuming spectral binning)

    for i in range(len(tmp_wave)):
        if minV < tmp_wave[i] < maxV:
            interp_wave.append(tmp_wave[i])
            tmp_flux2.append(tmp_flux[i])
            tmp_var2.append(tmp_var[i])

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_interp = interp1d(filter.wavelength, filter.transmission)(interp_wave)

    # And now calculate the magnitude and uncertainty

    c = 2.992792e18  # Angstrom/s
    Num = np.nansum(tmp_flux2 * trans_interp * interp_wave)
    Num_var = np.nansum(tmp_var2 * (trans_interp * interp_wave) ** 2)
    Den = np.nansum(trans_interp / interp_wave)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
            magABvar = 1.17882 * Num_var / (Num ** 2)
        except FloatingPointError:
            magAB = 99.
            magABvar = 99.

    return magAB, magABvar


def des_photo(photo, spectral_mjd, bands):

    """Takes in an mjd from the spectra, looks through a light curve file to find the nearest photometric epochs and
    performs linear interpolation to get estimate at date, return the photo mags.   """

    # Assumes dates are in chronological order!!!
    mags = np.zeros(3)
    errs = np.zeros(3)

    for l in range(len(photo['Date']) - 1):
        if photo['Band'][l] == bands[0] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            g_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            g_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            g_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])
        if photo['Band'][l] == bands[1] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            r_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            r_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            r_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])
        if photo['Band'][l] == bands[2] and photo['Date'][l] < spectral_mjd < photo['Date'][l + 1]:
            i_date_v = np.array([photo['Date'][l], photo['Date'][l + 1]])
            i_mag_v = np.array([photo['Mag'][l], photo['Mag'][l + 1]])
            i_err_v = np.array([photo['Mag_err'][l], photo['Mag_err'][l + 1]])

    mags[0], errs[0] = interpolatePhot(g_date_v, g_mag_v, g_err_v, spectral_mjd)
    mags[1], errs[1] = interpolatePhot(r_date_v, r_mag_v, r_err_v, spectral_mjd)
    mags[2], errs[2] = interpolatePhot(i_date_v, i_mag_v, i_err_v, spectral_mjd)

    return mags, errs


def des_photo_avg(photo, bands, numEpochs):

    """Performs an average on photometry data.
    For now this just averages over the whole file.  """

    # Assumes dates are in chronological order!!!
    mags = np.zeros((3, numEpochs))
    errs = np.zeros((3, numEpochs))

    g_mags = []
    r_mags = []
    i_mags = []

    for i, band in enumerate(photo['Band']):
        if band == bands[0]:
            g_mags.append(photo['Mag'][i])
        if band == bands[1]:
            r_mags.append(photo['Mag'][i])
        if band == bands[2]:
            i_mags.append(photo['Mag'][i])

    mags[0, :], errs[0, :] = np.mean(g_mags), np.std(g_mags)
    mags[1, :], errs[1, :] = np.mean(r_mags), np.std(r_mags)
    mags[2, :], errs[2, :] = np.mean(i_mags), np.std(i_mags)

    return mags, errs


def photo_from_fstars(photo, bands, numEpochs):
    """
    Performs an average on photometry data.
    For now this just averages over the whole file.

    """

    # Assumes dates are in chronological order!!!
    mags = np.zeros((3, numEpochs))
    errs = np.zeros((3, numEpochs))

    g_mags = []
    r_mags = []
    i_mags = []

    for i, band in enumerate(photo['Band']):
        if band == bands[0]:
            g_mags.append(photo['Mag'][i])
        if band == bands[1]:
            r_mags.append(photo['Mag'][i])
        if band == bands[2]:
            i_mags.append(photo['Mag'][i])

    mags[0, :], errs[0, :] = np.mean(g_mags), np.std(g_mags)
    mags[1, :], errs[1, :] = np.mean(r_mags), np.std(r_mags)
    mags[2, :], errs[2, :] = np.mean(i_mags), np.std(i_mags)

    # print("mags", mags)
    # print("errs", errs)

    return mags, errs


def photo_standard(photo, bands, numEpochs, uncertainty=0.01):
    """
    Makes the photometry arrays for sources that do not vary.

    """

    # Assumes dates are in chronological order!!!
    mags = np.zeros((3, numEpochs))
    errs = np.zeros((3, numEpochs))

    g_mags = photo['g']
    r_mags = photo['r']
    i_mags = photo['i']

    mags[0, :], errs[0, :] = g_mags, uncertainty
    mags[1, :], errs[1, :] = r_mags, uncertainty
    mags[2, :], errs[2, :] = i_mags, uncertainty

    return mags, errs


def des_photo_BBK(photo, dates, bands, numEpochs, plotFlag):
    """
    Finds nearest photometry on both sides of spectral observations and
    interpolates to find value at the time of the spectral observaitons using
    Brownian Bridge Gaussian processes. This is better for sparser data.

    """

    # Assumes dates are in chronological order!!!
    mags = np.zeros((3, numEpochs))

    errs = np.zeros((3, numEpochs))

    # Fit a Brownian Bridge Kernel to the data via Gaussian processes.
    for b in range(3):
        x = []  # Dates for each band
        y = []  # Mags for each band
        s = []  # Errors for each band

        # get data for each band
        for l in range(len(photo['Date']) - 1):
            if photo['Band'][l] == bands[b]:
                x.append(photo['Date'][l])
                y.append(photo['Mag'][l])
                s.append(photo['Mag_err'][l])

        x = np.array(x)
        y = np.array(y)
        s = np.array(s)

        # Define kernel for Gaussian process: Browning Bridge x Constant
        kernel1 = ut.BBK(length_scale=25, length_scale_bounds=(1, 1000))
        kernel2 = kernels.ConstantKernel(constant_value=1.0, constant_value_bounds=(0.001, 10.0))
        gp = GaussianProcessRegressor(kernel=kernel1 * kernel2, alpha=s ** 2, normalize_y=True)

        # Fit the data with the model
        xprime = np.atleast_2d(x).T
        yprime = np.atleast_2d(y).T
        gp.fit(xprime, yprime)
        print(plotFlag)

        if plotFlag != False:
            # Plot what the model looks like
            bname = ['_g', '_r', '_i']
            preddates = np.linspace(min(x) - 100, max(x) + 100, 3000)
            y_predAll, sigmaAll = gp.predict(np.atleast_2d(preddates).T, return_std=True)
            y_predAll = y_predAll.flatten()
            fig, ax1 = ut.makeFigSingle(plotFlag + bname[b], 'Date', 'Mag', [dates[0], dates[-1]])

            # I want to plot lines where the observations take place - only plot one per night though
            dateCull = dates.astype(int)
            dateCull = np.unique(dateCull)
            for e in range(len(dateCull)):
                ax1.axvline(dateCull[e], color='grey', alpha=0.5)
            ax1.errorbar(x, y, yerr=s, fmt='o', color='mediumblue', markersize='7')

            # Plot model with error bars.
            ax1.plot(preddates, y_predAll, color='black')
            ax1.fill_between(preddates, y_predAll - sigmaAll, y_predAll + sigmaAll, alpha=0.5, color='black')
            print(plotFlag + bname[b] + "_photoModel.png")
            plt.savefig(plotFlag + bname[b] + "_photoModel.png")
            plt.close(fig)

        # Predict photometry vales for each observation
        y_pred, sigma = gp.predict(np.atleast_2d(dates).T, return_std=True)
        mags[b, :] = y_pred.flatten()
        errs[b, :] = sigma[0]**2

    return mags, errs


def interpolatePhot(x, y, s, val):
    """
    Performs linear interpolation and propagates the unvertainty to return
    you a variance.

    """
    # takes sigma returns variance
    # x - x data points (list)
    # y - y data points (list)
    # s - sigma on y data points (list)
    # val - x value to interpolate to (number)

    mag = y[0] + (val - x[0]) * (y[1] - y[0])/(x[1] - x[0])

    err = s[0]**2 + (s[0]**2 + s[1]**2) * ((val - x[0])/(x[1] - x[0]))**2

    return mag, err


def scale_factors(mag_diff, mag_diff_var):
    """
    Calculates the scale factor and variance needed to change spectrscopically
    derived magnitude to the observed photometry.

    """

    flux_ratio = np.power(10., 0.4 * mag_diff)  # f_synthetic/f_photometry
    scale_factor = (1. / flux_ratio)
    scale_factor_sigma = mag_diff_var * (scale_factor * 0.4 * 2.3) ** 2   # ln(10) ~ 2.3

    return scale_factor, scale_factor_sigma


def warp_uncalib_spectra(scaling, flux, centers, wavelength):
    scale = InterpolatedUnivariateSpline(centers, scaling, k=2)
    return flux * scale(wavelength)


def uncalib_spectra_residuals(scaling, flux, coadd_flux, centers, wavelength):
    uncalib_spectra_scaled = warp_uncalib_spectra(scaling, flux, centers, wavelength)
    residuals = coadd_flux - uncalib_spectra_scaled
    return residuals


def spectra_log_likelihood(scaling, flux, coadd_flux, coadd_variance, centers, wavelength, **kwargs):
    """
    Get the log likelihood
    """
    residuals = uncalib_spectra_residuals(scaling, flux, coadd_flux, centers, wavelength)
    coadd_variance=0.1
    chi2 = np.nansum(residuals**2/coadd_variance) # Note variance is sigma^2
    return -0.5 * chi2


def fit_spectra_to_coadd(fit_spectra, coadd_spectra, centers, fit_method='emcee', index=None, **kwargs):
    """
    Fit a spectra to co-added spectra and determine uncertainties.

    Assumes that fit_spectra and coadd_spectra have the same wavelengths (x axis)

    Parameters
    ----------
    fit_spectra : Spectrumv18 object
        The spectra to be fitted.
    coadd_spectra : SpectrumCoadd object
        The co-added spectra that fit_spectra will be fitted to
    fit_method : str, optional
        Specify which method to fit with. Current supports emcee.
    **kwargs : dict keyword arguments, optional\
        Additional arguments used for interacting with the sampling method.

    Returns
    -------
    scale_params : array
        The best fit parameters for the scaling function.
        ``scale_params[0:3]`` is the best fit for g, r, i, respectively.
        ``scale_params[3:6]`` is the uncertainties for g, r, i, respectively.

    """

    if fit_method == 'emcee':

        initial_guess = [1]*len(centers)

        # print(fit_spectra.flux[:, index])
        # plt.plot(fit_spectra.wavelength, fit_spectra.flux[:, index])
        # plt.show()

        args = [fit_spectra.flux[:, index], coadd_spectra.combinedFlux.data, coadd_spectra.combinedVariance.data,
                    centers, fit_spectra.wavelength]

        ndim = len(initial_guess)


        # Obtain relevant kwargs for the sampler
        if 'nwalkers' in kwargs.keys():
            nwalkers = kwargs['nwalkers']
        else:
            nwalkers = 24

        if 'nsteps' in kwargs.keys():
            nsteps = kwargs['nwalkers']
        else:
            nsteps = 500

        if 'burnin' in kwargs.keys():
            burnin = kwargs['burnin']
        else:
            burnin = 200

        if 'progress' in kwargs.keys():
            progress = kwargs['progress']
        else:
            progress = True

        p0 = np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, spectra_log_likelihood, args=args)
        sampler.run_mcmc(p0, burnin+nsteps, progress=progress)

        c = ChainConsumer()
        c.configure(usetex=False)
        c.plotter.restore_rc_params()

        chains = sampler.get_chain()
        samples = sampler.chain[:, burnin:burnin+nsteps]
        samples = samples.reshape(-1, samples.shape[-1])

        c.add_chain(samples)

        # c.plotter.plot_walks()
        # c.plotter.plot()
        # plt.show()
        summary = c.analysis.get_summary()
        print(summary)

        max_likelihoods = [summary[key][1] for key in summary.keys()]
        print(max_likelihoods)

        plt.plot(fit_spectra.wavelength, coadd_spectra.fluxCoadd)
        plt.show()

    return scale_params


def warp_spectra(scaling, scaleErr, flux, variance, wavelength, centers, plotFlag, plotTitle=''):
    """
    Fits polynomial to scale factors and estimates associated uncertainties
    with Gaussian processes. If the plotFlag variable is not False it will
    save some diagnotic plots.

    """

    # associate scale factors with centers of bands and fit 2D polynomial to form scale function.
    scale = InterpolatedUnivariateSpline(centers, scaling, k=2)
    fluxScale = flux * scale(wavelength)

    # add in Gaussian process to estimate uncertainties, /10**-17 because it gets a bit panicky if you use small numbers
    stddev = (scaleErr ** 0.5) / 10 ** -17
    scale_v = scaling / 10 ** -17

    kernel = kernels.RBF(length_scale=300, length_scale_bounds=(.01, 2000.0))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=stddev**2)

    xprime = np.atleast_2d(centers).T
    yprime = np.atleast_2d(scale_v).T

    gp.fit(xprime, yprime)
    xplot_prime = np.atleast_2d(wavelength).T
    y_pred, sigma = gp.predict(xplot_prime, return_std=True)

    y_pred = y_pred[:,0]

    sigModel = (sigma/y_pred)*scale(wavelength)

    # now scale the original variance and combine with scale factor uncertainty
    varScale = variance * pow(scale(wavelength), 2) + sigModel ** 2

    if plotFlag != False:
        figa, ax1a, ax2a = ut.makeFigDouble(plotTitle, "Wavelength ($\AA$)", "f$_\lambda$ (arbitrary units)",
                                      "f$_\lambda$ (10$^{-17}$ erg/s/cm$^2$/$\AA$)", [wavelength[0], wavelength[-1]])
        # plt.title(plotTitle)

        ax1a.plot(wavelength, flux, color='black', label="Before Calibration")
        ax1a.legend(loc=1, frameon=False, prop={'size': 20})
        ax2a.plot(wavelength, fluxScale / 10 ** -17, color='black', label="After Calibration")
        ax2a.legend(loc=1, frameon=False, prop={'size': 20})
        plt.savefig(plotFlag + "_beforeAfter.png")
        plt.close(figa)

        figb, ax1b, ax2b = ut.makeFigDouble(plotTitle, "Wavelength ($\AA$)", "f$_\lambda$ (10$^{-17}$ erg/s/cm$^2$/$\AA$)",
                                         "% Uncertainty", [wavelength[0], wavelength[-1]])
        ax1b.plot(wavelength, fluxScale / 10 ** -17, color='black')

        ax2b.plot(wavelength, 100*abs(pow(varScale, 0.5)/fluxScale), color='black', linestyle='-', label='Total')
        ax2b.plot(wavelength, 100*abs(sigModel/fluxScale), color='blue', linestyle='-.', label='Warping')
        ax2b.legend(loc=1, frameon=False, prop={'size': 20})
        ax2b.set_ylim([0, 50])
        plt.savefig(plotFlag + "_uncertainty.png")
        plt.close(figb)

        figc, axc = ut.makeFigSingle(plotFlag, "Wavelength ($\AA$)", "Scale Factor (10$^{-17}$ erg/s/cm$^2$/$\AA$/counts)")
        axc.plot(wavelength, scale(wavelength)/10**-17, color='black')
        axc.errorbar(centers, scaling/10**-17, yerr=stddev, fmt='s', color='mediumblue')
        plt.savefig(plotFlag + "_scalefactors.png")
        plt.close(figc)

    return fluxScale, varScale


def coadd_output(obj_name, extensions, scaling, spectra, noPhotometry, badQC, photoName, outBase, plotFlag,
                 coaddFlag, redshift):
    """
    Coadds the observations based on run or night.
    """

    # Get a list of items (dates/runs) over which all observations will be coadded
    coaddOver = []

    for e in extensions:
        # OzDES runs 7,8 were close together in time and run 8 had bad weather so there was only observations of 1
        # field - coadd with run 7 to get better signal to noise
        if spectra.run[e] == 8:
            spectra.run[e] = 7

        if coaddFlag == 'Run':
            if spectra.run[e] not in coaddOver:
                coaddOver.append(spectra.run[e])

        if coaddFlag == 'Date':
            if int(spectra.dates[e]) not in coaddOver:
                coaddOver.append(int(spectra.dates[e]))


    coaddFlux = np.zeros((spectra.len_wavelength, len(coaddOver) + 1))
    coaddVar = np.zeros((spectra.len_wavelength, len(coaddOver) + 1))
    coaddBadPix = np.zeros((spectra.len_wavelength, len(coaddOver) + 1))

    speclistC = []  # For total coadd of observation
    index = 1

    for c in coaddOver:
        speclist = []
        for e in extensions:
            opt = ''
            if coaddFlag == 'Run':
                opt = spectra.run[e]
            if coaddFlag == 'Date':
                opt = int(spectra.dates[e])
            if opt == c:
                speclist.append(SingleSpec(obj_name, spectra.wavelength, spectra.flux[:,e], spectra.variance[:,e],
                                           spectra.badpix[:,e]))
                speclistC.append(SingleSpec(obj_name, spectra.wavelength, spectra.flux[:,e], spectra.variance[:,e],
                                            spectra.badpix[:,e]))

        if len(speclist) > 1:
            runCoadd = outlier_reject_and_coadd(obj_name, speclist)
            coaddFlux[:, index] = runCoadd.flux
            coaddVar[:, index] = runCoadd.variance
            coaddVar[:, index] = runCoadd.variance
            coaddBadPix[:,index] = runCoadd.isbad.astype('uint8')
        if len(speclist) == 1:
            coaddFlux[:, index] = speclist[0].flux
            coaddVar[:, index] = speclist[0].variance
            coaddBadPix[:, index] = speclist[0].isbad.astype('uint8')

        print(coaddFlux[:, index])
        index += 1

    if len(speclistC) > 1:
        allCoadd = outlier_reject_and_coadd(obj_name, speclistC)
        coaddFlux[:, 0] = allCoadd.flux
        coaddVar[:, 0] = allCoadd.variance
        coaddBadPix[:, 0] = allCoadd.isbad.astype('uint8')
    if len(speclistC) == 1:
        coaddFlux[:, 0] = speclistC[0].flux
        coaddVar[:, 0] = speclistC[0].variance
        coaddBadPix[:, 0] = speclistC[0].isbad.astype('uint8')

    mark_as_bad(coaddFlux, coaddVar)



    specio.create_output_coadd(obj_name, coaddOver, coaddFlux, coaddVar, coaddBadPix, extensions, scaling, spectra, redshift,
                        badQC, noPhotometry, photoName, outBase, coaddFlag)


    return


def mark_as_bad(fluxes, variances):
    """
    Occasionally you get some big spikes in the data that you do not want
    messing with your magnitude calculations. Remove these by
    looking at single bins that have a significantly 4.5 larger than
    averages fluxes or variances and change those to NaNs. NaNs will be
    interpolated over. The threshold should be chosen to weigh removing
    extreme outliers and removing noise.

    """

    number = int(fluxes.size/fluxes.shape[0])
    for epoch in range(number):
        if number == 1:
            flux = fluxes[:]
            variance = variances[:]
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        nBins = len(flux)
        # define the local average in flux and variance to compare outliers to
        for i in range(nBins):
            if i < 50:
                avg = np.nanmean(variance[0:99])
                avgf = np.nanmean(flux[0:99])
            elif i > nBins - 50:
                avg = np.nanmean(variance[i-50:nBins-1])
                avgf = np.nanmean(flux[i-50:nBins-1])
            else:
                avg = np.nanmean(variance[i-50:i+50])
                avgf = np.nanmean(flux[i-50:i+50])

            # find outliers and set that bin and the neighbouring ones to nan.

            if np.isnan(variance[i]) == False and variance[i] > 4.5*avg:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i - 1] = np.nan
                    flux[i - 2] = np.nan
                    flux[i - 3] = np.nan
                    flux[i + 1] = np.nan
                    flux[i + 2] = np.nan
                    flux[i + 3] = np.nan

            if np.isnan(flux[i]) == False and flux[i] > 4.5 * avgf:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan

            if np.isnan(flux[i]) == False and flux[i] < -4.5 * avgf:

                flux[i] = np.nan
                if i > 2 and i < 4996:
                    flux[i-1] = np.nan
                    flux[i-2] = np.nan
                    flux[i-3] = np.nan
                    flux[i+1] = np.nan
                    flux[i+2] = np.nan
                    flux[i+3] = np.nan

        # interpolates nans (added here and bad pixels in the data)
        filter_bad_pixels(flux, variance)
    return


def filter_bad_pixels(fluxes, variances):
    """
    Interpolates over NaNs in the spectrum.
    """
    number = int(fluxes.size/fluxes.shape[0])
    for epoch in range(number):
        if (number == 1):
            flux = fluxes[:]
            variance = variances[:]
        else:
            flux = fluxes[:, epoch]
            variance = variances[:, epoch]

        nBins = len(flux)

        flux[0] = np.nanmean(flux)/1000
        flux[-1] = np.nanmean(flux)/1000
        variance[0] = 100*np.nanmean(variance)
        variance[-1] = 100*np.nanmean(variance)

        bad_pixels = np.logical_or.reduce((np.isnan(flux), np.isnan(variance), variance < 0))

        bin = 0
        binEnd = 0

        while (bin < nBins):
            if (bad_pixels[bin] == True):
                binStart = bin
                binNext = bin + 1
                while (binNext < nBins):
                    if bad_pixels[binNext] == False:
                        binEnd = binNext - 1
                        binNext = nBins
                    binNext = binNext + 1

                ya = float(flux[binStart - 1])
                xa = float(binStart - 1)
                sa = variance[binStart - 1]
                yb = flux[binEnd + 1]
                xb = binEnd + 1
                sb = variance[binEnd + 1]

                step = binStart
                while (step < binEnd + 1):
                    flux[step] = ya + (yb - ya) * (step - xa) / (xb - xa)
                    variance[step] = sa + (sb + sa) * ((step - xa) / (xb - xa)) ** 2
                    step = step + 1
                bin = binEnd
            bin = bin + 1
    return


def outlier_reject_and_coadd(obj_name, speclist):
    """
    Reject outliers on single-object spectra to be coadded.
    Assumes input spectra have been resampled to a common wavelength grid,
    so this step needs to be done after joining and resampling.

    Inputs
        speclist:  list of SingleSpec instances on a common wavelength grid
        show:  boolean; show diagnostic plot?  (debug only; default=False)
        savefig:  boolean; save diagnostic plot?  (debug only; default=False)
    Output
        result:  SingleSpec instance of coadded spectrum, with bad pixels
            set to np.nan (runz requires this)
    """

    # Edge cases
    if len(speclist) == 0:
        print("outlier_reject:  empty spectrum list")
        return None
    elif len(speclist) == 1:
        tgname = speclist[0].name
        vmsg("Only one spectrum, no coadd needed for {0}".format(tgname))
        return speclist[0]

    # Have at least two spectra, so let's try to reject outliers
    # At this stage, all spectra have been mapped to a common wavelength scale
    wl = speclist[0].wavelength
    tgname = speclist[0].name
    # Retrieve single-object spectra and variance spectra.
    flux_2d = np.array([s.flux for s in speclist])
    fluxvar_2d = np.array([s.variance for s in speclist])
    badpix_2d = np.array([s.isbad for s in speclist])


    # Baseline parameters:
    #    outsig     Significance threshold for outliers (in sigma)
    #    nbin       Bin width for median rebinning
    #    ncoinc     Maximum number of spectra in which an artifact can appear
    outsig, nbin, ncoinc = 5, 25, 1
    nspec, nwl = flux_2d.shape

    # Run a median filter of the spectra to look for n-sigma outliers.
    # These incantations are kind of complicated but they seem to work
    # i) Compute the median of a wavelength section (nbin) along the observation direction
    # 0,1 : observation,wavelength, row index, column index
    # In moving to numpy v1.10.2, we replaced median with nanmedian
    fmed = np.reshape([np.nanmedian(flux_2d[:, j:j + nbin], axis=1)
                       for j in np.arange(0, nwl, nbin)], (-1, nspec)).T

    # Now expand fmed and flag pixels that are more than outsig off
    fmed_2d = np.reshape([fmed[:, int(j / nbin)] for j in np.arange(nwl)], (-1, nspec)).T

    resid = (flux_2d - fmed_2d) / np.sqrt(fluxvar_2d)
    # If the residual is nan, set flag_2d to 1
    nans = np.isnan(resid)

    flag_2d = np.zeros(nspec * nwl).reshape(nspec, nwl)
    flag_2d[nans] = 1
    flag_2d[~nans] = (np.abs(resid[~nans]) > outsig)

    # If a pixel is flagged in only one spectrum, it's probably a cosmic ray
    # and we should mark it as bad and add ito to badpix_2d.  Otherwise, keep it.
    # This may fail if we coadd many spectra and a cosmic appears in 2 pixels
    # For these cases, we could increase ncoinc
    flagsum = np.tile(np.sum(flag_2d, axis=0), (nspec, 1))
    # flag_2d, flagsum forms a tuple of 2 2d arrays
    # If flag_2d is true and if and flagsum <= ncoinc then set that pixel to bad.
    badpix_2d[np.all([flag_2d, flagsum <= ncoinc], axis=0)] = True


    # Remove bad pixels in the collection of spectra.  In the output they
    # must appear as NaN, but any wavelength bin which is NaN in one spectrum
    # will be NaN in the coadd.  So we need to set the bad pixel values to
    # something innocuous like the median flux, then set the weights of the
    # bad pixels to zero in the coadd.  If a wavelength bin is bad in all
    # the coadds, it's just bad and needs to be marked as NaN in the coadd.
    # In moving to numpy v1.10.2, we replaced median with nanmedian
    flux_2d[badpix_2d] = np.nanmedian(fluxvar_2d)
    fluxvar_2d[badpix_2d] = np.nanmedian(fluxvar_2d)
    badpix_coadd = np.all(badpix_2d, axis=0)
    # Derive the weights
    ## Use just the variance
    wi = 1.0 / (fluxvar_2d)
    # Set the weights of bad data to zero
    wi[badpix_2d] = 0.0
    # Why set the weight of the just first spectrum to np.nan?
    # If just one of the pixels is nan, then the result computed below is nan as well
    for i, val in enumerate(badpix_coadd):
        if val:  wi[0, i] = np.nan

    # Some coadd
    coaddflux = np.average(flux_2d, weights=wi, axis=0)
    coaddfluxvar = np.average(fluxvar_2d, weights=wi, axis=0) / nspec

    coaddflux[badpix_coadd] = np.nan
    coaddfluxvar[badpix_coadd] = np.nan

    # Return the coadded spectrum in a SingleSpectrum object
    return SingleSpec(obj_name, wl, coaddflux, coaddfluxvar, badpix_coadd)
