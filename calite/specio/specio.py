from astropy.io import fits
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.optimize import least_squares
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import matplotlib.pyplot as plt
import sys
import os
import emcee
from chainconsumer import ChainConsumer
from ..utils import BBK, build_path


def create_output_single(obj_name, extensions, scaling, spectra, noPhotometry, badQC, photoName, outName,
                         redshift):
    """Apply clibration to spectra and save to a new file.
    Parameters
    ----------
    obj_name : string
        The name of the object
    extensions : array, shape (n_samples_Y, n_features), (optional, default=None)
        Right argument of the returned kernel k(X, Y). If None, k(X, X)
        if evaluated instead.

    Returns
    -------
    K :
    """

    print("Saving Data to " + outName)

    hdulist = fits.HDUList(fits.PrimaryHDU())

    noPhotometryExt = []
    if len(noPhotometry) > 0:
        for i in range(len(noPhotometry)):
            noPhotometryExt.append(spectra.ext[noPhotometry[i]])

    badQCExt = []
    if len(badQC) > 0:
        for i in range(len(badQC)):
            badQCExt.append(spectra.ext[badQC[i]])

    index = 0
    # Create an HDU for each night
    for i in extensions:
        header = fits.Header()
        header['SOURCE'] = obj_name
        header['RA'] = spectra.RA
        header['DEC'] = spectra.DEC
        header['FIELD'] = spectra.field
        header['CRPIX1'] = spectra.crpix1
        header['CRVAL1'] = spectra.crval1
        header['CDELT1'] = spectra.cdelt1
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['EPOCHS'] = len(extensions)
        header['z'] = redshift[0]

        # save the names of the input data and the extensions ignored
        header['SFILE'] = spectra.filepath
        header['PFILE'] = photoName
        header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
        header['BADQC'] = ','.join(map(str, badQCExt))

        # save the original spectrum's extension number and some other details
        header["EXT"] = spectra.ext[i]
        header["UTMJD"] = spectra.dates[i]
        header["EXPOSE"] = spectra.exposed[i]
        header["QC"] = spectra.qc[i]

        # save scale factors/uncertainties
        header["SCALEG"] = scaling[0, i]
        header["ERRORG"] = scaling[3, i]
        header["SCALER"] = scaling[1, i]
        header["ERRORR"] = scaling[4, i]
        header["SCALEI"] = scaling[2, i]
        header["ERRORI"] = scaling[5, i]

        # save photometry/uncertainties used to calculate scale factors
        header["MAGG"] = scaling[8, i]
        header["MAGUG"] = scaling[9, i]
        header["MAGR"] = scaling[10, i]
        header["MAGUR"] = scaling[11, i]
        header["MAGI"] = scaling[12, i]
        header["MAGUI"] = scaling[13, i]
        if index == 0:
            hdulist[0].header['SOURCE'] = obj_name
            hdulist[0].header['RA'] = spectra.RA
            hdulist[0].header['DEC'] = spectra.DEC
            hdulist[0].header['CRPIX1'] = spectra.crpix1
            hdulist[0].header['CRVAL1'] = spectra.crval1
            hdulist[0].header['CDELT1'] = spectra.cdelt1
            hdulist[0].header['CTYPE1'] = 'wavelength'
            hdulist[0].header['CUNIT1'] = 'angstrom'
            hdulist[0].header['EPOCHS'] = len(extensions)

            # save the names of the input data and the extensions ignored
            hdulist[0].header['SFILE'] = spectra.filepath
            hdulist[0].header['PFILE'] = photoName
            hdulist[0].header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
            hdulist[0].header['BADQC'] = ','.join(map(str, badQCExt))

            # save the original spectrum's extension number and some other details
            hdulist[0].header["EXT"] = spectra.ext[i]
            hdulist[0].header["UTMJD"] = spectra.dates[i]
            hdulist[0].header["EXPOSE"] = spectra.exposed[i]
            hdulist[0].header["QC"] = spectra.qc[i]

            # save scale factors/uncertainties
            hdulist[0].header["SCALEG"] = scaling[0, i]
            hdulist[0].header["ERRORG"] = scaling[3, i]
            hdulist[0].header["SCALER"] = scaling[1, i]
            hdulist[0].header["ERRORR"] = scaling[4, i]
            hdulist[0].header["SCALEI"] = scaling[2, i]
            hdulist[0].header["ERRORI"] = scaling[5, i]

            # save photometry/uncertainties used to calculate scale factors
            hdulist[0].header["MAGG"] = scaling[8, i]
            hdulist[0].header["MAGUG"] = scaling[9, i]
            hdulist[0].header["MAGR"] = scaling[10, i]
            hdulist[0].header["MAGUR"] = scaling[11, i]
            hdulist[0].header["MAGI"] = scaling[12, i]
            hdulist[0].header["MAGUI"] = scaling[13, i]
            hdulist[0].data = spectra.flux[:, i]
            hdulist.append(fits.ImageHDU(data=spectra.variance[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.badpix[:, i], header=header))
            index = 2


        else:
            hdulist.append(fits.ImageHDU(data=spectra.flux[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.variance[:, i], header=header))
            hdulist.append(fits.ImageHDU(data=spectra.badpix[:, i], header=header))

    filepath = os.path.dirname(outName)
    if not os.path.exists(filepath):
        build_path(filepath)

    hdulist.writeto(outName, overwrite=True)
    hdulist.close()

    return


def create_output_coadd(obj_name, runList, fluxArray, varianceArray, badpixArray, extensions, scaling, spectra, redshift,
                        badQC, noPhotometry, photoName, outName, coaddFlag):
    """
    Outputs the warped and coadded spectra to a new fits file.

    """

    hdulist = fits.HDUList(fits.PrimaryHDU())

    noPhotometryExt = []
    if len(noPhotometry) > 0:
        for i in range(len(noPhotometry)):
            noPhotometryExt.append(spectra.ext[noPhotometry[i]])

    badQCExt = []
    if len(badQC) > 0:
        for i in range(len(badQC)):
            badQCExt.append(spectra.ext[badQC[i]])

    #print("Output Filename: %s \n" % (outName))
    # First save the total coadded spectrum for the source to the primary extension
    hdulist[0].data = fluxArray[:, 0]
    hdulist[0].header['CRPIX1'] = spectra.crpix1
    hdulist[0].header['CRVAL1'] = spectra.crval1
    hdulist[0].header['CDELT1'] = spectra.cdelt1
    hdulist[0].header['CTYPE1'] = 'wavelength'
    hdulist[0].header['CUNIT1'] = 'angstrom'
    hdulist[0].header['SOURCE'] = obj_name
    hdulist[0].header['RA'] = spectra.RA
    hdulist[0].header['DEC'] = spectra.DEC
    hdulist[0].header['FIELD'] = spectra.field
    hdulist[0].header['OBSNUM'] = len(runList)
    hdulist[0].header['z'] = redshift[0]
    hdulist[0].header['SFILE'] = spectra.filepath
    hdulist[0].header['PFILE'] = photoName
    hdulist[0].header['METHOD'] = coaddFlag
    hdulist[0].header['NOPHOTO'] = ','.join(map(str, noPhotometryExt))
    hdulist[0].header['BADQC'] = ','.join(map(str, badQCExt))

    # First extension is the total coadded variance
    header = fits.Header()
    header['EXTNAME'] = 'VARIANCE'
    header['CRPIX1'] = spectra.crpix1
    header['CRVAL1'] = spectra.crval1
    header['CDELT1'] = spectra.cdelt1
    header['CTYPE1'] = 'wavelength'
    header['CUNIT1'] = 'angstrom'
    hdulist.append(fits.ImageHDU(data=varianceArray[:, 0], header=header))

    # Second Extension is the total bad pixel map
    header = fits.Header()
    header['EXTNAME'] = 'BadPix'
    header['CRPIX1'] = spectra.crpix1
    header['CRVAL1'] = spectra.crval1
    header['CDELT1'] = spectra.cdelt1
    header['CTYPE1'] = 'wavelength'
    header['CUNIT1'] = 'angstrom'
    hdulist.append(fits.ImageHDU(data=badpixArray[:, 0], header=header))

    # Create an HDU for each night
    index1 = 1
    for k in runList:
        index = 0
        date = 0
        header = fits.Header()
        header['CRPIX1'] = spectra.crpix1
        header['CRVAL1'] = spectra.crval1
        header['CDELT1'] = spectra.cdelt1
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['RUN'] = k
        for i in extensions:
            here = False
            if coaddFlag == 'Run':
                if spectra.run[i] == k:
                    here = True

            if coaddFlag == 'Date':
                if int(spectra.dates[i]) == k:
                    here = True

            if here == True:
                head0 = "EXT" + str(index)
                header[head0] = spectra.ext[i]

                head1 = "UTMJD" + str(index)
                header[head1] = spectra.dates[i]
                date += spectra.dates[i]

                head2 = "EXPOSE" + str(index)
                header[head2] = spectra.exposed[i]

                head3 = "QC" + str(index)
                header[head3] = spectra.qc[i]

                head4 = "SCALEG" + str(index)
                header[head4] = scaling[0, i]

                head5 = "ERRORG" + str(index)
                header[head5] = scaling[3, i]

                head6 = "SCALER" + str(index)
                header[head6] = scaling[1, i]

                head7 = "ERRORR" + str(index)
                header[head7] = scaling[4, i]

                head8 = "SCALEI" + str(index)
                header[head8] = scaling[2, i]

                head9 = "ERRORI" + str(index)
                header[head9] = scaling[5, i]

                head10 = "MAGG" + str(index)
                header[head10] = scaling[8, i]

                head11 = "MAGUG" + str(index)
                header[head11] = scaling[9, i]

                head12 = "MAGR" + str(index)
                header[head12] = scaling[10, i]

                head13 = "MAGUR" + str(index)
                header[head13] = scaling[11, i]

                head14 = "MAGI" + str(index)
                header[head14] = scaling[12, i]

                head15 = "MAGUI" + str(index)
                header[head15] = scaling[13, i]

                index += 1

        if date > 0:
            header['OBSNUM'] = index
            header['AVGDATE'] = date / index

            hdu_flux = fits.ImageHDU(data=fluxArray[:, index1], header=header)
            hdu_fluxvar = fits.ImageHDU(data=varianceArray[:, index1], header=header)
            hdu_badpix = fits.ImageHDU(data=badpixArray[:, index1], header=header)
            hdulist.append(hdu_flux)
            hdulist.append(hdu_fluxvar)
            hdulist.append(hdu_badpix)
        index1 += 1

    filepath = os.path.dirname(outName)
    if not os.path.exists(filepath):
        build_path(filepath)

    hdulist.writeto(outName, overwrite=True)
    hdulist.close()

    return
