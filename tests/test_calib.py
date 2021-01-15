import numpy as np
import calite as cal

# First define where all of the data can/will be found

# Define where the transmission function is stored, the bands used, and the centers of each band
bands = ['g', 'r', 'i']

filepaths = ['../data/DES_g_y3a1.dat', '../data/DES_r_y3a1.dat', '../data/DES_i_y3a1.dat']

centers = [4730, 6420, 7840]

filters = cal.specstruct.FilterCurves(filepaths, bands, centers)

# Define where spectra are stored and file name format: name = spectraBase + ID + spectraEnd
spectraBase = "../data/SVA1_COADD-"
spectraEnd = ".fits"

# Define where photometry are stored and file name format
photoBase = "../data/"
photoEnd = "_lc.dat"

names = 'test'

# Define the name of the place you want the output data stored
outDir = "calib_test/"

# Do you want calibration plots - if so set the flag to the place where they should be saved, otherwise set it to false
# plotFlag = False
plotFlag = 'plots/'
# plotFlag = "../sparseTest/BBKSparse/"

# Do you want to coadd the spectra? If not the individual calibrated spectra will be save in a fits file
# (coaddFlag == False), otherwise the spectra will be coadded based on the flag chosen (Date: Everything on same mjd
# or Run: Everything on the same observing run)
# coaddFlag = False
coaddFlag = 'Date'
# coaddFlag = 'Run'

# When determining the DES photometric magnitudes at the same time of OzDES spectroscopic light curves the code normally
# just linearly interpolates between the photometry.  This works fine because there is generally such high sampling.
# However, if you have sparser data or what to forecast past when you have data you might want a more robust model.
# You can then use a Gaussian Processes to fit a Brownian Bridge model to the data.  You are allowed to forecast out to
# 28 days.  If you want to change this go to prevent_Excess.
interpFlag = 'linear'
# interpFlag = 'BBK'

# You can also give a file with labeled columns ID and z so the redshift data can be saved with the
# spectra. If you pass through False it will just be saved as -9.99
# redshifts = False
redshifts = False

obj_name = names

# Define input data names and read in spectra and photometric light curves
spectraName = spectraBase + obj_name + spectraEnd
photoName = photoBase + obj_name + photoEnd

print("Input Spectra Name: %s" % spectraName)
spectra = cal.specstruct.Spectrumv18(spectraName)

# Clean up the spectra.  Marks large isolated large variations in flux and variance as bad (nan) and linearly
# interpolates over all nans
cal.specalib.mark_as_bad(spectra.flux, spectra.variance)

print("Input Photometry Name: %s" % photoName)
photo = np.loadtxt(photoName, dtype={'names':('Date', 'Mag', 'Mag_err', 'Band'),
                                     'formats':(np.float, np.float, np.float, '|S15')}, skiprows=1)

if redshifts != False:
    if obj_name in zid:
        zi = np.where(zid == obj_name)
        redshift = red[zi]
    else:
        redshift = [-9.99]
else:
    redshift = [-9.99]


# Calls the main function which does the calibration
cal.specalib.calibSpec(obj_name, spectra, photo, photoName, outDir, filters, plotFlag,
               coaddFlag, interpFlag, redshift)
