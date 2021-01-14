import numpy as np
from astropy.io import fits
import os


class Spectra(object):
    """
    A spectra class which contains all of the attributes necessary to run
    calite.

    """
    def __init__(self, filepath):
        self.filepath = filepath

    @property
    def wavelength(self):
        """Define wavelength solution."""
        if getattr(self, '_wavelength', None) is None:
            wave = ((np.arange(self.n_pix) - self.crpix1) * self.cdelt1) + self.crval1
            self._wavelength = wave
        return self._wavelength

    @property
    def flux(self):
        if getattr(self, '_flux', None) is None:
            self._flux = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._flux[:, i] = self.data[i * 3 + 3].data
        return self._flux

    @property
    def variance(self):
        if getattr(self, '_variance', None) is None:
            self._variance = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._variance[:, i] = self.data[i * 3 + 4].data
        return self._variance

    @property
    def badpix(self):
        if getattr(self, '_badpix', None) is None:
            self._badpix = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._badpix[:, i] = self.data[i * 3 + 5].data
        return self._badpix

    @property
    def dates(self):
        if getattr(self, '_dates', None) is None:
            self._dates = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._dates[i] = round(self.data[i * 3 + 3].header['UTMJD'],3)
                # this give Modified Julian Date (UTC) that observation was taken
        return self._dates

    @property
    def len_wavelength(self):
        if getattr(self, '_len_wavelength', None) is None:
            self._len_wavelength = len(self.wavelength)
        return self._len_wavelength

    @classmethod
    def load_data(self, filepath):
        assert filepath != None, "No file name is specified."

        if not os.path.exists(filepath):
            print("Error: Cannot locate spectra file at {}".format(filepath))
            raise FileNotFoundError

        try:
            data = fits.open(filepath)
        except IOError:
            print("Error loading file {0}".format(filepath))
            raise IOError

        return data


class SpectrumCoadd(Spectra):
    """
    Spectrum class for latest version of the OzDES pipeline. This reads in
    calibrated spectra data assuming data is in the format provided by
    OzDES_calibSpec after coadding.

    """

    def __init__(self, filepath=None):
        super(Spectra, self).__init__()

        # Load the data and set self.data
        self.data = self.load_data(filepath)

        # Set other attributes
        self.combined = self.data[0]
        self.combinedVariance = self.data[1]
        self._wavelength = None
        self._flux = None
        self._variance = None
        self._fluxCoadd = None
        self._varianceCoadd = None
        self._dates = None
        self._runs = None
        self.numEpochs = int((np.size(self.data) - 3) / 3)
        self.redshift = self.combined.header['z']
        self.RA = self.combined.header['RA']
        self.DEC = self.combined.header['DEC']
        self.field = self.combined.header['FIELD']

    @property
    def fluxCoadd(self):
        if getattr(self, '_fluxCoadd', None) is None:
            self._fluxCoadd = np.zeros(self.len_wavelength, dtype=float)
            self._fluxCoadd[:] = self.data[0].data
        return self._fluxCoadd

    @property
    def varianceCoadd(self):
        if getattr(self, '_varianceCoadd', None) is None:
            self._varianceCoadd = np.zeros(self.len_wavelength, dtype=float)
            self._varianceCoadd[:] = self.data[1].data
        return self._varianceCoadd

    @property
    def runs(self):
        if getattr(self, '_runs', None) is None:
            self._runs = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._runs[i] = self.data[i * 3 + 3].header['RUN']  # this give the run number of the observation
        return self._runs


class Spectrumv18(Spectra):
    """
    Read in spectral data assuming the format from v18 of the OzDES reduction
    pipeline.
    """
    def __init__(self, filepath=None):
        super(Spectra, self).__init__()

        # Load the data and set self.data
        self.data = self.load_data(filepath)

        # Set other attributes
        self.combinedFlux = self.data[0]
        self.combinedVariance = self.data[1]
        self.combinedPixels = self.data[2]
        self.numEpochs = int((np.size(self.data) - 3) / 3)
        self.field = self.data[3].header['SOURCEF'][19:21]
        self.cdelt1 = self.combinedFlux.header['cdelt1']  # Wavelength interval between subsequent pixels
        self.crpix1 = self.combinedFlux.header['crpix1']
        self.crval1 = self.combinedFlux.header['crval1']
        self.n_pix = self.combinedFlux.header['NAXIS1']
        self.RA = self.combinedFlux.header['RA']
        self.DEC = self.combinedFlux.header['DEC']

        self.fluxCoadd = self.combinedFlux.data
        self.varianceCoadd = self.combinedVariance.data
        self.badpixCoadd = self.combinedPixels.data

        self._wavelength = None
        self._flux = None
        self._variance = None
        self._badpix = None
        self._dates = None
        self._run = None
        self._ext = None
        self._qc = None
        self._exposed = None

    @property
    def badpix(self):
        if getattr(self, '_badpix', None) is None:
            self._badpix = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._badpix[:, i] = self.data[i * 3 + 5].data
        return self._badpix

    @property
    def dates(self):
        if getattr(self, '_dates', None) is None:
            self._dates = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._dates[i] = round(self.data[i * 3 + 3].header['UTMJD'],3)
                # this give Modified Julian Date (UTC) that observation was taken
        return self._dates

    @property
    def ext(self):
        if getattr(self, '_ext', None) is None:
            self._ext = []
            for i in range(self.numEpochs):
                self._ext.append(i * 3 + 3)  # gives the extension in original fits file
        return self._ext

    @property
    def run(self):
        if getattr(self, '_run', None) is None:
            self._run = []
            for i in range(self.numEpochs):
                source = self.data[i * 3 + 3].header['SOURCEF']
                self._run.append(int(source[3:6]))  # this gives the run number of the observation
        return self._run

    @property
    def qc(self):
        if getattr(self, '_qc', None) is None:
            self._qc = []
            for i in range(self.numEpochs):
                self._qc.append(self.data[i * 3 + 3].header['QC'])
                # this tell you if there were any problems with the spectra that need to be masked out
        return self._qc

    @property
    def exposed(self):
        if getattr(self, '_exposed', None) is None:
            self._exposed = []
            for i in range(self.numEpochs):
                self._exposed.append(self.data[i * 3 + 3].header['EXPOSED'])
                # this will give you the exposure time of each observation
        return self._exposed


class SingleSpec(object):
    """
    Class representing a single spectrum for analysis
    """

    ## Added filename to SingleSpec
    def __init__(self, obj_name, wl, flux, fluxvar, badpix):

        self.name = obj_name
        # ---------------------------
        # self.pivot = int(fibrow[9])
        # self.xplate = int(fibrow[3])
        # self.yplate = int(fibrow[4])
        # self.ra = np.degrees(fibrow[1])
        # self.dec = np.degrees(fibrow[2])
        # self.mag=float(fibrow[10])
        # self.header=header

        self.wl = np.array(wl)
        self.flux = np.array(flux)
        self.fluxvar = np.array(fluxvar)

        # If there is a nan in either the flux, or the variance, mark it as bad

        # JKH: this was what was here originally, my version complains about it
        # self.fluxvar[fluxvar < 0] = np.nan

        for i in range(len(flux)):
            if (self.fluxvar[i] < 0):
                self.fluxvar[i] = np.nan

        # The following doesn't take into account
        #self.isbad = np.any([np.isnan(self.flux), np.isnan(self.fluxvar)], axis=0)
        self.isbad = badpix.astype(bool)
