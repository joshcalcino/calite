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
    def data_headers(self):
        if getattr(self, '_data_headers') is None:
            self._data_headers = []
            for i in range(self.numEpochs):
                header = {}
                for key in self.data[i * 3 + 3].header.keys():
                    header[key] = self.data[i * 3 + 3].header[key]
                self._data_headers.append(header)
        return self._data_headers

    @classmethod
    def load_data(self, filepath):
        assert filepath != None, "No file name is specified."

        if not os.path.exists(filepath):
            raise FileNotFoundError("Cannot locate spectra file at {}".format(filepath))

        try:
            data = fits.open(filepath)
        except IOError:
            raise IOError("Error loading file {0}".format(filepath))

        return data


class SpectraLite(object):
    """
    A spectra class which contains all of the attributes necessary to run
    calite.

    """
    def __init__(self, filepath=None, flux=None, wavelength=None):
        self.filepath = filepath
        self.flux = flux
        self.wavelength = wavelength


class SpectrumCoadd(Spectra):
    """
    Spectrum class for latest version of the OzDES pipeline. This reads in
    calibrated spectra data assuming data is in the format provided by
    OzDES_calibSpec after coadding.

    """

    def __init__(self, filepath=None):
        super(Spectra, self).__init__()

        self.filepath = filepath

        # Load the data and set self.data
        self.data = self.load_data(filepath)

        # Set other attributes
        self.combinedFlux = self.data[0]
        self.combinedVariance = self.data[1]
        self.combinedPixels = self.data[2]
        self._wavelength = None
        self._flux = None
        self._variance = None
        self._fluxCoadd = None
        self._varianceCoadd = None
        self._dates = None
        self._runs = None
        self.numEpochs = int((np.size(self.data) - 3) / 3)
        self.RA = self.combinedFlux.header['RA']
        self.DEC = self.combinedFlux.header['DEC']
        self.field = self.combinedFlux.header['FIELD']
        self.n_pix = self.combinedFlux.header['NAXIS1']
        self.cdelt1 = self.combinedFlux.header['cdelt1']  # Wavelength interval between subsequent pixels
        self.crpix1 = self.combinedFlux.header['crpix1']
        self.crval1 = self.combinedFlux.header['crval1']
        self.len_wavelength = len(self.wavelength)
        self._data_headers = None

        if 'z' in self.combinedFlux.header.keys():
            self.redshift = self.combinedFlux.header['z']
        else:
            self.redshift = 0.0

        # self.fluxCoadd = self.combinedFlux.data
        # self.varianceCoadd = self.combinedVariance.data
        self.badpixCoadd = self.combinedPixels.data
        self.len_wavelength = len(self.wavelength)

    @property
    def fluxCoadd(self):
        if getattr(self, '_fluxCoadd', None) is None:
            self._fluxCoadd = np.zeros(self.len_wavelength, dtype=np.float_)
            self._fluxCoadd[:] = self.data[0].data
        return self._fluxCoadd

    @property
    def varianceCoadd(self):
        if getattr(self, '_varianceCoadd', None) is None:
            self._varianceCoadd = np.zeros(self.len_wavelength, dtype=np.float_)
            self._varianceCoadd[:] = self.data[1].data
        return self._varianceCoadd

    @property
    def runs(self):
        if getattr(self, '_runs', None) is None:
            self._runs = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._runs[i] = self.data[i * 3 + 3].header['RUN']  # this give the run number of the observation
        return self._runs



class FitSpectrumCoadd(Spectra):
    """
    Spectrum class to hold the calibrated spectra that has been calibrated using
    the fstar fitting method.

    """

    def __init__(self, filepath=None):
        super(Spectra, self).__init__()

        self.filepath = filepath

        # Load the data and set self.data
        self.data = self.load_data(filepath)

        # Set other attributes
        self.combinedFlux = self.data[0]
        self.combinedVariance = self.data[1]
        self.combinedPixels = self.data[2]
        self.combined_scale_function = self.data[3]
        self.combined_scale_function_variance = self.data[4]
        self._wavelength = None
        self._flux = None
        self._variance = None
        self._scale_function = None
        self._scale_function_variance = None
        self._data_headers = None
        self._fluxCoadd = None
        self._varianceCoadd = None
        self._dates = None
        self._runs = None
        self.numEpochs = int((np.size(self.data) - 3) / 5)
        self.RA = self.combinedFlux.header['RA']
        self.DEC = self.combinedFlux.header['DEC']
        self.field = self.combinedFlux.header['FIELD']
        self.n_pix = self.combinedFlux.header['NAXIS1']
        self.cdelt1 = self.combinedFlux.header['cdelt1']  # Wavelength interval between subsequent pixels
        self.crpix1 = self.combinedFlux.header['crpix1']
        self.crval1 = self.combinedFlux.header['crval1']
        self.len_wavelength = len(self.wavelength)

        if 'z' in self.combinedFlux.header.keys():
            self.redshift = self.combinedFlux.header['z']
        else:
            self.redshift = 0.0

        # self.fluxCoadd = self.combinedFlux.data
        # self.varianceCoadd = self.combinedVariance.data
        self.badpixCoadd = self.combinedPixels.data
        self.len_wavelength = len(self.wavelength)

    @property
    def fluxCoadd(self):
        if getattr(self, '_fluxCoadd', None) is None:
            self._fluxCoadd = np.zeros(self.len_wavelength, dtype=np.float_)
            self._fluxCoadd[:] = self.data[0].data
        return self._fluxCoadd

    @property
    def varianceCoadd(self):
        if getattr(self, '_varianceCoadd', None) is None:
            self._varianceCoadd = np.zeros(self.len_wavelength, dtype=np.float_)
            self._varianceCoadd[:] = self.data[1].data
        return self._varianceCoadd

    @property
    def runs(self):
        if getattr(self, '_runs', None) is None:
            self._runs = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._runs[i] = self.data[i * 5 + 5].header['RUN']  # this give the run number of the observation
        return self._runs

    @property
    def flux(self):
        if getattr(self, '_flux', None) is None:
            self._flux = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._flux[:, i] = self.data[i * 5 + 5].data
        return self._flux

    @property
    def variance(self):
        if getattr(self, '_variance', None) is None:
            self._variance = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._variance[:, i] = self.data[i * 5 + 6].data
        return self._variance

    @property
    def badpix(self):
        if getattr(self, '_badpix', None) is None:
            self._badpix = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._badpix[:, i] = self.data[i * 5 + 7].data
        return self._badpix

    @property
    def scale_function(self):
        if getattr(self, '_scale_function', None) is None:
            self._scale_function = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._scale_function[:, i] = self.data[i * 5 + 8].data
        return self._scale_function

    @property
    def scaling_function_variance(self):
        if getattr(self, '_flux', None) is None:
            self._scale_function_variance = np.zeros((self.len_wavelength, self.numEpochs), dtype=float)
            for i in range(self.numEpochs):
                self._scale_function_variance[:, i] = self.data[i * 5 + 9].data
        return self._scale_function_variance

    @property
    def data_headers(self):
        if getattr(self, '_data_headers') is None:
            self._data_headers = []
            for i in range(self.numEpochs):
                header = {}
                for key in self.data[i * 5 + 5].header.keys():
                    header[key] = self.data[i * 5 + 5].header[key]
                self._data_headers.append(header)
        return self._data_headers

    @property
    def dates(self):
        if getattr(self, '_dates', None) is None:
            self._dates = np.zeros(self.numEpochs, dtype=float)
            for i in range(self.numEpochs):
                self._dates[i] = round(self.data[i * 5 + 3].header['UTMJD'],3)
                # this give Modified Julian Date (UTC) that observation was taken
        return self._dates



class Spectrumv18(Spectra):
    """
    Read in spectral data assuming the format from v18 of the OzDES reduction
    pipeline.

    """
    def __init__(self, filepath, obj_name=None):
        super(Spectra, self).__init__()

        self.filepath = filepath

        # Load the data and set self.data
        self.data = self.load_data(self.filepath)

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
        self.len_wavelength = len(self.wavelength)

        self._wavelength = None
        self._flux = None
        self._variance = None
        self._badpix = None
        self._dates = None
        self._run = None
        self._ext = None
        self._qc = None
        self._exposed = None
        self._data_headers = None

        self.name = obj_name

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

    @property
    def data_headers(self):
        if getattr(self, '_data_headers') is None:
            self._data_headers = []
            for i in range(self.numEpochs):
                header = {}
                for key in self.data[i * 3 + 3].header.keys():
                    header[key] = self.data[i * 3 + 3].header[key]
                self._data_headers.append(header)
        return self._data_headers


class SingleSpec(object):
    """
    Class representing a single spectrum for analysis
    """

    ## Added filename to SingleSpec
    def __init__(self, obj_name, wavelength, flux, variance, badpix):

        self.name = obj_name
        # ---------------------------
        # self.pivot = int(fibrow[9])
        # self.xplate = int(fibrow[3])
        # self.yplate = int(fibrow[4])
        # self.ra = np.degrees(fibrow[1])
        # self.dec = np.degrees(fibrow[2])
        # self.mag=float(fibrow[10])
        # self.header=header

        self.wavelength = np.array(wavelength)
        self.flux = np.array(flux)
        self.variance = np.array(variance)
        self.badpix = badpix

        # If there is a nan in either the flux, or the variance, mark it as bad

        # JKH: this was what was here originally, my version complains about it
        # self.fluxvar[fluxvar < 0] = np.nan

        for i in range(len(self.flux)):
            if (self.variance[i] < 0):
                self.variance[i] = np.nan

        # The following doesn't take into account
        #self.isbad = np.any([np.isnan(self.flux), np.isnan(self.fluxvar)], axis=0)
        self.isbad = self.badpix.astype(bool)


class Photo(object):
    """
    A class to hold photometry data.

    """

    def __init__(self, filepath=None, name=None, **kwargs):
        self.filepath = filepath
        self.name = name
        self.kwargs = kwargs

        self.data = None

        if self.filepath != None:
            self.read_data(filepath, **kwargs)


    def __getitem__(self, item):
        if isinstance(item, list):
            cols = []
            for i in item:
                try:
                    col = self.data[i]
                    cols.append(col)

                except ValueError:
                    print("{} is not a valid key for class Photo".format(item))
            return cols

        try:
            col = self.data[item]
            return col

        except ValueError:
            print("{} is not a valid key for class Photo".format(item))

    def read_data(self, filepath, **kwargs):
        """
        A generic read data function.

        """

        self.data = np.loadtxt(filepath, **kwargs)




class FilterCurve:
    """
    A class to hold the transmission function for a particular band.

    """

    def __init__(self, filepath, band, center, units='angstrom'):
        self.filepath = filepath
        self.band = band
        self.center = center

        # Set a factor
        self.factor = 1

        # This factor is used to convert the units to angstrom if necessary
        if self._read_units(units) in 'angstrom':
            self.factor = 10

        # Set the band
        self.band = band

        # Load the data
        self.read(filepath)



    def read(self, file):
        """
        Read in the filter curve file line by line and construct wavelength and
        transmission arrays.

        """

        self.wavelength = np.array([])
        self.transmission = np.array([])

        with open(file, 'r') as file:
            for line in file.readlines():
                if line[0] != '#':
                    entries = line.split()
                    self.wavelength = np.append(self.wavelength, float(entries[0]))
                    self.transmission = np.append(self.transmission, float(entries[1]))

        # Update wavelength with the relevant factor to get the units into ang
        self.wavelength = self.wavelength * self.factor

    def _read_units(self, units):
        """
        Determine the wavelength units of the filter curve.

        """

        # Make sure that the units is given as a string..
        try:
            assert (type(units) == str or type(units) == bytes)
        except AssertionError:
            raise TypeError("units must be string or bytes type")

        # Check if units is either angstroms or nanometers
        if units.lower() in ['angstroms', 'angstrom', 'ang', 'a']:
            return 'angstrom'

        elif units.lower() in ['nanometers', 'namometres', 'nms', 'nm', 'n']:
            return 'nm'

        else:
            raise RuntimeWarning("Warning in FilterCurve: Cannot determine units.\n \
                                Unit '{}' is not recognised. Assuming wavelength unit is Anstroms."\
                                .format(units))
            return 'angstrom'


class FilterCurves(FilterCurve):
    """
    A class which consolidates multiple filters into a single object.

    """

    def __init__(self, filepaths, bands, centers):
        self.filepaths = filepaths
        self.bands = bands

        self.bands = ['']*len(bands)

        for b, band in enumerate(bands):
            if type(band) == bytes or type(band) == np.byte:
                self.bands[b] = str(band, encoding='utf-8')
            else:
                self.bands[b] = band

        self.centers = centers

        # Sanity check to make sure that len(bands) is the same as len(filepaths)
        assert len(self.filepaths) == len(self.bands), "There must be the same number of \
                                                        bands as filepaths. "

        # Store each filter curve in in the FilterCurve class
        self._curves = []

        for i, filepath in enumerate(self.filepaths):
            self._curves.append(FilterCurve(filepath, self.bands[i], center=self.centers[i]))


    def __getitem__(self, band):
        """
        This allows us to access each band by simply indexing an instance of
        the FilterCurves class with the name of the band.

        e.g. `filtercurve = FilterCurves(filepaths, bands, centers)`
             `g_band = filtercurve['g']`

        """
        if type(band) == bytes:
            band = str(band, encoding='utf-8')

        try:
            col = self.bands.index(band)
            return self._curves[col]

        except ValueError:
            raise ValueError("{} is not a valid band in class FilterCurves".format(band))
