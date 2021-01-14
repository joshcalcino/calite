import numpy as np
import astropy.io.fits as fits
from sklearn.gaussian_process import kernels
import os


def norm_diff_spectra(spectra1, spectra2):
    """
    A function to obtain the normalised difference between two spectra.

    Parameters
    ----------
    spectra1 : Spectra-like object
        The spectra to compare with.
    spectra2 : Spectra-like object
        The spectra being compared.

    Returns
    -------
    norm_diff : numpy array
        The array ...

    """

    if np.shape(spectra1.flux) != np.shape(spectra2.flux):
        return np.array([np.nan])

    sum1 = np.array([np.sum(spectra1.flux[:, i]) for i in range(len(spectra1.flux[0, :]) )])
    sum2 = np.array([np.sum(spectra2.flux[:, i]) for i in range(len(spectra2.flux[0, :]) )])

    norm_diff = (sum1 - sum2)/sum1

    return norm_diff


def split_photo(photoBase, obj_name, photoEnd, timeName, time, outDir):
    photoName = photoBase + obj_name + photoEnd
    photo = np.loadtxt(photoName, dtype={'names':('Date', 'Mag', 'Mag_err', 'Band'),
                                         'formats':(np.float, np.float, np.float, 'U10')}, skiprows=1)

    new_photo_mask = [d > time[0] and d < time[1] for d in photo['Date']]

    new_photo = photo[new_photo_mask]

    newName = outDir + obj_name + '_' + timeName + photoEnd

    np.savetxt(newName, new_photo, header='MJD     MAG    MAGERR    BAND', fmt='%s')

    return


def split_spectra(spectraBase, obj_name, spectraEnd, timeName, time, outDir):
    # spectraBase: Path to the spectra
    # Obj_name: name of the object
    # spetraEnd: bit after the obj_name
    # timeName: what to call these files
    # time: array with t_min and t_max
    # outDir: where to save the file

    spectraName = spectraBase + obj_name + spectraEnd

    hdulist = fits.open(spectraName)
    numEpochs = int((np.size(hdulist) - 3) / 3)

    dates = [hdulist[i * 3 + 3].header['UTMJD'] for i in range(numEpochs)]

    new_hdu_mask = [d > time[0] and d < time[1] for d in dates]

    hdus = []
    hdus.append(hdulist[0])
    hdus.append(hdulist[1])
    hdus.append(hdulist[2])

    for i in range(numEpochs):
        if new_hdu_mask[i]:
            hdus.append(hdulist[i * 3 + 3])
            hdus.append(hdulist[i * 3 + 4])
            hdus.append(hdulist[i * 3 + 5])

    newName = outDir + obj_name + '_' + timeName + spectraEnd

    new_hdulist = fits.HDUList(hdus=hdus)
    new_hdulist.writeto(newName, overwrite=True)

    new_hdulist.close()
    hdulist.close()

    return


def build_path(filepath):
    """
    Builds a path if the path does not exist
    """

    # Split up the path into each directory
    path_split = os.path.split(filepath)
    tmp_path = ''

    for dir in path_split:
        tmp_path = os.path.join(tmp_path, dir)
        if not os.path.exists(tmp_path) and tmp_path != '':
            os.mkdir(tmp_path)

# -------------------------------------------------- #
# ----------------------- BBK ---------------------- #
# -------------------------------------------------- #
# -------------------------------------------------- #
# A Brownian Bridge Kernel to use with sklearn       #
# Gaussian Processes to interpolate between          #
# photometry.  I have really just copied             #
# Scikit-learn's RBF kernel and modified it to be a  #
# brownian bridge (sqeuclidian -> euclidian).        #
# -------------------------------------------------- #
class BBK(kernels.StationaryKernelMixin, kernels.NormalizedKernelMixin, kernels.Kernel):
    # Here I am slightly modifying scikit-learn's RBF Kernel to do
    # the brownian bridge.

    """Radial-basis function kernel (aka squared-exponential kernel).
    The RBF kernel is a stationary kernel. It is also known as the
    "squared exponential" kernel. It is parameterized by a length-scale
    parameter length_scale>0, which can either be a scalar (isotropic variant
    of the kernel) or a vector with the same number of dimensions as the inputs
    X (anisotropic variant of the kernel). The kernel is given by:
    k(x_i, x_j) = exp(-1 / 2 d(x_i / length_scale, x_j / length_scale)^2)
    This kernel is infinitely differentiable, which implies that GPs with this
    kernel as covariance function have mean square derivatives of all orders,
    and are thus very smooth.
    .. versionadded:: 0.18
    Parameters
    ----------
    length_scale : float or array with shape (n_features,), default: 1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used. If an array, an anisotropic kernel is used where each dimension
        of l defines the length-scale of the respective feature dimension.
    length_scale_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on length_scale
    """
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return kernels.Hyperparameter("length_scale", "numeric",
                                  self.length_scale_bounds,
                                  len(self.length_scale))
        return kernels.Hyperparameter(
            "length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        X = np.atleast_2d(X)
        length_scale = kernels._check_length_scale(X, self.length_scale)
        if Y is None:
            # JKH: All I changed was 'sqeuclidean' to 'euclidean'
            dists = pdist(X / length_scale, metric='euclidean')
            K = np.exp(-.5 * dists)
            # convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)
        else:
            if eval_gradient:
                raise ValueError(
                    "Gradient can only be evaluated when Y is None.")
            dists = cdist(X / length_scale, Y / length_scale,
                          metric='euclidean')
            K = np.exp(-.5 * dists)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                # Hyperparameter l kept fixed
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = \
                    (K * squareform(dists))[:, :, np.newaxis]
                return K, K_gradient
            elif self.anisotropic:
                # We need to recompute the pairwise dimension-wise distances
                K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 \
                    / (length_scale ** 2)
                K_gradient *= K[..., np.newaxis]
                return K, K_gradient
        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return "{0}(length_scale=[{1}])".format(
                self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                   self.length_scale)))
        else:  # isotropic
            return "{0}(length_scale={1:.3g})".format(
                self.__class__.__name__, np.ravel(self.length_scale)[0])
