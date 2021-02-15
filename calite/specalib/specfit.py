import numpy as np
import emcee
from chainconsumer import ChainConsumer
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import calite.specstruct as st
from . import calibutil as cu
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline


def fit_spectra_to_coadd(fit_spectra, template_spectra, filters, fit_method='emcee', index=None, **kwargs):
    """
    Fit a spectra to co-added spectra and determine uncertainties.

    Assumes that fit_spectra and coadd_spectra have the same wavelengths (x axis)

    Parameters
    ----------
    fit_spectra : Spectrumv18 object
        The spectra to be fitted.
    template_spectra : SpectrumCoadd or SpectraLite object
        The spectra that fit_spectra will be fitted to
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

        if 'order' in kwargs.keys():
            # numpy polynomial takes coeffs. A first order polynomial has 2 coeffs
            order = kwargs['order']+1
        else:
            order = len(filters.centers)

        fit_spectra_errors = np.sqrt(fit_spectra.variance[:, index]) # get_spectra_errors(fit_spectra.flux[:, index])

        fit_spectra_flux, fit_spectra_errors, fit_spectra_wavelength =\
                    cu.filter_nans(fit_spectra.flux[:, index], fit_spectra_errors, fit_spectra.wavelength)

        # plt.plot(fit_spectra_wavelength, fit_spectra_flux)
        # plt.plot(fit_spectra_wavelength, fit_spectra_errors)
        # plt.show()

        fit_spectra_flux = cu.smooth_spectra_savitsky_golay(fit_spectra_flux, window=21, order=3)

        if isinstance(template_spectra, st.SpectraLite):
            spectra_flux_interp = interp1d(template_spectra.wavelength, template_spectra.flux)
            spectra_flux = spectra_flux_interp(fit_spectra_wavelength) * 1e17

        elif isinstance(template_spectra, st.SpectrumCoadd):
            spectra_flux = template_spectra.fluxCoadd*1e17
            spectra_variance = template_spectra.varianceCoadd
            fit_spectra_flux, spectra_flux, spectra_variance =\
                    cu.filter_nans(fit_spectra_flux, spectra_flux, spectra_variance)

        fit_spectra_wavelength_fit = np.linspace(0, 1, len(fit_spectra_wavelength))

        initial_guess = [1]*order

        ndim = order

        # Obtain relevant kwargs for the sampler
        if 'nwalkers' in kwargs.keys():
            nwalkers = kwargs['nwalkers']
        else:
            nwalkers = 24

        if 'nsteps' in kwargs.keys():
            nsteps = kwargs['nsteps']
        else:
            nsteps = 1000*(1+order)

        if 'burnin' in kwargs.keys():
            burnin = kwargs['burnin']
        else:
            burnin = 500*(1+order)

        if 'progress' in kwargs.keys():
            progress = kwargs['progress']
        else:
            progress = True

        log_likelihood = spectra_log_likelihood
        args = [fit_spectra_flux, spectra_flux, fit_spectra_errors,
                    filters.centers, fit_spectra_wavelength_fit]

        p0 = np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=args, kwargs=kwargs)
        sampler.run_mcmc(p0, burnin+nsteps, progress=progress)

        c = ChainConsumer()
        # c.configure(usetex=False)
        # c.plotter.restore_rc_params()

        chains = sampler.get_chain()
        samples = sampler.chain[:, burnin:burnin+nsteps]
        samples = samples.reshape(-1, samples.shape[-1])

        c.add_chain(samples)

        # c.plotter.plot_walks()
        # # plt.savefig()
        # c.plotter.plot()
        # # plt.savefig('corner_4.png')
        # plt.show()

        summary = c.analysis.get_summary()
        # print(summary)

        max_likelihoods = [summary[key][1] for key in summary.keys()]
        lower_bound = [summary[key][0] for key in summary.keys()]
        upper_bound = [summary[key][2] for key in summary.keys()]

        if 'order' in kwargs.keys():
            best_fit_raw = warp_uncalib_spectra_pol(max_likelihoods, fit_spectra.flux[:, index], np.linspace(0, 1, len(fit_spectra.wavelength)))*1e-17
            best_fit = warp_uncalib_spectra_pol(max_likelihoods, fit_spectra_flux, fit_spectra_wavelength_fit)*1e-17
        else:
            best_fit = warp_uncalib_spectra(max_likelihoods, fit_spectra.flux[:, index], filters.centers, fit_spectra.wavelength)

        plt.figure()
        plt.plot(fit_spectra_wavelength, spectra_flux*1e-17, label='Template Spectra', alpha=.5)
        plt.plot(fit_spectra_wavelength, best_fit, label='Best Fit', alpha=0.5)
        plt.legend()
        plt.show()
        # plt.savefig(fit_spectra.name + '_' + str(index) + '.png')

        photo = np.zeros(3)
        photoU = np.zeros(3)

        nchain_samples = np.min([2000, nsteps])
        # print(samples)

        ids = range(nsteps)

        rand_ids = np.random.choice(ids, size=nchain_samples)
        rand_samples = samples[rand_ids]

        tmp_samples = np.zeros(nchain_samples)
        tmp_flux = np.zeros((nchain_samples, len(fit_spectra.wavelength)))

        for k in range(nchain_samples):
            tmp_flux[k] = warp_uncalib_spectra_pol(rand_samples[k], fit_spectra.flux[:, index], np.linspace(0, 1, len(fit_spectra.wavelength)))*1e-17

        model_variance = np.sqrt(np.std(tmp_flux, axis=0))
        total_variance = model_variance + fit_spectra.variance[:, index]

        tmp_flux_nanr = np.array([cu.filter_nans(arr) for arr in tmp_flux])

        for i, band in enumerate(filters.bands):
            photo[i] = cu.compute_mag_fast(filters[band], fit_spectra.wavelength, best_fit)

            for j in range(nchain_samples):
                tmp_samples[j] = cu.compute_mag_fast(filters[band], fit_spectra_wavelength, tmp_flux[j])

            # The histogram of this distribution is not very Gaussian..
            photoU[i] = np.std(tmp_samples)

        print("mock_photo", photo, photoU)

    return best_fit_raw, total_variance, photo, photoU


def curve_to_fit(x, *params):
    # x[0] is wavelength, x[1] is flux
    pol = np.polynomial.Polynomial(params[::-1])
    # val = x[0] * pol(x[1])
    # plt.figure()
    # plt.plot(x[0], val)
    # plt.show()
    return x[0] * pol(x[1])

def curve_to_fit2(x, params):
    # x[0] is wavelength, x[1] is flux
    pol = np.polynomial.Polynomial(params[::-1])
    return x[0] * pol(x[1])

def warp_uncalib_spectra(scaling, flux, centers, wavelength):
    scale = InterpolatedUnivariateSpline(centers, scaling, k=2)
    return flux * scale(wavelength)


def warp_uncalib_spectra_pol(scaling, flux, wavelength):
    pol = np.polynomial.Polynomial(scaling)
    return flux * pol(wavelength)


def uncalib_spectra_residuals(scaling, flux, coadd_flux, centers, wavelength, **kwargs):
    if 'order' in kwargs.keys():
        uncalib_spectra_scaled = warp_uncalib_spectra_pol(scaling, flux, wavelength)
    else:
        uncalib_spectra_scaled = warp_uncalib_spectra(scaling, flux, centers, wavelength)

    residuals = coadd_flux - uncalib_spectra_scaled
    return residuals


def spectra_log_likelihood(scaling, flux, coadd_flux, errors, centers, wavelength, **kwargs):
    """
    Get the log likelihood
    """
    residuals = uncalib_spectra_residuals(scaling, flux, coadd_flux, centers, wavelength, **kwargs)
    chi2 = np.nansum(residuals**2/errors**2) # Note variance is sigma^2
    return -0.5 * chi2


def get_spectra_errors(flux, nbins=25):
    """ Compute the uncertainties in a given spectra assuming the uncertainties
    can be derived from the scatter in the spectra """

    spectra_len = len(flux)

    nan_mask = np.array([np.isnan(f) for f in flux])
    nan_free_spectra = flux[~nan_mask]

    chunks = np.array_split(nan_free_spectra, nbins)

    std_err = np.zeros(len(nan_free_spectra))

    spectra_error = np.zeros(spectra_len)
    spectra_error[nan_mask] = np.nan

    counter = 0

    for chunk in chunks:
        std_err[counter:counter+len(chunk)] = np.std(chunk)
        counter += len(chunk)

    spectra_error[~nan_mask] = [std_err[i] for i in range(len(std_err))]

    return spectra_error
