import numpy as np
import emcee
from chainconsumer import ChainConsumer
from scipy.interpolate import interp1d
from .. import specstruct as st
import matplotlib.pyplot as plt


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
            order = kwargs['order']
        else:
            order = len(filters.centers)-1

        if isinstance(template_spectra, st.SpectraLite):
            spectra_flux_interp = interp1d(template_spectra.wavelength, template_spectra.flux)
            spectra_flux = spectra_flux_interp(fit_spectra.wavelength)
            spectra_variance = np.zeros(len(fit_spectra.wavelength))

        elif isinstance(template_spectra, st.SpectrumCoadd):
            spectra_flux = template_spectra.fluxCoadd*1e17
            spectra_variance = template_spectra.varianceCoadd

        initial_guess = [1]*order

        ndim = order


        # Obtain relevant kwargs for the sampler
        if 'nwalkers' in kwargs.keys():
            nwalkers = kwargs['nwalkers']
        else:
            nwalkers = 24

        if 'nsteps' in kwargs.keys():
            nsteps = kwargs['nwalkers']
        else:
            nsteps = 2000

        if 'burnin' in kwargs.keys():
            burnin = kwargs['burnin']
        else:
            burnin = 4000

        if 'progress' in kwargs.keys():
            progress = kwargs['progress']
        else:
            progress = True

        if 'fit_function' in kwargs.keys():
            if kwargs['fit_function'] == 'full':
                log_likelihood = spectra_log_likelihood
                args = [fit_spectra.flux[:, index], spectra_flux, spectra_variance,
                            filters.centers, fit_spectra.wavelength]

            elif kwargs['fit_function'] == 'partial':
                log_likelihood = partial_spectra_log_likelihood
                coadd_mags = np.empty(len(filters.bands))
                coadd_vars = np.empty(len(filters.bands))

                for i, filter in enumerate(filters.bands):
                    coadd_mags[i], coadd_vars[i] = cu.computeABmag(filters[filter], fit_spectra.wavelength,
                                                spectra_flux,
                                                spectra_variance)

                args = [fit_spectra.flux[:, index], np.array(coadd_mags), np.array(coadd_vars), filters, fit_spectra.wavelength]

        else:
            log_likelihood = spectra_log_likelihood
            args = [fit_spectra.flux[:, index], spectra_flux, spectra_variance,
                        filters.centers, fit_spectra.wavelength]


        p0 = np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=args, kwargs=kwargs)
        sampler.run_mcmc(p0, burnin+nsteps, progress=progress)

        c = ChainConsumer()
        c.configure(usetex=False)
        c.plotter.restore_rc_params()

        chains = sampler.get_chain()
        samples = sampler.chain[:, burnin:burnin+nsteps]
        samples = samples.reshape(-1, samples.shape[-1])

        c.add_chain(samples)

        summary = c.analysis.get_summary()
        print(summary)

        max_likelihoods = [summary[key][1] for key in summary.keys()]
        print(max_likelihoods)

        if 'order' in kwargs.keys():
            best_fit = warp_uncalib_spectra_pol(max_likelihoods, fit_spectra.flux[:, index], fit_spectra.wavelength)
        else:
            best_fit = warp_uncalib_spectra(max_likelihoods, fit_spectra.flux[:, index], filters.centers, fit_spectra.wavelength)

        print("spectra flux", spectra_flux)
        print("best fit", best_fit)

        plt.plot(fit_spectra.wavelength, spectra_flux)
        plt.plot(fit_spectra.wavelength, best_fit)
        plt.show()

    return scale_params


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


def uncalib_partial_spectra_residuals(scaling, flux, coadd_mags, filters, wavelength):
    uncalib_spectra_scaled = warp_uncalib_spectra(scaling, flux, filters.centers, wavelength)
    uncalib_spectra_mags = np.empty(len(filters.bands))

    for i, filter in enumerate(filters.bands):
        tmp, tmp_var = cu.computeABmag(filters[filter], wavelength, uncalib_spectra_scaled, np.zeros(len(uncalib_spectra_scaled)))
        uncalib_spectra_mags[i] = tmp

    residuals = coadd_mags - uncalib_spectra_mags
    return residuals


def spectra_log_likelihood(scaling, flux, coadd_flux, coadd_variance, centers, wavelength, **kwargs):
    """
    Get the log likelihood
    """
    residuals = uncalib_spectra_residuals(scaling, flux, coadd_flux, centers, wavelength, **kwargs)
    coadd_variance=1e-2
    chi2 = np.nansum(residuals**2/coadd_variance) # Note variance is sigma^2
    # print(chi2)
    return -0.5 * chi2


def partial_spectra_log_likelihood(scaling, flux, coadd_mags, coadd_variance, filters, wavelength, **kwargs):
    """
    Get the log likelihood
    """
    residuals = uncalib_partial_spectra_residuals(scaling, flux, coadd_mags, filters, wavelength)
    coadd_variance=0.2
    chi2 = np.nansum(residuals**2/coadd_variance) # Note variance is sigma^2
    # print(chi2)
    return -0.5 * chi2
