# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import json
import numpy as np
from collections import OrderedDict
import scipy.stats as sps
from scipy.stats import truncnorm

from enterprise.signals import parameter
from enterprise.signals import signal_base
from enterprise.signals import deterministic_signals
from enterprise.signals import gp_signals

from enterprise_extensions.blocks import (
    white_noise_block,
    red_noise_block,
    dm_noise_block,
    chromatic_noise_block,
    common_red_noise_block,
)


def BoundNormPrior(value, mu=0, sigma=1, pmin=-1, pmax=1):
    """Prior function for InvGamma parameters."""
    low, up = (pmin - mu) / sigma, (pmax - mu) / sigma
    return truncnorm.pdf(value, loc=mu, scale=sigma, a=low, b=up)


def BoundNormSampler(mu=0, sigma=1, pmin=-1, pmax=1, size=None):
    """Sampling function for Uniform parameters."""
    low, up = (pmin - mu) / sigma, (pmax - mu) / sigma
    return truncnorm.rvs(loc=mu, scale=sigma, a=low, b=up, size=size)


def BoundedNormal(mu=0, sigma=1, pmin=-1, pmax=1, size=None):
    """Class factory for bounded Normal parameters."""

    class BoundedNormal(parameter.Parameter):
        _prior = parameter.Function(
            BoundNormPrior, mu=mu, sigma=sigma, pmin=pmin, pmax=pmax
        )
        _sampler = staticmethod(BoundNormSampler)
        _size = size
        _mu = mu
        _sigma = sigma
        _pmin = pmin
        _pmax = pmax

        def __repr__(self):
            return "{}: BoundedNormal({},{}, [{},{}])".format(
                self.name, mu, sigma, pmin, pmax
            ) + ("" if self._size is None else "[{}]".format(self._size))

    return BoundedNormal


# NE2001 DM Dist data prior.
def NE2001DMDist_Prior(value):
    """Prior function for NE2001DMDist parameters."""
    return px_rv.pdf(value)


def NE2001DMDist_Sampler(size=None):
    """Sampling function for NE2001DMDist parameters."""
    return px_rv.rvs(size=size)


def NE2001DMDist_Parameter(size=None):
    """Class factory for NE2001DMDist parameters."""

    class NE2001DMDist_Parameter(parameter.Parameter):
        _size = size
        _typename = parameter._argrepr("NE2001DMDist")
        _prior = parameter.Function(NE2001DMDist_Prior)
        _sampler = staticmethod(NE2001DMDist_Sampler)

    return NE2001DMDist_Parameter


# Scipy defined RV for NE2001 DM Dist data.
defpath = os.path.dirname(__file__)
data_file = defpath + "/px_prior_1.txt"
px_prior = np.loadtxt(data_file)
px_hist = np.histogram(px_prior, bins=100, density=True)
px_rv = sps.rv_histogram(px_hist)


def get_default_physical_tm_priors():
    """
    Fills dictionary with physical bounds on timing parameters
    """
    physical_tm_priors = {}
    physical_tm_priors["E"] = {"pmin": 0.0, "pmax": 0.9999}
    physical_tm_priors["ECC"] = {"pmin": 0.0, "pmax": 0.9999}
    physical_tm_priors["SINI"] = {"pmin": 0.0, "pmax": 1.0}
    physical_tm_priors["COSI"] = {"pmin": 0.0, "pmax": 1.0}
    physical_tm_priors["PX"] = {"pmin": 0.0}
    physical_tm_priors["M2"] = {"pmin": 1e-10}

    return physical_tm_priors


def get_pardict(psrs, datareleases):
    """assigns a parameter dictionary for each psr per dataset the parfile values/errors
    :param psrs: enterprise pulsar instances corresponding to datareleases
    :param datareleases: list of datareleases
    """
    pardict = {}
    for psr, dataset in zip(psrs, datareleases):
        pardict[psr.name] = {}
        pardict[psr.name][dataset] = {}
        for par, vals, errs in zip(
            psr.fitpars[1:],
            np.longdouble(psr.t2pulsar.vals()),
            np.longdouble(psr.t2pulsar.errs()),
        ):
            pardict[psr.name][dataset][par] = {}
            pardict[psr.name][dataset][par]["val"] = vals
            pardict[psr.name][dataset][par]["err"] = errs
    return pardict


def get_astrometric_priors(astrometric_px_file="../parallaxes.json"):
    # astrometric_px_file = '../parallaxes.json'
    astrometric_px = {}
    with open(astrometric_px_file, "r") as pxf:
        astrometric_px = json.load(pxf)
        pxf.close()

    return astrometric_px


def get_prior(
    prior_type,
    prior_sigma,
    prior_lower_bound,
    prior_upper_bound,
    mu=0.0,
    num_params=None,
):
    """
    Returns the requested prior for a parameter
    :param prior_type: prior on timing parameters.
    :param prior_sigma: Sets the sigma on timing parameters for normal distribution draws
    :param prior_lower_bound: Sets the lower bound on timing parameters for bounded normal and uniform distribution draws
    :param prior_upper_bound: Sets the upper bound on timing parameters for bounded normal and uniform distribution draws
    :param mu: Sets the mean/central value of prior if bounded normal is selected
    :param num_params: number of timing parameters assigned to prior. Default is None (ie. only one)
    """
    if prior_type == "bounded-normal":
        return BoundedNormal(
            mu=mu,
            sigma=prior_sigma,
            pmin=prior_lower_bound,
            pmax=prior_upper_bound,
            size=num_params,
        )
    elif prior_type == "uniform":
        return parameter.Uniform(prior_lower_bound, prior_upper_bound, size=num_params)
    elif prior_type == "dm_dist_px_prior":
        return NE2001DMDist_Parameter(size=num_params)
    else:
        raise ValueError(
            "prior_type can only be uniform, bounded-normal, or dm_dist_px_prior, not ",
            prior_type,
        )


def filter_Mmat(psr, ltm_list=[]):
    """Filters the pulsar's design matrix of parameters
    :param psr: Pulsar object
    :param ltm_list: a list of parameters that will linearly varied, default is to vary anything not in tm_param_list
    :return: A new pulsar object with the filtered design matrix
    """
    idx_lin_pars = [psr.fitpars.index(p) for p in psr.fitpars if p in ltm_list]
    psr.fitpars = list(np.array(psr.fitpars)[idx_lin_pars])
    psr._designmatrix = psr._designmatrix[:, idx_lin_pars]
    return psr


# timing model delay
@signal_base.function
def tm_delay(t2pulsar, tm_params_orig, **kwargs):
    """
    Compute difference in residuals due to perturbed timing model.

    :param t2pulsar: libstempo pulsar object
    :param tm_params_orig: dictionary of TM parameter tuples, (val, err)

    :return: difference between new and old residuals in seconds
    """
    residuals = t2pulsar.residuals()
    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    for tm_scaled_key, tm_scaled_val in kwargs.items():
        if "DMX" in tm_scaled_key:
            tm_param = "_".join(tm_scaled_key.split("_")[-2:])
        else:
            tm_param = tm_scaled_key.split("_")[-1]

        if tm_param == "COSI":
            orig_params["SINI"] = np.longdouble(tm_params_orig["SINI"][0])
        else:
            orig_params[tm_param] = np.longdouble(tm_params_orig[tm_param][0])

        if "physical" in tm_params_orig[tm_param]:
            # User defined priors are assumed to not be scaled
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - tm_scaled_val**2)
                )
            else:
                tm_params_rescaled[tm_param] = np.longdouble(tm_scaled_val)
        else:
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                rescaled_COSI = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - rescaled_COSI**2)
                )
            else:
                tm_params_rescaled[tm_param] = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )

    # set to new values
    t2pulsar.vals(tm_params_rescaled)
    new_res = np.longdouble(t2pulsar.residuals().copy())

    # remeber to set values back to originals
    t2pulsar.vals(orig_params)

    # Return the time-series for the pulsar
    return new_res - residuals


# Model component building blocks #


def timing_block(
    psr,
    tm_param_list=["F0", "F1"],
    ltm_list=["Offset"],
    prior_type="uniform",
    prior_mu=0.0,
    prior_sigma=2.0,
    prior_lower_bound=-5.0,
    prior_upper_bound=5.0,
    tm_param_dict={},
    fit_remaining_pars=True,
    fixed_dict={},
    wideband_kwargs={},
):
    """
    Returns the timing model block of the model
    :param psr: a pulsar object on which to construct the timing model
    :param tm_param_list: a list of parameters to vary nonlinearly in the model
    :param ltm_list: a list of parameters to vary linearly in the model
    :param prior_type: the function used for the priors ['uniform','bounded-normal']
    :param prior_mu: the mean/central vlaue for the prior if ``prior_type`` is 'bounded-normal'
    :param prior_sigma: the sigma for the prior if ``prior_type`` is 'bounded-normal'
    :param prior_lower_bound: the lower bound for the prior
    :param prior_upper_bound: the upper bound for the prior
    :param tm_param_dict: a dictionary of physical parameters for nonlinearly varied timing model parameters, used to sample in non-sigma-scaled parameter space
    :param fit_remaining_pars: fits any timing model parameter in the linear regime if not in ``tm_param_list`` or ``tm_param_dict``
    :param fixed_dict: if not ``fit_remaining_pars``, the tm_params_orig dictionary for the parameters and value will be set to those in this dictionary.
    :param wideband_kwargs: extra kwargs for ``gp_signals.WidebandTimingModel``
    """
    # If param in tm_param_dict not in tm_param_list, add it
    for key in tm_param_dict.keys():
        if key not in tm_param_list:
            tm_param_list.append(key)

    physical_tm_priors = get_default_physical_tm_priors()

    # Get values and errors as pulled by libstempo from par file. Replace with fixed parameters if in fixed_dict
    psr.tm_params_orig = OrderedDict()
    for par, val, err in zip(
        psr.t2pulsar.pars(), psr.t2pulsar.vals(), psr.t2pulsar.errs()
    ):
        fixed = False
        for fixed_par in fixed_dict.keys():
            if par == fixed_par.split("_")[-1]:
                fixed = True
                print("par:", par, "parfile val:", val)
                print("par:", par, "fixed val:", fixed_dict[fixed_par])
                psr.tm_params_orig[par] = [fixed_dict[fixed_par], err, "normalized"]
        if not fixed:
            psr.tm_params_orig[par] = [val, err, "normalized"]
    """
    # Get values and errors as pulled by libstempo from par file.
    ptypes = ["normalized" for ii in range(len(psr.t2pulsar.pars()))]

    psr.tm_params_orig = OrderedDict(
        zip(
            psr.t2pulsar.pars(),
            map(
                list,
                zip(
                    np.longdouble(psr.t2pulsar.vals()),
                    np.longdouble(psr.t2pulsar.errs()),
                    ptypes,
                ),
            ),
        )
    )
    """
    # Check to see if nan or inf in pulsar parameter errors.
    # The refit will populate the incorrect errors, but sometimes
    # changes the values by too much, which is why it is done in this order.
    if np.any(np.isnan(psr.t2pulsar.errs())) or np.any(
        [err == 0.0 for err in psr.t2pulsar.errs()]
    ):
        eidxs = np.where(
            np.logical_or(np.isnan(psr.t2pulsar.errs()), psr.t2pulsar.errs() == 0.0)
        )[0]
        psr.t2pulsar.fit()
        for idx in eidxs:
            par = psr.t2pulsar.pars()[idx]
            psr.tm_params_orig[par][1] = np.longdouble(psr.t2pulsar.errs()[idx])

    tm_delay_kwargs = {}
    default_prior_params = [
        prior_mu,
        prior_sigma,
        prior_lower_bound,
        prior_upper_bound,
        prior_type,
    ]
    for par in tm_param_list:
        if par == "Offset":
            raise ValueError(
                "TEMPO2 does not support modeling the phase offset: 'Offset'."
            )
        elif par in tm_param_dict.keys():
            # Overwrite default priors if new ones defined for the parameter in tm_param_dict
            if par in psr.tm_params_orig.keys():
                psr.tm_params_orig[par][-1] = "physical"
                val, err, _ = psr.tm_params_orig[par]
            elif "COSI" in par and "SINI" in psr.tm_params_orig.keys():
                print("COSI added to tm_params_orig for to work with tm_delay.")
                sin_val, sin_err, _ = psr.tm_params_orig["SINI"]
                val = np.longdouble(np.sqrt(1 - sin_val**2))
                err = np.longdouble(
                    np.sqrt((sin_err * sin_val) ** 2 / (1 - sin_val**2))
                )
                # psr.tm_params_orig["SINI"][-1] = "physical"
                psr.tm_params_orig[par] = [val, err, "physical"]
            else:
                raise ValueError(par, "not in psr.tm_params_orig.")

            if "prior_mu" in tm_param_dict[par].keys():
                prior_mu = tm_param_dict[par]["prior_mu"]
            else:
                prior_mu = default_prior_params[0]
            if "prior_sigma" in tm_param_dict[par].keys():
                prior_sigma = tm_param_dict[par]["prior_sigma"]
            else:
                prior_sigma = default_prior_params[1]
            if "prior_lower_bound" in tm_param_dict[par].keys():
                prior_lower_bound = tm_param_dict[par]["prior_lower_bound"]
            else:
                prior_lower_bound = np.float(val + err * prior_lower_bound)
            if "prior_upper_bound" in tm_param_dict[par].keys():
                prior_upper_bound = tm_param_dict[par]["prior_upper_bound"]
            else:
                prior_upper_bound = np.float(val + err * prior_upper_bound)

            if "prior_type" in tm_param_dict[par].keys():
                prior_type = tm_param_dict[par]["prior_type"]
            else:
                prior_type = default_prior_params[4]
        else:
            prior_mu = default_prior_params[0]
            prior_sigma = default_prior_params[1]
            prior_lower_bound = default_prior_params[2]
            prior_upper_bound = default_prior_params[3]
            prior_type = default_prior_params[4]

        if par in physical_tm_priors.keys():
            if par in tm_param_dict.keys():
                if "pmin" in physical_tm_priors[par].keys():
                    if prior_lower_bound < physical_tm_priors[par]["pmin"]:
                        prior_lower_bound = physical_tm_priors[par]["pmin"]
                if "pmax" in physical_tm_priors[par].keys():
                    if prior_upper_bound > physical_tm_priors[par]["pmax"]:
                        prior_upper_bound = physical_tm_priors[par]["pmax"]
            else:
                if par in psr.tm_params_orig.keys():
                    val, err, _ = psr.tm_params_orig[par]
                else:
                    # Switch for sampling in COSI, but using SINI in libstempo
                    if "COSI" in par and "SINI" in psr.tm_params_orig.keys():
                        print("COSI added to tm_params_orig for to work with tm_delay.")
                        sin_val, sin_err, _ = psr.tm_params_orig["SINI"]
                        val = np.longdouble(np.sqrt(1 - sin_val**2))
                        err = np.longdouble(
                            np.sqrt((sin_err * sin_val) ** 2 / (1 - sin_val**2))
                        )
                        psr.tm_params_orig[par] = [val, err, "normalized"]
                    else:
                        raise ValueError("{} not in psr.tm_params_orig".format(par))

                if (
                    "pmin" in physical_tm_priors[par].keys()
                    and "pmax" in physical_tm_priors[par].keys()
                ):
                    if val + err * prior_lower_bound < physical_tm_priors[par]["pmin"]:
                        psr.tm_params_orig[par][-1] = "physical"
                        prior_lower_bound = physical_tm_priors[par]["pmin"]

                    if val + err * prior_upper_bound > physical_tm_priors[par]["pmax"]:
                        if psr.tm_params_orig[par][-1] != "physical":
                            psr.tm_params_orig[par][-1] = "physical"
                            # Need to change lower bound to a non-normed prior too
                            prior_lower_bound = np.float(val + err * prior_lower_bound)
                        prior_upper_bound = physical_tm_priors[par]["pmax"]
                    else:
                        if psr.tm_params_orig[par][-1] == "physical":
                            prior_upper_bound = np.float(val + err * prior_upper_bound)
                elif (
                    "pmin" in physical_tm_priors[par].keys()
                    or "pmax" in physical_tm_priors[par].keys()
                ):
                    if "pmin" in physical_tm_priors[par].keys():
                        if (
                            val + err * prior_lower_bound
                            < physical_tm_priors[par]["pmin"]
                        ):
                            psr.tm_params_orig[par][-1] = "physical"
                            prior_lower_bound = physical_tm_priors[par]["pmin"]
                            # Need to change lower bound to a non-normed prior too
                            prior_upper_bound = np.float(val + err * prior_upper_bound)
                    elif "pmax" in physical_tm_priors[par].keys():
                        if (
                            val + err * prior_upper_bound
                            > physical_tm_priors[par]["pmax"]
                        ):
                            psr.tm_params_orig[par][-1] = "physical"
                            prior_upper_bound = physical_tm_priors[par]["pmax"]
                            # Need to change lower bound to a non-normed prior too
                            prior_lower_bound = np.float(val + err * prior_lower_bound)

        tm_delay_kwargs[par] = get_prior(
            prior_type,
            prior_sigma,
            prior_lower_bound,
            prior_upper_bound,
            mu=prior_mu,
        )
    # timing model
    tm_func = tm_delay(**tm_delay_kwargs)
    tm = deterministic_signals.Deterministic(tm_func, name="timing_model")

    # filter design matrix of all but linear params
    if fit_remaining_pars:
        if not ltm_list:
            ltm_list = [p for p in psr.fitpars if p not in tm_param_list]
        filter_Mmat(psr, ltm_list=ltm_list)
        if any(["DMX" in x for x in ltm_list]) and wideband_kwargs:
            ltm = gp_signals.WidebandTimingModel(
                name="wideband_timing_model",
                **wideband_kwargs,
            )
        else:
            ltm = gp_signals.TimingModel(coefficients=False)
        tm += ltm

    return tm


def model_nltm(
    psr,
    tm_var=False,
    tm_linear=False,
    tm_param_list=[],
    ltm_list=[],
    tm_param_dict={},
    tm_prior="uniform",
    fit_remaining_pars=True,
    fixed_dict={},
    red_var=True,
    psd="powerlaw",
    red_select=None,
    noisedict=None,
    tm_svd=False,
    tm_norm=True,
    white_vary=True,
    components=30,
    upper_limit=False,
    is_wideband=False,
    use_dmdata=False,
    dmjump_var=False,
    gamma_val=None,
    dm_var=False,
    dm_type="gp",
    dmgp_kernel="diag",
    dm_psd="powerlaw",
    dm_nondiag_kernel="periodic",
    dmx_data=None,
    dm_annual=False,
    gamma_dm_val=None,
    chrom_gp=False,
    chrom_gp_kernel="nondiag",
    chrom_psd="powerlaw",
    chrom_idx=4,
    chrom_kernel="periodic",
    chrom_dt=15,
    dm_expdip=False,
    dmexp_sign="negative",
    dm_expdip_idx=2,
    dm_expdip_tmin=None,
    dm_expdip_tmax=None,
    num_dmdips=1,
    dmdip_seqname=None,
    dm_cusp=False,
    dm_cusp_sign="negative",
    dm_cusp_idx=2,
    dm_cusp_sym=False,
    dm_cusp_tmin=None,
    dm_cusp_tmax=None,
    num_dm_cusps=1,
    dm_dt=15,
    dm_cusp_seqname=None,
    dm_dual_cusp=False,
    dm_dual_cusp_tmin=None,
    dm_dual_cusp_tmax=None,
    dm_dual_cusp_sym=False,
    dm_dual_cusp_idx1=2,
    dm_dual_cusp_idx2=4,
    dm_dual_cusp_sign="negative",
    num_dm_dual_cusps=1,
    dm_dual_cusp_seqname=None,
    dm_sw_deter=False,
    dm_sw_gp=False,
    swgp_prior=None,
    swgp_basis=None,
    coefficients=False,
    extra_sigs=None,
    select="backend",
):
    """
    Single pulsar noise model
    :param psr: enterprise pulsar object
    :param tm_var: explicitly vary the timing model parameters
    :param tm_linear: vary the timing model in the linear approximation
    :param tm_param_list: an explicit list of timing model parameters to vary
    :param ltm_list: a list of parameters that will linearly varied, default is to vary anything not in tm_param_list
    :param tm_param_dict: a nested dictionary of parameters to vary in the model and their user defined values and priors
    :param tm_prior: prior type on varied timing model parameters {'uniform','bounded-normal'}
    :param fit_remaining_pars: boolean to switch combined non-linear + linear timing models on, only works for tm_var True
    :param fixed_dict: if not ``fit_remaining_pars``, the tm_params_orig dictionary for the parameters and value will be set to those in this dictionary.
    :param red var: include red noise in the model
    :param psd: red noise psd model
    :param noisedict: dictionary of noise parameters
    :param tm_svd: boolean for svd-stabilised timing model design matrix
    :param tm_norm: normalize the timing model, or provide custom normalization
    :param white_vary: boolean for varying white noise or keeping fixed
    :param components: number of modes in Fourier domain processes
    :param upper_limit: whether to do an upper-limit analysis
    :param is_wideband: whether input TOAs are wideband TOAs; will exclude
           ecorr from the white noise model
    :param use_dmdata: whether to use DM data (WidebandTimingModel) if
           is_wideband
    :param gamma_val: red noise spectral index to fix
    :param dm_var: whether to explicitly model DM-variations
    :param dm_type: gaussian process ('gp') or dmx ('dmx')
    :param dmgp_kernel: diagonal in frequency or non-diagonal
    :param dm_psd: power-spectral density of DM variations
    :param dm_nondiag_kernel: type of time-domain DM GP kernel
    :param dmx_data: supply the DMX data from par files
    :param dm_annual: include an annual DM signal
    :param dm_dt: linear_interp_basis size in units of days
    :param gamma_dm_val: spectral index of power-law DM variations
    :param chrom_gp: include general chromatic noise
    :param chrom_gp_kernel: GP kernel type to use in chrom ['diag','nondiag']
    :param chrom_psd: power-spectral density of chromatic noise
        ['powerlaw','tprocess','free_spectrum']
    :param chrom_idx: frequency scaling of chromatic noise
    :param chrom_kernel: Type of 'nondiag' time-domain chrom GP kernel to use
        ['periodic', 'sq_exp','periodic_rfband', 'sq_exp_rfband']
    :param chrom_dt: linear_interp_basis size in units of days
    :param dm_expdip: inclue a DM exponential dip
    :param dmexp_sign: set the sign parameter for dip
    :param dm_expdip_idx: chromatic index of exponential dip
    :param dm_expdip_tmin: sampling minimum of DM dip epoch
    :param dm_expdip_tmax: sampling maximum of DM dip epoch
    :param num_dmdips: number of dm exponential dips
    :param dmdip_seqname: name of dip sequence
    :param dm_cusp: include a DM exponential cusp
    :param dm_cusp_sign: set the sign parameter for cusp
    :param dm_cusp_idx: chromatic index of exponential cusp
    :param dm_cusp_tmin: sampling minimum of DM cusp epoch
    :param dm_cusp_tmax: sampling maximum of DM cusp epoch
    :param dm_cusp_sym: make exponential cusp symmetric
    :param num_dm_cusps: number of dm exponential cusps
    :param dm_cusp_seqname: name of cusp sequence
    :param dm_dual_cusp: include a DM cusp with two chromatic indices
    :param dm_dual_cusp_tmin: sampling minimum of DM dual cusp epoch
    :param dm_dual_cusp_tmax: sampling maximum of DM dual cusp epoch
    :param dm_dual_cusp_idx1: first chromatic index of DM dual cusp
    :param dm_dual_cusp_idx2: second chromatic index of DM dual cusp
    :param dm_dual_cusp_sym: make dual cusp symmetric
    :param dm_dual_cusp_sign: set the sign parameter for dual cusp
    :param num_dm_dual_cusps: number of DM dual cusps
    :param dm_dual_cusp_seqname: name of dual cusp sequence
    :param dm_scattering: whether to explicitly model DM scattering variations
    :param dm_sw_deter: use the deterministic solar wind model
    :param dm_sw_gp: add a Gaussian process perturbation to the deterministic
        solar wind model.
    :param swgp_prior: prior is currently set automatically
    :param swgp_basis: ['powerlaw', 'periodic', 'sq_exp']
    :param coefficients: explicitly include latent coefficients in model
    :param extra_sigs: Any additional `enterprise` signals to be added to the
        model.

    :return s: single pulsar noise model
    """
    amp_prior = "uniform" if upper_limit else "log-uniform"

    # timing model
    wideband_kwargs = {}
    if is_wideband and use_dmdata:
        if dmjump_var:
            wideband_kwargs["dmjump"] = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            wideband_kwargs["dmjump"] = parameter.Constant()
        if white_vary:
            wideband_kwargs["dmefac"] = parameter.Uniform(pmin=0.1, pmax=10.0)
            wideband_kwargs["log10_dmequad"] = parameter.Uniform(pmin=-7.0, pmax=0.0)
            # dmjump = parameter.Uniform(pmin=-0.005, pmax=0.005)
        else:
            wideband_kwargs["dmefac"] = parameter.Constant()
            wideband_kwargs["log10_dmequad"] = parameter.Constant()
            # dmjump = parameter.Constant()
        wideband_kwargs["dmefac_selection"] = selections.Selection(
            selections.by_backend
        )
        wideband_kwargs["log10_dmequad_selection"] = selections.Selection(
            selections.by_backend
        )
        wideband_kwargs["dmjump_selection"] = selections.Selection(
            selections.by_frontend
        )
    if not tm_var:
        if is_wideband and use_dmdata:
            s = gp_signals.WidebandTimingModel(**wideband_kwargs)
        else:
            s = gp_signals.TimingModel(
                use_svd=tm_svd, normed=tm_norm, coefficients=coefficients
            )
    else:
        if tm_linear:
            # create new attribute for enterprise pulsar object
            # UNSURE IF NECESSARY
            psr.tm_params_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
            for key in psr.tm_params_orig:
                psr.tm_params_orig[key] = (psr.t2pulsar[key].val, psr.t2pulsar[key].err)
            s = gp_signals.TimingModel(
                use_svd=tm_svd, normed=tm_norm, coefficients=coefficients
            )
        else:
            s = timing_block(
                psr,
                tm_param_list=tm_param_list,
                ltm_list=ltm_list,
                prior_type=tm_prior,
                prior_sigma=2.0,
                prior_lower_bound=-5.0,
                prior_upper_bound=5.0,
                tm_param_dict=tm_param_dict,
                fit_remaining_pars=fit_remaining_pars,
                fixed_dict=fixed_dict,
                wideband_kwargs=wideband_kwargs,
            )

    # red noise
    if red_var:
        s += red_noise_block(
            psd=psd,
            prior=amp_prior,
            components=components,
            gamma_val=gamma_val,
            coefficients=coefficients,
            select=red_select,
        )

    # DM variations
    if dm_var:
        if dm_type == "gp":
            if dmgp_kernel == "diag":
                s += dm_noise_block(
                    gp_kernel=dmgp_kernel,
                    psd=dm_psd,
                    prior=amp_prior,
                    components=components,
                    gamma_val=gamma_dm_val,
                    coefficients=coefficients,
                )
            elif dmgp_kernel == "nondiag":
                s += dm_noise_block(
                    gp_kernel=dmgp_kernel,
                    dm_dt=dm_dt,
                    nondiag_kernel=dm_nondiag_kernel,
                    coefficients=coefficients,
                )
        elif dm_type == "dmx":
            s += chrom.dmx_signal(dmx_data=dmx_data[psr.name])
        if dm_annual:
            s += chrom.dm_annual_signal()
        if chrom_gp:
            s += chromatic_noise_block(
                gp_kernel=chrom_gp_kernel,
                psd=chrom_psd,
                idx=chrom_idx,
                components=components,
                nondiag_kernel=chrom_kernel,
                coefficients=coefficients,
                chrom_dt=chrom_dt,
            )

        if dm_expdip:
            if dm_expdip_tmin is None and dm_expdip_tmax is None:
                tmin = [psr.toas.min() / 86400 for ii in range(num_dmdips)]
                tmax = [psr.toas.max() / 86400 for ii in range(num_dmdips)]
            else:
                tmin = (
                    dm_expdip_tmin
                    if isinstance(dm_expdip_tmin, list)
                    else [dm_expdip_tmin]
                )
                tmax = (
                    dm_expdip_tmax
                    if isinstance(dm_expdip_tmax, list)
                    else [dm_expdip_tmax]
                )
            if dmdip_seqname is not None:
                dmdipname_base = (
                    ["dmexp_" + nm for nm in dmdip_seqname]
                    if isinstance(dmdip_seqname, list)
                    else ["dmexp_" + dmdip_seqname]
                )
            else:
                dmdipname_base = [
                    "dmexp_{0}".format(ii + 1) for ii in range(num_dmdips)
                ]

            dm_expdip_idx = (
                dm_expdip_idx if isinstance(dm_expdip_idx, list) else [dm_expdip_idx]
            )
            for dd in range(num_dmdips):
                s += chrom.dm_exponential_dip(
                    tmin=tmin[dd],
                    tmax=tmax[dd],
                    idx=dm_expdip_idx[dd],
                    sign=dmexp_sign,
                    name=dmdipname_base[dd],
                )
        if dm_cusp:
            if dm_cusp_tmin is None and dm_cusp_tmax is None:
                tmin = [psr.toas.min() / 86400 for ii in range(num_dm_cusps)]
                tmax = [psr.toas.max() / 86400 for ii in range(num_dm_cusps)]
            else:
                tmin = (
                    dm_cusp_tmin if isinstance(dm_cusp_tmin, list) else [dm_cusp_tmin]
                )
                tmax = (
                    dm_cusp_tmax if isinstance(dm_cusp_tmmax, list) else [dm_cusp_tmax]
                )
            if dm_cusp_seqname is not None:
                cusp_name_base = "dm_cusp_" + dm_cusp_seqname + "_"
            else:
                cusp_name_base = "dm_cusp_"
            dm_cusp_idx = (
                dm_cusp_idx if isinstance(dm_cusp_idx, list) else [dm_cusp_idx]
            )
            for dd in range(1, num_dm_cusps + 1):
                s += chrom.dm_exponential_cusp(
                    tmin=tmin[dd - 1],
                    tmax=tmax[dd - 1],
                    idx=dm_cusp_idx,
                    sign=dm_cusp_sign,
                    symmetric=dm_cusp_sym,
                    name=cusp_name_base + str(dd),
                )
        if dm_dual_cusp:
            if dm_dual_cusp_tmin is None and dm_cusp_tmax is None:
                tmin = psr.toas.min() / 86400
                tmax = psr.toas.max() / 86400
            else:
                tmin = dm_dual_cusp_tmin
                tmax = dm_dual_cusp_tmax
            if dm_dual_cusp_seqname is not None:
                dual_cusp_name_base = "dm_dual_cusp_" + dm_cusp_seqname + "_"
            else:
                dual_cusp_name_base = "dm_dual_cusp_"
            for dd in range(1, num_dm_dual_cusps + 1):
                s += chrom.dm_dual_exp_cusp(
                    tmin=tmin,
                    tmax=tmax,
                    idx1=dm_dual_cusp_idx1,
                    idx2=dm_dual_cusp_idx2,
                    sign=dm_dual_cusp_sign,
                    symmetric=dm_dual_cusp_sym,
                    name=dual_cusp_name_base + str(dd),
                )
        if dm_sw_deter:
            Tspan = psr.toas.max() - psr.toas.min()
            s += solar_wind_block(
                ACE_prior=True,
                include_swgp=dm_sw_gp,
                swgp_prior=swgp_prior,
                swgp_basis=swgp_basis,
                Tspan=Tspan,
            )

    if extra_sigs is not None:
        s += extra_sigs
    # adding white-noise, and acting on psr objects
    if (
        "NANOGrav" in psr.flags["pta"] or "CHIME" in psr.flags["f"]
    ) and not is_wideband:
        s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True, select=select)
        model = s2(psr)
    else:
        s3 = s + white_noise_block(vary=white_vary, inc_ecorr=False, select=select)
        model = s3(psr)

    # set up PTA
    pta = signal_base.PTA([model])

    # set white noise parameters
    if not white_vary or (is_wideband and use_dmdata):
        if noisedict is None:
            print("No noise dictionary provided!...")
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta
