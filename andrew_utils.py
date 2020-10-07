# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import os, glob, ephem, json, copy
import numpy as np
from collections import defaultdict, OrderedDict
from enterprise.signals import parameter, signal_base, deterministic_signals, gp_signals

from enterprise_extensions import model_utils, sampler
from enterprise_extensions.blocks import (
    white_noise_block,
    red_noise_block,
    dm_noise_block,
    chromatic_noise_block,
    common_red_noise_block,
)
from scipy.stats import truncnorm


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


def get_timing_groups(pta):
    """Utility function to get parameter groups for timing sampling.
    These groups should be appended to the usual get_parameter_groups()
    output.
    """
    timing_pars = ["pos", "spin", "kep", "gr", "pm", "DMX"]

    groups = []
    for pars in timing_pars:
        group = sampler.group_from_params(pta, [pars])
        if len(group):
            groups.append(group)

    return groups


def get_default_physical_tm_priors():
    """
    "RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"
    "PMDEC", "PMRA", "PMELONG", "PMELAT", "PMRV", "PMBETA", "PMLAMBDA"
    "F", "F0", "F1", "F2", "P", "P1","PB","T0","A1","OM","EPS1","EPS2",
    "EPS1DOT","EPS2DOT","FB","MTOT","M2","XDOT","X2DOT","EDOT","H3",
    "H4","OMDOT","OM2DOT","XOMDOT","PBDOT","XPBDOT","GAMMA","PPNGAMMA",
    "DR","DTHETA"
    """
    default_tm_priors = {}
    default_tm_priors["E"] = {"pmin": 0.0, "pmax": 1.0}
    default_tm_priors["ECC"] = {"pmin": 0.0, "pmax": 1.0}
    default_tm_priors["SINI"] = {"pmin": 0.0, "pmax": 1.0}
    return


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
    else:
        raise ValueError(
            "prior_type can only be uniform or bounded-normal, not ", prior_type
        )


def get_par_errors(t2psr, par):
    """
    Prevents nans in errors for some pulsars

    :param psr: pulsar to pull error for
    :param par: parameter to pull error from par file
    """
    filename = t2psr.parfile.split("/")[-1]
    file = glob.glob("../../*/par/" + filename)[0]

    with open(file, "r") as f:
        for line in f.readlines():
            if par == "ELONG":
                # enterprise renames LAMBDA
                if line.split()[0] in [par, "LAMBDA"]:
                    error = ephem.degrees(line.split()[-1])
                    return error
            elif par == "ELAT":
                # enterprise renames BETA
                if line.split()[0] in [par, "BETA"]:
                    error = ephem.degrees(line.split()[-1])
                    return error
            else:
                if line.split()[0] == par:
                    error = ephem.degrees(line.split()[-1])
                    return error
                else:
                    raise ValueError(par, " not in file!")


def filter_Mmat(psr, ltm_exclude_list=[], exclude=True):
    """Filters the pulsar's design matrix of parameters
    :param psr: Pulsar object
    :param ltm_exclude_list: a list of parameters that will be excluded from being varied linearly
        if exlude is True; if exclude is False they are the only parameters to include in the linear model
    :param exclude: bool, whether to include or exlude parameters given in ltm_exclude_list

    :return: A new pulsar object with the filtered design matrix
    """
    if exclude:
        idx_lin_pars = [
            psr.fitpars.index(p) for p in psr.fitpars if p not in ltm_exclude_list
        ]
    else:
        idx_lin_pars = [
            psr.fitpars.index(p) for p in psr.fitpars if p in ltm_exclude_list
        ]
    print(len(psr.fitpars))
    psr.fitpars = list(np.array(psr.fitpars)[idx_lin_pars])
    print(len(psr.fitpars))
    print(psr.Mmat.shape)
    psr._designmatrix = psr._designmatrix[:, idx_lin_pars]
    print(psr.Mmat.shape)
    return psr


# timing model delay
@signal_base.function
def tm_delay(t2pulsar, tm_params_orig, tm_param_dict={}, **kwargs):
    """
    Compute difference in residuals due to perturbed timing model.

    :param residuals: original pulsar residuals from Pulsar object
    :param t2pulsar: libstempo pulsar object
    :param tm_params_orig: dictionary of TM parameter tuples, (val, err)
    :param tm_params: new timing model parameters, rescaled to be in sigmas
    :param which: option to have all or only named TM parameters varied

    :return: difference between new and old residuals in seconds
    """
    """OUTLINE:
    take in parameters in par file
    save to dictionary
    Based on params in input param list, set parameter prior distribution
    Feed the priors and param list into tm_delay function
    """
    residuals = t2pulsar.residuals()
    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    error_pos = {}
    for tm_scaled_key, tm_scaled_val in kwargs.items():
        tm_param = tm_scaled_key.split("_")[-1]
        orig_params[tm_param] = tm_params_orig[tm_param][0]

        if tm_param in tm_param_dict.keys():
            # User defined priors are assumed to not be scaled
            tm_params_rescaled[tm_param] = tm_scaled_val
        else:
            # Section because there are incorrect handlings of errors for ecliptic coordinates, idk why
            if tm_param in ["ELONG", "LAMBDA"]:
                error_pos["ELONG"] = get_par_errors(t2pulsar, tm_param)
            elif tm_param in ["ELAT", "BETA"]:
                error_pos["ELAT"] = get_par_errors(t2pulsar, tm_param)

            if tm_param in ["ELONG", "LAMBDA", "ELAT", "BETA"] and error_pos.keys() >= {
                "ELAT",
                "ELONG",
            }:
                ec_errors = ephem.Ecliptic(
                    error_pos["ELONG"]["err"], error_pos["ELAT"]["err"]
                )

                tm_params_rescaled["ELONG"] = (
                    tm_scaled_val * np.double(ec_errors.lon)
                    + tm_params_orig["ELONG"][0]
                )
                tm_params_rescaled["ELAT"] = (
                    tm_scaled_val * np.double(ec_errors.lat) + tm_params_orig["ELAT"][0]
                )
                # End of handling section
                """
                for key in error_pos.keys():
                    print(key,': ')
                    print(' Original Value: ', orig_params[key])
                    print(' normed_params: ', normed_params[error_pos[key]["param_iter"]])
                    if key == "ELONG":
                        print(' tm_param Errors: ',np.double(ec_errors.lon))
                    else:
                        print(' tm_param Errors: ',np.double(ec_errors.lat))
                    print(' tm_param Value: ',tm_params_orig[key][0])
                    print(' New Value: ',tm_params_rescaled[key])
                """
            else:
                tm_params_rescaled[tm_param] = (
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )
                """
                # Making sanity checks
                if tm_param in ["E", "ECC"]:  # ,"SINI"]:
                    if tm_params_rescaled[tm_param] <= 0.0:
                        tm_params_rescaled[tm_param] = 1e-9
                    elif tm_params_rescaled[tm_param] >= 1.0:
                        tm_params_rescaled[tm_param] = 1.0 - 1e-9
                if tm_param in ["PX"]:
                    if tm_params_rescaled[tm_param] <= 0.0:
                        tm_params_rescaled[tm_param] = 1e-9
                
                if tm_param not in ["ELONG","ELAT"]:
                    print(tm_param,': ')
                    print(' Original Value: ', orig_params[tm_param])
                    print(' normed_params: ', normed_params)
                    print(' tm_param Errors: ',tm_params_orig[tm_param][1])
                    print(' tm_param Value: ',tm_params_orig[tm_param][0])
                    print(' New Value: ',tm_params_rescaled[tm_param])
                """

    # set to new values
    t2pulsar.vals(tm_params_rescaled)
    new_res = t2pulsar.residuals()

    # remmeber to set values back to originals
    t2pulsar.vals(orig_params)

    # Return the time-series for the pulsar
    return new_res - residuals


# Model component building blocks #


def timing_block(
    tm_param_list=["RAJ", "DECJ", "F0", "F1", "PMRA", "PMDEC", "PX"],
    prior_type="uniform",
    prior_mu=0.0,
    prior_sigma=2.0,
    prior_lower_bound=-3.0,
    prior_upper_bound=3.0,
    tm_param_dict={},
):
    """
    Returns the timing model block of the model

    :param tm_param_list: a list of parameters to vary in the model
    :param prior_type: prior on timing parameters. Default is a bounded normal, can be "uniform"
    :param prior_sigma: Sets the center value on timing parameters for normal distribution draws
    :param prior_sigma: Sets the sigma on timing parameters for normal distribution draws
    :param prior_lower_bound: Sets the lower bound on timing parameters for bounded normal and uniform distribution draws
    :param prior_upper_bound: Sets the upper bound on timing parameters for bounded normal and uniform distribution draws
    :param tm_param_dict: a nested dictionary of parameters to vary in the model and their user defined values and priors:
        e.g. {'PX':{'prior_sigma':prior_sigma,'prior_lower_bound':prior_lower_bound,'prior_upper_bound':prior_upper_bound}}
        The priors cannot be normalized by sigma if there are uneven error bounds!
    """
    # If param in tm_param_dict not in tm_param_list, add it
    for key in tm_param_dict.keys():
        if key not in tm_param_list:
            tm_param_list.append(key)

    tm_delay_kwargs = {}
    default_prior_params = [prior_mu, prior_sigma, prior_lower_bound, prior_upper_bound]
    for par in tm_param_list:
        if par in tm_param_dict.keys():
            # Overwrite default priors if new ones defined for the parameter in tm_param_dict
            if "prior_mu" in tm_param_dict[par].keys():
                prior_mu = tm_param_dict[par]["prior_mu"]
            if "prior_sigma" in tm_param_dict[par].keys():
                prior_sigma = tm_param_dict[par]["prior_sigma"]
            if "prior_lower_bound" in tm_param_dict[par].keys():
                prior_lower_bound = tm_param_dict[par]["prior_lower_bound"]
            if "prior_upper_bound" in tm_param_dict[par].keys():
                prior_upper_bound = tm_param_dict[par]["prior_upper_bound"]
        else:
            prior_mu = default_prior_params[0]
            prior_sigma = default_prior_params[1]
            prior_lower_bound = default_prior_params[2]
            prior_upper_bound = default_prior_params[3]

        if par in ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]:
            key_string = "pos_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type,
                prior_sigma,
                prior_lower_bound,
                prior_upper_bound,
                mu=prior_mu,
            )

        elif par in [
            "PMDEC",
            "PMRA",
            "PMELONG",
            "PMELAT",
            "PMRV",
            "PMBETA",
            "PMLAMBDA",
        ]:
            key_string = "pm_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type,
                prior_sigma,
                prior_lower_bound,
                prior_upper_bound,
                mu=prior_mu,
            )
        elif par in ["F", "F0", "F1", "F2", "P", "P1"]:
            key_string = "spin_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type,
                prior_sigma,
                prior_lower_bound,
                prior_upper_bound,
                mu=prior_mu,
            )
        elif par in [
            "PB",
            "T0",
            "A1",
            "OM",
            "E",
            "ECC",
            "EPS1",
            "EPS2",
            "EPS1DOT",
            "EPS2DOT",
            "FB",
            "SINI",
            "MTOT",
            "M2",
            "XDOT",
            "X2DOT",
            "EDOT",
            "KOM",
            "KIN",
            "TASC",
        ]:
            key_string = "kep_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type,
                prior_sigma,
                prior_lower_bound,
                prior_upper_bound,
                mu=prior_mu,
            )
        elif par in [
            "H3",
            "H4",
            "OMDOT",
            "OM2DOT",
            "XOMDOT",
            "PBDOT",
            "XPBDOT",
            "GAMMA",
            "PPNGAMMA",
            "DR",
            "DTHETA",
        ]:
            key_string = "gr_param_" + par
            tm_delay_kwargs[key_string] = get_prior(
                prior_type,
                prior_sigma,
                prior_lower_bound,
                prior_upper_bound,
                mu=prior_mu,
            )
        else:
            if "DMX" in ["".join(list(x)[0:3]) for x in par.split("_")][0]:
                key_string = "dmx_param_" + par
                tm_delay_kwargs[key_string] = get_prior(
                    prior_type,
                    prior_sigma,
                    prior_lower_bound,
                    prior_upper_bound,
                    mu=prior_mu,
                )
            else:
                print(par, " is not currently a modelled parameter.")

    # timing model
    tm_func = tm_delay(tm_param_dict=tm_param_dict, **tm_delay_kwargs)
    tm = deterministic_signals.Deterministic(tm_func, name="non_linear_timing_model")

    return tm


def model_nltm(
    psrs,
    tm_var=False,
    tm_linear=False,
    tm_param_list=[],
    ltm_exclude_list=[],
    exclude=True,
    tm_param_dict={},
    tm_prior="uniform",
    nltm_plus_ltm=False,
    common_var=True,
    common_psd="powerlaw",
    red_psd="powerlaw",
    orf=None,
    common_components=30,
    red_components=30,
    dm_components=30,
    modes=None,
    wgts=None,
    logfreq=False,
    nmodes_log=10,
    noisedict=None,
    tm_svd=False,
    tm_norm=True,
    gamma_common=None,
    upper_limit=False,
    upper_limit_red=None,
    upper_limit_dm=None,
    upper_limit_common=None,
    bayesephem=False,
    be_type="orbel",
    wideband=False,
    dm_var=False,
    dm_type="gp",
    dm_psd="powerlaw",
    dm_annual=False,
    white_vary=False,
    gequad=False,
    dm_chrom=False,
    dmchrom_psd="powerlaw",
    dmchrom_idx=4,
    red_var=True,
    red_select=None,
    red_breakflat=False,
    red_breakflat_fq=None,
    coefficients=False,
    pshift=False,
):
    """
    Reads in list of enterprise Pulsar instance and returns a PTA
    instantiated with model 2A from the analysis paper:

    per pulsar:
        1. fixed EFAC per backend/receiver system
        2. fixed EQUAD per backend/receiver system
        3. fixed ECORR per backend/receiver system
        4. Red noise modeled as a power-law with 30 sampling frequencies
        5. Linear timing model.

    global:
        1.Common red noise modeled with user defined PSD with
        30 sampling frequencies. Available PSDs are
        ['powerlaw', 'turnover' 'spectrum']
        2. Optional physical ephemeris modeling.

    :param psd:
        PSD to use for common red noise signal. Available options
        are ['powerlaw', 'turnover' 'spectrum']. 'powerlaw' is default
        value.
    :param tm_var: explicitly vary the timing model parameters
    :param tm_linear: vary the timing model in the linear approximation
    :param tm_param_list: an explicit list of timing model parameters to vary
    :param ltm_exclude_list: a list of parameters that will be excluded from being varied linearly
        if exlude is True; if exclude is False they are the only parameters to include in the linear model
    :param exclude: bool, whether to include or exlude parameters given in ltm_exclude_list
    :param tm_param_dict: a nested dictionary of parameters to vary in the model and their user defined values and priors
    :param tm_prior: prior type on varied timing model parameters {'Uniform','bounded-normal'}
    :param nltm_plus_ltm: boolean to switch combined non-linear + linear timing models on, only works for tm_var True
    :param noisedict:
        Dictionary of pulsar noise properties. Can provide manually,
        or the code will attempt to find it.
    :param gamma_common:
        Fixed common red process spectral index value. By default we
        vary the spectral index over the range [0, 7].
    :param upper_limit:
        Perform upper limit on common red noise amplitude. By default
        this is set to False. Note that when perfoming upper limits it
        is recommended that the spectral index also be fixed to a specific
        value.
    :param bayesephem:
        Include BayesEphem model. Set to False by default
    :param be_type:
        orbel, orbel-v2, setIII
    :param wideband:
        Use wideband par and tim files. Ignore ECORR. Set to False by default.
    """

    amp_prior = "uniform" if upper_limit else "log-uniform"
    gp_priors = [upper_limit_red, upper_limit_dm, upper_limit_common]
    if all(ii is None for ii in gp_priors):
        amp_prior_red = amp_prior
        amp_prior_dm = amp_prior
        amp_prior_common = amp_prior
    else:
        amp_prior_red = "uniform" if upper_limit_red else "log-uniform"
        amp_prior_dm = "uniform" if upper_limit_dm else "log-uniform"
        amp_prior_common = "uniform" if upper_limit_common else "log-uniform"

    # timing model
    if not tm_var:
        s = gp_signals.TimingModel(
            use_svd=tm_svd, normed=tm_norm, coefficients=coefficients
        )
    else:
        # create new attribute for enterprise pulsar object
        for p in psrs:
            p.tm_params_orig = OrderedDict.fromkeys(p.t2pulsar.pars())
            for key in p.tm_params_orig:
                p.tm_params_orig[key] = (p.t2pulsar[key].val, p.t2pulsar[key].err)
        if tm_linear and not nltm_plus_ltm:
            s = gp_signals.TimingModel(
                use_svd=tm_svd, normed=tm_norm, coefficients=coefficients
            )
        elif not tm_linear and not nltm_plus_ltm:
            s = timing_block(
                tm_param_list=tm_param_list,
                prior_type=tm_prior,
                prior_sigma=2.0,
                prior_lower_bound=-3.0,
                prior_upper_bound=3.0,
                tm_param_dict=tm_param_dict,
            )
        elif not tm_linear and nltm_plus_ltm:
            # Don't need this catch!
            if len(ltm_exclude_list) == 0:
                ltm_exclude_list = tm_param_list
            psrs2 = [
                filter_Mmat(psr, ltm_exclude_list=ltm_exclude_list, exclude=exclude)
                for psr in psrs
            ]
            s = gp_signals.TimingModel(
                use_svd=tm_svd, normed=tm_norm, coefficients=coefficients
            )
            s += timing_block(
                tm_param_list=tm_param_list,
                prior_type=tm_prior,
                prior_sigma=2.0,
                prior_lower_bound=-3.0,
                prior_upper_bound=3.0,
                tm_param_dict=tm_param_dict,
            )
        else:
            pass

    # find the maximum time span to set GW frequency sampling
    Tspan = model_utils.get_tspan(psrs)

    if logfreq:
        fmin = 10.0
        modes, wgts = model_utils.linBinning(
            Tspan, nmodes_log, 1.0 / fmin / Tspan, common_components, nmodes_log
        )
        wgts = wgts ** 2.0

    if red_var:
        # red noise
        s += red_noise_block(
            psd=red_psd,
            prior=amp_prior_red,
            Tspan=Tspan,
            components=red_components,
            modes=modes,
            wgts=wgts,
            coefficients=coefficients,
            select=red_select,
            break_flat=red_breakflat,
            break_flat_fq=red_breakflat_fq,
        )

    if common_var:
        # common red noise block
        if orf is None:
            s += common_red_noise_block(
                psd=common_psd,
                prior=amp_prior_common,
                Tspan=Tspan,
                components=common_components,
                coefficients=coefficients,
                gamma_val=gamma_common,
                name="gw",
            )
        elif orf == "hd":
            s += common_red_noise_block(
                psd=common_psd,
                prior=amp_prior_common,
                Tspan=Tspan,
                components=common_components,
                coefficients=coefficients,
                gamma_val=gamma_common,
                orf="hd",
                name="gw",
            )

    # DM variations
    if dm_var:
        if dm_type == "gp":
            s += dm_noise_block(
                gp_kernel="diag",
                psd=dm_psd,
                prior=amp_prior_dm,
                components=dm_components,
                gamma_val=None,
                coefficients=coefficients,
            )
        if dm_annual:
            s += chrom.dm_annual_signal()
        if dm_chrom:
            s += chromatic_noise_block(
                psd=dmchrom_psd,
                idx=dmchrom_idx,
                name="chromatic",
                components=dm_components,
                coefficients=coefficients,
            )

    # ephemeris model
    if bayesephem:
        s += deterministic_signals.PhysicalEphemerisSignal(
            use_epoch_toas=True, model=be_type
        )
    fixed_bayesephem = False
    if fixed_bayesephem:
        # TODO: Allow for fixed ephem params in enterprise.deterministic_signals
        # s += deterministic_signals.PhysicalEphemerisSignal(
        #    use_epoch_toas=True, model=be_type
        # )
        frame_drift_rate = parameter.Uniform(-1e-9, 1e-9)("frame_drift_rate")

        d_jupiter_mass = parameter.Normal(0, 1.54976690e-11)("d_jupiter_mass")

        d_saturn_mass = parameter.Normal(0, 8.17306184e-12)("d_saturn_mass")

        d_uranus_mass = parameter.Normal(0, 5.71923361e-11)("d_uranus_mass")

        d_neptune_mass = parameter.Normal(0, 7.96103855e-11)("d_neptune_mass")

        jup_orb_elements = parameter.Uniform(-0.05, 0.05, size=6)("jup_orb_elements")

        sat_orb_elements = parameter.Uniform(-0.5, 0.5, size=6)("sat_orb_elements")

        # note: default prior for dynamical model is Uniform(-1e-4, 1e-4)
        #       for each element.

        times, jup_orbit, sat_orbit = utils.get_planet_orbital_elements(model)
        wf = utils.physical_ephem_delay(
            frame_drift_rate=frame_drift_rate,
            d_jupiter_mass=d_jupiter_mass,
            d_saturn_mass=d_saturn_mass,
            d_uranus_mass=d_uranus_mass,
            d_neptune_mass=d_neptune_mass,
            jup_orb_elements=jup_orb_elements,
            sat_orb_elements=sat_orb_elements,
            times=times,
            jup_orbit=jup_orbit,
            sat_orbit=sat_orbit,
        )

    # adding white-noise, and acting on psr objects
    models = []
    if tm_var and not tm_linear and nltm_plus_ltm:
        for p in psrs2:
            if "NANOGrav" in p.flags["pta"] and not wideband:
                s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
                if gequad:
                    s2 += white_signals.EquadNoise(
                        log10_equad=parameter.Uniform(-8.5, -5),
                        selection=selections.Selection(selections.no_selection),
                        name="gequad",
                    )
                if "1713" in p.name and dm_var:
                    tmin = p.toas.min() / 86400
                    tmax = p.toas.max() / 86400
                    s3 = s2 + chrom.dm_exponential_dip(
                        tmin=tmin, tmax=tmax, idx=2, sign=False, name="dmexp"
                    )
                    models.append(s3(p))
                else:
                    models.append(s2(p))
            else:
                s4 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
                if gequad:
                    s4 += white_signals.EquadNoise(
                        log10_equad=parameter.Uniform(-8.5, -5),
                        selection=selections.Selection(selections.no_selection),
                        name="gequad",
                    )
                if "1713" in p.name and dm_var:
                    tmin = p.toas.min() / 86400
                    tmax = p.toas.max() / 86400
                    s5 = s4 + chrom.dm_exponential_dip(
                        tmin=tmin, tmax=tmax, idx=2, sign=False, name="dmexp"
                    )
                    models.append(s5(p))
                else:
                    models.append(s4(p))
    else:
        for p in psrs:
            if "NANOGrav" in p.flags["pta"] and not wideband:
                s2 = s + white_noise_block(vary=white_vary, inc_ecorr=True)
                if gequad:
                    s2 += white_signals.EquadNoise(
                        log10_equad=parameter.Uniform(-8.5, -5),
                        selection=selections.Selection(selections.no_selection),
                        name="gequad",
                    )
                if "1713" in p.name and dm_var:
                    tmin = p.toas.min() / 86400
                    tmax = p.toas.max() / 86400
                    s3 = s2 + chrom.dm_exponential_dip(
                        tmin=tmin, tmax=tmax, idx=2, sign=False, name="dmexp"
                    )
                    models.append(s3(p))
                else:
                    models.append(s2(p))
            else:
                s4 = s + white_noise_block(vary=white_vary, inc_ecorr=False)
                if gequad:
                    s4 += white_signals.EquadNoise(
                        log10_equad=parameter.Uniform(-8.5, -5),
                        selection=selections.Selection(selections.no_selection),
                        name="gequad",
                    )
                if "1713" in p.name and dm_var:
                    tmin = p.toas.min() / 86400
                    tmax = p.toas.max() / 86400
                    s5 = s4 + chrom.dm_exponential_dip(
                        tmin=tmin, tmax=tmax, idx=2, sign=False, name="dmexp"
                    )
                    models.append(s5(p))
                else:
                    models.append(s4(p))

    # set up PTA
    pta = signal_base.PTA(models)

    # set white noise parameters
    if not white_vary:
        if noisedict is None:
            print("No noise dictionary provided!...")
        else:
            noisedict = noisedict
            pta.set_default_params(noisedict)

    return pta


def get_noise_from_file(noisefile, ecorr_switch=False):
    """Gets noise parameters from file that is in a list and
    converts it to a dictionary of parameters

    Parameters
    ----------
    noisefile : str
        The location of the particular noisefile to convert, (ie. '/path/to/file/psrname_rest_of_filename.txt')

    Returns
    -------
    params : dict
        The dictionary of parameters corresponding to the particular pulsar

    """
    psrname = noisefile.split("/")[-1].split("_noise.txt")[0]
    fin = open(noisefile, "r")
    lines = fin.readlines()
    params = {}
    for line in lines:
        ln = line.split()
        if "efac" in line:
            par = "efac"
            flag = ln[0].split("efac-")[-1]
        elif "equad" in line:
            par = "log10_equad"
            flag = ln[0].split("equad-")[-1]
        elif "jitter_q" in line:
            par = "log10_ecorr"
            flag = ln[0].split("jitter_q-")[-1]
            if ecorr_switch:
                flag = "basis_ecorr_" + flag
        elif "RN-Amplitude" in line:
            par = "red_noise_log10_A"
            flag = ""
        elif "RN-spectral-index" in line:
            par = "red_noise_gamma"
            flag = ""
        else:
            break
        if flag:
            name = [psrname, flag, par]
        else:
            name = [psrname, par]
        pname = "_".join(name)
        params.update({pname: float(ln[1])})
    return params
