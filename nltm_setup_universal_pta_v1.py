import numpy as np
import glob, os, sys, pickle, json, inspect
from collections import OrderedDict

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
# top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
e_path = top_dir + "/enterprise/"
ptmcmc_path = top_dir + "/PTMCMCSampler"

sys.path.insert(0, ptmcmc_path)
sys.path.insert(0, e_path)
sys.path.insert(0, e_e_path)

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import utils
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import gp_signals

import enterprise_extensions as e_e
from enterprise_extensions import sampler
from enterprise_extensions import models
from enterprise_extensions.sampler import (
    JumpProposal,
    get_parameter_groups,
    get_timing_groups,
    group_from_params,
)
from enterprise_extensions.timing import timing_block
from enterprise_extensions.blocks import channelized_backends


def pta_setup(
    psr,
    datarelease,
    psr_name,
    tm_var=True,
    red_var=False,
    white_var=True,
    fit_remaining_pars=True,
    lin_dmx_jump_fd=True,
    wideband=False,
    fixed_remaining_pars=False,
    sample_cos=True,
    tm_linear=False,
    tm_prior="uniform",
    incTimingModel=True,
    Ecorr_gp_basis=False,
    pal2_priors=True,
):

    nltm_params = []
    ltm_list = []
    fixed_list = []
    tm_param_dict = {}
    for par in psr.fitpars:
        if par == "Offset":
            ltm_list.append(par)
        elif "DMX" in par and any([lin_dmx_jump_fd, wideband, fixed_remaining_pars]):
            if fixed_remaining_pars:
                fixed_list.append(par)
            else:
                ltm_list.append(par)
        elif "JUMP" in par and any([lin_dmx_jump_fd, fixed_remaining_pars]):
            if fixed_remaining_pars:
                fixed_list.append(par)
            else:
                ltm_list.append(par)
        elif "FD" in par and any([lin_dmx_jump_fd, fixed_remaining_pars]):
            if fixed_remaining_pars:
                fixed_list.append(par)
            else:
                ltm_list.append(par)
        elif par == "SINI" and sample_cos:
            nltm_params.append("COSI")
        else:
            nltm_params.append(par)

        if par in ["PBDOT", "XDOT"] and hasattr(psr, "t2pulsar"):
            par_val = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
            par_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
            if np.log10(par_sigma) > -10.0:
                print(
                    f"USING PHYSICAL {par}. Val: ", par_val, "Err: ", par_sigma * 1e-12
                )
                lower = par_val - 50 * par_sigma * 1e-12
                upper = par_val + 50 * par_sigma * 1e-12
                # lower = pbdot - 5 * pbdot_sigma * 1e-12
                # upper = pbdot + 5 * pbdot_sigma * 1e-12
                tm_param_dict[par] = {
                    "prior_mu": par_val,
                    "prior_sigma": par_sigma * 1e-12,
                    "prior_lower_bound": lower,
                    "prior_upper_bound": upper,
                }

        elif par == "SINI" and datarelease == "5yr" and psr_name == "J1640+2224":
            # Use the priors from Vigeland and Vallisneri 2014
            if hasattr(psr, "t2pulsar"):
                sini_mu = np.double(
                    psr.t2pulsar.vals()[psr.t2pulsar.pars().index("SINI")]
                )
                sini_err = np.double(
                    psr.t2pulsar.errs()[psr.t2pulsar.pars().index("SINI")]
                )
            elif hasattr(psr, "model"):
                sini_mu = np.double(getattr(psr.model, par).value)
                sini_err = np.double(getattr(psr.model, par).uncertainty_value)
                print(sini_mu, sini_err)
            if sample_cos:
                cosi_mu = np.sqrt(1 - sini_mu**2)
                cosi_err = np.double(
                    np.sqrt((sini_err * sini_mu) ** 2 / (1 - sini_mu**2))
                )
                tm_param_dict["COSI"] = {
                    "prior_mu": cosi_mu,
                    "prior_sigma": cosi_err,
                    "prior_lower_bound": 0.0,
                    "prior_upper_bound": 1.0,
                }
            else:
                tm_param_dict["SINI"] = {
                    "prior_mu": sini_mu,
                    "prior_sigma": sini_err,
                    "prior_lower_bound": 0.0,
                    "prior_upper_bound": 1.0,
                }
        elif par == "PX" and datarelease == "5yr" and psr_name == "J1640+2224":
            # Use the priors from Vigeland and Vallisneri 2014
            if hasattr(psr, "t2pulsar"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(
                        psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_sigma": np.double(
                        psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_type": "dm_dist_px_prior",
                }
            elif hasattr(psr, "model"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(getattr(psr.model, par).value),
                    "prior_sigma": np.double(getattr(psr.model, par).uncertainty_value),
                    "prior_type": "dm_dist_px_prior",
                }

        elif par == "M2" and datarelease == "5yr" and psr_name == "J1640+2224":
            # Use the priors from Vigeland and Vallisneri 2014
            if hasattr(psr, "t2pulsar"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(
                        psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_sigma": np.double(
                        psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_lower_bound": 1e-10,
                    "prior_upper_bound": 10.0,
                }
            elif hasattr(psr, "model"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(getattr(psr.model, par).value),
                    "prior_sigma": np.double(getattr(psr.model, par).uncertainty_value),
                    "prior_lower_bound": 1e-10,
                    "prior_upper_bound": 10.0,
                }

    if not tm_linear and tm_var:
        print(
            "Non-linearly varying these values: ",
            nltm_params,
            "\n in pulsar ",
            psr_name,
        )
    elif tm_linear and tm_var:
        print("Using linear approximation for all timing parameters.")
    else:
        print("Not varying timing parameters.")

    if fit_remaining_pars and tm_var:
        print("Linearly varying these values: ", ltm_list)

    if fixed_remaining_pars:
        print("Fixing these parameters: ", fixed_list)

    print("Using ", tm_prior, " prior.")
    """
    # define selection by observing backend
    if datarelease == "5yr":
        s = timing_block(
            psr,
            tm_param_list=nltm_params,
            ltm_list=ltm_list,
            prior_type=tm_prior,
            prior_sigma=2.0,
            prior_lower_bound=-5.0,
            prior_upper_bound=5.0,
            tm_param_dict=tm_param_dict,
            fit_remaining_pars=fit_remaining_pars,
        )
        select = "none"
        if select == "backend":
            backend = selections.Selection(selections.by_backend)
        else:
            # define no selection
            backend = selections.Selection(selections.no_selection)
        # white noise parameters
        efac = parameter.Uniform(0.01, 10.0)

        # white noise signals
        s += white_signals.MeasurementNoise(efac=efac, selection=backend, name=None)

        model = s(psr)

        # set up PTA
        pta = signal_base.PTA([model])
    else:
    """
    if tm_var and not tm_linear:
        if pal2_priors:
            s = timing_block(
                psr,
                tm_param_list=nltm_params,
                ltm_list=ltm_list,
                prior_type=tm_prior,
                prior_sigma=2.0,
                prior_lower_bound=-500.0,
                prior_upper_bound=500.0,
                tm_param_dict=tm_param_dict,
                fit_remaining_pars=fit_remaining_pars,
                wideband_kwargs={},
            )

            # red noise
            if red_var:
                s += red_noise_block(
                    psd="powerlaw",
                    prior="uniform",
                    components=30,
                    gamma_val=None,
                    coefficients=False,
                    select=None,
                )
            # define selection by observing backend
            backend = selections.Selection(selections.by_backend)
            # define selection by nanograv backends
            backend_ng = selections.Selection(selections.nanograv_backends)
            backend_ch = selections.Selection(channelized_backends)

            # white noise parameters
            efac = parameter.Uniform(0.001, 10.0)
            equad = parameter.Uniform(-10.0, -4.0)
            ecorr = parameter.Uniform(-8.5, -4.0)

            # white noise signals
            ef = white_signals.MeasurementNoise(efac=efac, selection=backend, name=None)
            eq = white_signals.EquadNoise(
                log10_equad=equad, selection=backend, name=None
            )
            if Ecorr_gp_basis:
                ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=backend_ch)
            else:
                ec = white_signals.EcorrKernelNoise(
                    log10_ecorr=ecorr, selection=backend_ch
                )

            # combine signals
            s += ef + eq + ec
            model = s(psr)

            # set up PTA
            pta = signal_base.PTA([model])
        else:
            model_args = inspect.getfullargspec(models.model_singlepsr_noise)
            model_keys = model_args[0][1:]
            model_vals = model_args[3]
            model_kwargs = dict(zip(model_keys, model_vals))
            model_kwargs.update(
                {
                    "tm_var": tm_var,
                    "tm_linear": tm_linear,
                    "tm_param_list": nltm_params,
                    "ltm_list": ltm_list,
                    "tm_param_dict": tm_param_dict,
                    "tm_prior": tm_prior,
                    "normalize_prior_bound": 50.0,
                    "fit_remaining_pars": fit_remaining_pars,
                    "red_var": red_var,
                    "noisedict": None,
                    "white_vary": white_var,
                    "is_wideband": wideband,
                    "use_dmdata": wideband,
                    "dmjump_var": wideband,
                    "coefficients": False,
                }
            )
            # print(model_kwargs)

            pta = models.model_singlepsr_noise(psr, **model_kwargs)
    else:
        if incTimingModel:
            # create new attribute for enterprise pulsar object
            # UNSURE IF NECESSARY
            psr.tm_params_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
            for key in psr.tm_params_orig:
                psr.tm_params_orig[key] = (
                    psr.t2pulsar[key].val,
                    psr.t2pulsar[key].err,
                )
            s = gp_signals.TimingModel(use_svd=False, normed=True, coefficients=False)

        # define selection by observing backend
        backend = selections.Selection(selections.by_backend)
        # define selection by nanograv backends
        backend_ng = selections.Selection(selections.nanograv_backends)
        backend_ch = selections.Selection(channelized_backends)

        # white noise parameters
        if pal2_priors:
            efac = parameter.Uniform(0.001, 10.0)
            equad = parameter.Uniform(-10.0, -4.0)
            ecorr = parameter.Uniform(-8.5, -4.0)
        else:
            efac = parameter.Uniform(0.01, 10.0)
            equad = parameter.Uniform(-8.5, -5.0)
            ecorr = parameter.Uniform(-8.5, -5.0)

        # white noise signals
        ef = white_signals.MeasurementNoise(efac=efac, selection=backend, name=None)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=backend, name=None)
        if Ecorr_gp_basis:
            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=backend_ch)
        else:
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=backend_ch)

        # combine signals
        if incTimingModel:
            s += ef + eq + ec
        else:
            s = ef + eq + ec

        model = s(psr)

        # set up PTA
        pta = signal_base.PTA([model])
    return pta


def get_tm_param_dict(psr, datarelease, psr_name):
    tm_param_dict = {}
    for par in psr.fitpars:
        if par in ["PBDOT", "XDOT"] and hasattr(psr, "t2pulsar"):
            par_val = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
            par_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
            if np.log10(par_sigma) > -10.0:
                print(
                    f"USING PHYSICAL {par}. Val: ", par_val, "Err: ", par_sigma * 1e-12
                )
                lower = par_val - 50 * par_sigma * 1e-12
                upper = par_val + 50 * par_sigma * 1e-12
                # lower = pbdot - 5 * pbdot_sigma * 1e-12
                # upper = pbdot + 5 * pbdot_sigma * 1e-12
                tm_param_dict[par] = {
                    "prior_mu": par_val,
                    "prior_sigma": par_sigma * 1e-12,
                    "prior_lower_bound": lower,
                    "prior_upper_bound": upper,
                }

        elif par == "SINI" and datarelease == "5yr" and psr_name == "J1640+2224":
            # Use the priors from Vigeland and Vallisneri 2014
            if hasattr(psr, "t2pulsar"):
                sini_mu = np.double(
                    psr.t2pulsar.vals()[psr.t2pulsar.pars().index("SINI")]
                )
                sini_err = np.double(
                    psr.t2pulsar.errs()[psr.t2pulsar.pars().index("SINI")]
                )
            elif hasattr(psr, "model"):
                sini_mu = np.double(getattr(psr.model, par).value)
                sini_err = np.double(getattr(psr.model, par).uncertainty_value)
                print(sini_mu, sini_err)
            if sample_cos:
                cosi_mu = np.sqrt(1 - sini_mu**2)
                cosi_err = np.double(
                    np.sqrt((sini_err * sini_mu) ** 2 / (1 - sini_mu**2))
                )
                tm_param_dict["COSI"] = {
                    "prior_mu": cosi_mu,
                    "prior_sigma": cosi_err,
                    "prior_lower_bound": 0.0,
                    "prior_upper_bound": 1.0,
                }
            else:
                tm_param_dict["SINI"] = {
                    "prior_mu": sini_mu,
                    "prior_sigma": sini_err,
                    "prior_lower_bound": 0.0,
                    "prior_upper_bound": 1.0,
                }
        elif par == "PX" and datarelease == "5yr" and psr_name == "J1640+2224":
            # Use the priors from Vigeland and Vallisneri 2014
            if hasattr(psr, "t2pulsar"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(
                        psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_sigma": np.double(
                        psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_type": "dm_dist_px_prior",
                }
            elif hasattr(psr, "model"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(getattr(psr.model, par).value),
                    "prior_sigma": np.double(getattr(psr.model, par).uncertainty_value),
                    "prior_type": "dm_dist_px_prior",
                }

        elif par == "M2" and datarelease == "5yr" and psr_name == "J1640+2224":
            # Use the priors from Vigeland and Vallisneri 2014
            if hasattr(psr, "t2pulsar"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(
                        psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_sigma": np.double(
                        psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)]
                    ),
                    "prior_lower_bound": 1e-10,
                    "prior_upper_bound": 10.0,
                }
            elif hasattr(psr, "model"):
                tm_param_dict[par] = {
                    "prior_mu": np.double(getattr(psr.model, par).value),
                    "prior_sigma": np.double(getattr(psr.model, par).uncertainty_value),
                    "prior_lower_bound": 1e-10,
                    "prior_upper_bound": 10.0,
                }
    return tm_param_dict


def get_initial_sample(psr, pta, tm_param_dict, zero_start=True):
    if zero_start:
        x0_list = []
        for p in pta.params:
            if "timing" in p.name:
                if "DMX" in p.name:
                    p_name = ("_").join(p.name.split("_")[-2:])
                else:
                    p_name = p.name.split("_")[-1]
                if psr.tm_params_orig[p_name][-1] == "normalized":
                    x0_list.append(np.double(0.0))
                else:
                    if p_name in tm_param_dict.keys():
                        x0_list.append(np.double(tm_param_dict[p_name]["prior_mu"]))
                    else:
                        x0_list.append(np.double(psr.tm_params_orig[p_name][0]))
            else:
                x0_list.append(p.sample())
        x0 = np.asarray(x0_list)
    else:
        x0 = np.hstack([p.sample() for p in pta.params])

    return x0


def setup_sampler(
    pta,
    outdir="chains",
    resume=False,
    empirical_distr=None,
    groups=None,
    human=None,
    save_ext_dists=False,
    timing=False,
    psr=None,
    loglkwargs={},
    logpkwargs={},
):
    """
    Sets up an instance of PTMCMC sampler.

    We initialize the sampler the likelihood and prior function
    from the PTA object. We set up an initial jump covariance matrix
    with fairly small jumps as this will be adapted as the MCMC runs.

    We will setup an output directory in `outdir` that will contain
    the chain (first n columns are the samples for the n parameters
    and last 4 are log-posterior, log-likelihood, acceptance rate, and
    an indicator variable for parallel tempering but it doesn't matter
    because we aren't using parallel tempering).

    We then add several custom jump proposals to the mix based on
    whether or not certain parameters are in the model. These are
    all either draws from the prior distribution of parameters or
    draws from uniform distributions.

    save_ext_dists: saves distributions that have been extended to
    cover priors as a pickle to the outdir folder. These can then
    be loaded later as distributions to save a minute at the start
    of the run.
    """

    # dimension of parameter space
    params = pta.param_names
    ndim = len(params)

    # initial jump covariance matrix
    cov = np.diag(np.ones(ndim) * 0.1**2)

    # parameter groupings
    if groups is None:
        groups = get_parameter_groups(pta)

    if timing:
        groups.extend(get_timing_groups(pta))
        groups.append(
            group_from_params(
                pta,
                [
                    x
                    for x in pta.param_names
                    if any(y in x for y in ["timing_model", "ecorr"])
                ],
            )
        )

    sampler = ptmcmc(
        ndim,
        pta.get_lnlikelihood,
        pta.get_lnprior,
        cov,
        groups=groups,
        outDir=outdir,
        resume=resume,
        loglkwargs=loglkwargs,
        logpkwargs=logpkwargs,
    )

    # additional jump proposals
    jp = JumpProposal(
        pta,
        empirical_distr=empirical_distr,
        save_ext_dists=save_ext_dists,
        outdir=outdir,
        timing=timing,
        psr=psr,
        sampler=sampler,
    )
    sampler.jp = jp

    # always add draw from prior
    sampler.addProposalToCycle(jp.draw_from_prior, 15)

    # try adding empirical proposals
    if empirical_distr is not None:
        print("Attempting to add empirical proposals...\n")
        sampler.addProposalToCycle(jp.draw_from_empirical_distr, 30)

    # Red noise prior draw
    if "red noise" in jp.snames:
        print("Adding red noise prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

    # DM GP noise prior draw
    if "dm_gp" in jp.snames:
        print("Adding DM GP noise prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 10)

    # DM annual prior draw
    if "dm_s1yr" in jp.snames:
        print("Adding DM annual prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dm1yr_prior, 10)

    # DM dip prior draw
    if "dmexp" in jp.snames:
        print("Adding DM exponential dip prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dmexpdip_prior, 10)

    # DM cusp prior draw
    if "dm_cusp" in jp.snames:
        print("Adding DM exponential cusp prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dmexpcusp_prior, 10)

    # DMX prior draw
    if "dmx_signal" in jp.snames:
        print("Adding DMX prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dmx_prior, 10)

    # Ephemeris prior draw
    if "d_jupiter_mass" in pta.param_names:
        print("Adding ephemeris model prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

    # GWB uniform distribution draw
    if np.any([("gw" in par and "log10_A" in par) for par in pta.param_names]):
        print("Adding GWB uniform distribution draws...\n")
        sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

    # Dipole uniform distribution draw
    if "dipole_log10_A" in pta.param_names:
        print("Adding dipole uniform distribution draws...\n")
        sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

    # Monopole uniform distribution draw
    if "monopole_log10_A" in pta.param_names:
        print("Adding monopole uniform distribution draws...\n")
        sampler.addProposalToCycle(jp.draw_from_monopole_log_uniform_distribution, 10)

    # Altpol uniform distribution draw
    if "log10Apol_tt" in pta.param_names:
        print("Adding alternative GW-polarization uniform distribution draws...\n")
        sampler.addProposalToCycle(jp.draw_from_altpol_log_uniform_distribution, 10)

    # BWM prior draw
    if "bwm_log10_A" in pta.param_names:
        print("Adding BWM prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

    # FDM prior draw
    if "fdm_log10_A" in pta.param_names:
        print("Adding FDM prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_fdm_prior, 10)

    # CW prior draw
    if "cw_log10_h" in pta.param_names:
        print("Adding CW strain prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)
    if "cw_log10_Mc" in pta.param_names:
        print("Adding CW prior draws...\n")
        sampler.addProposalToCycle(jp.draw_from_cw_distribution, 10)

    # Non Linear Timing Draws
    if "timing_model" in jp.snames:
        print("Adding timing model jump proposal...\n")
        sampler.addProposalToCycle(jp.draw_from_timing_model, 25)
    if "timing_model" in jp.snames:
        print("Adding timing model prior draw...\n")
        sampler.addProposalToCycle(jp.draw_from_timing_model_prior, 10)

    if timing:
        # SCAM and AM Draws
        # add SCAM
        print("Adding SCAM Jump Proposal...\n")
        sampler.addProposalToCycle(jp.covarianceJumpProposalSCAM, 30)

        # add AM
        print("Adding AM Jump Proposal...\n")
        sampler.addProposalToCycle(jp.covarianceJumpProposalAM, 15)

        # add DE
        print("Adding DE Jump Proposal...\n")
        sampler.addProposalToCycle(jp.DEJump, 30)

    return sampler
