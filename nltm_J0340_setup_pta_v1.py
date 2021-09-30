import numpy as np
import glob, os, sys, pickle, json
from collections import OrderedDict


import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import utils
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import signal_base
from enterprise.signals import selections
from enterprise.signals import gp_signals


from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
# top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
noise_path = top_dir + "/pta_sim/pta_sim"
sys.path.insert(0, noise_path)
sys.path.insert(0, e_e_path)
import enterprise_extensions as e_e
from enterprise_extensions import sampler
from enterprise_extensions import models
from enterprise_extensions.sampler import JumpProposal
from enterprise_extensions.timing import timing_block
from enterprise_extensions.blocks import channelized_backends

import noise
import argparse


def pta_setup(
    psr,
    datarelease="9yr",
    psr_name="J0340+4130",
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
):

    nltm_params = []
    ltm_list = []
    fixed_list = []
    tm_param_dict = {}
    for par in psr.fitpars:
        if par == "Offset":
            # nltm_params.append(par)
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
        elif par == "SINI":
            if sample_cos:
                nltm_params.append("COSI")
                if datarelease == "5yr":
                    tm_param_dict["COSI"] = {
                        "prior_lower_bound": 0.0,
                        "prior_upper_bound": 1.0,
                    }
            else:
                nltm_params.append(par)
                if datarelease == "5yr":
                    tm_param_dict[par] = {
                        "prior_lower_bound": 0.0,
                        "prior_upper_bound": 1.0,
                    }
        elif par == "PX":
            nltm_params.append(par)
            if datarelease == "5yr":
                tm_param_dict[par] = {
                    "prior_type": "dm_dist_px_prior",
                }
        elif par == "M2":
            nltm_params.append(par)
            if datarelease == "5yr":
                tm_param_dict[par] = {
                    "prior_lower_bound": 0.0,
                    "prior_upper_bound": 10.0,
                }
        elif (
            par in ["ELONG", "ELAT", "F0", "F1"]
            and datarelease == "9yr"
            and lin_dmx_jump_fd
        ):
            ltm_list.append(par)
        else:
            nltm_params.append(par)

        if par == "PBDOT":
            pbdot = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
            pbdot_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
            print("USING PHYSICAL PBDOT. Val: ", pbdot, "Err: ", pbdot_sigma * 1e-12)
            lower = pbdot - 5 * pbdot_sigma * 1e-12
            upper = pbdot + 5 * pbdot_sigma * 1e-12
            tm_param_dict["PBDOT"] = {
                "prior_lower_bound": lower,
                "prior_upper_bound": upper,
            }
        elif par == "XDOT":
            xdot = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
            xdot_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
            print("USING PHYSICAL XDOT. Val: ", xdot, "Err: ", xdot_sigma * 1e-12)
            lower = xdot - 5 * xdot_sigma * 1e-12
            upper = xdot + 5 * xdot_sigma * 1e-12
            tm_param_dict["XDOT"] = {
                "prior_lower_bound": lower,
                "prior_upper_bound": upper,
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
        if tm_var and not tm_linear:
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
            if incTimingModel:
                # create new attribute for enterprise pulsar object
                # UNSURE IF NECESSARY
                psr.tm_params_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
                for key in psr.tm_params_orig:
                    psr.tm_params_orig[key] = (
                        psr.t2pulsar[key].val,
                        psr.t2pulsar[key].err,
                    )
                s = gp_signals.TimingModel(
                    use_svd=False, normed=True, coefficients=False
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
            if incTimingModel:
                s += ef + eq + ec
            else:
                s = ef + eq + ec

            model = s(psr)

            # set up PTA
            pta = signal_base.PTA([model])
    return pta
