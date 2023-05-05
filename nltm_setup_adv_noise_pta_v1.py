import numpy as np
import scipy

import glob, os, sys, pickle, json, inspect, copy, string
from collections import OrderedDict

current_path = os.getcwd()
splt_path = current_path.split("/")
# top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
e_path = top_dir + "/enterprise/"
ptmcmc_path = top_dir + "/PTMCMCSampler"

sys.path.insert(0, e_e_path)
sys.path.insert(0, ptmcmc_path)
sys.path.insert(0, e_path)

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
from enterprise_extensions.sampler import JumpProposal
from enterprise_extensions.timing import timing_block
from enterprise_extensions.blocks import channelized_backends
from enterprise_extensions.hypermodel import HyperModel

# from hypermodel_timing import TimingHyperModel

import argparse


def dm_funk(t, dm1, dm2, DMEPOCH):
    """Used to refit for DM1 and DM2. t in units of MJD"""
    # DM(t)=DM+DM1*(t-DMEPOCH)+DM2*(t-DMEPOCH)^2
    return dm1 * (t - DMEPOCH) + dm2 * (t - DMEPOCH) ** 2


def pta_setup(
    psr,
    datarelease,
    psr_name,
    parfile,
    dmx_file,
    tm_var=True,
    red_var=False,
    white_var=True,
    fit_remaining_pars=True,
    lin_dmx_jump_fd=False,
    wideband=False,
    fixed_remaining_pars=False,
    sample_cos=True,
    tm_linear=False,
    tm_prior="uniform",
    incTimingModel=True,
    Ecorr_gp_basis=False,
    model_kwargs_file="",
):

    if os.path.isfile(parfile):
        parfile = parfile
        # Load raw parfile to get DMEPOCH
        DMEPOCH = 0
        with open(parfile, "r") as f:
            for line in f.readlines():
                if "DMEPOCH" in [x for x in line.split()]:
                    DMEPOCH = np.double(line.split()[-1])
        if DMEPOCH == 0:
            raise ValueError(
                "DMEPOCH not in parfile. Please add it to the parfile so DM1/DM2 fitting can work."
            )
        globals()[DMEPOCH] = DMEPOCH
    else:
        raise ValueError(f"{parfile} does not exist. Please pick a real parfile.")
    if os.path.isfile(dmx_file):
        # Load DMX values
        dtypes = {
            "names": (
                "DMXEP",
                "DMX_value",
                "DMX_var_err",
                "DMXR1",
                "DMXR2",
                "DMXF1",
                "DMXF2",
                "DMX_bin",
            ),
            "formats": ("f4", "f4", "f4", "f4", "f4", "f4", "f4", "U6"),
        }
        try:
            dmx = np.loadtxt(dmx_file, skiprows=4, dtype=dtypes)
        except:
            with open(dmx_file, "r") as f:
                for i in range(4):
                    dmx_pars = f.readline()
            dmx_pars = dmx_pars.split(": ")[-1].split("\n")[0].split(" ")
            dtypes_2 = {}
            tmp_names = []
            tmp_formats = []
            for nam, typ in zip(dtypes["names"], dtypes["formats"]):
                if nam in dmx_pars:
                    tmp_names.append(nam)
                    tmp_formats.append(typ)

            dtypes_2["names"] = tuple(tmp_names)
            dtypes_2["formats"] = tuple(tmp_formats)
            dmx = np.loadtxt(dmx_file, skiprows=4, dtype=dtypes_2)
    else:
        raise ValueError(f"{dmx_file} does not exist. Please pick a real dmx_file.")

    nltm_params = []
    ltm_list = []
    fixed_list = []
    refit_pars = []
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
            if sample_cos:
                nltm_params.append("COSI")
            else:
                nltm_params.append(par)
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

        elif par in ["DM1", "DM2"] and par not in refit_pars:
            popt, pcov = scipy.optimize.curve_fit(
                dm_funk, dmx["DMXEP"], dmx["DMX_value"]
            )
            perr = np.sqrt(np.diag(pcov))
            for ii, parr in enumerate(["DM1", "DM2"]):
                refit_pars.append(parr)
                print(f"USING REFIT {parr}. Val:  {popt[ii]}, Err: {perr[ii]}")
                lower = popt[ii] - 1e4 * perr[ii]
                upper = popt[ii] + 1e4 * perr[ii]
                tm_param_dict[f"{parr}"] = {
                    "prior_mu": popt[ii],
                    "prior_sigma": perr[ii],
                    "prior_lower_bound": lower,
                    "prior_upper_bound": upper,
                }
    print(tm_param_dict)
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

    if os.path.isfile(model_kwargs_file):
        print("loading model kwargs from file...")
        with open(model_kwargs_file, "r") as fin:
            model_dict = json.load(fin)
        # Instantiate single pulsar noise model
        for ct, mod in enumerate(model_dict.keys()):
            ptas[ct] = models.model_singlepsr_noise(psr, **mod)
    else:
        model_args = inspect.getfullargspec(models.model_singlepsr_noise)
        model_keys = model_args[0][1:]
        model_vals = model_args[3]
        model_kwargs = dict(zip(model_keys, model_vals))
        """
        #First Round:
        """
        """
        red_psd = "powerlaw"
        dm_nondiag_kernel = ["None", "sq_exp", "periodic"]
        dm_sw_gp = [True, False]
        dm_annual = False
        """
        """
        #Second Round (Didn't do for J0740):
        """
        """
        red_psd = 'powerlaw'
        dm_nondiag_kernel = ['periodic','sq_exp','periodic_rfband','sq_exp_rfband']
        dm_sw_gp = [True,False] #Depends on Round 1
        dm_annual = False
        """
        """
        #Third Round (Second for J0740):
        """
        # red_psd = "powerlaw"
        # dm_sw_gp = False
        # dm_annual = False
        # dm_sw = False
        # Round 3a
        # dm_nondiag_kernel = ['sq_exp','sq_exp_rfband']
        # Round 3b
        # dm_nondiag_kernel = ['periodic','periodic_rfband']
        # Almost round 4a
        # dm_nondiag_kernel = ["periodic", "sq_exp"]
        # chrom_gps = [True, False]
        # chrom_gp_kernel = "nondiag"
        # chrom_kernels = ["periodic", "sq_exp"]
        """
        #Fourth Round (Third for J0740):
        """
        """
        red_psd = "powerlaw"
        dm_sw_gp = False
        dm_annual = False
        dm_sw = False
        # Round 3a
        # dm_nondiag_kernel = ['sq_exp','sq_exp_rfband']
        # Round 3b
        # dm_nondiag_kernel = ['periodic','periodic_rfband']
        # Almost round 4a
        dm_nondiag_kernel = ["periodic","periodic_rfband", "sq_exp","sq_exp_rfband"]
        chrom_gps = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic", "sq_exp"]
        """
        """
        #Fifth Round (Fourth for J0740):
        """

        red_psd = "powerlaw"
        dm_sw_gp = False
        dm_annual = False
        dm_sw = False
        # Round 3a
        # dm_nondiag_kernel = ['sq_exp','sq_exp_rfband']
        # Round 3b
        # dm_nondiag_kernel = ['periodic','periodic_rfband']
        # Almost round 4a
        dm_nondiag_kernel = [
            "sq_exp_rfband",
            "periodic_rfband",
        ]
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic", "sq_exp"]

        # Create list of pta models for our model selection
        # nmodels = len(dm_annuals) * len(dm_nondiag_kernel)
        nmodels = 4
        # nmodels = len(chrom_indices) * len(dm_nondiag_kernel)
        mod_index = np.arange(nmodels)

        ptas = dict.fromkeys(mod_index)
        model_dict = {}
        model_labels = []
        ct = 0
        for dm in dm_nondiag_kernel:
            # for add_cusp in dm_cusp:
            # for dm_sw in dm_sw_gp:
            # for chrom_gp in chrom_gps:
            for chrom_kernel in chrom_kernels:
                if dm == "None":
                    dm_var = False
                else:
                    dm_var = True
                # Copy template kwargs dict and replace values we are changing.
                kwargs = copy.deepcopy(model_kwargs)

                kwargs.update(
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
                        "dm_var": dm_var,
                        "dmgp_kernel": "nondiag",
                        "psd": red_psd,
                        "dm_nondiag_kernel": dm,
                        "dm_sw_deter": True,
                        "dm_sw_gp": dm_sw,
                        "dm_annual": dm_annual,
                        "swgp_basis": "powerlaw",
                        "chrom_gp_kernel": chrom_gp_kernel,
                        "chrom_kernel": chrom_kernel,
                        "chrom_gp": chrom_gp,
                        #'chrom_idx':chrom_index,
                        #'dm_cusp':dm_cusp,
                        #'dm_cusp_idx':cusp_idxs[:num_cusp],
                        #'num_dm_cusps':num_cusp,
                        #'dm_cusp_sign':cusp_signs[:num_cusp]
                        "dm_df": None,
                        "chrom_df": None,
                    }
                )
                # if dm == "None" and dm_sw:
                #    pass
                # if not chrom_gp and chrom_kernel == "sq_exp":
                #    pass
                # else:
                # Instantiate single pulsar noise model
                ptas[ct] = models.model_singlepsr_noise(psr, **kwargs)
                # Add labels and kwargs to save for posterity and plotting.
                # model_labels.append([string.ascii_uppercase[ct], dm, dm_sw])
                model_labels.append([string.ascii_uppercase[ct], dm, chrom_kernel])
                model_dict.update({str(ct): kwargs})
                ct += 1

    changed_params_list = []
    for j, key in enumerate(model_dict["0"].keys()):
        # print(key)
        # print('\t model 0',model_dict['0'][key])
        for other_model in model_dict.keys():
            if "0" != other_model:
                # print('\t model',other_model,model_dict[other_model][key])
                if model_dict[other_model][key] != model_dict["0"][key]:
                    changed_params_list.append(key)
    changed_params = {}
    for model in model_dict.keys():
        changed_params[model] = {}
        for param in changed_params_list:
            changed_params[model][param] = model_dict[model][param]
    print("")
    print(changed_params)
    print("")
    print(model_labels)

    # Instantiate a collection of models
    return HyperModel(ptas)
