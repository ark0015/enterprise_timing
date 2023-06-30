import numpy as np
import glob, os, sys, pickle, json
import string, inspect, copy
from collections import OrderedDict
import scipy

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


def add_bool_arg(parser, name, help, default):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--" + name, dest=name, action="store_true", help=help)
    group.add_argument("--no-" + name, dest=name, action="store_false", help=help)
    parser.set_defaults(**{name: default})


parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--psr_name", required=True, type=str, help="name of pulsar used for search"
)
parser.add_argument("--datarelease", required=True, help="What dataset to use")
parser.add_argument("--run_num", required=True, help="Label at end of output file.")
parser.add_argument(
    "--tm_prior",
    choices=["uniform", "bounded"],
    default="uniform",
    help="Use either uniform or bounded for ephemeris modeling? (DEFAULT: uniform)",
)
parser.add_argument(
    "--ephem", default="DE436", help="Ephemeris option (DEFAULT: DE436)"
)
add_bool_arg(parser, "white_var", "Vary the white noise? (DEFAULT: TRUE)", True)
add_bool_arg(parser, "red_var", "Vary the red noise? (DEFAULT: TRUE)", True)
add_bool_arg(parser, "resume", "Whether to resume the chains. (DEFAULT: FALSE", False)
add_bool_arg(
    parser,
    "wideband",
    "Whether to use wideband timing for DMX parameters. (DEFAULT: FALSE",
    False,
)
add_bool_arg(
    parser,
    "coefficients",
    "Whether to keep track of linear components. (DEFAULT: FALSE",
    False,
)
add_bool_arg(parser, "tm_var", "Whether to vary timing model. (DEFAULT: True)", True)
add_bool_arg(
    parser,
    "tm_linear",
    "Whether to use only the linear timing model. (DEFAULT: FALSE)",
    False,
)
add_bool_arg(
    parser,
    "fit_remaining_pars",
    "Whether to use non-linear plus linear timing model variations. (DEFAULT: True)",
    True,
)
add_bool_arg(
    parser,
    "fixed_remaining_pars",
    "Whether to use non-linear plus fixed timing model parameters. (DEFAULT: FALSE)",
    False,
)
add_bool_arg(
    parser,
    "writeHotChains",
    "Whether to write out the parallel tempering chains. (DEFAULT: TRUE)",
    True,
)
add_bool_arg(
    parser,
    "reallyHotChain",
    "Whether to include a really hot chain in the parallel tempering runs. (DEFAULT: FALSE)",
    False,
)
parser.add_argument("--N", default=int(1e6), help="Number of samples (DEFAULT: 1e6)")
add_bool_arg(
    parser,
    "lin_dmx_jump_fd",
    "Whether to use linear timing for DMX, JUMP, and FD parameters. (DEFAULT: FALSE)",
    False,
)
add_bool_arg(
    parser,
    "sample_cos",
    "Whether to sample inclination in COSI or SINI. (DEFAULT: FALSE)",
    True,
)
add_bool_arg(
    parser,
    "Ecorr_gp_basis",
    "Whether to use the gp_signals or white_signals ECORR. (DEFAULT: FALSE)",
    False,
)
add_bool_arg(
    parser,
    "incTimingModel",
    "Whether to include the timing model. (DEFAULT: TRUE)",
    True,
)
add_bool_arg(
    parser,
    "zero_start",
    "Whether to start the timing parameters at the parfile value. (DEFAULT: TRUE",
    True,
)
add_bool_arg(
    parser,
    "tnequad",
    "Whether to use old tempo2 version of equad (DEFAULT: False)",
    False,
)
parser.add_argument(
    "--parfile",
    default="",
    help="Location of parfile </PATH/TO/FILE/PARFILE.par>",
    required=True,
)
parser.add_argument(
    "--timfile",
    default="",
    help="Location of timfile </PATH/TO/FILE/TIMFILE.tim>",
    required=True,
)
parser.add_argument(
    "--model_kwargs_file", default="", help="Location of model_kwargs_file"
)
parser.add_argument(
    "--emp_dist_path", default="", help="Location of empirical distribution"
)
parser.add_argument(
    "--dmx_file", help="Location of dmx file to fit DM1/DM2", required=True
)
parser.add_argument(
    "--timing_package",
    default="tempo2",
    help="Whether to use PINT or Tempo2 (DEFAULT: tempo2)",
)
add_bool_arg(
    parser,
    "restrict_pulsar_mass",
    "Whether to have a hard upper limit of 3 M_sun on the pulsar mass (DEFAULT: True)",
    True,
)
parser.add_argument(
    "--dt",
    default=15,
    help="Sets time window for dm_dt and chrom_dt (DEFAULT: 15)",
)

args = parser.parse_args()

if not isinstance(args.N, int):
    N = int(float(args.N))
else:
    N = args.N

if not isinstance(args.dt, int):
    dt = int(float(args.dt))
else:
    dt = args.dt

if os.path.isfile(args.parfile):
    parfile = args.parfile
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
else:
    raise ValueError(f"{args.parfile} does not exist. Please pick a real parfile.")


if os.path.isfile(args.timfile):
    timfile = args.timfile
else:
    raise ValueError(f"{args.timfile} does not exist. Please pick a real timfile.")

if os.path.isfile(args.dmx_file):
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
        dmx = np.loadtxt(args.dmx_file, skiprows=4, dtype=dtypes)
    except:
        with open(args.dmx_file, "r") as f:
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
        dmx = np.loadtxt(args.dmx_file, skiprows=4, dtype=dtypes_2)
else:
    raise ValueError(f"{args.dmx_file} does not exist. Please pick a real dmx_file.")

if args.fit_remaining_pars and args.tm_var:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(args.psr_name, args.datarelease)
        + args.psr_name
        + "_{}_{}_nltm_ltm_adv_noise_mod_{}".format(
            "_".join(args.tm_prior.split("-")), args.ephem, args.run_num
        )
    )
else:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(args.psr_name, args.datarelease)
        + args.psr_name
        + "_{}_{}_nltm_adv_noise_mod_{}".format(
            "_".join(args.tm_prior.split("-")), args.ephem, args.run_num
        )
    )

if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)
else:
    if not args.resume:
        print("nothing!")
        # raise ValueError("{} already exists!".format(outdir))

noisedict = None

# filter
is_psr = False
if args.psr_name in parfile:
    if args.timing_package.lower() == "tempo2":
        psr = Pulsar(
            parfile,
            timfile,
            ephem=args.ephem,
            clk=None,
            drop_t2pulsar=False,
            timing_package="tempo2",
        )
    elif args.timing_package.lower() == "pint":
        psr = Pulsar(
            parfile,
            timfile,
            ephem=args.ephem,
            clk=None,
            drop_pintpsr=False,
            timing_package="pint",
        )
    is_psr = True

if not is_psr:
    raise ValueError(
        "{} does not exist in {} datarelease.".format(args.psr_name, args.datarelease)
    )

nltm_params = []
ltm_list = []
fixed_list = []
refit_pars = []
tm_param_dict = {}
for par in psr.fitpars:
    if par == "Offset":
        ltm_list.append(par)
    elif "DMX" in par and any(
        [args.lin_dmx_jump_fd, args.wideband, args.fixed_remaining_pars]
    ):
        if args.fixed_remaining_pars:
            fixed_list.append(par)
        else:
            ltm_list.append(par)
    elif "JUMP" in par and any([args.lin_dmx_jump_fd, args.fixed_remaining_pars]):
        if args.fixed_remaining_pars:
            fixed_list.append(par)
        else:
            ltm_list.append(par)
    elif "FD" in par and any([args.lin_dmx_jump_fd, args.fixed_remaining_pars]):
        if args.fixed_remaining_pars:
            fixed_list.append(par)
        else:
            ltm_list.append(par)
    elif par == "SINI" and args.sample_cos:
        if args.sample_cos:
            nltm_params.append("COSI")
        else:
            nltm_params.append(par)
    else:
        nltm_params.append(par)

    if par in ["PBDOT", "XDOT"] and hasattr(psr, "t2pulsar"):
        par_val = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
        par_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
        if np.log10(par_sigma) > -10.0:
            print(f"USING PHYSICAL {par}. Val: ", par_val, "Err: ", par_sigma * 1e-12)
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

    elif par in ["DM", "DM1", "DM2"]:
        raise ValueError("These are fit internally with ANM! Please remove them from parfile/make them fixed.")

print(tm_param_dict)
if not args.tm_linear and args.tm_var:
    print(
        "Non-linearly varying these values: ",
        nltm_params,
        "\n in pulsar ",
        args.psr_name,
    )
elif args.tm_linear and args.tm_var:
    print("Using linear approximation for all timing parameters.")
else:
    print("Not varying timing parameters.")

if args.fit_remaining_pars and args.tm_var:
    print("Linearly varying these values: ", ltm_list)

if args.fixed_remaining_pars:
    print("Fixing these parameters: ", fixed_list)

print("Using ", args.tm_prior, " prior.")

model_args = inspect.getfullargspec(models.model_singlepsr_noise)
model_keys = model_args[0][1:]
model_vals = model_args[3]
model_kwargs = dict(zip(model_keys, model_vals))

if os.path.isfile(args.model_kwargs_file):
    print("loading model kwargs from file...")
    with open(args.model_kwargs_file, "r") as fin:
        model_dict = json.load(fin)

    if "0" in model_dict.keys():
        # Hypermodel
        ptas = dict.fromkeys(np.array([int(x) for x in model_dict.keys()]))
        for ct, mod in enumerate(model_dict.keys()):
            model_dict[mod]["dmx_data"] = dmx
            ptas[ct] = models.model_singlepsr_noise(psr, **model_dict[mod])
        print("Using tm_param_dict from input model_kwargs_file")
        tm_param_dict = model_dict["0"]["tm_param_dict"]
        print(tm_param_dict)
    else:
        # Take out parameters not in model_kwargs
        del_pars = [x for x in model_dict.keys() if x not in model_kwargs.keys()]
        if del_pars:
            for dp in del_pars:
                del model_dict[dp]
            # print(model_kwargs)
            model_dict.update(
                {
                    "tm_var": args.tm_var,
                    "tm_linear": args.tm_linear,
                    "tm_param_list": nltm_params,
                    "ltm_list": ltm_list,
                    "tm_param_dict": tm_param_dict,
                    "tm_prior": args.tm_prior,
                    "normalize_prior_bound": 50.0,
                    "fit_remaining_pars": args.fit_remaining_pars,
                    "dmepoch":DMEPOCH,
                    "dmx_data":dmx,
                    "red_var": args.red_var,
                    "noisedict": noisedict,
                    "white_vary": args.white_var,
                    "is_wideband": args.wideband,
                    "use_dmdata": args.wideband,
                    "dmjump_var": args.wideband,
                    "coefficients": args.coefficients,
                    "dm_dt": dt,
                    "chrom_dt": dt,
                    "dm_df": None,
                    "chrom_df": None,
                }
            )
        pta = models.model_singlepsr_noise(psr, **model_dict)
elif not os.path.isfile(args.model_kwargs_file) and len(args.model_kwargs_file) > 0:
    raise ValueError(f"{args.model_kwargs_file} does not exist!")
else:
    ###########################
    # First Round:
    ###########################
    """
    red_psd = "powerlaw"
    dm_nondiag_kernel = ["None", "sq_exp", "periodic"]
    dm_sw_gp = [True, False]
    dm_annual = False
    nmodels = len(dm_nondiag_kernel)*len(dm_sw_gp)
    """
    ###########################
    # Second Round:
    ###########################
    """
    red_psd = 'powerlaw'
    dm_nondiag_kernel = ['None','periodic','sq_exp','periodic_rfband','sq_exp_rfband']
    dm_sw = False
    #dm_sw_gp = [True,False] #Depends on Round 1
    dm_annual = False
    chrom_gp = False
    chrom_gp_kernel = "nondiag"
    chrom_kernel = "periodic"
    nmodels = len(dm_nondiag_kernel)#*len(dm_sw_gp)
    """
    ###########################
    # Third Round (Second for J0740):
    ###########################
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
    dm_nondiag_kernel = ["periodic", "sq_exp"]
    chrom_gps = [True, False]
    chrom_gp_kernel = "nondiag"
    chrom_kernels = ["periodic", "sq_exp"]
    nmodels = 6
    """
    ###########################
    # Fourth Round (Third for J0740):
    ###########################
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
    chrom_gp = True
    chrom_gp_kernel = "nondiag"
    chrom_kernels = ["periodic", "sq_exp"]
    nmodels = len(chrom_kernels) * len(dm_nondiag_kernel)
    """
    ###########################
    # Fifth Round (Fourth for J0740):
    ###########################
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
    nmodels = 6
    """
    ###########################
    # Sixth Round (Fifth for J0740)
    ###########################
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
    dm_nondiag_kernel = "periodic_rfband"
    chrom_gp = True
    chrom_gp_kernel = "nondiag"
    chrom_kernels = ["periodic", "sq_exp"]
    dm_cusp = [True, False]
    dm_cusp_tmin = 57200
    dm_cusp_tmax = 57500
    nmodels = 6
    """
    ###########################
    # Seventh Round
    ###########################
    """
    red_psd = "powerlaw"
    dm_sw_gp = False
    dm_annual = False
    dm_sw = False
    # Round 7a
    dm_nondiag_kernel = ['sq_exp','sq_exp_rfband']
    # Round 7b
    # dm_nondiag_kernel = ['periodic','periodic_rfband']
    chrom_gp = True
    chrom_gp_kernel = "nondiag"
    chrom_kernels = ["periodic", "periodic_rfband", "sq_exp", "sq_exp_rfband"]
    nmodels = len(chrom_kernels) * len(dm_nondiag_kernel)
    """
    ###########################
    # Most Complex Single Round START HERE
    ###########################
    """
    red_psd = "powerlaw"
    dm_sw = False
    dm_annual = False
    dm_nondiag_kernel = ['periodic_rfband']
    chrom_gp = True
    chrom_gp_kernel = "nondiag"
    chrom_kernels = ["periodic_rfband"]
    nmodels = 1
    """
    ###########################
    # J2043 12.5yr - Most Complex Single Round
    ###########################
    if args.psr_name == "J2043+1711" and args.datarelease == "12p5yr":
        ###########################
        # Round 1
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['periodic_rfband']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic_rfband"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
        ###########################
        # Round 2
        ###########################
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['sq_exp']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["sq_exp"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
    ###########################
    # J1600 12.5yr - Most Complex Single Round
    ###########################
    if args.psr_name == "J1600-3053" and args.datarelease == "12p5yr":
        ###########################
        # Round 1
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = True
        dm_annual = False
        dm_nondiag_kernel = ['periodic_rfband']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic_rfband"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
        ###########################
        # Round 2
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = True
        dm_annual = False
        dm_nondiag_kernel = ['periodic_rfband']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic_rfband"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
        ###########################
        # Round 3
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = True
        dm_annual = False
        dm_nondiag_kernel = ['periodic']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic_rfband"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
        ###########################
        # DM Fit Round 2
        ###########################

        red_psd = "powerlaw"
        dm_sw = True
        dm_annual = False
        dm_nondiag_kernel = ['periodic']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1

    # J1640 12.5yr - Most Complex Single Round
    ###########################
    if args.psr_name == "J1640+2224" and args.datarelease == "12p5yr":
        ###########################
        # Round 1
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['periodic_rfband']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic_rfband"]
        add_expdip = True
        dm_expdip_tmin = 55500
        dm_expdip_tmax = 56500
        nmodels = 1
        """
        ###########################
        # Round 2
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['sq_exp_rfband']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["sq_exp_rfband"]
        nmodels = 1
        """
        ###########################
        # Round 3
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ["sq_exp"]
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["sq_exp"]
        nmodels = 1
        add_expdip = True
        dm_expdip_tmin = 55500
        dm_expdip_tmax = 56500
        """
        ###########################
        # DM Fit Round 2
        ###########################
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ["sq_exp"]
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["sq_exp"]
        nmodels = 1
        add_expdip = True
        dm_expdip_tmin = 55500
        dm_expdip_tmax = 56500
    ###########################
    # J0740 CFR+19 - Most Complex Single Round
    ###########################
    if args.psr_name == "J0740+6620" and args.datarelease == "cfr+19":
        ###########################
        # Round 1
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['periodic_rfband']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic_rfband"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
        ###########################
        # Round 2
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['sq_exp']
        chrom_gp = False
        chrom_gp_kernel = "nondiag" # Not Used
        chrom_kernels = ["sq_exp_rfband"] # Not Used
        nmodels = 1
        """
        ###########################
        # Round 3
        ###########################
        
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ["sq_exp"]
        chrom_gp = False
        chrom_gp_kernel = "nondiag"  # Not Used
        chrom_kernels = ["sq_exp_rfband"]  # Not Used
        nmodels = 1
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        
        ###########################
        # DM Fit Round 2
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['periodic']
        chrom_gp = False
        chrom_gp_kernel = "nondiag"  # Not Used
        chrom_kernels = ["periodic_rfband"]  # Not Used
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
    ###########################
    # J0740 FCP+21 - Most Complex Single Round
    ###########################
    if args.psr_name == "J0740+6620" and args.datarelease == "fcp+21":
        ###########################
        # Round 1
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['periodic_rfband']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic_rfband"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
        ###########################
        # Round 2
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['periodic']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic"]
        nmodels = 1
        """
        ###########################
        # Round 3
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['sq_exp']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["sq_exp"]
        nmodels = 1
        add_expdip = False # Not Used
        dm_expdip_tmin = 55500 # Not Used
        dm_expdip_tmax = 56500 # Not Used
        """
        ###########################
        # Round 4
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ["periodic"]
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["sq_exp"]
        nmodels = 1
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        """
        ###########################
        # DM Fit Round 2
        ###########################
        """
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['periodic']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["periodic"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1
        """
        ###########################
        # DM Fit Round 3
        ###########################
        red_psd = "powerlaw"
        dm_sw = False
        dm_annual = False
        dm_nondiag_kernel = ['sq_exp']
        chrom_gp = True
        chrom_gp_kernel = "nondiag"
        chrom_kernels = ["sq_exp"]
        add_expdip = False  # Not Used
        dm_expdip_tmin = 55500  # Not Used
        dm_expdip_tmax = 56500  # Not Used
        nmodels = 1

    # Create list of pta models for our model selection
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
            # if dm == "None":
            #     dm_var = False
            # else:
            dm_var = True
            # Copy template kwargs dict and replace values we are changing.
            kwargs = copy.deepcopy(model_kwargs)

            kwargs.update(
                {
                    "tm_var": args.tm_var,
                    "tm_linear": args.tm_linear,
                    "tm_param_list": nltm_params,
                    "ltm_list": ltm_list,
                    "tm_param_dict": tm_param_dict,
                    "tm_prior": args.tm_prior,
                    "normalize_prior_bound": 50.0,
                    "fit_remaining_pars": args.fit_remaining_pars,
                    "dmepoch":DMEPOCH,
                    "dmx_data":dmx,
                    "red_var": args.red_var,
                    "noisedict": noisedict,
                    "white_vary": args.white_var,
                    "is_wideband": args.wideband,
                    "use_dmdata": args.wideband,
                    "dmjump_var": args.wideband,
                    "coefficients": args.coefficients,
                    "dm_var": dm_var,
                    "dmgp_kernel": "nondiag",
                    "psd": red_psd,
                    # "dm_nondiag_kernel": dm_nondiag_kernel,
                    "dm_nondiag_kernel": dm,
                    "dm_sw_deter": True,
                    "dm_sw_gp": dm_sw,
                    "dm_annual": dm_annual,
                    "swgp_basis": "powerlaw",
                    "chrom_gp_kernel": chrom_gp_kernel,
                    "chrom_kernel": chrom_kernel,
                    "chrom_gp": chrom_gp,
                    #'chrom_idx':chrom_index,
                    # "dm_cusp": add_cusp,
                    # "dm_cusp_tmin":dm_cusp_tmin,
                    # "dm_cusp_tmax":dm_cusp_tmax,
                    "dm_expdip": add_expdip,
                    "dm_expdip_tmin": dm_expdip_tmin,
                    "dm_expdip_tmax": dm_expdip_tmax,
                    #'dm_cusp_idx':cusp_idxs[:num_cusp],
                    #'num_dm_cusps':num_cusp,
                    #'dm_cusp_sign':cusp_signs[:num_cusp]
                    "dm_dt": dt,
                    "chrom_dt": dt,
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
            model_labels.append(
                [string.ascii_uppercase[ct], dm, chrom_gp, chrom_kernel]
            )
            # model_labels.append([string.ascii_uppercase[ct], dm, chrom_kernel])
            # model_labels.append([string.ascii_uppercase[ct], dm, dm_var])
            # model_labels.append([string.ascii_uppercase[ct], add_cusp, chrom_kernel])
            model_dict.update({str(ct): kwargs})
            ct += 1
    with open(outdir + "/model_labels.json", "w") as fout:
        json.dump(model_labels, fout, sort_keys=True, indent=4, separators=(",", ": "))
    print(model_labels)

if os.path.isfile(args.emp_dist_path):
    emp_dist_path = args.emp_dist_path
else:
    print("No empirical distribution used.")
    emp_dist_path = None

print(model_dict.keys())
if "0" in model_dict.keys():
    # Hypermodel

    # Instantiate a collection of models
    # super_model = TimingHyperModel(ptas)
    super_model = HyperModel(ptas)

    psampler = super_model.setup_sampler(
        outdir=outdir,
        resume=args.resume,
        timing=args.incTimingModel,
        empirical_distr=emp_dist_path,
        psr=psr,
        restrict_mass=args.restrict_pulsar_mass,
    )
    model_params = {}

    for ky, pta in ptas.items():
        model_params.update({str(ky): pta.param_names})

    with open(outdir + "/model_params.json", "w") as fout:
        json.dump(model_params, fout, sort_keys=True, indent=4, separators=(",", ": "))

    x0 = super_model.initial_sample(
        tm_params_orig=psr.tm_params_orig,
        tm_param_dict=tm_param_dict,
        zero_start=args.zero_start,
    )

else:
    psampler = sampler.setup_sampler(
        pta,
        outdir=outdir,
        resume=args.resume,
        timing=args.incTimingModel,
        psr=psr,
        restrict_mass=args.restrict_pulsar_mass,
    )

    if args.zero_start:
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
                    # Could be a problem if input model_kwargs['tm_param_dict'] is different than tm_param_dict
                    if p_name in tm_param_dict.keys():
                        x0_list.append(np.double(tm_param_dict[p_name]["prior_mu"]))
                    else:
                        x0_list.append(np.double(psr.tm_params_orig[p_name][0]))
            elif "dm_model" in p.name:
                if "mu" in str(p):
                    x0_list.append(float(str(p).split("(")[1].split(",")[0].split("=")[-1]))
                else:
                    x0_list.append(p.sample())
            else:
                x0_list.append(p.sample())
        x0 = np.asarray(x0_list)
    else:
        x0 = np.hstack([p.sample() for p in pta.params])


with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
    pickle.dump(psr.tm_params_orig, fout)

if tm_param_dict:
    with open(outdir + "/tm_param_dict.json", "w") as fout:
        json.dump(tm_param_dict, fout, sort_keys=True, indent=4, separators=(",", ": "))

#Get rid of dmx_data to allow JSON saving
if "0" in model_dict.keys():
    for key in model_dict.keys():
        if "dmx_data" in model_dict[key].keys():
            model_dict[key].pop("dmx_data")
else:
    if "dmx_data" in model_dict.keys():
        model_dict.pop("dmx_data")

print(model_dict)
with open(outdir + "/model_kwargs.json", "w") as fout:
    json.dump(model_dict, fout, sort_keys=True, indent=4, separators=(",", ": "))

if args.restrict_pulsar_mass:
    psampler.sample(
        x0,
        N,
        SCAMweight=0,
        AMweight=0,
        DEweight=0,
        writeHotChains=args.writeHotChains,
        hotChain=args.reallyHotChain,
    )
else:
    psampler.sample(
        x0,
        N,
        SCAMweight=20,
        AMweight=20,
        DEweight=20,
        writeHotChains=args.writeHotChains,
        hotChain=args.reallyHotChain,
    )
