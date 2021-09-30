import numpy as np
import glob, os, sys, pickle, json
import cloudpickle
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
# top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
top_path_idx = splt_path.index("ark0015")
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
    "Whether to start the timing parameters at the parfile value. (DEFAULT: TRUE)",
    True,
)

args = parser.parse_args()

if not isinstance(args.N, int):
    N = int(float(args.N))
else:
    N = args.N

if args.fit_remaining_pars and args.tm_var:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(args.psr_name, args.datarelease)
        + args.psr_name
        + "_{}_{}_nltm_ltm_{}".format(
            "_".join(args.tm_prior.split("-")), args.ephem, args.run_num
        )
    )
else:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(args.psr_name, args.datarelease)
        + args.psr_name
        + "_{}_{}_tm_{}".format(
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
parfile = current_path + "/J1600-3053_12yV3_dmgp.par"
timfile = top_dir + "/{}/narrowband/tim/{}_NANOGrav_12yv3.tim".format(
    args.datarelease, args.psr_name
)

# filter
is_psr = False
if args.psr_name in parfile:
    psr = Pulsar(parfile, timfile, ephem=args.ephem, clk=None, drop_t2pulsar=False)
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

    if par == "PBDOT":
        pbdot = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
        pbdot_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
        print("USING PHYSICAL PBDOT. Val: ", pbdot, "Err: ", pbdot_sigma * 1e-12)
        lower = pbdot - 500 * pbdot_sigma * 1e-12
        upper = pbdot + 500 * pbdot_sigma * 1e-12
        tm_param_dict["PBDOT"] = {
            "prior_lower_bound": lower,
            "prior_upper_bound": upper,
        }
    elif par == "XDOT":
        xdot = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
        xdot_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
        print("USING PHYSICAL XDOT. Val: ", xdot, "Err: ", xdot_sigma * 1e-12)
        lower = xdot - 500 * xdot_sigma * 1e-12
        upper = xdot + 500 * xdot_sigma * 1e-12
        tm_param_dict["XDOT"] = {
            "prior_lower_bound": lower,
            "prior_upper_bound": upper,
        }
    elif par in ["DM", "DM1", "DM2"] and par not in refit_pars:
        orig_vals = {p: v for p, v in zip(psr.t2pulsar.pars(), psr.t2pulsar.vals())}
        orig_errs = {p: e for p, e in zip(psr.t2pulsar.pars(), psr.t2pulsar.errs())}
        if np.any(np.isnan(psr.t2pulsar.errs())) or np.any(
            [err == 0.0 for err in psr.t2pulsar.errs()]
        ):
            eidxs = np.where(
                np.logical_or(np.isnan(psr.t2pulsar.errs()), psr.t2pulsar.errs() == 0.0)
            )[0]
            psr.t2pulsar.fit()
            for idx in eidxs:
                parr = psr.t2pulsar.pars()[idx]
                if parr in ["DM", "DM1", "DM2"]:
                    refit_pars.append(parr)
                    parr_val = np.double(psr.t2pulsar.vals()[idx])
                    parr_sigma = np.double(psr.t2pulsar.errs()[idx])
                    print(f"USING REFIT {parr}. Val: ", parr_val, "Err: ", parr_sigma)
                    lower = parr_val - 500 * parr_sigma
                    upper = parr_val + 500 * parr_sigma
                    tm_param_dict[f"{parr}"] = {
                        "prior_lower_bound": lower,
                        "prior_upper_bound": upper,
                    }
        psr.t2pulsar.vals(orig_vals)
        psr.t2pulsar.errs(orig_errs)
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

model_kwargs_path = current_path + "/J1600-3053_model_kwargs.json"
with open(model_kwargs_path, "r") as fin:
    model_kwargs = json.load(fin)

if "tmparam_list" in model_kwargs.keys():
    del model_kwargs["tmparam_list"]
# print(model_kwargs)
model_kwargs.update(
    {
        "tm_var": args.tm_var,
        "tm_linear": args.tm_linear,
        "tm_param_list": nltm_params,
        "ltm_list": ltm_list,
        "tm_param_dict": tm_param_dict,
        "tm_prior": args.tm_prior,
        "normalize_prior_bound": 500.0,
        "fit_remaining_pars": args.fit_remaining_pars,
        "red_var": args.red_var,
        "noisedict": noisedict,
        "white_vary": args.white_var,
        "is_wideband": args.wideband,
        "use_dmdata": args.wideband,
        "dmjump_var": args.wideband,
        "coefficients": args.coefficients,
    }
)
# print(model_kwargs)

pta = models.model_singlepsr_noise(psr, **model_kwargs)


emp_dist_path = current_path + "/ng12p5yr_J1600-3053_advnoise_freespec_emp_dist.pkl"
print("Empirical Distribution?", os.path.isfile(emp_dist_path))

psampler = sampler.setup_sampler(
    pta, outdir=outdir, empirical_distr=emp_dist_path, resume=args.resume, timing=True
)
with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
    pickle.dump(psr.tm_params_orig, fout)

if args.coefficients:
    x0_list = []
    for p in pta.params:
        if "coefficients" not in p.name:
            x0_list.append(p.sample())
    x0 = np.asarray(x0_list)
else:
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
                    if p_name in tm_param_dict.keys():
                        x0_list.append(np.double(tm_param_dict[p_name]["prior_mu"]))
                    else:
                        x0_list.append(np.double(psr.tm_params_orig[p_name][0]))
            else:
                x0_list.append(p.sample())
        x0 = np.asarray(x0_list)
    else:
        x0 = np.hstack([p.sample() for p in pta.params])

with open(outdir + "/model_kwargs.json", "w") as fout:
    json.dump(model_kwargs, fout, sort_keys=True, indent=4, separators=(",", ": "))

psampler.sample(
    x0,
    N,
    SCAMweight=30,
    AMweight=15,
    DEweight=30,
    writeHotChains=args.writeHotChains,
    hotChain=args.reallyHotChain,
)
