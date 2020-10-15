from __future__ import division

import numpy as np
import glob, os, sys, pickle, json

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import utils

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
    "coefficients",
    "Whether to keep track of linear components. (DEFAULT: FALSE",
    False,
)

add_bool_arg(
    parser,
    "fit_remaining_pars",
    "Whether to use non-linear plus linear timing model variations. (DEFAULT: TRUE)",
    True,
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

args = parser.parse_args()

if not isinstance(args.N, int):
    N = int(float(args.N))
else:
    N = args.N

if args.psr_name != "J0740+6620":
    raise ValueError("Only used for J0740! Not {}".format(args.psr_name))

if args.datarelease == "12p5yr":
    datadir = top_dir + "/{}".format(args.datarelease)
    parfiles = sorted(glob.glob(datadir + "/*.par"))
    timfiles = sorted(glob.glob(datadir + "/*.tim"))
else:
    datadir = top_dir + "/{}".format(args.datarelease)
    parfiles = sorted(glob.glob(datadir + "/par/*.par"))
    timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))

if args.fit_remaining_pars:
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

noisedict = {}
if args.datarelease in ["12p5yr"]:
    noisefiles = sorted(glob.glob(top_dir + "/{}/*.json".format(args.datarelease)))
    for noisefile in noisefiles:
        tmpnoisedict = {}
        with open(noisefile, "r") as fin:
            tmpnoisedict.update(json.load(fin))
        for key in tmpnoisedict.keys():
            if key.split("_")[0] == args.psr_name:
                noisedict[key] = tmpnoisedict[key]
else:
    noisefiles = sorted(glob.glob(datadir + "/noisefiles/*.txt"))
    for noisefile in noisefiles:
        tmpnoisedict = {}
        tmpnoisedict = noise.get_noise_from_file(noisefile)
        for og_key in tmpnoisedict.keys():
            split_key = og_key.split("_")
            psr_name = split_key[0]
            if psr_name == args.psr_name:
                if args.datarelease in ["5yr"]:
                    param = "_".join(split_key[1:])
                    new_key = "_".join([psr_name, "_".join(param.split("-"))])
                    noisedict[new_key] = tmpnoisedict[og_key]
                else:
                    noisedict[og_key] = tmpnoisedict[og_key]

# filter
is_psr = False
for p, t in zip(parfiles, timfiles):
    if p.split("/")[-1].split(".")[0].split("_")[0] == args.psr_name:
        psr = Pulsar(p, t, ephem=args.ephem, clk=None, drop_t2pulsar=False)
        is_psr = True

if not is_psr:
    raise ValueError(
        "{} does not exist in {} datarelease.".format(args.psr_name, args.datarelease)
    )
nltm_params = []
ltm_list = []
for par in psr.fitpars:
    if par != "Offset":
        nltm_params.append(par)
    else:
        ltm_list.append(par)

print("Non-linearly varying these values: ", nltm_params)

if args.fit_remaining_pars:
    print("Linearly varying these values: ", ltm_list)

print("Using ", args.tm_prior, " prior.")

#pbdot = 9.40616956524680049e-13
pbdot = 9.613818e-13
#pbdot_sigma = 1.697e-13
pbdot_sigma = 1.832471e-13

lower = pbdot - 5 * pbdot_sigma
upper = pbdot + 5 * pbdot_sigma
tm_param_dict={'PBDOT':{'prior_lower_bound':lower,
                      'prior_upper_bound':upper}}

pta = models.model_singlepsr_noise(
    psr,
    tm_var=True,
    tm_linear=False,
    tm_param_list=nltm_params,
    ltm_list=ltm_list,
    tm_param_dict=tm_param_dict,
    tm_prior=args.tm_prior,
    fit_remaining_pars=args.fit_remaining_pars,
    red_var=args.red_var,
    psd="powerlaw",
    red_select=None,
    noisedict=noisedict,
    tm_svd=False,
    tm_norm=True,
    white_vary=args.white_var,
    components=30,
    upper_limit=False,
    wideband=False,
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
    coefficients=args.coefficients,
    extra_sigs=None,
)

psampler = sampler.setup_sampler(pta, outdir=outdir, resume=args.resume, timing=True)

with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
    pickle.dump(psr.tm_params_orig, fout)

if args.coefficients:
    x0_list = []
    for p in pta.params:
        if "coefficients" not in p.name:
            x0_list.append(p.sample())
    x0 = np.asarray(x0_list)
else:
    x0 = np.hstack([p.sample() for p in pta.params])

psampler.sample(
    x0,
    N,
    SCAMweight=30,
    AMweight=15,
    DEweight=50,
    writeHotChains=args.writeHotChains,
    hotChain=args.reallyHotChain,
)
