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

parser.add_argument("--psrlist", required=True, help="name of pulsar used for search")

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
add_bool_arg(parser, "tm_var", "Whether to vary timing model. (DEFAULT: True)", True)
add_bool_arg(
    parser,
    "tm_linear",
    "Whether to use only the linear timing model. (DEFAULT: FALSE)",
    False,
)

add_bool_arg(
    parser,
    "nltm_plus_ltm",
    "Whether to use non-linear plus linear timing model variations. (DEFAULT: FALSE)",
    False,
)
add_bool_arg(
    parser,
    "fullnltm",
    "Whether to include all fitparameters in the non-linear model, or just cool ones. (DEFAULT: FALSE)",
    False,
)

add_bool_arg(
    parser,
    "exclude",
    "Whether to exclude non-linear parameters from "
    + "linear timing model variations, or only include some parameters in linear timing model. (DEFAULT: TRUE)",
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
if isinstance(args.psrlist, str):
    psrlist = [args.psrlist]
elif isinstance(args.psrlist, list):
    psrlist = args.psrlist
else:
    raise ValueError("Pulsar name must be a string or list of strings.")

if not isinstance(args.N, int):
    N = int(float(args.N))
else:
    N = args.N


# psrlist = ["J1744-1134"]
# datarelease = '5yr'
# tm_prior = "uniform"
# ephem = 'DE438'
# white_vary = True
# red_var = True

# run_num = 2
# resume = True
# sampler for N steps
# N = int(1e6)

# coefficients = False
# tm_var=True
# nltm_plus_ltm = False
# exclude = True

# writeHotChains = True
# reallyHotChain = False
datadir = top_dir + "/{}".format(args.datarelease)

if args.nltm_plus_ltm:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(psrlist[0], args.datarelease)
        + psrlist[0]
        + "_{}_{}_nltm_ltm_{}".format(
            "_".join(args.tm_prior.split("-")), args.ephem, args.run_num
        )
    )
else:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(psrlist[0], args.datarelease)
        + psrlist[0]
        + "_{}_{}_tm_{}".format(
            "_".join(args.tm_prior.split("-")), args.ephem, args.run_num
        )
    )
    # outdir = current_path + "/chains/{}/".format(args.datarelease) + psrlist[0] +\
    # "_{}_{}_nltm_{}/".format("_".join(args.tm_prior.split('-')),args.ephem,args.run_num)
# outdir = current_path + "/chains/{}/".format(args.datarelease) + psrlist[0] + "_testing_uniform_tm_3/"


if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)
else:
    if not args.resume:
        raise ValueError("{} already exists!".format(outdir))

parfiles = sorted(glob.glob(datadir + "/par/*.par"))
timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))

noisedict = {}
if args.datarelease in ["12p5yr"]:
    noisefiles = sorted(glob.glob(top_dir + "/{}/*.json".format(args.datarelease)))
    for noisefile in noisefiles:
        tmpnoisedict = {}
        with open(noisefile, "r") as fin:
            tmpnoisedict.update(json.load(fin))
        for key in tmpnoisedict.keys():
            if key.split("_")[0] in psrlist:
                noisedict[key] = tmpnoisedict[key]
else:
    noisefiles = sorted(glob.glob(datadir + "/noisefiles/*.txt"))
    for noisefile in noisefiles:
        tmpnoisedict = {}
        tmpnoisedict = noise.get_noise_from_file(noisefile)
        for og_key in tmpnoisedict.keys():
            split_key = og_key.split("_")
            psr_name = split_key[0]
            if psr_name in psrlist:
                if args.datarelease in ["5yr"]:
                    param = "_".join(split_key[1:])
                    new_key = "_".join([psr_name, "_".join(param.split("-"))])
                    noisedict[new_key] = tmpnoisedict[og_key]
                else:
                    noisedict[og_key] = tmpnoisedict[og_key]

# filter
parfiles = [
    x for x in parfiles if x.split("/")[-1].split(".")[0].split("_")[0] in psrlist
]
timfiles = [
    x for x in timfiles if x.split("/")[-1].split(".")[0].split("_")[0] in psrlist
]

psrs = []
for p, t in zip(parfiles, timfiles):
    psr = Pulsar(p, t, ephem=args.ephem, clk=None, drop_t2pulsar=False)
    psrs.append(psr)

if len(psrs) == 0:
    raise ValueError(
        "{} does not exist in {} datarelease.".format(psrlist[0], args.datarelease)
    )
nltm_params = []
ltm_exclude_list = []
for psr in psrs:
    for par in psr.fitpars:
        if args.fullnltm:
            nltm_params.append(par)
        else:
            if "DMX" in ["".join(list(x)[0:3]) for x in par.split("_")][0]:
                pass
            elif "FD" in ["".join(list(x)[0:2]) for x in par.split("_")][0]:
                pass
            elif "JUMP" in ["".join(list(x)[0:4]) for x in par.split("_")][0]:
                pass
            elif par in ["Offset", "TASC"]:
                pass
            elif par in ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA"]:
                ltm_exclude_list.append(par)
            elif par in ["F0"]:
                ltm_exclude_list.append(par)
            # elif par in ["PMRA", "PMDEC", "PMELONG", "PMELAT", "PMBETA", "PMLAMBDA"]:
            #    pass
            else:
                nltm_params.append(par)

if not args.tm_linear and args.tm_var:
    print(
        "Non-linearly varying these values: ", nltm_params, "\n in pulsar ", psrlist[0]
    )
elif args.tm_linear and args.tm_var:
    print("Using linear approximation for all timing parameters.")
else:
    print("Not varying timing parameters.")

if args.nltm_plus_ltm:
    if args.exclude:
        ltm_exclude_list = nltm_params
        print(
            "Linearly varying everything but these values: ",
            ltm_exclude_list,
            "\n in pulsar ",
            psrlist[0],
        )
    else:
        print(
            "Linearly varying only these values: ",
            ltm_exclude_list,
            "\n in pulsar ",
            psrlist[0],
        )

print("Using ", args.tm_prior, " prior.")

pta = models.model_general(
    psrs,
    tm_var=args.tm_var,
    tm_linear=args.tm_linear,
    tm_param_list=nltm_params,
    ltm_exclude_list=ltm_exclude_list,
    exclude=args.exclude,
    tm_param_dict={},
    tm_prior=args.tm_prior,
    nltm_plus_ltm=args.nltm_plus_ltm,
    common_psd="powerlaw",
    red_psd="powerlaw",
    orf=None,
    common_var=False,
    common_components=30,
    red_components=30,
    dm_components=30,
    modes=None,
    wgts=None,
    logfreq=False,
    nmodes_log=10,
    noisedict=noisedict,
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
    white_vary=args.white_var,
    gequad=False,
    dm_chrom=False,
    dmchrom_psd="powerlaw",
    dmchrom_idx=4,
    red_var=args.red_var,
    red_select=None,
    red_breakflat=False,
    red_breakflat_fq=None,
    coefficients=args.coefficients,
)

# dimension of parameter space
params = pta.param_names
ndim = len(params)

# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.1**2)

# parameter groupings
groups = sampler.get_parameter_groups(pta)
tm_groups = sampler.get_timing_groups(pta)
for tm_group in tm_groups:
    groups.append(tm_group)

wn_pars = ["ecorr", "equad", "efac"]
groups.append(sampler.group_from_params(pta, wn_pars))

psampler = ptmcmc(
    ndim,
    pta.get_lnlikelihood,
    pta.get_lnprior,
    cov,
    groups=groups,
    outDir=outdir,
    resume=args.resume,
)

np.savetxt(outdir + "/pars.txt", list(map(str, pta.param_names)), fmt="%s")
np.savetxt(
    outdir + "/priors.txt",
    list(map(lambda x: str(x.__repr__()), pta.params)),
    fmt="%s",
)

if args.tm_var and not args.tm_linear:
    jp = JumpProposal(pta)
    psampler.addProposalToCycle(jp.draw_from_signal("non_linear_timing_model"), 30)
    for p in pta.params:
        for cat in ["pos", "pm", "spin", "kep", "gr"]:
            if cat in p.name.split("_"):
                psampler.addProposalToCycle(jp.draw_from_par_prior(p.name), 30)

tmp = True
# if args.coefficients:
"""if tmp:
    x0_dict = {}
    cpar = []
    for p in pta.params:
        print(p)
        if "coefficients" in p.name:
            cpar.append(p)
        else:
            x0_dict.update({p.name:p.sample()})

    pr2 = cpar[0].get_logpdf(params=x0_dict)
    print(pr2)
    psc = utils.get_coefficients(pta, x0_dict)
    print(psc)"""
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
