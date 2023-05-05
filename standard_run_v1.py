import numpy as np
import glob, os, sys, pickle, json, inspect
from collections import OrderedDict

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

current_path = os.getcwd()
splt_path = current_path.split("/")
# top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
noise_path = top_dir + "/pta_sim/pta_sim"
# e_path = top_dir + "/enterprise/"

sys.path.insert(0, noise_path)
sys.path.insert(0, e_e_path)
# sys.path.insert(0, e_path)

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
    "fact_like",
    "Whether to do a factorized likelihood run (DEFAULT: False)",
    False,
)
parser.add_argument(
    "--parfile", default="", help="Location of parfile </PATH/TO/FILE/PARFILE.par>"
)
parser.add_argument(
    "--timfile", default="", help="Location of timfile </PATH/TO/FILE/TIMFILE.tim>"
)
parser.add_argument(
    "--timing_package",
    default="tempo2",
    help="Whether to use PINT or Tempo2 (DEFAULT: tempo2)",
)

args = parser.parse_args()

if not isinstance(args.N, int):
    N = int(float(args.N))
else:
    N = args.N

if len(args.parfile):
    parfile = args.parfile
if not os.path.isfile(parfile):
    raise ValueError(f"{parfile} does not exist. Please pick a real parfile.")

if len(args.timfile):
    timfile = args.timfile
if not os.path.isfile(timfile):
    raise ValueError(f"{timfile} does not exist. Please pick a real timfile.")

outdir = (
    current_path
    + f"/{args.psr_name}/chains/{args.datarelease}/{args.psr_name}_{args.ephem}_standard_run_{args.run_num}"
)

if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)
else:
    if not args.resume:
        print("nothing!")
        # raise ValueError("{} already exists!".format(outdir))

noisedict = {}
if args.datarelease in ["12p5yr", "cfr+19"]:
    noisefiles = sorted(glob.glob(top_dir + "/12p5yr/*.json"))
    for noisefile in noisefiles:
        tmpnoisedict = {}
        with open(noisefile, "r") as fin:
            tmpnoisedict.update(json.load(fin))
        for key in tmpnoisedict.keys():
            if key.split("_")[0] == args.psr_name:
                noisedict[key] = tmpnoisedict[key]
elif args.datarelease in ["5yr", "9yr", "11yr"]:
    noisefiles = sorted(glob.glob(datadir + "/noisefiles/*.txt"))
    for noisefile in noisefiles:
        tmpnoisedict = {}
        tmpnoisedict = noise.get_noise_from_file(noisefile)
        for og_key in tmpnoisedict.keys():
            split_key = og_key.split("_")
            psr_name = split_key[0]
            if psr_name == args.psr_name or args.datarelease == "5yr":
                if args.datarelease == "5yr":
                    param = "_".join(split_key[1:])
                    new_key = "_".join([psr_name, "_".join(param.split("-"))])
                    noisedict[new_key] = tmpnoisedict[og_key]
                else:
                    noisedict[og_key] = tmpnoisedict[og_key]
else:
    noisedict = None

if not args.white_var:
    with open(parfile, "r") as f:
        lines = f.readlines()
    noisedict = {}
    for line in lines:
        splt_line = line.split()
        if "T2EFAC" in splt_line[0]:
            noisedict[f"{args.psr_name}_{splt_line[2]}_efac"] = np.float64(splt_line[3])
        if "T2EQUAD" in splt_line[0]:
            noisedict[f"{args.psr_name}_{splt_line[2]}_log10_equad"] = np.log10(
                np.float64(splt_line[3])
            )
        if "ECORR" in splt_line[0]:
            noisedict[f"{args.psr_name}_{splt_line[2]}_log10_ecorr"] = np.log10(
                np.float64(splt_line[3])
            )

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


model_args = inspect.getfullargspec(models.model_singlepsr_noise)
model_keys = model_args[0][1:]
model_vals = model_args[3]
model_kwargs = dict(zip(model_keys, model_vals))
model_kwargs.update(
    {
        "red_var": args.red_var,
        "noisedict": noisedict,
        "white_vary": args.white_var,
        "is_wideband": args.wideband,
        "use_dmdata": args.wideband,
        "dmjump_var": args.wideband,
    }
)
if args.fact_like:
    if args.datarelease == "12p5yr":
        Tspan = 407576851.48121357
        print(Tspan / (365.25 * 24 * 3600), " yrs")
    else:
        raise ValueError("Only have Tspan for 12.5-yr.")
    model_kwargs.update(
        {
            "factorized_like": True,
            "Tspan": Tspan,
            "fact_like_gamma": 13.0 / 3,
            "gw_components": 5,
            "psd": "powerlaw",
        }
    )
# print(model_kwargs)

pta = models.model_singlepsr_noise(psr, **model_kwargs)

psampler = sampler.setup_sampler(pta, outdir=outdir, resume=args.resume, timing=False)

x0 = np.hstack([p.sample() for p in pta.params])

psampler.sample(
    x0,
    N,
    SCAMweight=30,
    AMweight=15,
    DEweight=30,
    writeHotChains=args.writeHotChains,
    hotChain=args.reallyHotChain,
)
