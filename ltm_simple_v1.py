import numpy as np
import os, sys, inspect

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
# top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
e_path = top_dir + "/enterprise/"

sys.path.insert(0, e_e_path)
sys.path.insert(0, e_path)

from enterprise.pulsar import Pulsar

from enterprise_extensions import sampler
from enterprise_extensions import models

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

parfile = top_dir + "/{}/narrowband/par/{}_NANOGrav_12yv4.gls.par".format(
    args.datarelease, args.psr_name
)
timfile = top_dir + "/{}/narrowband/tim/{}_NANOGrav_12yv4.tim".format(
    args.datarelease, args.psr_name
)
print("Using {} Narrowband data".format(args.datarelease))


if len(args.parfile):
    parfile = args.parfile
if not os.path.isfile(parfile):
    raise ValueError(f"{parfile} does not exist. Please pick a real parfile.")

if len(args.timfile):
    timfile = args.timfile
if not os.path.isfile(timfile):
    raise ValueError(f"{timfile} does not exist. Please pick a real timfile.")

if args.tm_var:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(args.psr_name, args.datarelease)
        + args.psr_name
        + "_{}_nltm_ltm_{}".format(args.ephem, args.run_num
        )
    )
else:
    outdir = (
        current_path
        + "/{}/chains/{}/".format(args.psr_name, args.datarelease)
        + args.psr_name
        + "_{}_tm_{}".format(args.ephem, args.run_num
        )
    )

if not os.path.isdir(outdir):
    os.makedirs(outdir, exist_ok=True)
else:
    if not args.resume:
        print("nothing!")
        # raise ValueError("{} already exists!".format(outdir))

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

if args.tm_linear and args.tm_var:
    print("Using linear approximation for all timing parameters.")
else:
    print("Not varying timing parameters.")

model_args = inspect.getfullargspec(models.model_singlepsr_noise)
model_keys = model_args[0][1:]
model_vals = model_args[3]
model_kwargs = dict(zip(model_keys, model_vals))
model_kwargs.update(
    {
        "tm_var": args.tm_var,
        "tm_linear": args.tm_linear,
        "red_var": args.red_var,
        "noisedict": noisedict,
        "white_vary": args.white_var,
        "coefficients": args.coefficients,
    }
)

pta = models.model_singlepsr_noise(psr, **model_kwargs)

psampler = sampler.setup_sampler(
    pta,
    outdir=outdir,
    resume=args.resume,
    timing=False,
    psr=psr,
    restrict_mass=False,
)

x0 = np.hstack([p.sample() for p in pta.params])

psampler.sample(
    x0,
    N,
    SCAMweight=20,
    AMweight=20,
    DEweight=20,
    writeHotChains=args.writeHotChains,
    hotChain=args.reallyHotChain,
)
