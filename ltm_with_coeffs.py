import numpy as np
import os, sys
from collections import OrderedDict

from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index("nanograv")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
e_path = top_dir + "/enterprise"
sys.path.insert(0, e_e_path)
sys.path.insert(0, e_path)

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
parser.add_argument("--run_num", required=True, help="Label at end of output file.")
parser.add_argument(
    "--ephem", default="DE436", help="Ephemeris option (DEFAULT: DE436)"
)
parser.add_argument("--N", default=int(1e6), help="Number of samples (DEFAULT: 1e6)")
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
parser.add_argument(
    "--parfile", default="", help="Location of parfile </PATH/TO/FILE/PARFILE.par>"
)
parser.add_argument(
    "--timfile", default="", help="Location of timfile </PATH/TO/FILE/TIMFILE.tim>"
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

# filter
is_psr = False
if args.psr_name in parfile:
    psr = Pulsar(parfile, timfile, ephem=args.ephem, clk=None, drop_t2pulsar=False)
    is_psr = True

if not is_psr:
    raise ValueError(f"{args.psr_name} does not exist in {parfile} or {timfile}.")

outdir = f"{current_path}/{args.psr_name}/chains/misc/{args.run_num}"
# create new attribute for enterprise pulsar object
# UNSURE IF NECESSARY
psr.tm_params_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
for key in psr.tm_params_orig:
    psr.tm_params_orig[key] = (psr.t2pulsar[key].val, psr.t2pulsar[key].err)
s = gp_signals.TimingModel(use_svd=False, normed=False, coefficients=args.coefficients)

# define selection by observing backend
backend = selections.Selection(selections.by_backend)
# define selection by nanograv backends
backend_ng = selections.Selection(selections.nanograv_backends)
backend_ch = selections.Selection(channelized_backends)

# white noise parameters
efac = parameter.Uniform(0.01, 10.0)
equad = parameter.Uniform(-8.5, -5.0)
ecorr = parameter.Uniform(-8.5, -5.0)

# white noise signals
ef = white_signals.MeasurementNoise(efac=efac, selection=backend, name=None)
eq = white_signals.EquadNoise(log10_equad=equad, selection=backend, name=None)

# ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=backend_ch,coefficients=args.coefficients)
ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=backend_ch)

# combine signals
s += ef + eq + ec

model = s(psr)

# set up PTA
pta = signal_base.PTA([model])
psampler = sampler.setup_sampler(pta, outdir=outdir, resume=args.resume, timing=True)


for p in pta.params:
    print(p.name)
    try:
        print(p.sample())
    except:
        print(p.size)
        print("Can't sample parameter")
    print("--------------")

x0_dict = {}
cpar = []
for p in pta.params:
    print(p)
    if "coefficients" in p.name:
        # x0_dict.update({p.name:np.random.randn(p.size)})
        print("not adding")
    else:
        x0_dict.update({p.name: p.sample()})

print(x0_dict)
print("----------------------")
psc = utils.get_coefficients(pta, x0_dict, variance=False)
print(psc)
for key, val in psc.items():
    print(key)
    if not isinstance(val, (float, int)):
        print(len(val))
    else:
        print(val)
    print(val)
    print("")
"""
psampler.sample(
    [x0 for x0 in x0_dict.values()],
    N,
    SCAMweight=30,
    AMweight=15,
    DEweight=30,
    writeHotChains=args.writeHotChains,
    hotChain=args.reallyHotChain,
)
"""
