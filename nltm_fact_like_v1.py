import numpy as np
import glob, os, sys, pickle, json
import string, inspect, copy
from collections import OrderedDict

import enterprise
from enterprise.pulsar import Pulsar

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
add_bool_arg(parser, "resume", "Whether to resume the chains. (DEFAULT: FALSE", False)
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
    "zero_start",
    "Whether to start the timing parameters at the parfile value. (DEFAULT: TRUE",
    True,
)
parser.add_argument(
    "--parfile", default="", help="Location of parfile </PATH/TO/FILE/PARFILE.par>"
)
parser.add_argument(
    "--timfile", default="", help="Location of timfile </PATH/TO/FILE/TIMFILE.tim>"
)
parser.add_argument(
    "--model_kwargs_file", default="", help="Location of model_kwargs_file"
)
parser.add_argument(
    "--emp_dist_path", default="", help="Location of empirical distribution"
)

args = parser.parse_args()

if not isinstance(args.N, int):
    N = int(float(args.N))
else:
    N = args.N

if args.datarelease == "all" and args.psr_name == "J0740+6620":
    timfile = top_dir + "/12p5yr/J0740+6620/J0740+6620.prenoise.all.nchan64.tim"
    print("Using All Data (CHIME+12.5yr+Cromartie et al. 2019)")
elif args.datarelease == "fcp+21" and args.psr_name == "J0740+6620":
    timfile = top_dir + "/12p5yr/J0740+6620/J0740+6620.FCP+21.nb.tim"
    print("Using Data From Fonseca+21")
elif args.datarelease == "cfr+19" and args.psr_name == "J0740+6620":
    timfile = top_dir + "/12p5yr/J0740+6620/J0740+6620.cfr+19.tim"
    print("Using Cromartie et al. 2019 data")
elif args.datarelease == "12p5yr":
    Tspan = 407576851.48121357
    print(Tspan / (365.25 * 24 * 3600), " yrs")
    timfile = top_dir + "/{}/narrowband/tim/{}_NANOGrav_12yv3.tim".format(
        args.datarelease, args.psr_name
    )
    print("Using {} Narrowband data".format(args.datarelease))
elif args.datarelease == "prelim15yr":
    timfile = top_dir + "/{}/{}.working.tim".format(args.datarelease, args.psr_name)
    print("Using {} data".format(args.datarelease))
elif args.datarelease == "15yr" and args.psr_name == "J0709+0458":
    timfile = top_dir + "/{}/{}/J0709+0458.combined.nb.tim".format(
        args.datarelease, args.psr_name
    )
    # timfile = top_dir + "/{}/{}/J0709+0458.L-wide.PUPPI.15y.x.nb.tim".format(args.datarelease, args.psr_name)
    print("Using {} data".format(args.datarelease))
else:
    datadir = top_dir + "/{}".format(args.datarelease)
    timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))
    timfile = [tfile for tfile in timfiles if args.psr_name in tfile][0]
    print("Using {} data".format(args.datarelease))

if len(args.parfile):
    parfile = args.parfile
if not os.path.isfile(parfile):
    raise ValueError(f"{parfile} does not exist. Please pick a real parfile.")

if len(args.timfile):
    timfile = args.timfile
if not os.path.isfile(timfile):
    raise ValueError(f"{timfile} does not exist. Please pick a real timfile.")

outdir = f"{current_path}/{args.psr_name}/chains/{args.datarelease}/{args.psr_name}_factorized_like_run_{args.ephem}_{args.run_num}"

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
    psr = Pulsar(parfile, timfile, ephem=args.ephem, clk=None, drop_t2pulsar=False)
    is_psr = True

if not is_psr:
    raise ValueError(
        "{} does not exist in {} datarelease.".format(args.psr_name, args.datarelease)
    )

if os.path.isfile(args.model_kwargs_file):
    print("loading model kwargs from file...")
    with open(args.model_kwargs_file, "r") as fin:
        model_dict = json.load(fin)

    model_args = inspect.getfullargspec(models.model_singlepsr_noise)
    model_keys = model_args[0][1:]
    model_vals = model_args[3]
    model_kwargs = dict(zip(model_keys, model_vals))
    tmp_dict = copy.deepcopy(model_dict)
    model_kwargs.update(tmp_dict)

    model_kwargs.update(
        {
            "factorized_like": True,
            "Tspan": Tspan,
            "fact_like_gamma": 13.0 / 3,
            "gw_components": 5,
            "psd": "powerlaw",
        }
    )
    # Instantiate single pulsar noise model
    pta = models.model_singlepsr_noise(psr, **model_kwargs)
else:
    raise ValueError(
        "Must use previous model kwargs by pointing to the model json file using --model_kwargs_file = <PATH/TO/FILE>"
    )

if os.path.isfile(args.emp_dist_path):
    emp_dist_path = args.emp_dist_path
else:
    print("No empirical distribution used.")
    emp_dist_path = None

psampler = sampler.setup_sampler(
    pta, outdir=outdir, resume=args.resume, timing=True, empirical_distr=emp_dist_path
)

achrom_freqs_fL = np.linspace(1 / Tspan, 10 / Tspan, 10)
np.savetxt(outdir + "/achrom_rn_freqs.txt", achrom_freqs_fL, fmt="%.18e")

with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
    pickle.dump(psr.tm_params_orig, fout)

with open(outdir + "/model_kwargs.json", "w") as fout:
    json.dump(model_kwargs, fout, sort_keys=True, indent=4, separators=(",", ": "))

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
                if p_name in model_kwargs["tm_param_dict"].keys():
                    x0_list.append(
                        np.double(model_kwargs["tm_param_dict"][p_name]["prior_mu"])
                    )
                else:
                    x0_list.append(np.double(psr.tm_params_orig[p_name][0]))
        else:
            x0_list.append(p.sample())
    x0 = np.asarray(x0_list)
else:
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
