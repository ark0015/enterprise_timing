import numpy as np
import glob, os, sys, pickle, json
import string, inspect, copy
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

from hypermodel_timing import TimingHyperModel
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
    if args.wideband:
        timfile = top_dir + "/{}/wideband/tim/{}_NANOGrav_12yv3.wb.tim".format(
            args.datarelease, args.psr_name
        )
        print("Using {} Wideband data".format(args.datarelease))
    else:
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
        lower = pbdot - 50 * pbdot_sigma * 1e-12
        upper = pbdot + 50 * pbdot_sigma * 1e-12
        # lower = pbdot - 5 * pbdot_sigma * 1e-12
        # upper = pbdot + 5 * pbdot_sigma * 1e-12
        tm_param_dict["PBDOT"] = {
            "prior_mu": pbdot,
            "prior_sigma": pbdot_sigma,
            "prior_lower_bound": lower,
            "prior_upper_bound": upper,
        }
    elif par == "XDOT":
        xdot = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
        xdot_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
        print("USING PHYSICAL XDOT. Val: ", xdot, "Err: ", xdot_sigma * 1e-12)
        lower = xdot - 50 * xdot_sigma * 1e-12
        upper = xdot + 50 * xdot_sigma * 1e-12
        # lower = xdot - 5 * xdot_sigma * 1e-12
        # upper = xdot + 5 * xdot_sigma * 1e-12
        tm_param_dict["XDOT"] = {
            "prior_mu": xdot,
            "prior_sigma": xdot_sigma,
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
                    lower = parr_val - 50 * parr_sigma
                    upper = parr_val + 50 * parr_sigma
                    tm_param_dict[f"{parr}"] = {
                        "prior_mu": parr_val,
                        "prior_sigma": parr_sigma,
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
                    "red_var": args.red_var,
                    "noisedict": noisedict,
                    "white_vary": args.white_var,
                    "is_wideband": args.wideband,
                    "use_dmdata": args.wideband,
                    "dmjump_var": args.wideband,
                    "coefficients": args.coefficients,
                }
            )
        pta = models.model_singlepsr_noise(psr, **model_dict)
elif not os.path.isfile(args.model_kwargs_file) and len(args.model_kwargs_file) > 0:
    raise ValueError(f"{args.model_kwargs_file} does not exist!")
else:
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
    #Second Round:
    """
    """
    red_psd = 'powerlaw'
    dm_nondiag_kernel = ['periodic','sq_exp','periodic_rfband','sq_exp_rfband']
    dm_sw_gp = [True,False] #Depends on Round 1
    dm_annual = False
    """
    """
    #Third Round:
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
    """
    """

    # Create list of pta models for our model selection
    # nmodels = len(dm_annuals) * len(dm_nondiag_kernel)
    nmodels = 6
    # nmodels = len(chrom_indices) * len(dm_nondiag_kernel)
    mod_index = np.arange(nmodels)

    ptas = dict.fromkeys(mod_index)
    model_dict = {}
    model_labels = []
    ct = 0
    for dm in dm_nondiag_kernel:
        # for add_cusp in dm_cusp:
        # for dm_sw in dm_sw_gp:
        for chrom_gp in chrom_gps:
            for chrom_kernel in chrom_kernels:
                if dm == "None":
                    dm_var = False
                else:
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
                    }
                )
                # if dm == "None" and dm_sw:
                #    pass
                if not chrom_gp and chrom_kernel == "sq_exp":
                    pass
                else:
                    # Instantiate single pulsar noise model
                    ptas[ct] = models.model_singlepsr_noise(psr, **kwargs)
                    # Add labels and kwargs to save for posterity and plotting.
                    # model_labels.append([string.ascii_uppercase[ct], dm, dm_sw])
                    model_labels.append(
                        [string.ascii_uppercase[ct], dm, chrom_gp, chrom_kernel]
                    )
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

if "0" in model_dict.keys():
    # Hypermodel
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
    # Instantiate a collection of models
    super_model = TimingHyperModel(ptas)

    psampler = super_model.setup_sampler(
        outdir=outdir, resume=args.resume, timing=True, empirical_distr=emp_dist_path
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
        pta, outdir=outdir, resume=args.resume, timing=args.incTimingModel
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
            else:
                x0_list.append(p.sample())
        x0 = np.asarray(x0_list)
    else:
        x0 = np.hstack([p.sample() for p in pta.params])


with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
    pickle.dump(psr.tm_params_orig, fout)

with open(outdir + "/model_kwargs.json", "w") as fout:
    json.dump(model_dict, fout, sort_keys=True, indent=4, separators=(",", ": "))

psampler.sample(
    x0,
    N,
    SCAMweight=30,
    AMweight=15,
    DEweight=30,
    writeHotChains=args.writeHotChains,
    hotChain=args.reallyHotChain,
)
