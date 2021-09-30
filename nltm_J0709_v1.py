import numpy as np
import glob, os, sys, pickle, json, inspect
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
    "Whether to start the timing parameters at the parfile value. (DEFAULT: TRUE",
    True,
)
add_bool_arg(
    parser, "pal2_priors", "Whether to use PAL2 WN priors (DEFAULT: False)", False,
)

args = parser.parse_args()

if not isinstance(args.N, int):
    N = int(float(args.N))
else:
    N = args.N

if args.datarelease == "all":
    parfile = top_dir + "/12p5yr/J0740+6620/J0740+6620.prenoise.all.nchan64.par"
    timfile = top_dir + "/12p5yr/J0740+6620/J0740+6620.prenoise.all.nchan64.tim"
    print("Using All Data (CHIME+12.5yr+Cromartie et al. 2019)")
elif args.datarelease == "cfr+19":
    parfile = top_dir + "/12p5yr/J0740+6620/J0740+6620.cfr+19.par"
    timfile = top_dir + "/12p5yr/J0740+6620/J0740+6620.cfr+19.tim"
    print("Using Cromartie et al. 2019 data")
elif args.datarelease == "12p5yr":
    if args.wideband:
        if args.psr_name == "J1713+0747":
            parfile = (
                top_dir
                + "/{}/wideband/par/{}_NANOGrav_12yv3.wb.gls.t2.par".format(
                    args.datarelease, args.psr_name
                )
            )
        else:
            parfile = top_dir + "/{}/wideband/par/{}_NANOGrav_12yv3.wb.gls.par".format(
                args.datarelease, args.psr_name
            )
        timfile = top_dir + "/{}/wideband/tim/{}_NANOGrav_12yv3.wb.tim".format(
            args.datarelease, args.psr_name
        )
        print("Using {} Wideband data".format(args.datarelease))
    else:
        if args.psr_name == "J1713+0747":
            parfile = (
                top_dir
                + "/{}/narrowband/par/{}_NANOGrav_12yv3.gls.t2.par".format(
                    args.datarelease, args.psr_name
                )
            )
        else:
            parfile = top_dir + "/{}/narrowband/par/{}_NANOGrav_12yv3.gls.par".format(
                args.datarelease, args.psr_name
            )
        timfile = top_dir + "/{}/narrowband/tim/{}_NANOGrav_12yv3.tim".format(
            args.datarelease, args.psr_name
        )
        print("Using {} Narrowband data".format(args.datarelease))
elif args.datarelease == "prelim15yr":
    parfile = top_dir + "/{}/{}.working.par".format(args.datarelease, args.psr_name)
    timfile = top_dir + "/{}/{}.working.tim".format(args.datarelease, args.psr_name)
    print("Using {} data".format(args.datarelease))
elif args.datarelease == "15yr":
    parfile = top_dir + "/{}/{}/T2ized_J0709+0458_PINT_20210303.nb.par".format(
        args.datarelease, args.psr_name
    )
    timfile = top_dir + "/{}/{}/J0709+0458.combined.nb.tim".format(
        args.datarelease, args.psr_name
    )
    # timfile = top_dir + "/{}/{}/J0709+0458.L-wide.PUPPI.15y.x.nb.tim".format(args.datarelease, args.psr_name)
    print("Using {} data".format(args.datarelease))
else:
    datadir = top_dir + "/{}".format(args.datarelease)
    parfiles = sorted(glob.glob(datadir + "/par/*.par"))
    timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))
    if args.psr_name == "J1713+0747" and args.datarelease != "5yr":
        parfile = [
            pfile for pfile in parfiles if args.psr_name in pfile and "t2" in pfile
        ][0]
    elif args.psr_name == "J1640+2224" and args.datarelease == "5yr":
        parfiles = sorted(glob.glob(datadir + "/par/*_nltm.par"))
        parfile = [pfile for pfile in parfiles if args.psr_name in pfile][0]
    else:
        parfile = [pfile for pfile in parfiles if args.psr_name in pfile][0]
    timfile = [tfile for tfile in timfiles if args.psr_name in tfile][0]
    print("Using {} data".format(args.datarelease))

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
        lower = pbdot - 5 * pbdot_sigma * 1e-12
        upper = pbdot + 5 * pbdot_sigma * 1e-12
        tm_param_dict["PBDOT"] = {
            "prior_lower_bound": lower,
            "prior_upper_bound": upper,
        }
    elif par == "XDOT":
        xdot = np.double(psr.t2pulsar.vals()[psr.t2pulsar.pars().index(par)])
        xdot_sigma = np.double(psr.t2pulsar.errs()[psr.t2pulsar.pars().index(par)])
        print("USING PHYSICAL XDOT. Val: ", xdot, "Err: ", xdot_sigma * 1e-12)
        lower = xdot - 5 * xdot_sigma * 1e-12
        upper = xdot + 5 * xdot_sigma * 1e-12
        tm_param_dict["XDOT"] = {
            "prior_lower_bound": lower,
            "prior_upper_bound": upper,
        }

if args.tm_var and not args.tm_linear:
    if args.pal2_priors:
        s = timing_block(
            psr,
            tm_param_list=nltm_params,
            ltm_list=ltm_list,
            prior_type=args.tm_prior,
            prior_sigma=2.0,
            prior_lower_bound=-500.0,
            prior_upper_bound=500.0,
            tm_param_dict=tm_param_dict,
            fit_remaining_pars=args.fit_remaining_pars,
            wideband_kwargs={},
        )

        # red noise
        if args.red_var:
            s += red_noise_block(
                psd="powerlaw",
                prior="uniform",
                components=30,
                gamma_val=None,
                coefficients=False,
                select=None,
            )
        # define selection by observing backend
        backend = selections.Selection(selections.by_backend)
        # define selection by nanograv backends
        backend_ng = selections.Selection(selections.nanograv_backends)
        backend_ch = selections.Selection(channelized_backends)

        # white noise parameters
        efac = parameter.Uniform(0.001, 10.0)
        equad = parameter.Uniform(-10.0, -4.0)
        ecorr = parameter.Uniform(-8.5, -4.0)

        # white noise signals
        ef = white_signals.MeasurementNoise(efac=efac, selection=backend, name=None)
        eq = white_signals.EquadNoise(log10_equad=equad, selection=backend, name=None)

        if args.Ecorr_gp_basis:
            ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=backend_ch)
        else:
            ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=backend_ch)

        # combine signals
        s += ef + eq + ec
        model = s(psr)

        # set up PTA
        pta = signal_base.PTA([model])
    else:
        model_args = inspect.getfullargspec(models.model_singlepsr_noise)
        model_keys = model_args[0][1:]
        model_vals = model_args[3]
        model_kwargs = dict(zip(model_keys, model_vals))
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

    psampler = sampler.setup_sampler(
        pta, outdir=outdir, resume=args.resume, timing=True
    )
    with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
        pickle.dump(psr.tm_params_orig, fout)
else:
    if args.incTimingModel:
        # create new attribute for enterprise pulsar object
        # UNSURE IF NECESSARY
        psr.tm_params_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
        for key in psr.tm_params_orig:
            psr.tm_params_orig[key] = (psr.t2pulsar[key].val, psr.t2pulsar[key].err)
        s = gp_signals.TimingModel(use_svd=False, normed=True, coefficients=False)

    # define selection by observing backend
    backend = selections.Selection(selections.by_backend)
    # define selection by nanograv backends
    backend_ng = selections.Selection(selections.nanograv_backends)
    backend_ch = selections.Selection(channelized_backends)

    # white noise parameters
    efac = parameter.Uniform(0.001, 10.0)
    equad = parameter.Uniform(-10.0, -4.0)
    ecorr = parameter.Uniform(-8.5, -4.0)

    # white noise signals
    ef = white_signals.MeasurementNoise(efac=efac, selection=backend, name=None)
    eq = white_signals.EquadNoise(log10_equad=equad, selection=backend, name=None)

    if args.Ecorr_gp_basis:
        ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=backend_ch)
    else:
        ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=backend_ch)

    # combine signals
    if args.incTimingModel:
        s += ef + eq + ec
    else:
        s = ef + eq + ec

    model = s(psr)

    # set up PTA
    pta = signal_base.PTA([model])
    psampler = sampler.setup_sampler(
        pta, outdir=outdir, resume=args.resume, timing=args.incTimingModel
    )

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
