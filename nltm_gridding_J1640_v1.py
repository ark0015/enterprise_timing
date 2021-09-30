from __future__ import division

import numpy as np
import glob, os, sys, pickle, json

import enterprise
from enterprise.pulsar import Pulsar
from enterprise.signals import utils
from enterprise.signals import parameter
from enterprise.signals import white_signals
from enterprise.signals import signal_base
from enterprise.signals import selections

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
from enterprise_extensions.timing import timing_block

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
    "Whether to use non-linear plus linear timing model variations. (DEFAULT: TRUE)",
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
    False,
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
        parfile = top_dir + "/{}/wideband/par/{}_NANOGrav_12yv3.wb.gls.par".format(
            args.datarelease, args.psr_name
        )
        timfile = top_dir + "/{}/wideband/tim/{}_NANOGrav_12yv3.wb.tim".format(
            args.datarelease, args.psr_name
        )
        print("Using {} Wideband data".format(args.datarelease))
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
elif args.datarelease == "12p5yr_working":
    datadir = (
        top_dir + f"/enterprise_timing/{args.psr_name}/Fonseca_PAL2/J1640+2224/m2sini"
    )
    parfile = datadir + f"/{args.psr_name}.working.par"
    timfile = datadir + f"/{args.psr_name}.working.tim"
    print("Using {} data".format(args.datarelease))
else:
    datadir = top_dir + "/{}".format(args.datarelease)
    if args.datarelease == "5yr":
        parfiles = sorted(glob.glob(datadir + "/par/*_nltm.par"))
        print(parfiles)
    else:
        parfiles = sorted(glob.glob(datadir + "/par/*.par"))
    timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))
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
    elif par == "SINI":
        if args.sample_cos:
            nltm_params.append("COSI")
            if args.datarelease == "5yr":
                tm_param_dict["COSI"] = {
                    "prior_lower_bound": 0.0,
                    "prior_upper_bound": 1.0,
                }
        else:
            nltm_params.append(par)
            if args.datarelease == "5yr":
                tm_param_dict[par] = {
                    "prior_lower_bound": 0.0,
                    "prior_upper_bound": 1.0,
                }
    elif par == "PX":
        nltm_params.append(par)
        if args.datarelease == "5yr":
            tm_param_dict[par] = {
                "prior_type": "dm_dist_px_prior",
            }
    elif par == "M2":
        nltm_params.append(par)
        if args.datarelease == "5yr":
            tm_param_dict[par] = {
                "prior_lower_bound": 0.0,
                "prior_upper_bound": 10.0,
            }
    elif (
        par in ["ELONG", "ELAT", "F0", "F1"]
        and args.datarelease == "12p5yr_working"
        and args.lin_dmx_jump_fd
    ):
        ltm_list.append(par)
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

# define selection by observing backend
if args.datarelease == "5yr":
    s = timing_block(
        psr,
        tm_param_list=nltm_params,
        ltm_list=ltm_list,
        prior_type=args.tm_prior,
        prior_sigma=2.0,
        prior_lower_bound=-5.0,
        prior_upper_bound=5.0,
        tm_param_dict=tm_param_dict,
        fit_remaining_pars=args.fit_remaining_pars,
    )
    select = "none"
    if select == "backend":
        backend = selections.Selection(selections.by_backend)
    else:
        # define no selection
        backend = selections.Selection(selections.no_selection)
    # white noise parameters
    efac = parameter.Uniform(0.01, 10.0)

    # white noise signals
    s += white_signals.MeasurementNoise(efac=efac, selection=backend, name=None)

    model = s(psr)

    # set up PTA
    pta = signal_base.PTA([model])

    psampler = sampler.setup_sampler(
        pta, outdir=outdir, resume=args.resume, timing=True
    )

    with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
        pickle.dump(psr.tm_params_orig, fout)

else:
    pta = models.model_singlepsr_noise(
        psr,
        tm_var=args.tm_var,
        tm_linear=args.tm_linear,
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
        is_wideband=args.wideband,
        use_dmdata=args.wideband,
        dmjump_var=args.wideband,
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
    if args.tm_var:
        psampler = sampler.setup_sampler(
            pta, outdir=outdir, resume=args.resume, timing=True
        )

        with open(outdir + "/orig_timing_pars.pkl", "wb") as fout:
            pickle.dump(psr.tm_params_orig, fout)
    else:
        psampler = sampler.setup_sampler(
            pta, outdir=outdir, resume=args.resume, timing=False
        )

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
