from __future__ import division

import numpy as np
import glob, os, sys, pickle, json


from enterprise.pulsar import Pulsar


import corner
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc

current_path = os.getcwd()
splt_path = current_path.split("/")
# top_path_idx = splt_path.index("akaiser")
# top_path_idx = splt_path.index("nanograv")
top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
noise_path = top_dir + "/pta_sim/pta_sim"
sys.path.insert(0, noise_path)
sys.path.insert(0, e_e_path)
import enterprise_extensions as e_e
from enterprise_extensions import sampler
from enterprise_extensions import models_2 as models
from enterprise_extensions.timing_2 import get_astrometric_priors
from enterprise_extensions.sampler import JumpProposal
import noise

# psrlist = ["J2317+1439"]
# psrlist = ["J1640+2224"]  # J1643-1224 J1640+2224 J1909-3744
psrlist = ["J1713+0747"]
# psrlist = ["J2145-0750"]

datarelease = "9yr"
tm_prior = "uniform"
ephem = "DE436"
white_vary = True
red_var = True
run_num = 1
resume = True

writeHotChains = True
reallyHotChain = False
datadir = top_dir + "/{}".format(datarelease)
outdir = (
    current_path
    + "/chains/{}/".format(datarelease)
    + psrlist[0]
    + "_{}_{}_PX_tm_{}/".format("_".join(tm_prior.split("-")), ephem, run_num)
)
# outdir = current_path + "/chains/" + "messing_around/"

parfiles = sorted(glob.glob(datadir + "/par/*.par"))
timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))

noisedict = {}
if datarelease in ["12p5yr"]:
    noisefiles = sorted(glob.glob(top_dir + "/{}/*.json".format(datarelease)))
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
                if datarelease in ["5yr"]:
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
    psr = Pulsar(p, t, ephem=ephem, clk=None, drop_t2pulsar=False)
    psrs.append(psr)

tm_params_nodmx = []
for psr in psrs:
    for par in psr.fitpars:
        if "DMX" in ["".join(list(x)[0:3]) for x in par.split("_")][0]:
            pass
        elif "FD" in ["".join(list(x)[0:2]) for x in par.split("_")][0]:
            pass
        elif "JUMP" in ["".join(list(x)[0:4]) for x in par.split("_")][0]:
            pass
        elif par in ["Offset", "TASC"]:
            pass
        elif par in ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA"]:
            pass
        elif par in ["F0"]:
            pass
        # elif par in ["PMRA", "PMDEC", "PMELONG", "PMELAT", "PMBETA", "PMLAMBDA"]:
        #    pass
        else:
            tm_params_nodmx.append(par)

# tm_param_list = ['F0', 'F1', 'PX', 'PB', 'A1', 'EPS1', 'EPS2', 'EPS1DOT', 'EPS2DOT']
# tm_param_list = [ 'PB', 'A1', 'XDOT', 'TASC', 'EPS1', 'EPS2', 'H3', 'H4']
# tm_param_list = [ 'PB', 'A1', 'EPS1', 'EPS2', 'EPS1DOT', 'EPS2DOT']
# tm_param_list = [ 'PB', 'A1', 'EPS1', 'EPS2']
# tm_param_list = ['F0', 'F1', 'PB', 'T0', 'A1', 'OM', 'ECC', 'M2']
tm_param_list = tm_params_nodmx
print("Sampling these values:\n", tm_param_list, "\nin pulsar ", psrlist[0])
print("Using ", tm_prior, " prior and astrometric priors.")
px_priors = get_astrometric_priors(
    astrometric_px_file=top_dir + "/enterprise_timing/parallaxes.json"
)

tm_val = float(px_priors[psrlist[0]]["PI"][0])
tm_sigma = 2.0 * max(
    float(px_priors[psrlist[0]]["eminus"][0]), float(px_priors[psrlist[0]]["eplus"][0])
)
tm_lower_bound = tm_val + 3.0 * float(px_priors[psrlist[0]]["eminus"][0])
tm_upper_bound = tm_val + 3.0 * float(px_priors[psrlist[0]]["eplus"][0])

tm_param_dict = {
    "PX": {
        "prior_mu": tm_val,
        "prior_sigma": tm_sigma,
        "prior_lower_bound": tm_lower_bound,
        "prior_upper_bound": tm_upper_bound,
    }
}

pta = models.model_general(
    psrs,
    tm_var=True,
    tm_linear=False,
    tm_param_list=tm_param_list,
    tm_param_dict=tm_param_dict,
    tm_prior=tm_prior,
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
    white_vary=white_vary,
    gequad=False,
    dm_chrom=False,
    dmchrom_psd="powerlaw",
    dmchrom_idx=4,
    red_var=red_var,
    red_select=None,
    red_breakflat=False,
    red_breakflat_fq=None,
    coefficients=False,
)

# dimension of parameter space
params = pta.param_names
ndim = len(params)
# initial jump covariance matrix
cov = np.diag(np.ones(ndim) * 0.1 ** 2)

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
    resume=resume,
)
np.savetxt(outdir + "/pars.txt", list(map(str, pta.param_names)), fmt="%s")
np.savetxt(
    outdir + "/priors.txt",
    list(map(lambda x: str(x.__repr__()), pta.params)),
    fmt="%s",
)
jp = JumpProposal(pta)
psampler.addProposalToCycle(jp.draw_from_signal("non_linear_timing_model"), 30)
for p in pta.params:
    for cat in ["pos", "pm", "spin", "kep", "gr"]:
        if cat in p.name.split("_"):
            psampler.addProposalToCycle(jp.draw_from_par_prior(p.name), 30)

# sampler for N steps
N = int(4e6)
x0 = np.hstack(p.sample() for p in pta.params)
psampler.sample(
    x0,
    N,
    SCAMweight=30,
    AMweight=15,
    DEweight=50,
    writeHotChains=writeHotChains,
    hotChain=reallyHotChain,
)
