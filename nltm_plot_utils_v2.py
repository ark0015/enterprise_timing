import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import numpy as np
import scipy.stats as sps
import json, pickle, os, corner, glob
import enterprise.signals.utils as utils
from enterprise.pulsar import Pulsar
import arviz as az
from collections import defaultdict
import pandas as pd

import la_forge.diagnostics as dg
import la_forge.core as co
from la_forge.rednoise import plot_rednoise_spectrum, plot_free_spec
from la_forge.utils import epoch_ave_resid

color_cycle_wong = [
    "#000000",
    "#E69F00",
    "#009E73",
    "#56B4E9",
    "#0072B2",
    "#F0E442",
    "#D55E00",
    "#CC79A7",
]
mpl.rcParams["axes.prop_cycle"] = cycler(color=color_cycle_wong)

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index("akaiser")
# top_path_idx = splt_path.index("nanograv")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])


def get_psrs(psrlist, datareleases):
    """Loads in par and tim files for all psrs in psrlist for each datarelease in datareleases.
    :psrlist:
        list, list of pulsars
    :datareleases:
        list, list of datareleases
    """
    datadir_list = []
    parfiles = []
    timfiles = []
    for datarelease in datareleases:
        datadir = top_dir + "/{}".format(datarelease)
        tmp_parfiles = sorted(glob.glob(datadir + "/par/*.par"))
        tmp_timfiles = sorted(glob.glob(datadir + "/tim/*.tim"))
        # filter
        parfiles = np.concatenate(
            (
                parfiles,
                [
                    x
                    for x in tmp_parfiles
                    if x.split("/")[-1].split(".")[0].split("_")[0] in psrlist
                ],
            ),
            axis=0,
        )
        timfiles = np.concatenate(
            (
                timfiles,
                [
                    x
                    for x in tmp_timfiles
                    if x.split("/")[-1].split(".")[0].split("_")[0] in psrlist
                ],
            ),
            axis=0,
        )

    psrs = []
    for p, t in zip(parfiles, timfiles):
        psr = Pulsar(p, t, ephem="DE436", clk=None, drop_t2pulsar=False)
        psrs.append(psr)

    return psrs


def get_pardict(psrs, datareleases):
    """assigns a parameter dictionary for each psr per dataset the parfile values/errors
    :psrs:
        objs, enterprise pulsar instances corresponding to datareleases
    :datareleases:
        list, list of datareleases
    """
    pardict = {}
    for psr, dataset in zip(psrs, datareleases):
        pardict[dataset] = {}
        print(dataset)
        pardict[dataset][psr.name] = {}
        for par, vals, errs in zip(
            psr.fitpars[1:], psr.t2pulsar.vals(), psr.t2pulsar.errs()
        ):
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
            # elif par in ["PMRA", "PMDEC", "PMELONG", "PMELAT", "PMBETA", "PMLAMBDA"]:
            #    pass
            else:
                pardict[dataset][psr.name][par] = {}
                pardict[dataset][psr.name][par]["val"] = vals
                pardict[dataset][psr.name][par]["err"] = errs
    return pardict


def get_chaindir_indices(chaindir_list):
    """separates list of chains by priors, puts corresponding indices in a list in a dictionary
    :chaindir_list:
        list, list of chain locations
    """
    chaindir_indices = {}

    for i, chaindir in enumerate(chaindir_list):
        sep = chaindir.split("/")
        dataset = sep[8]
        name = sep[-1]
        splt_name = name.split("_")
        if "uniform" in splt_name and not "timing" in splt_name:
            if "uniform" not in chaindir_indices.keys():
                chaindir_indices["uniform"] = {}
            if "PX" in splt_name and not any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                if "vlbi_priors" not in chaindir_indices["uniform"].keys():
                    chaindir_indices["uniform"]["vlbi_priors"] = defaultdict(list)
                chaindir_indices["uniform"]["vlbi_priors"][dataset].append(i)
            elif "PX" in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "vlbi_priors_" + item
                        if new_item not in chaindir_indices["uniform"].keys():
                            chaindir_indices["uniform"][new_item] = defaultdict(list)
                        chaindir_indices["uniform"][new_item][dataset].append(i)
            elif "PX" not in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "parfile_priors_" + item
                        if new_item not in chaindir_indices["uniform"].keys():
                            chaindir_indices["uniform"][new_item] = defaultdict(list)
                        chaindir_indices["uniform"][new_item][dataset].append(i)
            else:
                if "parfile_priors" not in chaindir_indices["uniform"].keys():
                    chaindir_indices["uniform"]["parfile_priors"] = defaultdict(list)
                chaindir_indices["uniform"]["parfile_priors"][dataset].append(i)
        elif "bounded" in splt_name:
            if "bounded" not in chaindir_indices.keys():
                chaindir_indices["bounded"] = {}
            if "PX" in splt_name and not any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                if "vlbi_priors" not in chaindir_indices["bounded"].keys():
                    chaindir_indices["bounded"]["vlbi_priors"] = defaultdict(list)
                chaindir_indices["bounded"]["vlbi_priors"][dataset].append(i)
            elif "PX" in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "vlbi_priors_" + item
                        if new_item not in chaindir_indices["bounded"].keys():
                            chaindir_indices["bounded"][new_item] = defaultdict(list)
                        chaindir_indices["bounded"][new_item][dataset].append(i)
            elif "PX" not in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "parfile_priors_" + item
                        if new_item not in chaindir_indices["bounded"].keys():
                            chaindir_indices["bounded"][new_item] = defaultdict(list)
                        chaindir_indices["bounded"][new_item][dataset].append(i)
            else:
                if "parfile_priors" not in chaindir_indices["bounded"].keys():
                    chaindir_indices["bounded"]["parfile_priors"] = defaultdict(list)
                chaindir_indices["bounded"]["parfile_priors"][dataset].append(i)
        else:
            if "misc" not in chaindir_indices.keys():
                chaindir_indices["misc"] = {}
            chaindir_indices["misc"][dataset].append(i)
    return chaindir_indices


def get_chain_tmparam_lists(chaindir_list, burn=None):
    """separates list of chains by priors, puts corresponding indices in a list in a dictionary
    :chaindir_list:
        list, list of chain locations
    """
    chain_list = []
    tmparam_list = []
    for chaindir in chaindir_list:
        tmp_tmparam = []
        tmp_tmparam.extend(np.loadtxt(chaindir + "/pars.txt", dtype="S").astype("U"))
        tmp_tmparam.extend(("lnlike", "lnprior", "chain accept", "pt chain accept"))
        tmparam_list.append(tmp_tmparam)

        chainpaths = sorted(glob.glob(chaindir + "/chain*.txt"))
        chain = pd.read_csv(chainpaths[0], sep="\t", dtype=float, header=None).values
        hot_chains = {}
        if len(chainpaths) > 1:
            for chp in chainpaths[1:]:
                try:
                    ch = pd.read_csv(chp, sep="\t", dtype=float, header=None).values
                    ky = chp.split("/")[-1].split("_")[-1].replace(".txt", "")
                    hot_chains.update({ky: ch})
                except:
                    print(chp, "cant be loaded.")

        if not isinstance(burn, int):
            if burn is None:
                burn = int(0.25 * chain.shape[0])
            else:
                burn = int(burn)
        chain_list.append(chain[burn:])


def get_chain_tmparam_dict(chaindir_list, **kwargs):
    """separates list of chains by priors, puts corresponding chains and timing params
         in a list in a dictionary.
    :chaindir_list:
        list, list of chain locations
    """
    chain_dict = {}

    for chaindir in chaindir_list:
        sep = chaindir.split("/")
        dataset = sep[8]
        name = sep[-1]
        splt_name = name.split("_")

        tmp_tmparam = []
        tmp_tmparam.extend(np.loadtxt(chaindir + "/pars.txt", dtype="S").astype("U"))
        tmp_tmparam.extend(("lnlike", "lnprior", "chain accept", "pt chain accept"))

        chainpaths = sorted(glob.glob(chaindir + "/chain*.txt"))
        chain = pd.read_csv(chainpaths[0], sep="\t", dtype=float, header=None).values
        hot_chains = {}
        if len(chainpaths) > 1:
            for chp in chainpaths[1:]:
                try:
                    ch = pd.read_csv(chp, sep="\t", dtype=float, header=None).values
                    ky = chp.split("/")[-1].split("_")[-1].replace(".txt", "")
                    hot_chains.update({ky: ch})
                except:
                    print(chp, "cant be loaded.")

        burn = kwargs.get("burn", int(0.25 * chain.shape[0]))
        if not isinstance(burn, int):
            burn = int(burn)

        if "uniform" in splt_name and not "timing" in splt_name:
            if "uniform" not in chain_dict.keys():
                chain_dict["uniform"] = {}
            if "PX" in splt_name and not any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                if "vlbi_priors" not in chain_dict["uniform"].keys():
                    chain_dict["uniform"]["vlbi_priors"] = {}
                if dataset not in chain_dict["uniform"]["vlbi_priors"].keys():
                    chain_dict["uniform"]["vlbi_priors"][dataset] = defaultdict(list)
                chain_dict["uniform"]["vlbi_priors"][dataset]["chains"].append(
                    chain[burn:]
                )
                chain_dict["uniform"]["vlbi_priors"][dataset]["tmparams"].append(
                    tmp_tmparam
                )
                if hot_chains:
                    chain_dict["uniform"]["vlbi_priors"][dataset][
                        "hot_chains"
                    ] = hot_chains

            elif "PX" in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "vlbi_priors_" + item
                        if new_item not in chain_dict["uniform"].keys():
                            chain_dict["uniform"][new_item] = {}
                        if dataset not in chain_dict["uniform"][new_item].keys():
                            chain_dict["uniform"][new_item][dataset] = defaultdict(list)
                        chain_dict["uniform"][new_item][dataset]["chains"].append(
                            chain[burn:]
                        )
                        chain_dict["uniform"][new_item][dataset]["tmparams"].append(
                            tmp_tmparam
                        )
                        if hot_chains:
                            chain_dict["uniform"][new_item][dataset][
                                "hot_chains"
                            ] = hot_chains

            elif "PX" not in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "parfile_priors_" + item
                        if new_item not in chain_dict["uniform"].keys():
                            chain_dict["uniform"][new_item] = {}
                        if dataset not in chain_dict["uniform"][new_item].keys():
                            chain_dict["uniform"][new_item][dataset] = defaultdict(list)
                        chain_dict["uniform"][new_item][dataset]["chains"].append(
                            chain[burn:]
                        )
                        chain_dict["uniform"][new_item][dataset]["tmparams"].append(
                            tmp_tmparam
                        )
                        if hot_chains:
                            chain_dict["uniform"][new_item][dataset][
                                "hot_chains"
                            ] = hot_chains

            else:
                if "parfile_priors" not in chain_dict["uniform"].keys():
                    chain_dict["uniform"]["parfile_priors"] = {}
                if dataset not in chain_dict["uniform"]["parfile_priors"].keys():
                    chain_dict["uniform"]["parfile_priors"][dataset] = defaultdict(list)
                chain_dict["uniform"]["parfile_priors"][dataset]["chains"].append(
                    chain[burn:]
                )
                chain_dict["uniform"]["parfile_priors"][dataset]["tmparams"].append(
                    tmp_tmparam
                )
                if hot_chains:
                    chain_dict["uniform"]["parfile_priors"][dataset][
                        "hot_chains"
                    ] = hot_chains

        elif "bounded" in splt_name:
            if "bounded" not in chain_dict.keys():
                chain_dict["bounded"] = {}
            if "PX" in splt_name and not any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                if "vlbi_priors" not in chain_dict["bounded"].keys():
                    chain_dict["bounded"]["vlbi_priors"] = {}
                if dataset not in chain_dict["bounded"]["vlbi_priors"].keys():
                    chain_dict["bounded"]["vlbi_priors"][dataset] = defaultdict(list)
                chain_dict["bounded"]["vlbi_priors"][dataset]["chains"].append(
                    chain[burn:]
                )
                chain_dict["bounded"]["vlbi_priors"][dataset]["tmparams"].append(
                    tmp_tmparam
                )
                if hot_chains:
                    chain_dict["bounded"]["vlbi_priors"][dataset][
                        "hot_chains"
                    ] = hot_chains

            elif "PX" in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "vlbi_priors_" + item
                        if new_item not in chain_dict["bounded"].keys():
                            chain_dict["bounded"][new_item] = {}
                        if dataset not in chain_dict["bounded"][new_item].keys():
                            chain_dict["bounded"][new_item][dataset] = defaultdict(list)
                        chain_dict["bounded"][new_item][dataset]["chains"].append(
                            chain[burn:]
                        )
                        chain_dict["bounded"][new_item][dataset]["tmparams"].append(
                            tmp_tmparam
                        )
                        if hot_chains:
                            chain_dict["bounded"][new_item][dataset][
                                "hot_chains"
                            ] = hot_chains

            elif "PX" not in splt_name and any(
                ephem in ["DE405", "DE421", "DE436", "DE438"] for ephem in splt_name
            ):
                for item in splt_name:
                    if item in ["DE405", "DE421", "DE436", "DE438"]:
                        new_item = "parfile_priors_" + item
                        if new_item not in chain_dict["bounded"].keys():
                            chain_dict["bounded"][new_item] = {}
                        if dataset not in chain_dict["bounded"][new_item].keys():
                            chain_dict["bounded"][new_item][dataset] = defaultdict(list)
                        chain_dict["bounded"][new_item][dataset]["chains"].append(
                            chain[burn:]
                        )
                        chain_dict["bounded"][new_item][dataset]["tmparams"].append(
                            tmp_tmparam
                        )
                        if hot_chains:
                            chain_dict["bounded"][new_item][dataset][
                                "hot_chains"
                            ] = hot_chains

            else:
                if "parfile_priors" not in chain_dict["bounded"].keys():
                    chain_dict["bounded"]["parfile_priors"] = {}
                if dataset not in chain_dict["bounded"]["parfile_priors"].keys():
                    chain_dict["bounded"]["parfile_priors"][dataset] = defaultdict(list)
                chain_dict["bounded"]["parfile_priors"][dataset]["chains"].append(
                    chain[burn:]
                )
                chain_dict["bounded"]["parfile_priors"][dataset]["tmparams"].append(
                    tmp_tmparam
                )
                if hot_chains:
                    chain_dict["bounded"]["parfile_priors"][dataset][
                        "hot_chains"
                    ] = hot_chains

        else:
            if dataset not in chain_dict["misc"].keys():
                chain_dict["misc"][dataset] = defaultdict(list)
            chain_dict["misc"][dataset]["chains"].append(chain[burn:])
            chain_dict["misc"][dataset]["tmparams"].append(tmp_tmparam)
            if hot_chains:
                chain_dict["misc"][dataset]["hot_chains"] = hot_chains

        print("\r" + dataset + " " + name + " Loaded.      ")
    return chain_dict


def get_trimmed_chain_tmparam_dict(chain_dict):
    """Makes all chains in chain_dict the same length"""
    trimmed_chain_dict = {}
    i = 0
    for prior, prior_dict in chain_dict.items():
        for px_prior, px_prior_dict in prior_dict.items():
            for dataset in px_prior_dict.keys():
                for chain in chain_dict[prior][px_prior][dataset]["chains"]:
                    if i == 0:
                        min_chain_len_idx = (np.shape(chain)[0], i)
                    else:
                        if np.shape(chain)[0] < min_chain_len_idx[0]:
                            min_chain_len_idx = (np.shape(chain)[0], i)
                    i += 1
    i = 0
    for prior, prior_dict in chain_dict.items():
        trimmed_chain_dict[prior] = {}
        for px_prior, px_prior_dict in prior_dict.items():
            trimmed_chain_dict[prior][px_prior] = {}
            for dataset in chain_dict[prior][px_prior].keys():
                if dataset not in trimmed_chain_dict[prior][px_prior].keys():
                    trimmed_chain_dict[prior][px_prior][dataset] = defaultdict(list)
                for chain, tmparam in zip(
                    chain_dict[prior][px_prior][dataset]["chains"],
                    chain_dict[prior][px_prior][dataset]["tmparams"],
                ):
                    if i == min_chain_len_idx[1]:
                        trimmed_chain_dict[prior][px_prior][dataset]["chains"].append(
                            chain
                        )
                    else:
                        trim = np.shape(chain)[0] - min_chain_len_idx[0]
                        trimmed_chain_dict[prior][px_prior][dataset]["chains"].append(
                            chain[trim:]
                        )
                    trimmed_chain_dict[prior][px_prior][dataset]["tmparams"].append(
                        tmparam
                    )
                    i += 1
    return trimmed_chain_dict


def get_combined_arviz_obj_from_list(chain_list):
    comb_chain_dict = {}
    for i, chain in enumerate(chain_list):
        print(np.shape(chain))
        for j, par in enumerate(tmparam_list[i]):
            if par.split("_")[0] in [psr.name for psr in psrs]:
                if list(par)[0] != "J":
                    par = "J" + par
            if par in comb_chain_dict.keys():
                comb_chain_dict[par] = np.concatenate(
                    (comb_chain_dict[par], [chain[:, j]]), axis=0
                )
            else:
                comb_chain_dict[par] = [chain[:, j]]
    return az.convert_to_inference_data(comb_chain_dict)


def get_combined_arviz_obj_from_dict(chain_dict, psrs, return_dict=False):
    comb_chain_dict = {}
    for prior, prior_dict in chain_dict.items():
        for px_prior, px_prior_dict in prior_dict.items():
            for dataset, dataset_dict in px_prior_dict.items():
                for i, chain in enumerate(
                    chain_dict[prior][px_prior][dataset]["chains"]
                ):
                    print(np.shape(chain))
                    for j, par in enumerate(
                        chain_dict[prior][px_prior][dataset]["tmparams"][i]
                    ):
                        if par.split("_")[0] in [psr.name for psr in psrs]:
                            if list(par)[0] != "J":
                                par = "J" + par
                        if par in comb_chain_dict.keys():
                            try:
                                comb_chain_dict[par] = np.concatenate(
                                    (comb_chain_dict[par], [chain[:, j]]), axis=0
                                )
                            except:
                                print(
                                    "Exluding ",
                                    prior,
                                    px_prior,
                                    dataset,
                                    par,
                                    "because its chain length {} doesnt match {}".format(
                                        np.shape(chain[:, j]),
                                        np.shape(comb_chain_dict[par][0]),
                                    ),
                                )
                        else:
                            comb_chain_dict[par] = [chain[:, j]]
    if return_dict:
        return az.convert_to_inference_data(comb_chain_dict), comb_chain_dict
    else:
        return az.convert_to_inference_data(comb_chain_dict)


def get_rescaled_chain_dict(
    chains, tmparams, pardict, dataset, old_dict=None, px_priors=False
):
    if not isinstance(chains, np.ndarray):
        chains = np.asarray(chains)

    if not isinstance(tmparams, np.ndarray):
        tmparams = np.array(tmparams)

    rescaled_chain_dict = {}
    for i in range(np.shape(tmparams)[0]):
        for j, par in enumerate(tmparams[i]):
            split_string = par.split("_")
            psr_name = split_string[0]
            if "timing" in split_string:
                og_par = split_string[-1]
                if px_priors and og_par == "PX":
                    rescaled_chain = chains[i][:, j]
                else:
                    rescaled_chain = (
                        chains[i][:, j] * pardict[dataset][psr_name][og_par]["err"]
                        + pardict[dataset][psr_name][og_par]["val"]
                    )
            else:
                if psr_name in ["lnlike", "lnprior", "chain accept", "pt chain accept"]:
                    og_par = par
                    rescaled_chain = chains[i][:, j]
                else:
                    if list(par)[0] != "J":
                        og_par = "J" + par
                    else:
                        og_par = par
                rescaled_chain = chains[i][:, j]

            if og_par in rescaled_chain_dict.keys():
                rescaled_chain_dict[og_par] = np.concatenate(
                    (rescaled_chain_dict[og_par], [rescaled_chain]), axis=0
                )
            else:
                rescaled_chain_dict[og_par] = [rescaled_chain]

    if old_dict is not None:
        for old_key in old_dict.keys():
            if old_key in rescaled_chain_dict.keys():
                rescaled_chain_dict[old_key] = np.concatenate(
                    (rescaled_chain_dict[old_key], old_dict[old_key]), axis=0
                )
            else:
                rescaled_chain_dict[old_key] = old_dict[old_key]

    return rescaled_chain_dict


def get_combined_rescaled_chain_dict(chain_dict, pardict, dataset, **kwargs):
    px_priors = kwargs.get("px_priors", [])

    if "priors" not in kwargs.keys():
        priors = []
        if len(px_priors) == 0:
            for p in chain_dict.keys():
                for px_p in chain_dict[p].keys():
                    if dataset in chain_dict[p][px_p].keys():
                        if p not in priors:
                            priors.append(p)
                        if px_p not in px_priors:
                            px_priors.append(px_p)
                    else:
                        print(dataset, "not in", p, px_p, chain_dict[p][px_p].keys())
        else:
            for p in chain_dict.keys():
                for px_p in px_priors:
                    if px_p in chain_dict[p].keys():
                        if dataset in chain_dict[p][px_p].keys():
                            if p not in priors:
                                priors.append(p)
                        else:
                            print(
                                dataset, "not in", p, px_p, chain_dict[p][px_p].keys()
                            )
                    else:
                        print(px_p, "not in", p, chain_dict[p].keys())
    else:
        priors = kwargs["priors"]

    rescaled_chain_dict = {}
    for prior in priors:
        for px_prior in px_priors:
            if px_prior in chain_dict[prior].keys():
                for tmparams, chains in zip(
                    chain_dict[prior][px_prior][dataset]["tmparams"],
                    chain_dict[prior][px_prior][dataset]["chains"],
                ):
                    if "vlbi" in px_prior.split("_"):
                        if not rescaled_chain_dict:
                            rescaled_chain_dict = get_rescaled_chain_dict(
                                [chains], [tmparams], pardict, dataset, px_priors=True
                            )
                        else:
                            rescaled_chain_dict_comb = get_rescaled_chain_dict(
                                [chains],
                                [tmparams],
                                pardict,
                                dataset,
                                px_priors=True,
                                old_dict=rescaled_chain_dict,
                            )
                            del rescaled_chain_dict
                            rescaled_chain_dict = rescaled_chain_dict_comb
                    else:
                        if not rescaled_chain_dict:
                            rescaled_chain_dict = get_rescaled_chain_dict(
                                [chains], [tmparams], pardict, dataset
                            )
                        else:
                            rescaled_chain_dict_comb = get_rescaled_chain_dict(
                                [chains],
                                [tmparams],
                                pardict,
                                dataset,
                                old_dict=rescaled_chain_dict,
                            )
                            del rescaled_chain_dict
                            rescaled_chain_dict = rescaled_chain_dict_comb
            else:
                print(
                    px_prior,
                    "not in",
                    prior,
                    "dict:",
                    chain_dict[prior].keys(),
                )

    return rescaled_chain_dict_comb


def get_rescaled_chains_as_core_list(chain_dict, label, burn=0):
    common_dict = {}
    max_size = (0, 0)
    for val in chain_dict.values():
        size = np.shape(val)
        if np.shape(val)[0] > max_size[0]:
            max_size = size
    for key, val in chain_dict.items():
        if np.shape(val) == max_size:
            common_dict[key] = val
        else:
            print(
                "Exluding ",
                key,
                " because its shape {} doesnt match others".format(np.shape(val)),
            )

    param_list = list(common_dict.keys())
    [num_chains, len_chains] = np.shape(common_dict[param_list[0]])
    core_list = []
    for i in range(num_chains):
        if i != 0:
            del stacked_chains
        for chains in common_dict.values():
            if "stacked_chains" not in locals():
                stacked_chains = [chains[i]]
            else:
                stacked_chains = np.concatenate((stacked_chains, [chains[i]]), axis=0)

        core_list.append(
            co.Core(
                label="{}".format(label),
                chain=stacked_chains.T,
                params=param_list,
                burn=burn,
            )
        )
    return core_list


def plot_common_chains(
    core_list, chaindir_list, priors, px_priors, dataset, plot_kwargs={}, misc_kwargs={}
):
    """Uses la_forge to plot chains
    :prior:
        str, {'uniform','bounded'}
    :px_prior:
        str, {'px_priors','other'}
    :dataset:
        str, {'5yr','9yr','11yr'}
    """
    if "hist" not in plot_kwargs.keys():
        plot_kwargs["hist"] = True
    if "ncols" not in plot_kwargs.keys():
        plot_kwargs["ncols"] = 4
    if "bins" not in plot_kwargs.keys():
        plot_kwargs["bins"] = 10
    if "hist_kwargs" not in plot_kwargs.keys():
        plot_kwargs["hist_kwargs"] = dict(fill=False)

    legend_labels = misc_kwargs.get("legend_labels",[])

    if legend_labels:
        user_legend = True
    else:
        user_legend = False

    legend_loc = misc_kwargs.get("legend_loc",None)

    if isinstance(priors, str):
        priors = [priors]
    if isinstance(px_priors, str):
        px_priors = [px_priors]

    if isinstance(dataset, str):
        datasets = [dataset]
    else:
        datasets = dataset

    chaindir_indices = get_chaindir_indices(chaindir_list)

    if len(priors) == 1 and len(px_priors) == 1:
        if np.shape(chaindir_indices[priors[0]][px_priors[0]][datasets[0]])[0] > 1:
            common_pars = []
            for i in range(len(chaindir_indices[priors[0]][px_priors[0]][datasets[0]])):
                if i == 0:
                    for param in core_list[
                        chaindir_indices[priors[0]][px_priors[0]][datasets[0]][i]
                    ].params:
                        if param not in [
                            "lnlike",
                            "lnprior",
                            "chain_accept",
                            "pt_chain_accept",
                            "chain accept",
                            "pt chain accept",
                        ]:
                            common_pars.append(param)
                else:
                    noncommon_pars = []
                    for com_par in common_pars:
                        if (
                            com_par
                            not in core_list[
                                chaindir_indices[priors[0]][px_priors[0]][datasets[0]][
                                    i
                                ]
                            ].params
                        ):
                            noncommon_pars.append(com_par)
                    for ncom_par in noncommon_pars:
                        del common_pars[common_pars.index(ncom_par)]
                if not user_legend:
                    legend_labels.append(
                        datasets[0]
                        + ": "
                        + priors[0]
                        + " prior, "
                        + px_priors[0]
                        + " v{}".format(i + 1)
                    )

            dg.plot_chains(
                [
                    core_list[x]
                    for x in chaindir_indices[priors[0]][px_priors[0]][datasets[0]]
                ],
                pars=common_pars,
                legend_labels=legend_labels,
                legend_loc=legend_loc,
                **plot_kwargs
            )
        else:
            if not user_legend:
                legend_labels.append(
                    datasets[0] + ": " + priors[0] + " prior, " + px_priors[0]
                )
            dg.plot_chains(
                [
                    core_list[x]
                    for x in chaindir_indices[priors[0]][px_priors[0]][datasets[0]]
                ],
                legend_labels=legend_labels,
                legend_loc=legend_loc,
                **plot_kwargs
            )
    else:
        common_pars = []
        for prior in priors:
            if prior in chaindir_indices.keys():
                for px_prior in px_priors:
                    if px_prior in chaindir_indices[prior].keys():
                        for dataset in datasets:
                            if dataset in chaindir_indices[prior][px_prior].keys():
                                if len(common_pars) == 0:
                                    comb_indices = chaindir_indices[prior][px_prior][
                                        dataset
                                    ]
                                    for idx1 in comb_indices:
                                        if len(common_pars) == 0:
                                            for core_param in core_list[idx1].params:
                                                if core_param not in [
                                                    "lnlike",
                                                    "lnprior",
                                                    "chain_accept",
                                                    "pt_chain_accept",
                                                    "chain accept",
                                                    "pt chain accept",
                                                ]:
                                                    common_pars.append(core_param)
                                        else:
                                            noncommon_pars = []
                                            for com_par in common_pars:
                                                if (
                                                    com_par
                                                    not in core_list[idx1].params
                                                ):
                                                    noncommon_pars.append(com_par)
                                            for ncom_par in noncommon_pars:
                                                del common_pars[
                                                    common_pars.index(ncom_par)
                                                ]
                                else:
                                    comb_indices = sorted(
                                        np.concatenate(
                                            (
                                                comb_indices,
                                                chaindir_indices[prior][px_prior][
                                                    dataset
                                                ],
                                            ),
                                            axis=0,
                                        )
                                    )
                                    for idx2 in comb_indices:
                                        noncommon_pars = []
                                        for com_par in common_pars:
                                            if com_par not in core_list[idx2].params:
                                                noncommon_pars.append(com_par)
                                        for ncom_par in noncommon_pars:
                                            del common_pars[common_pars.index(ncom_par)]
                            else:
                                print(
                                    dataset,
                                    "not in",
                                    prior,
                                    "[",
                                    px_prior,
                                    "] dict:",
                                    chaindir_indices[prior][px_prior].keys(),
                                )
                    else:
                        print(
                            px_prior,
                            "not in",
                            prior,
                            "dict:",
                            chaindir_indices[prior].keys(),
                        )
            else:
                print(prior, "not in", chaindir_indices.keys())
        if not user_legend:
            for i in comb_indices:
                for prior in chaindir_indices.keys():
                    for px_prior in chaindir_indices[prior].keys():
                        for dataset in chaindir_indices[prior][px_prior].keys():
                            if (
                                i
                                in chaindir_indices[prior][px_prior][
                                    dataset
                                ]
                            ):
                                j = 1
                                label = (
                                    dataset
                                    + ": "
                                    + prior
                                    + " prior, "
                                    + px_prior
                                    + " v{}".format(j)
                                )
                                if label in legend_labels:
                                    versioning = True
                                    j += 1
                                    while versioning is True:
                                        new_label = (
                                            dataset
                                            + ": "
                                            + prior
                                            + " prior, "
                                            + px_prior
                                            + " v{}".format(j)
                                        )
                                        if new_label not in legend_labels:
                                            legend_labels.append(new_label)
                                            versioning = False
                                        j += 1
                                else:
                                    legend_labels.append(label)
        dg.plot_chains(
            [core_list[x] for x in comb_indices],
            pars=common_pars,
            legend_labels=legend_labels,
            legend_loc=legend_loc,
            **plot_kwargs
        )
