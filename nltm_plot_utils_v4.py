import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from collections import defaultdict, OrderedDict
from copy import deepcopy

import pandas as pd
import corner
import acor

# import pymc3

import la_forge
import la_forge.diagnostics as dg
from la_forge.core import TimingCore, Core


def get_pardict(psrs, datareleases):
    """assigns a parameter dictionary for each psr per dataset the parfile values/errors
    :param psrs: enterprise pulsar instances corresponding to datareleases
    :param datareleases: list of datareleases
    """
    pardict = {}
    for psr, dataset in zip(psrs, datareleases):
        pardict[psr.name] = {}
        pardict[psr.name][dataset] = {}
        for par, vals, errs in zip(
            psr.fitpars[1:],
            np.longdouble(psr.t2pulsar.vals()),
            np.longdouble(psr.t2pulsar.errs()),
        ):
            pardict[psr.name][dataset][par] = {}
            pardict[psr.name][dataset][par]["val"] = vals
            pardict[psr.name][dataset][par]["err"] = errs
    return pardict


def make_dmx_file(parfile):
    dmx_dict = {}
    with open(parfile, "r") as f:
        lines = f.readlines()

    for line in lines:
        splt_line = line.split()
        if "DMX" in splt_line[0] and splt_line[0] != "DMX":
            for dmx_group in [
                y.split()
                for y in lines
                if str(splt_line[0].split("_")[-1]) in str(y.split()[0])
            ]:
                # Columns: DMXEP DMX_value DMX_var_err DMXR1 DMXR2 DMXF1 DMXF2 DMX_bin
                lab = f"DMX_{dmx_group[0].split('_')[-1]}"
                if lab not in dmx_dict.keys():
                    dmx_dict[lab] = {}
                if "DMX_" in dmx_group[0]:
                    if isinstance(dmx_group[1], str):
                        dmx_dict[lab]["DMX_value"] = np.double(
                            ("e").join(dmx_group[1].split("D"))
                        )
                    else:
                        dmx_dict[lab]["DMX_value"] = np.double(dmx_group[1])
                    if isinstance(dmx_group[-1], str):
                        dmx_dict[lab]["DMX_var_err"] = np.double(
                            ("e").join(dmx_group[-1].split("D"))
                        )
                    else:
                        dmx_dict[lab]["DMX_var_err"] = np.double(dmx_group[-1])
                    dmx_dict[lab]["DMX_bin"] = "DX" + dmx_group[0].split("_")[-1]
                else:
                    dmx_dict[lab][dmx_group[0].split("_")[0]] = np.double(dmx_group[1])
    for dmx_name, dmx_attrs in dmx_dict.items():
        if any([key for key in dmx_attrs.keys() if "DMXEP" not in key]):
            dmx_dict[dmx_name]["DMXEP"] = (
                dmx_attrs["DMXR1"] + dmx_attrs["DMXR2"]
            ) / 2.0
    dmx_df = pd.DataFrame.from_dict(dmx_dict, orient="index")
    neworder = [
        "DMXEP",
        "DMX_value",
        "DMX_var_err",
        "DMXR1",
        "DMXR2",
        "DMXF1",
        "DMXF2",
        "DMX_bin",
    ]
    final_order = []
    for order in neworder:
        if order in dmx_dict["DMX_0001"]:
            final_order.append(order)
    dmx_df = dmx_df.reindex(columns=final_order)
    new_dmx_file = (".dmx").join(parfile.split(".par"))
    with open(new_dmx_file, "w") as f:
        f.write(
            f"# {parfile.split('/')[-1].split('.par')[0]} dispersion measure variation\n"
        )
        f.write(
            f"# Mean DMX value = {np.mean([dmx_dict[x]['DMX_value'] for x in dmx_dict.keys()])} \n"
        )
        f.write(
            f"# Uncertainty in average DM = {np.std([dmx_dict[x]['DMX_value'] for x in dmx_dict.keys()])} \n"
        )
        f.write(f"# Columns: {(' ').join(final_order)}\n")
        dmx_df.to_csv(f, sep=" ", index=False, header=False)


def get_map_param(core, params, to_burn=True):
    map_idx = np.argmax(core.get_param("lnpost", to_burn=True))
    """
    print(stats.mode(core.get_param('lnpost',to_burn=to_burn)))
    values, counts = np.unique(core.get_param('lnpost',to_burn=to_burn), return_counts=True)
    ind = np.argmax(counts)
    print(values[ind])  # prints the most frequent element
    print(map_idx)
    """
    if isinstance(params, str):
        params = [params]
    else:
        if not isinstance(params, list):
            raise ValueError("params need to be string or list")
    map_params = {}
    for par in params:
        if isinstance(core, TimingCore):
            if not any(
                [
                    x in par
                    for x in ["ecorr", "equad", "efac", "lnpost", "lnlike", "accept"]
                ]
            ):
                if "DMX" in par:
                    tm_par = "_".join(par.split("_")[-2:])
                else:
                    tm_par = par.split("_")[-1]

                if core.tm_pars_orig[tm_par][-1] == "normalized":
                    unscaled_param = core.get_param(
                        par, to_burn=to_burn, tm_convert=False
                    )
                else:
                    unscaled_param = core.get_param(
                        par, to_burn=to_burn, tm_convert=True
                    )

                map_params[tm_par] = unscaled_param[map_idx]
        elif isinstance(core, Core):
            unscaled_param = core.get_param(par, to_burn=to_burn)
            map_params[par] = unscaled_param[map_idx]
    return map_params


def tm_delay(t2pulsar, tm_params_orig, new_params):
    """
    Compute difference in residuals due to perturbed timing model.

    :param t2pulsar: libstempo pulsar object
    :param tm_params_orig: dictionary of TM parameter tuples, (val, err)

    :return: difference between new and old residuals in seconds
    """
    residuals = np.longdouble(t2pulsar.residuals().copy())
    # grab original timing model parameters and errors in dictionary
    orig_params = {}
    tm_params_rescaled = {}
    error_pos = {}
    for tm_scaled_key, tm_scaled_val in new_params.items():
        if "DMX" in tm_scaled_key:
            tm_param = "_".join(tm_scaled_key.split("_")[-2:])
        else:
            tm_param = tm_scaled_key.split("_")[-1]

        if tm_param == "COSI":
            orig_params["SINI"] = np.longdouble(tm_params_orig["SINI"][0])
        else:
            orig_params[tm_param] = np.longdouble(tm_params_orig[tm_param][0])

        if "physical" in tm_params_orig[tm_param]:
            # User defined priors are assumed to not be scaled
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - tm_scaled_val ** 2)
                )
            else:
                tm_params_rescaled[tm_param] = np.longdouble(tm_scaled_val)
        else:
            if tm_param == "COSI":
                # Switch for sampling in COSI, but using SINI in libstempo
                rescaled_COSI = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )
                tm_params_rescaled["SINI"] = np.longdouble(
                    np.sqrt(1 - rescaled_COSI ** 2)
                )
                # print("Rescaled COSI used to find SINI", np.longdouble(rescaled_COSI))
                # print("rescaled SINI", tm_params_rescaled["SINI"])
            else:
                tm_params_rescaled[tm_param] = np.longdouble(
                    tm_scaled_val * tm_params_orig[tm_param][1]
                    + tm_params_orig[tm_param][0]
                )
    # set to new values
    # print(tm_params_rescaled)
    # TODO: Find a way to not do this every likelihood call bc it doesn't change and it is in enterprise.psr._isort
    # Sort residuals by toa to match with get_detres() call
    isort = np.argsort(t2pulsar.toas(), kind="mergesort")
    t2pulsar.vals(tm_params_rescaled)
    new_res = np.longdouble(t2pulsar.residuals().copy())

    # remeber to set values back to originals
    t2pulsar.vals(orig_params)

    plotres(t2pulsar, new_res, residuals, tm_params_rescaled)

    # Return the time-series for the pulsar
    # return -(new_res[isort] - residuals[isort])


def plotres(psr, new_res, old_res, par_dict, deleted=False, group=None, **kwargs):
    """Plot residuals, compute unweighted rms residual."""

    t, errs = psr.toas(), psr.toaerrs

    meannewres = np.sqrt(np.mean(new_res ** 2)) / 1e-6
    meanoldres = np.sqrt(np.mean(old_res ** 2)) / 1e-6

    if (not deleted) and np.any(psr.deleted != 0):
        new_res, old_res, t, errs = (
            new_res[psr.deleted == 0],
            old_res[psr.deleted == 0],
            t[psr.deleted == 0],
            errs[psr.deleted == 0],
        )
        print("Plotting {0}/{1} nondeleted points.".format(len(new_res), psr.nobs))
    if group is None:
        i = np.argsort(t)
        pars = [x for x in par_dict.keys()]
        vals = []
        for p, v in zip(psr.pars(), psr.vals()):
            for pa in pars:
                if p == pa:
                    vals.append(v)

        plt.errorbar(
            t[i],
            old_res[i] / 1e-6,
            yerr=errs[i],
            fmt="x",
            label="Old Residuals",
            **kwargs,
        )
        plt.errorbar(
            t[i],
            new_res[i] / 1e-6,
            # yerr=errs[i],
            fmt="+",
            label="New Residuals",
            **kwargs,
        )
        # plt.legend()
    else:
        if (not deleted) and np.any(psr.deleted):
            flagmask = psr.flagvals(group)[~psr.deleted]
        else:
            flagmask = psr.flagvals(group)

        unique = list(set(flagmask))

        for flagval in unique:
            f = flagmask == flagval
            flagnewres, flagoldresflagt, flagerrs = (
                new_res[f],
                old_res[f],
                t[f],
                errs[f],
            )
            i = np.argsort(flagt)

            plt.errorbar(
                flagt[i],
                flagoldres[i] / 1e-6,
                yerr=flagerrs[i],
                fmt="d",
                label="Old Residuals",
                **kwargs,
            )
            plt.errorbar(
                flagt[i],
                flagnewres[i] / 1e-6,
                yerr=flagerrs[i],
                fmt="x",
                label="New Residuals",
                **kwargs,
            )

        # plt.legend(unique,numpoints=1,bbox_to_anchor=(1.1,1.1))
    plt.legend()
    plt.xlabel(r"MJD")
    plt.ylabel(r"res [$\mu s$]")
    plt.title(
        fr"{psr.name}: RMS, Old Res = {meanoldres:.3f} $\mu s$, New Res = {meannewres:.3f} $\mu s$"
    )


def check_convergence(core_list):
    cut_off_idx = -3
    for core in core_list:
        lp = np.unique(np.max(core.get_param("lnpost")))
        ll = np.unique(np.max(core.get_param("lnlike")))
        print("-------------------------------")
        print(f"core: {core.label}")
        print(f"\t lnpost: {lp[0]}, lnlike: {ll[0]}")
        try:
            grub = grubin(core.chain[:, :cut_off_idx])
            if len(grub[1]) > 0:
                print(
                    "\t Params exceed rhat threshold: ",
                    [core.params[p] for p in grub[1]],
                )
        except:
            print("\t Can't run Grubin test")
            pass
        try:
            geweke = geweke_check(core.chain[:, :cut_off_idx], burn_frac=0.25)
            if len(geweke) > 0:
                print("\t Params fail Geweke test: ", [core.params[p] for p in geweke])
        except:
            print("\t Can't run Geweke test")
            pass
        try:
            max_acl = np.unique(np.max(get_param_acorr(core.chain[:, :cut_off_idx])))[0]
            print(
                f"\t Max autocorrelation length: {max_acl}, Effective sample size: {core.chain.shape[0]/max_acl}"
            )
        except:
            print("\t Can't run Autocorrelation test")
            pass
        print("")


def summary_comparison(psr_name, core, par_sigma={}, selection="all"):
    """Makes comparison table of the form:
    Par Name | Old Value | New Value | Difference | Old Sigma | New Sigma 
    TODO: allow for selection of subparameters"""
    pd.set_option("max_rows", None)
    plot_params = get_param_groups(core, selection=selection)
    summary_dict = {}
    for pnames, title in zip(plot_params["par"], plot_params["title"]):
        if "timing" in pnames:
            if isinstance(core, TimingCore):
                param_vals = core.get_param(pnames, tm_convert=True, to_burn=True)
            elif isinstance(core, Core):
                param_vals = core.get_param(pnames, to_burn=True)

            summary_dict[title] = {}
            summary_dict[title]["new_val"] = np.median(param_vals)
            summary_dict[title]["new_sigma"] = np.std(param_vals)
            if title in core.tm_pars_orig:
                summary_dict[title]["old_val"] = core.tm_pars_orig[title][0]
                summary_dict[title]["old_sigma"] = core.tm_pars_orig[title][1]
                summary_dict[title]["difference"] = (
                    summary_dict[title]["new_val"] - core.tm_pars_orig[title][0]
                )
                if abs(summary_dict[title]["difference"]) > core.tm_pars_orig[title][1]:
                    summary_dict[title]["big"] = True
                else:
                    summary_dict[title]["big"] = False
                if summary_dict[title]["new_sigma"] < summary_dict[title]["old_sigma"]:
                    summary_dict[title]["constrained"] = True
                else:
                    summary_dict[title]["constrained"] = False
            else:
                summary_dict[title]["old_val"] = "-"
                summary_dict[title]["old_sigma"] = "-"
                summary_dict[title]["difference"] = "-"
                summary_dict[title]["big"] = "-"
                summary_dict[title]["constrained"] = "-"
    return pd.DataFrame(
        np.asarray(
            [
                [x for x in summary_dict.keys()],
                [summary_dict[x]["old_val"] for x in summary_dict.keys()],
                [summary_dict[x]["new_val"] for x in summary_dict.keys()],
                [summary_dict[x]["difference"] for x in summary_dict.keys()],
                [summary_dict[x]["old_sigma"] for x in summary_dict.keys()],
                [summary_dict[x]["new_sigma"] for x in summary_dict.keys()],
                [summary_dict[x]["big"] for x in summary_dict.keys()],
                [summary_dict[x]["constrained"] for x in summary_dict.keys()],
            ]
        ).T,
        columns=[
            "Parameter",
            "Old Value",
            "New Median Value",
            "Difference",
            "Old Sigma",
            "New Std Dev",
            ">1 sigma change?",
            "More Constrained?",
        ],
    )


def residual_comparison(
    t2pulsar, core, use_mean_median_map="median", use_tm_pars_orig=False
):
    """Used to compare old residuals to new residuals."""
    core_titles = get_titles(t2pulsar.name, core)
    core_timing_dict_unscaled = OrderedDict()

    if use_mean_median_map == "map":
        map_idx_e_e = np.argmax(core.get_param("lnpost", to_burn=True))
    elif use_mean_median_map not in ["map", "median", "mean"]:
        raise ValueError(
            "use_mean_median_map can only be either 'map','median', or 'mean"
        )

    for par in core.params:
        unscaled_param = core.get_param(par, to_burn=True, tm_convert=True)
        if use_mean_median_map == "map":
            core_timing_dict_unscaled[par] = unscaled_param[map_idx_e_e]
        elif use_mean_median_map == "mean":
            core_timing_dict_unscaled[par] = np.mean(unscaled_param)
        elif use_mean_median_map == "median":
            core_timing_dict_unscaled[par] = np.median(unscaled_param)

    core_timing_dict = deepcopy(core_timing_dict_unscaled)
    if use_tm_pars_orig:
        for p in core_timing_dict.keys():
            if "timing" in p:
                core_timing_dict.update(
                    {p: np.double(core.tm_pars_orig[p.split("_")[-1]][0])}
                )

    chain_tm_params_orig = deepcopy(core.tm_pars_orig)
    chain_tm_delay_kwargs = {}
    for par in t2pulsar.pars():
        if par == "SINI" and "COSI" in core.tm_pars_orig.keys():
            sin_val, sin_err, _ = chain_tm_params_orig[par]
            val = np.longdouble(np.sqrt(1 - sin_val ** 2))
            err = np.longdouble(np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err ** 2))
            chain_tm_params_orig["COSI"] = [val, err, "physical"]
            chain_tm_delay_kwargs["COSI"] = core_timing_dict[
                core.params[core_titles.index("COSI")]
            ]
        elif par in core.tm_pars_orig.keys():
            chain_tm_params_orig[par][-1] = "physical"
            chain_tm_delay_kwargs[par] = core_timing_dict[
                core.params[core_titles.index(par)]
            ]
        else:
            print(f"{par} not in t2pulsar pars")

    tm_delay(t2pulsar, chain_tm_params_orig, chain_tm_delay_kwargs)
    plt.show()


def refit_errs(psr, tm_params_orig):
    # Check to see if nan or inf in pulsar parameter errors.
    # The refit will populate the incorrect errors, but sometimes
    # changes the values by too much, which is why it is done in this order.
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
            par = psr.t2pulsar.pars()[idx]
            print(par, np.longdouble(psr.t2pulsar.errs()[idx]))
            tm_params_orig[par][1] = np.longdouble(psr.t2pulsar.errs()[idx])
    psr.t2pulsar.vals(orig_vals)
    psr.t2pulsar.errs(orig_errs)


def reorder_columns(e_e_chaindir, outdir):
    chain_e_e = pd.read_csv(
        e_e_chaindir + "/chain_1.0.txt", sep="\t", dtype=float, header=None
    )
    switched_chain = chain_e_e[[0, 3, 1, 4, 2, 5, 6, 7, 8, 9, 10, 11, 12]]
    np.savetxt(outdir + "/chain_1.txt", switched_chain.values, delimiter="\t")


def get_new_PAL2_params(psr_name, core):
    new_PAL2_params = []
    for par in core.params:
        if "efac" in par:
            new_par = psr_name + "_" + par.split("efac-")[-1] + "_efac"
        elif "jitter" in par:
            new_par = psr_name + "_" + par.split("jitter_q-")[-1] + "_log10_ecorr"
        elif "equad" in par:
            new_par = psr_name + "_" + par.split("equad-")[-1] + "_log10_equad"
        elif par in ["lnpost", "lnlike", "chain_accept", "pt_chain_accept"]:
            new_par = par
        else:
            new_par = psr_name + "_timing_model_" + par
        new_PAL2_params.append(new_par)
    return new_PAL2_params


def get_titles(psr_name, core):
    titles = []
    for core_param in core.params:
        if "timing" in core_param.split("_"):
            if "DMX" in core_param.split("_"):
                titles.append(("_").join(core_param.split("_")[-2:]))
            else:
                titles.append(core_param.split("_")[-1])
        else:
            if psr_name in core_param.split("_"):
                titles.append((" ").join(core_param.split("_")[1:]))
            else:
                titles.append(core_param)
    """
    else:
        unparams=['lnpost','lnlike','chain_accept','pt_chain_accept']
        for com_par in params:
            if (
                com_par
                not in core.params
            ):
                unparams.append(com_par)
        for ncom_par in unparams:
            if ncom_par in params:
                del titles[params.index(ncom_par)]
                del params[
                    params.index(ncom_par)
                ]
    """
    return titles


def get_common_params_titles(psr_name, core_list, exclude=True):
    common_params = []
    common_titles = []
    for core in core_list:
        if len(common_params) == 0:
            for core_param in core.params:
                common_params.append(core_param)
                if "timing" in core_param.split("_"):
                    if "DMX" in core_param.split("_"):
                        common_titles.append(("_").join(core_param.split("_")[-2:]))
                    else:
                        common_titles.append(core_param.split("_")[-1])
                else:
                    if psr_name in core_param.split("_"):
                        common_titles.append((" ").join(core_param.split("_")[1:]))
                    else:
                        common_titles.append(core_param)
        else:
            if exclude:
                uncommon_params = [
                    "lnpost",
                    "lnlike",
                    "chain_accept",
                    "pt_chain_accept",
                ]
            else:
                uncommon_params = ["chain_accept", "pt_chain_accept"]

            for com_par in common_params:
                if com_par not in core.params:
                    uncommon_params.append(com_par)
            for ncom_par in uncommon_params:
                if ncom_par in common_params:
                    del common_titles[common_params.index(ncom_par)]
                    del common_params[common_params.index(ncom_par)]
    return common_params, common_titles


def get_other_param_overlap(psr_name, core_list):
    """Returns a dictionary of params with list of indices for corresponding core in core_list"""
    boring_params = ["lnpost", "lnlike", "chain_accept", "pt_chain_accept"]
    common_params_all, _ = get_common_params_titles(psr_name, core_list, exclude=False)
    com_par_dict = defaultdict(list)
    for j, core in enumerate(core_list):
        for param in core.params:
            if param not in common_params_all + boring_params:
                com_par_dict[param].append(j)

    return com_par_dict


def plot_all_param_overlap(
    psr_name,
    core_list,
    core_list_legend=None,
    com_par_dict=None,
    exclude=True,
    real_tm_pars=True,
    conf_int=None,
    close=True,
    par_sigma={},
    ncols=3,
    hist_kwargs={},
    fig_kwargs={},
):
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)

    hist_core_list_kwargs = {
        "hist": True,
        "ncols": ncols,
        "title_y": 1.4,
        "hist_kwargs": hist_kwargs,
        "linewidth": 3.0,
    }

    if "suptitle" not in fig_kwargs.keys():
        suptitle = f"{psr_name} Mass Plots"
    else:
        suptitle = fig_kwargs["suptitle"]
    if "suptitlefontsize" not in fig_kwargs.keys():
        suptitlefontsize = 24
    else:
        suptitlefontsize = fig_kwargs["suptitlefontsize"]
    if "suptitleloc" not in fig_kwargs.keys():
        suptitleloc = (0.25, 1.05)
    else:
        suptitleloc = fig_kwargs["suptitleloc"]
    if "legendloc" not in fig_kwargs.keys():
        legendloc = (0.45, 0.97)
    else:
        legendloc = fig_kwargs["legendloc"]
    if "legendfontsize" not in fig_kwargs.keys():
        legendfontsize = 12
    else:
        legendfontsize = fig_kwargs["legendfontsize"]
    if "colors" not in fig_kwargs.keys():
        colors = [f"C{ii}" for ii in range(len(core_list_legend))]
    else:
        colors = fig_kwargs["colors"]
    if "wspace" not in fig_kwargs.keys():
        wspace = 0.1
    else:
        wspace = fig_kwargs["wspace"]
    if "hspace" not in fig_kwargs.keys():
        hspace = 0.2
    else:
        hspace = fig_kwargs["hspace"]

    common_params_all, common_titles_all = get_common_params_titles(
        psr_name, core_list, exclude=exclude
    )
    hist_core_list_kwargs["title_y"] = 1.0 + 0.025 * len(core_list)
    if len(core_list) < 3:
        y = 0.98 - 0.01 * len(core_list)
        hist_core_list_kwargs["title_y"] = 1.0 + 0.02 * len(core_list)
    elif len(core_list) >= 3 and len(core_list) < 5:
        y = 0.95 - 0.01 * len(core_list)
        # hist_core_list_kwargs['title_y'] = 1.+.05*len(core_list)
    elif len(core_list) >= 3 and len(core_list) < 10:
        y = 0.95 - 0.01 * len(core_list)
        # hist_core_list_kwargs['title_y'] = 1.+.025*len(core_list)
    else:
        y = 0.95 - 0.01 * len(core_list)
        # hist_core_list_kwargs['title_y'] = 1.+.05*len(core_list)

    if close:
        dg.plot_chains(
            core_list,
            suptitle=suptitle,
            pars=common_params_all,
            titles=common_titles_all,
            real_tm_pars=real_tm_pars,
            show=False,
            close=False,
            **hist_core_list_kwargs,
        )

        if conf_int or par_sigma:
            fig = plt.gcf()
            allaxes = fig.get_axes()
            for ax in allaxes:
                splt_key = ax.get_title()
                if par_sigma:
                    if splt_key in par_sigma:
                        val = par_sigma[splt_key][0]
                        err = par_sigma[splt_key][1]
                        fill_space_x = np.linspace(val - err, val + err, 20)
                        ax.fill_between(
                            fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                        )
                        ax.axvline(val, color="k", linestyle="--")
                    elif splt_key == "COSI" and "SINI" in par_sigma:
                        sin_val, sin_err, _ = par_sigma["SINI"]
                        val = np.longdouble(np.sqrt(1 - sin_val ** 2))
                        err = np.longdouble(
                            np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err ** 2)
                        )
                        fill_space_x = np.linspace(val - err, val + err, 20)
                        ax.fill_between(
                            fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                        )
                        ax.axvline(val, color="k", linestyle="--")

                if conf_int:
                    if isinstance(conf_int, (float, int)):
                        if conf_int < 1.0 or conf_int > 99.0:
                            raise ValueError("conf_int must be between 1 and 99")
                    else:
                        raise ValueError("conf_int must be an int or float")

                    for i, core in enumerate(core_list):
                        # elif splt_key == 'COSI' and 'SINI' in par_sigma:
                        for com_par, com_title in zip(
                            common_params_all, common_titles_all
                        ):
                            if splt_key == com_title:
                                low, up = core.get_param_confint(
                                    com_par, interval=conf_int
                                )
                                ax.fill_between(
                                    [low, up],
                                    ax.get_ylim()[1],
                                    color=f"C{i}",
                                    alpha=0.1,
                                )
        else:
            fig = plt.gcf()
        patches = []
        for jj, lab in enumerate(core_list_legend):
            patches.append(
                mpl.patches.Patch(color=colors[jj], label=lab)
            )  # .split(":")[-1]))

        fig.legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
        fig.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.suptitle(
            suptitle, fontsize=suptitlefontsize, x=suptitleloc[0], y=suptitleloc[1],
        )
        plt.show()
        plt.close()
    else:
        dg.plot_chains(
            core_list,
            suptitle=psr_name,
            pars=common_params_all,
            titles=common_titles_all,
            legend_labels=core_list_legend,
            legend_loc=(0.0, y),
            real_tm_pars=real_tm_pars,
            **hist_core_list_kwargs,
        )


def plot_other_param_overlap(
    psr_name,
    core_list,
    core_list_legend=None,
    com_par_dict=None,
    real_tm_pars=True,
    close=False,
    par_sigma={},
):
    if not com_par_dict:
        com_par_dict = get_other_param_overlap(psr_name, core_list)
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)
    hist_core_list_kwargs = {
        "hist": True,
        "ncols": 1,
        "title_y": 1.4,
        "hist_kwargs": dict(fill=False),
        "linewidth": 3.0,
    }
    plotted_cosi = False
    for key, vals in com_par_dict.items():
        if close:
            if "SINI" in key or "COSI" in key:
                if not plotted_cosi:
                    if (
                        "SINI" in key
                        and ("_").join(key.split("_")[:-1] + ["COSI"])
                        in com_par_dict.keys()
                    ):
                        cosi_chains = []
                        cosi_chains_labels = []
                        for c in vals:
                            gpar_kwargs = dg._get_gpar_kwargs(
                                core_list[c], real_tm_pars
                            )
                            cosi_chains.append(
                                np.sqrt(
                                    1 - core_list[c].get_param(key, **gpar_kwargs) ** 2
                                )
                            )
                            cosi_chains_labels.append(core_list_legend[c])

                        for c in com_par_dict[
                            ("_").join(key.split("_")[:-1] + ["COSI"])
                        ]:
                            gpar_kwargs = dg._get_gpar_kwargs(
                                core_list[c], real_tm_pars
                            )
                            cosi_chains.append(
                                core_list[c].get_param("COSI", **gpar_kwargs)
                            )
                            cosi_chains_labels.append(core_list_legend[c])
                    elif (
                        "COSI" in key
                        and ("_").join(key.split("_")[:-1] + ["SINI"])
                        in com_par_dict.keys()
                    ):
                        cosi_chains_labels = []
                        cosi_chains = []
                        for c in vals:
                            gpar_kwargs = dg._get_gpar_kwargs(
                                core_list[c], real_tm_pars
                            )
                            cosi_chains.append(
                                core_list[c].get_param(key, **gpar_kwargs)
                            )
                            cosi_chains_labels.append(core_list_legend[c])

                        for c in com_par_dict[
                            ("_").join(key.split("_")[:-1] + ["SINI"])
                        ]:
                            gpar_kwargs = dg._get_gpar_kwargs(
                                core_list[c], real_tm_pars
                            )
                            cosi_chains.append(
                                np.sqrt(
                                    1
                                    - core_list[c].get_param("SINI", **gpar_kwargs) ** 2
                                )
                            )
                            cosi_chains_labels.append(core_list_legend[c])

                    fig = plt.figure(figsize=[15, 4])
                    axis = fig.add_subplot(1, 1, 1)
                    for i, chain in enumerate(cosi_chains):
                        plt.hist(
                            chain,
                            bins=40,
                            density=True,
                            log=False,
                            linestyle="-",
                            histtype="step",
                            linewidth=hist_core_list_kwargs["linewidth"],
                            label=cosi_chains_labels[i],
                        )
                    plt.legend()
                    plt.title("COSI")
            else:
                hist_core_list_kwargs["title_y"] = 1.0 + 0.05 * len(core_list)
                y = 0.95 - 0.03 * len(core_list)
                dg.plot_chains(
                    [core_list[c] for c in vals],
                    suptitle=psr_name,
                    pars=[key],
                    legend_labels=[core_list_legend[c] for c in vals],
                    legend_loc=(0.0, y),
                    real_tm_pars=real_tm_pars,
                    show=False,
                    close=False,
                    **hist_core_list_kwargs,
                )
            if par_sigma:
                splt_key = key.split("_")[-1]
                if splt_key == "COSI" or splt_key == "SINI" and "SINI" in par_sigma:
                    if not plotted_cosi:
                        sin_val, sin_err, _ = par_sigma["SINI"]
                        val = np.longdouble(np.sqrt(1 - sin_val ** 2))
                        err = np.longdouble(
                            np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err ** 2)
                        )
                        fill_space_x = np.linspace(val - err, val + err, 20)
                        plt.fill_between(
                            fill_space_x,
                            plt.gca().get_ylim()[1],
                            color="grey",
                            alpha=0.2,
                        )
                        plt.axvline(val, color="k", linestyle="--")
                        plotted_cosi = True
                else:
                    if splt_key in par_sigma:
                        val = par_sigma[splt_key][0]
                        err = par_sigma[splt_key][1]
                        fill_space_x = np.linspace(val - err, val + err, 20)
                        plt.fill_between(
                            fill_space_x,
                            plt.gca().get_ylim()[1],
                            color="grey",
                            alpha=0.2,
                        )
                        plt.axvline(val, color="k", linestyle="--")
            plt.show()
            plt.close()
        else:
            hist_core_list_kwargs["title_y"] = 1.0 + 0.05 * len(core_list)
            y = 0.95 - 0.03 * len(core_list)
            dg.plot_chains(
                [core_list[c] for c in vals],
                suptitle=psr_name,
                pars=[key],
                legend_labels=[core_list_legend[c] for c in vals],
                legend_loc=(0.0, y),
                real_tm_pars=real_tm_pars,
                **hist_core_list_kwargs,
            )
        print("")


def get_fancy_labels(labels, overline=False):
    fancy_labels = []
    for lab in labels:
        if lab == "A1":
            fancy_labels.append(r"$x-\overline{x}$ (lt-s)")
        elif lab == "EPS1":
            fancy_labels.append(r"$\epsilon_{1}-\overline{\epsilon_{1}}$")
        elif lab == "EPS2":
            fancy_labels.append(r"$\epsilon_{2}-\overline{\epsilon_{2}}$")
        elif lab == "M2":
            fancy_labels.append(r"$m_{\mathrm{c}}-\overline{m_{\mathrm{c}}}$")
        elif lab == "COSI":
            fancy_labels.append(r"$\mathrm{cos}i-\overline{\mathrm{cos}i}$")
            # fancy_labels.append(r'$\mathrm{cos}i$')
        elif lab == "PB":
            fancy_labels.append(r"$P_{\mathrm{b}}-\overline{P_{\mathrm{b}}}$")
            # fancy_labels.append(r'$P_{\mathrm{b}}-\overline{P_{\mathrm{b}}}$ (days)')
        elif lab == "TASC":
            fancy_labels.append(r"$T_{\mathrm{asc}}-\overline{T_{\mathrm{asc}}}$")
            # fancy_labels.append(r'$T_{\mathrm{asc}}-\overline{T_{\mathrm{asc}}}$ (MJD)')
        elif lab == "ELONG":
            fancy_labels.append(r"$\lambda-\overline{\lambda}$")
            # fancy_labels.append(r'$\lambda-\overline{\lambda}$ (degrees)')
        elif lab == "ELAT":
            fancy_labels.append(r"$\beta-\overline{\beta}$")
            # fancy_labels.append(r'$\beta-\overline{\beta}$ (degrees)')
        elif lab == "PMELONG":
            fancy_labels.append(r"$\mu_{\lambda}-\overline{\mu_{\lambda}}$")
            # fancy_labels.append(r'$\mu_{\lambda}-\overline{\mu_{\lambda}}$ (mas/yr)')
        elif lab == "PMELAT":
            fancy_labels.append(r"$\mu_{\beta}-\overline{\mu_{\beta}}$")
            # fancy_labels.append(r'$\mu_{\beta}-\overline{\mu_{\beta}}$ (mas/yr)')
        elif lab == "F0":
            fancy_labels.append(r"$\nu-\overline{\nu}$")
            # fancy_labels.append(r'$\nu-\overline{\nu}~(\mathrm{s}^{-1})$')
        elif lab == "F1":
            fancy_labels.append(r"$\dot{\nu}-\overline{\dot{\nu}}$")
            # fancy_labels.append(r'$\dot{\nu}-\overline{\dot{\nu}}~(\mathrm{s}^{-2})$')
        elif lab == "PX":
            # fancy_labels.append(r'$\pi-\overline{\pi}$ (mas)')
            fancy_labels.append(r"$\pi$ (mas)")
        elif "efac" in lab:
            fancy_labels.append(r"EFAC")
        elif "equad" in lab:
            fancy_labels.append(r"$\mathrm{log}_{10}$EQUAD")
        elif "ecorr" in lab:
            fancy_labels.append(r"$\mathrm{log}_{10}$ECORR")
        else:
            fancy_labels.append(lab)
    return fancy_labels


def fancy_plot_all_param_overlap(
    psr_name,
    core_list,
    core_list_legend=None,
    com_par_dict=None,
    exclude=True,
    par_sigma={},
    conf_int=False,
    preliminary=True,
    ncols=4,
    real_tm_pars=True,
    selection="all",
    hist_kwargs={},
    fig_kwargs={},
):
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)

    if not hist_kwargs:
        hist_kwargs = {
            "linewidth": 3.0,
            "density": True,
            "histtype": "step",
            "bins": 40,
        }
    if "suptitle" not in fig_kwargs.keys():
        suptitle = f"{psr_name} Mass Plots"
    else:
        suptitle = fig_kwargs["suptitle"]
    if "labelfontsize" not in fig_kwargs.keys():
        labelfontsize = 18
    else:
        labelfontsize = fig_kwargs["labelfontsize"]
    if "titlefontsize" not in fig_kwargs.keys():
        titlefontsize = 16
    else:
        titlefontsize = fig_kwargs["titlefontsize"]
    if "suptitlefontsize" not in fig_kwargs.keys():
        suptitlefontsize = 24
    else:
        suptitlefontsize = fig_kwargs["suptitlefontsize"]
    if "suptitleloc" not in fig_kwargs.keys():
        suptitleloc = (0.35, 0.94)
    else:
        suptitleloc = fig_kwargs["suptitleloc"]
    if "legendloc" not in fig_kwargs.keys():
        legendloc = (0.5, 0.93)
    else:
        legendloc = fig_kwargs["legendloc"]
    if "legendfontsize" not in fig_kwargs.keys():
        legendfontsize = 12
    else:
        legendfontsize = fig_kwargs["legendfontsize"]
    if "colors" not in fig_kwargs.keys():
        colors = [f"C{ii}" for ii in range(len(core_list_legend))]
    else:
        colors = fig_kwargs["colors"]
    if "wspace" not in fig_kwargs.keys():
        wspace = 0.1
    else:
        wspace = fig_kwargs["wspace"]
    if "hspace" not in fig_kwargs.keys():
        hspace = 0.4
    else:
        hspace = fig_kwargs["hspace"]

    common_params_all, common_titles_all = get_common_params_titles(
        psr_name, core_list, exclude=exclude
    )

    fancy_labels = get_fancy_labels(common_titles_all)
    selected_params = get_param_groups(core_list[0], selection=selection)

    selected_common_params = []
    selected_common_titles = []
    selected_fancy_labels = []
    for cpa, cta, fl in zip(common_params_all, common_titles_all, fancy_labels):
        if cpa in selected_params["par"]:
            selected_common_params.append(cpa)
            selected_common_titles.append(cta)
            selected_fancy_labels.append(fl)

    if preliminary:
        txt = "PRELIMINARY"
        txt_loc = (0.15, 0.15)
        txt_kwargs = {"fontsize": 180, "alpha": 0.25, "rotation": 55}

    L = len(selected_common_params)
    nrows = int(L // ncols)
    if L % ncols > 0:
        nrows += 1
    fig = plt.figure(figsize=[15, 4 * nrows])
    for (ii, par), label in zip(
        enumerate(selected_common_params), selected_fancy_labels
    ):
        if par in selected_params["par"]:
            cell = ii + 1
            axis = fig.add_subplot(nrows, ncols, cell)
            for co in core_list:
                if isinstance(co, TimingCore):
                    plt.hist(co.get_param(par, tm_convert=real_tm_pars), **hist_kwargs)
                elif isinstance(co, Core):
                    plt.hist(co.get_param(par), **hist_kwargs)
            if "efac" in selected_common_titles[ii]:
                axis.set_title(
                    selected_common_titles[ii].split("efac")[0], fontsize=titlefontsize
                )
                axis.set_xlabel(label, fontsize=labelfontsize - 2)
            elif "equad" in selected_common_titles[ii]:
                axis.set_title(
                    selected_common_titles[ii].split("log10")[0], fontsize=titlefontsize
                )
                axis.set_xlabel(label, fontsize=labelfontsize - 2)
            elif "ecorr" in selected_common_titles[ii]:
                axis.set_title(
                    selected_common_titles[ii].split("log10")[0], fontsize=titlefontsize
                )
                axis.set_xlabel(label, fontsize=labelfontsize - 2)
            else:
                axis.set_xlabel(label, fontsize=labelfontsize)

            if "DMX" in par:
                splt_key = ("_").join(par.split("_")[-2:])
            else:
                splt_key = par.split("_")[-1]
            if par_sigma:
                if splt_key in par_sigma:
                    val = par_sigma[splt_key][0]
                    err = par_sigma[splt_key][1]
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    axis.fill_between(
                        fill_space_x, axis.get_ylim()[1], color="grey", alpha=0.2
                    )
                    axis.axvline(val, color="k", linestyle="--")
                elif splt_key == "COSI" and "SINI" in par_sigma:
                    sin_val, sin_err, _ = par_sigma["SINI"]
                    val = np.longdouble(np.sqrt(1 - sin_val ** 2))
                    err = np.longdouble(
                        np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err ** 2)
                    )
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    axis.fill_between(
                        fill_space_x, axis.get_ylim()[1], color="grey", alpha=0.2
                    )
                    axis.axvline(val, color="k", linestyle="--")

            if conf_int:
                if isinstance(conf_int, (float, int)):
                    if conf_int < 1.0 or conf_int > 99.0:
                        raise ValueError("conf_int must be between 1 and 99")
                else:
                    raise ValueError("conf_int must be an int or float")

                for i, core in enumerate(core_list):
                    # elif splt_key == 'COSI' and 'SINI' in par_sigma:
                    for com_par, com_title in zip(
                        selected_common_params, selected_common_titles
                    ):
                        if splt_key == com_title:
                            low, up = core.get_param_confint(com_par, interval=conf_int)
                            axis.fill_between(
                                [low, up], axis.get_ylim()[1], color=f"C{i}", alpha=0.1,
                            )
        axis.set_yticks([])
    patches = []
    for jj, lab in enumerate(core_list_legend):
        patches.append(mpl.patches.Patch(color=colors[jj], label=lab))

    if preliminary:
        fig.text(txt_loc[0], txt_loc[1], txt, **txt_kwargs)
    fig.legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    # fig.subplots_adjust(top=0.96)
    plt.suptitle(
        suptitle, fontsize=suptitlefontsize, x=suptitleloc[0], y=suptitleloc[1],
    )
    # plt.savefig(f'Figures/{psr_name}_cfr19_common_pars_2.png', dpi=150, bbox_inches='tight')
    # plt.savefig(f'Figures/{psr_name}_12p5yr_common_pars.png', dpi=150, bbox_inches='tight')
    plt.show()


def get_param_groups(core, selection="kep"):
    """selection = 'all', or 'kep','gr','spin','pos','noise', 'dm', 'chrom' 'dmx' all joined by underscores"""
    if selection == "all":
        selection = "kep_binary_gr_pm_spin_pos_noise_dm_chrom_dmx"
    kep_pars = [
        "PB",
        "PBDOT",
        "T0",
        "A1",
        "OM",
        "E",
        "ECC",
        "EPS1",
        "EPS2",
        "EPS1DOT",
        "EPS2DOT",
        "FB",
        "SINI",
        "COSI" "MTOT",
        "M2",
        "XDOT",
        "X2DOT",
        "EDOT",
        "KOM",
        "KIN",
        "TASC",
    ]

    mass_pars = ["M2", "SINI", "COSI", "PB", "A1"]

    noise_pars = ["efac", "ecorr", "equad", "gamma", "A"]

    pos_pars = ["RAJ", "DECJ", "ELONG", "ELAT", "BETA", "LAMBDA", "PX"]

    spin_pars = ["F", "F0", "F1", "F2", "P", "P1", "Offset"]

    gr_pars = [
        "H3",
        "H4",
        "OMDOT",
        "OM2DOT",
        "XOMDOT",
        "PBDOT",
        "XPBDOT",
        "GAMMA",
        "PPNGAMMA",
        "DR",
        "DTHETA",
    ]

    pm_pars = ["PMDEC", "PMRA", "PMELONG", "PMELAT", "PMRV", "PMBETA", "PMLAMBDA"]

    dm_pars = [
        "dm_gp_log10_sigma",
        "dm_gp_log10_ell",
        "dm_gp_log10_gam_p",
        "dm_gp_log10_p",
        "dm_gp_log10_ell2",
        "dm_gp_log10_alpha_wgt",
        "n_earth",
    ]

    chrom_gp_pars = [
        "chrom_gp_log10_sigma",
        "chrom_gp_log10_ell",
        "chrom_gp_log10_gam_p",
        "chrom_gp_log10_p",
        "chrom_gp_log10_ell2",
        "chrom_gp_log10_alpha_wgt",
    ]

    excludes = ["lnlike", "lnprior", "chain_accept", "pt_chain_accept"]

    selection_list = selection.split("_")
    plot_params = defaultdict(list)
    for param in core.params:
        split_param = param.split("_")[-1]
        if "kep" in selection_list:
            if split_param in kep_pars and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "mass" in selection_list:
            if split_param in mass_pars and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "pos" in selection_list:
            if split_param in pos_pars and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "noise" in selection_list:
            if split_param in noise_pars and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append((" ").join(param.split("_")[1:]))
        if "spin" in selection_list:
            if split_param in spin_pars and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "gr" in selection_list:
            if split_param in gr_pars and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "pm" in selection_list:
            if split_param in pm_pars and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(split_param)
        if "dm" in selection_list:
            if ("_").join(param.split("_")[1:]) in dm_pars and ("_").join(
                param.split("_")[1:]
            ) not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(param)  # (" ").join(param.split("_")[-2:]))
        if "chrom" in selection_list:
            if ("_").join(param.split("_")[1:]) in chrom_gp_pars and ("_").join(
                param.split("_")[1:]
            ) not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(param)
            elif param in dm_pars and param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(param)
        if "dmx" in selection_list:
            if "DMX_" in param and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(("_").join(param.split("_")[-2:]))
        if "excludes" in selection_list:
            if split_param in excludes and split_param not in plot_params:
                plot_params["par"].append(param)
                plot_params["title"].append(param)
    return plot_params


def corner_plots(
    psr_name, core, selection="kep", save=False, real_tm_pars=False, truths=True
):

    plot_params = get_param_groups(core, selection=selection)

    if truths:
        plot_truths = defaultdict(list)
        for pp in plot_params["par"]:
            if "timing" in pp:
                val, err, normed = core.tm_pars_orig[pp.split("_")[-1]]
                if normed == "normalized":
                    plot_truths["val"].append(0.0)
                    plot_truths["min"].append(-1.0)
                    plot_truths["max"].append(1.0)
                elif pp.split("_")[-1] in ["XDOT", "PBDOT"]:
                    plot_truths["val"].append(val)
                    plot_truths["min"].append(val - err * 1e-12)
                    plot_truths["max"].append(val + err * 1e-12)
                else:
                    plot_truths["val"].append(val)
                    plot_truths["min"].append(val - err)
                    plot_truths["max"].append(val + err)
            else:
                plot_truths["val"].append(-40.0)
                plot_truths["min"].append(-50.0)
                plot_truths["max"].append(-30.0)
    else:
        plot_truths = None

    hist2d_kwargs = {
        "plot_density": False,
        "no_fill_contours": True,
        "data_kwargs": {"alpha": 0.02},
    }
    hist_kwargs = {"range": (-5, 5)}
    ranges = np.ones(len(plot_params)) * 0.98
    if isinstance(core, TimingCore):
        plt_param = core.get_param(plot_params["par"], tm_convert=real_tm_pars)
    elif isinstance(core, Core):
        plt_param = core.get_param(plot_params["par"])
    fig = corner.corner(
        plt_param,
        truths=plot_truths["val"],
        truth_color="k",
        color="C0",
        ranges=ranges,
        labels=plot_params["title"],
        levels=[0.68, 0.95],
        label_kwargs={"fontsize": 20, "rotation": 45},
        **hist2d_kwargs,
    )

    ndim = len(plot_params["par"])
    axes = np.array(fig.axes).reshape((ndim, ndim))

    if truths:
        # Loop over the histograms
        for yi in range(ndim):
            for xi in range(yi):
                ax = axes[yi, xi]
                if plot_truths["min"][xi] == -50:
                    fill_space_x = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 20)
                else:
                    fill_space_x = np.linspace(
                        plot_truths["min"][xi], plot_truths["max"][xi], 20
                    )

                if plot_truths["min"][yi] == -50:
                    ax.fill_between(
                        fill_space_x,
                        ax.get_ylim()[0],
                        ax.get_ylim()[1],
                        color="y",
                        alpha=0.4,
                    )
                else:
                    ax.fill_between(
                        fill_space_x,
                        plot_truths["min"][yi],
                        plot_truths["max"][yi],
                        color="y",
                        alpha=0.4,
                    )

    fig.suptitle(f"{psr_name}, {selection} Parameters", fontsize=24)
    if save:
        plt.savefig(
            f"Figures/{selection}_Corner_{core.label}.png", dpi=150, bbox_inches="tight"
        )
    plt.show()


def mass_pulsar(PB, A1, SINI, M2, errors={}):
    """
    Computes the companion mass from the Keplerian mass function. This
    function uses a Newton-Raphson method since the equation is
    transcendental.

    Computes Keplerian mass function, given projected size and orbital period.
    Inputs:
        - PB = orbital period [days]
        - A1 = projected semimajor axis [lt-s]
    Output:
        - mass function [solar mass]
    """
    T_sun = 4.925490947e-6  # conversion from solar masses to seconds
    nb = 2 * np.pi / PB / 86400
    mf = nb ** 2 * A1 ** 3 / T_sun

    if errors:
        mp_err_sqrd = (
            ((3 / 2) * np.sqrt(M2 * SINI ** 3 / mf) - 1) ** 2 * errors["M2"] ** 2
            + (((3 / 2) * np.sqrt(M2 ** 3 * SINI / mf)) ** 2 * errors["SINI"] ** 2)
            + (
                (np.sqrt(M2 ** 3 * SINI ** 3 / (2 * np.pi) ** 2)) ** 2
                * (errors["PB"] / 8600) ** 2
            )
            + (
                ((3 / 2) * np.sqrt(M2 ** 2 * SINI / nb ** 2 / T_sun / A1)) ** 2
                * errors["A1"] ** 2
            )
        )

        return np.sqrt((M2 * SINI) ** 3 / mf) - M2, np.sqrt(mp_err_sqrd)
    else:
        return np.sqrt((M2 * SINI) ** 3 / mf) - M2


def mass_plot(
    psr_name,
    core_list,
    core_list_legend=None,
    com_par_dict=None,
    exclude=True,
    real_tm_pars=True,
    preliminary=False,
    conf_int=None,
    close=True,
    par_sigma={},
    hist_kwargs={},
    fig_kwargs={},
):
    if not core_list_legend:
        core_list_legend = []
        for core in core_list:
            core_list_legend.append(core.label)

    if not hist_kwargs:
        hist_kwargs = {
            "linewidth": 3.0,
            "density": True,
            "histtype": "step",
            "bins": 40,
        }

    if "suptitle" not in fig_kwargs.keys():
        suptitle = f"{psr_name} Mass Plots"
    else:
        suptitle = fig_kwargs["suptitle"]
    if "suptitlefontsize" not in fig_kwargs.keys():
        suptitlefontsize = 24
    else:
        suptitlefontsize = fig_kwargs["suptitlefontsize"]
    if "suptitleloc" not in fig_kwargs.keys():
        suptitleloc = (0.25, 1.01)
    else:
        suptitleloc = fig_kwargs["suptitleloc"]
    if "legendloc" not in fig_kwargs.keys():
        legendloc = (0.45, 0.925)
    else:
        legendloc = fig_kwargs["legendloc"]
    if "legendfontsize" not in fig_kwargs.keys():
        legendfontsize = 12
    else:
        legendfontsize = fig_kwargs["legendfontsize"]
    if "colors" not in fig_kwargs.keys():
        colors = [f"C{ii}" for ii in range(len(core_list_legend))]
    else:
        colors = fig_kwargs["colors"]
    if "wspace" not in fig_kwargs.keys():
        wspace = 0.1
    else:
        wspace = fig_kwargs["wspace"]
    if "hspace" not in fig_kwargs.keys():
        hspace = 0.4
    else:
        hspace = fig_kwargs["hspace"]
    if "figsize" not in fig_kwargs.keys():
        figsize = (15, 10)
    else:
        figsize = fig_kwargs["figsize"]

    fig, axes = plt.subplots(3, 1, figsize=figsize)
    co_labels = [
        r"Pulsar Mass$~(\mathrm{M}_{\odot})$",
        r"Companion Mass$~(\mathrm{M}_{\odot})$",
        r"$\mathrm{cos}i$",
    ]
    for i, coco in enumerate(core_list):
        if isinstance(coco, TimingCore):
            co_Mc = coco.get_param(
                f"{psr_name}_timing_model_M2", to_burn=True, tm_convert=True
            )
            if f"{psr_name}_timing_model_COSI" in coco.params:
                co_COSI = coco.get_param(
                    f"{psr_name}_timing_model_COSI", to_burn=True, tm_convert=True
                )
                co_Mp = mass_pulsar(
                    coco.get_param(
                        f"{psr_name}_timing_model_PB", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_A1", to_burn=True, tm_convert=True
                    ),
                    np.sqrt((1 - co_COSI ** 2)),
                    coco.get_param(
                        f"{psr_name}_timing_model_M2", to_burn=True, tm_convert=True
                    ),
                )
            else:
                co_Mp = mass_pulsar(
                    coco.get_param(
                        f"{psr_name}_timing_model_PB", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_A1", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_SINI", to_burn=True, tm_convert=True
                    ),
                    coco.get_param(
                        f"{psr_name}_timing_model_M2", to_burn=True, tm_convert=True
                    ),
                )
                co_COSI = np.sqrt(
                    1
                    - coco.get_param(
                        f"{psr_name}_timing_model_SINI", to_burn=True, tm_convert=True
                    )
                    ** 2
                )
        elif isinstance(coco, Core):
            co_Mc = coco.get_param(f"{psr_name}_timing_model_M2", to_burn=True)
            if f"{psr_name}_timing_model_COSI" in coco.params:
                co_COSI = coco.get_param(f"{psr_name}_timing_model_COSI", to_burn=True)
                co_Mp = mass_pulsar(
                    coco.get_param(f"{psr_name}_timing_model_PB", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_A1", to_burn=True),
                    np.sqrt((1 - co_COSI ** 2)),
                    coco.get_param(f"{psr_name}_timing_model_M2", to_burn=True),
                )
            else:
                co_Mp = mass_pulsar(
                    coco.get_param(f"{psr_name}_timing_model_PB", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_A1", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_SINI", to_burn=True),
                    coco.get_param(f"{psr_name}_timing_model_M2", to_burn=True),
                )
                co_COSI = co_COSI = np.sqrt(
                    1
                    - coco.get_param(f"{psr_name}_timing_model_SINI", to_burn=True) ** 2
                )
        co_bins = [co_Mp, co_Mc, co_COSI]
        print(coco.label)
        print("----------------")
        for j, ax in enumerate(axes):
            ax.hist(co_bins[j], label=core_list_legend[i], **hist_kwargs)
            ax.set_xlabel(co_labels[j], fontsize=24)
            ax.get_yaxis().set_visible(False)
            ax.tick_params(axis="x", labelsize=16)
            if conf_int:
                if isinstance(conf_int, (float, int)):
                    if conf_int < 1.0 or conf_int > 99.0:
                        raise ValueError("conf_int must be between 1 and 99")
                else:
                    raise ValueError("conf_int must be an int or float")
                lower_q = (100 - conf_int) / 2
                lower = np.percentile(co_bins[j], q=lower_q)
                upper = np.percentile(co_bins[j], q=100 - lower_q)
                ax.fill_between(
                    [lower, upper], ax.get_ylim()[1], color=f"C{i}", alpha=0.1
                )

                print(co_labels[j])
                if j == 2:
                    print(f"Median: {np.arccos(np.median(co_bins[j]))*180/np.pi}")
                    print(f"Upper: {np.arccos(lower)*180/np.pi}")
                    print(f"Lower: {np.arccos(upper)*180/np.pi}")
                    print(
                        f"Diff Upper: {(np.arccos(lower)-np.arccos(np.median(co_bins[j])))*180/np.pi}"
                    )
                    print(
                        f"Diff Lower: {(np.arccos(np.median(co_bins[j]))-np.arccos(upper))*180/np.pi}"
                    )
                    print("")
                else:
                    print(f"Median: {np.median(co_bins[j])}")
                    print(f"Lower: {lower}")
                    print(f"Upper: {upper}")
                    print(f"Diff Lower: {np.median(co_bins[j])-lower}")
                    print(f"Diff Upper: {upper-np.median(co_bins[j])}")
                    print("")

    fig = plt.gcf()
    allaxes = fig.get_axes()
    if par_sigma:
        for ax, splt_key in zip(allaxes, ["Mp", "M2", "COSI"]):
            if par_sigma:
                if splt_key in par_sigma:
                    val = par_sigma[splt_key][0]
                    err = par_sigma[splt_key][1]
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    ax.fill_between(
                        fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                    )
                    ax.axvline(val, color="k", linestyle="--")
                elif splt_key == "COSI" and "SINI" in par_sigma:
                    sin_val, sin_err, _ = par_sigma["SINI"]
                    val = np.longdouble(np.sqrt(1 - sin_val ** 2))
                    err = np.longdouble(
                        np.sqrt((np.abs(sin_val / val)) ** 2 * sin_err ** 2)
                    )
                    fill_space_x = np.linspace(val - err, val + err, 20)
                    ax.fill_between(
                        fill_space_x, ax.get_ylim()[1], color="grey", alpha=0.2
                    )
                    ax.axvline(val, color="k", linestyle="--")

    # fig = plt.gcf()
    patches = []
    for jj, lab in enumerate(core_list_legend):
        patches.append(
            mpl.patches.Patch(color=colors[jj], label=lab)
        )  # .split(":")[-1]))

    if preliminary:
        txt = "PRELIMINARY"
        txt_loc = (0.05, 0.1)
        txt_kwargs = {"fontsize": 165, "alpha": 0.25, "rotation": 30}
        fig.text(txt_loc[0], txt_loc[1], txt, **txt_kwargs)
    allaxes[0].legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle(
        suptitle, fontsize=suptitlefontsize, x=suptitleloc[0], y=suptitleloc[1],
    )


########################################
# Taken from mcmc_diagnostics
def geweke_check(chain, burn_frac=None, threshold=0.25):
    """
    Function to check for stationarity of MCMC chain using the Geweke diagnostic from arviz.geweke
    
    Parameters
    ============
    chain -- N-dimensional MCMC posterior chain. Assumes rows = samples, columns = parameters.
    burn_frac -- Burn-in fraction; Default: None
    threshold -- Threshold to determine failure of stationarity for given chain; Default: 0.25
    
    Returns
    ============
    nc_idx -- index of parameters in the chain whose samples fail the Geweke test
    """
    if burn_frac is not None:
        burn_len = int(chain.shape[0] * burn_frac)
        test_chain = chain[burn_len:, :].copy()
    else:
        test_chain = chain[:, :].copy()

    nsamp, nparam = test_chain.shape

    nc_flag = np.full((nparam), False)

    # print(nparam)

    for ii in range(nparam):

        # print(ii)

        gs = np.array([[5, 5]])

        trial_starts = np.array([0.1, 0.2, 0.3, 0.4])

        jj = 0

        while sum(np.abs(gs[:, 1]) > threshold) > 0:

            if jj == len(trial_starts):
                nc_flag[ii] = True
                break

            first = trial_starts[jj]
            last = 0.5

            gs = pymc3.geweke(test_chain[:, ii], first=first, last=last, intervals=10)

            jj += 1

    nc_idx = np.arange(0, test_chain.shape[1])[nc_flag]

    return nc_idx


def geweke_plot(arr, threshold=0.25, first=0.1, last=0.5, intervals=10):
    """
    Function to plot the z-score from arviz.geweke with threshold
    
    Parameters
    ===========
    arr -- Input (1-D) array
    threshold -- Threshold for geweke test; Default: 0.25
    first -- First fraction of arr; Default: 0.1
    last -- Last fraction of arr; Default: 0.5
    intervals -- Number of intervals of `first` fraction of arr to compute z-score
    
    Returns
    ===========
    Plots the Geweke diagnostic plot. For stationary chains, the z-score should oscillate between
    the `threshold` values, but not exceed it.
    """
    gs = pymc3.geweke(arr, first=first, last=last, intervals=10)

    pl.plot(gs[:, 0], gs[:, 1], marker="o", ls="-")

    pl.axhline(threshold)
    pl.axhline(-1 * threshold)

    pl.ylabel("Z-score")
    pl.xlabel("No. of samples")

    pl.show()

    return None


def plot_dist_evolution(arr, nbins=20, fracs=np.array([0.1, 0.2, 0.3, 0.4]), last=0.5):
    """
    Function to plot histograms of different fractions used in Geweke test
    
    Parameters
    ===========
    arr -- Input (1-D) array
    nbins -- Number of bins in histograms; Default: 20
    fracs -- Starting fractions of arr to plot; Default: [0.1, 0.2, 0.3, 0.4]
    last -- Final fraction of arr to plot; Default: 0.5
    
    Returns
    ===========
    Plots of histograms of given fractions overlayed together.
    """
    fracs = fracs

    last = last
    last_subset = arr[int(last * arr.shape[0]) :]

    for ff in fracs:

        subset = arr[: int(ff * arr.shape[0])]

        pl.hist(
            subset, nbins, histtype="step", density=True, label="f=0.0--{}".format(ff)
        )

    pl.hist(
        last_subset,
        nbins,
        histtype="step",
        density=True,
        label="f={}--1.0".format(last),
    )

    pl.legend(loc="best", ncol=3)

    pl.show()

    return None


def get_param_acorr(arr, burn=None):
    """
    Function to get the autocorrelation length for each parameter in a ndim array

    Parameters
    ==========
    arr -- array, optional
        Array that contains samples from an MCMC array that is samples x param
        in shape.
    burn -- int, optional
        Number of samples burned from beginning of array. Used when calculating
        statistics and plotting histograms. If None, burn is `len(samples)/4`

    Returns
    =======
    Array of autocorrelation lengths for each parameter
    """
    if burn is None:
        burn = int(0.25 * arr.shape[0])

    # Do not want autocorr of acceptance rate or pt acceptance rate
    len_rel_params = arr.shape[1] - 2
    tau_arr = np.zeros(len_rel_params)
    for param_idx in range(len_rel_params):
        indv_param = arr[burn:, param_idx]
        tau_arr[param_idx], _, _ = acor.acor(indv_param)
    return tau_arr


def trim_array_acorr(arr, burn=None):
    """
    Function to trim an array by the longest autocorrelation length of all parameters in a ndim array

    Parameters
    ==========
    arr -- array, optional
        Array that contains samples from an MCMC chain that is samples x param
        in shape.
    burn -- int, optional
        Number of samples burned from beginning of chain. Used when calculating
        statistics and plotting histograms. If None, burn is `len(samples)/4`.

    Returns
    =======
    Thinned array
    """
    if burn is None:
        burn = int(0.25 * arr.shape[0])
    acorr_lens = get_param_acorr(arr, burn=burn)
    max_acorr = int(np.unique(np.max(acorr_lens)))
    return arr[burn::max_acorr]


def grubin(data, M=4, burn=0.25, threshold=1.1):
    """
    Gelman-Rubin split R hat statistic to verify convergence.

    See section 3.1 of https://arxiv.org/pdf/1903.08008.pdf.
    Values > 1.1 => recommend continuing sampling due to poor convergence.

    Input:
        data (ndarray): consists of entire chain file
        pars (list): list of parameters for each column
        M (integer): number of times to split the chain
        burn (float): percent of chain to cut for burn-in
        threshold (float): Rhat value to tell when chains are good

    Output:
        Rhat (ndarray): array of values for each index
        idx (ndarray): array of indices that are not sampled enough (Rhat > threshold)
    """
    burn = int(burn * data.shape[0])  # cut off burn-in
    # data_split = np.split(data[burn:,:-2], M)  # cut off last two columns
    try:
        data_split = np.split(data[burn:, :-2], M)  # cut off last two columns
    except:
        # this section is to make everything divide evenly into M arrays
        P = int(np.floor((len(data[:, 0]) - burn) / M))  # nearest integer to division
        X = len(data[:, 0]) - burn - M * P  # number of additional burn in points
        burn += X  # burn in to the nearest divisor
        burn = int(burn)

        data_split = np.split(data[burn:, :-2], M)  # cut off last two columns

    N = len(data[burn:, 0])
    data = np.array(data_split)

    # print(data_split.shape)

    theta_bar_dotm = np.mean(data, axis=1)  # mean of each subchain
    theta_bar_dotdot = np.mean(theta_bar_dotm, axis=0)  # mean of between chains
    B = (
        N / (M - 1) * np.sum((theta_bar_dotm - theta_bar_dotdot) ** 2, axis=0)
    )  # between chains

    # do some clever broadcasting:
    sm_sq = 1 / (N - 1) * np.sum((data - theta_bar_dotm[:, None, :]) ** 2, axis=1)
    W = 1 / M * np.sum(sm_sq, axis=0)  # within chains

    var_post = (N - 1) / N * W + 1 / N * B
    Rhat = np.sqrt(var_post / W)

    idx = np.where(Rhat > threshold)[0]  # where Rhat > threshold

    return Rhat, idx
