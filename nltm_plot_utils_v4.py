import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
from collections import defaultdict, OrderedDict
from copy import deepcopy

import pandas as pd
import corner

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
    return -(new_res[isort] - residuals[isort])


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
            label=f"Old Residual, {pars[0]}={vals[0]:.2e}",
            **kwargs,
        )
        plt.errorbar(
            t[i],
            new_res[i] / 1e-6,
            # yerr=errs[i],
            fmt="+",
            label=f"New Residual, {pars[0]}={par_dict[pars[0]]:.2e}",
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
                label=f"Old Residual, {pars[0]}={vals[0]:.2e}",
                **kwargs,
            )
            plt.errorbar(
                flagt[i],
                flagnewres[i] / 1e-6,
                yerr=flagerrs[i],
                fmt="x",
                label=f"New Residual, {pars[0]}={par_dict[pars[0]]:.2e}",
                **kwargs,
            )

        # plt.legend(unique,numpoints=1,bbox_to_anchor=(1.1,1.1))

    plt.xlabel("MJD")
    plt.ylabel("res [us]")
    plt.title(
        f"{psr.name} - rms old res = {meanoldres:.3f} us new res {meannewres:.3f} us"
    )


def summary_comparison(core, par_sigma={}):
    """Makes comparison table of the form:
    Par Name | Old Value | New Value | Difference | Old Sigma | New Sigma 
    TODO: allow for selection of subparameters"""
    # pd.set_option('max_rows', None)
    titles = nltm.get_titles(psr_name, core)
    summary_dict = {}
    for pnames, title in zip(core.params, titles):
        if "timing" in pnames:
            param_vals = core.get_param(pnames, tm_convert=True, to_burn=True)
            # else:
            #    param_vals = core.get_param(pnames,tm_convert=False,to_burn=True)

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
            "New Value",
            "Difference",
            "Old Sigma",
            "New Sigma",
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
        unscaled_param = core.get_param(par, to_burn=True)
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
            suptitle=psr_name,
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
            f"{psr_name} Common Parameters",
            fontsize=suptitlefontsize,
            x=suptitleloc[0],
            y=suptitleloc[1],
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

    if preliminary:
        txt = "PRELIMINARY"
        txt_loc = (0.15, 0.15)
        txt_kwargs = {"fontsize": 180, "alpha": 0.25, "rotation": 55}

    L = len(common_params_all)
    nrows = int(L // ncols)
    if L % ncols > 0:
        nrows += 1
    fig = plt.figure(figsize=[15, 4 * nrows])
    for (ii, par), label in zip(enumerate(common_params_all), fancy_labels):
        cell = ii + 1
        axis = fig.add_subplot(nrows, ncols, cell)
        for co in core_list:
            if isinstance(co, TimingCore):
                plt.hist(co.get_param(par, tm_convert=real_tm_pars), **hist_kwargs)
            elif isinstance(co, Core):
                plt.hist(co.get_param(par), **hist_kwargs)
        if "efac" in common_titles_all[ii]:
            axis.set_title(
                common_titles_all[ii].split("efac")[0], fontsize=titlefontsize
            )
            axis.set_xlabel(label, fontsize=labelfontsize - 2)
        elif "equad" in common_titles_all[ii]:
            axis.set_title(
                common_titles_all[ii].split("log10")[0], fontsize=titlefontsize
            )
            axis.set_xlabel(label, fontsize=labelfontsize - 2)
        elif "ecorr" in common_titles_all[ii]:
            axis.set_title(
                common_titles_all[ii].split("log10")[0], fontsize=titlefontsize
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
                for com_par, com_title in zip(common_params_all, common_titles_all):
                    if splt_key == com_title:
                        low, up = core.get_param_confint(com_par, interval=conf_int)
                        axis.fill_between(
                            [low, up], axis.get_ylim()[1], color=f"C{i}", alpha=0.1,
                        )
        axis.set_yticks([])
    patches = []
    for jj, lab in enumerate(core_list_legend):
        patches.append(mpl.patches.Patch(color=colors[jj], label=lab.split(":")[-1]))

    if preliminary:
        fig.text(txt_loc[0], txt_loc[1], txt, **txt_kwargs)
    fig.legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    # fig.subplots_adjust(top=0.96)
    plt.suptitle(
        f"{psr_name} Common Parameters",
        fontsize=suptitlefontsize,
        x=suptitleloc[0],
        y=suptitleloc[1],
    )
    # plt.savefig(f'Figures/{psr_name}_cfr19_common_pars_2.png', dpi=150, bbox_inches='tight')
    # plt.savefig(f'Figures/{psr_name}_12p5yr_common_pars.png', dpi=150, bbox_inches='tight')
    plt.show()


def get_param_groups(core, selection="kep"):
    """selection = 'all', or 'kep','gr','spin','pos','noise', 'dm' all joined by underscores"""
    if selection == "all":
        selection = "kep_binary_gr_spin_pos_noise_dm"
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
        "MTOT",
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
        "log10_sigma",
        "log10_ell",
        "log10_gam_p",
        "log10_p",
        "log10_ell2",
        "log10_alpha_wgt",
    ]

    excludes = ["lnlike", "lnprior", "chain_accept", "pt_chain_accept"]

    selection_list = selection.split("_")
    plot_params = defaultdict(list)
    for param in core.params:
        split_param = param.split("_")[-1]
        if "kep" in selection_list:
            for p in kep_pars:
                if p == split_param and p not in plot_params:
                    plot_params["par"].append(param)
                    plot_params["title"].append(param.split("_")[-1])
        if "mass" in selection_list:
            for p in mass_pars:
                if p == split_param and "kep" not in selection_list:
                    plot_params["par"].append(param)
                    plot_params["title"].append(param.split("_")[-1])
        if "pos" in selection_list:
            for p in pos_pars:
                if p == split_param and p not in plot_params:
                    plot_params["par"].append(param)
                    plot_params["title"].append(param.split("_")[-1])
        if "noise" in selection_list:
            for p in noise_pars:
                if p == split_param and p not in plot_params:
                    plot_params["par"].append(param)
                    plot_params["title"].append((" ").join(param.split("_")[1:]))
        if "spin" in selection_list:
            for p in spin_pars:
                if p == split_param and p not in plot_params:
                    plot_params["par"].append(param)
                    plot_params["title"].append(param.split("_")[-1])
        if "gr" in selection_list:
            for p in gr_pars:
                if p == split_param and p not in plot_params:
                    plot_params["par"].append(param)
                    plot_params["title"].append(param.split("_")[-1])
        if "pm" in selection_list:
            for p in pm_pars:
                if p == split_param and p not in plot_params:
                    plot_params["par"].append(param)
                    plot_params["title"].append(param.split("_")[-1])
        if "dm" in selection_list:
            for p in dm_pars:
                if p in ("_").join(param.split("_")[-2:]) and p not in plot_params:
                    plot_params["par"].append(param)
                    plot_params["title"].append((" ").join(param.split("_")[-2:]))
        if "excludes" in selection_list:
            for p in excludes:
                if p in param and p not in plot_params:
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


def mass_pulsar(PB, A1, SINI, M2):
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
        for j, ax in enumerate(axes):
            ax.hist(co_bins[j], label=core_list_legend[i], **hist_kwargs)
            ax.set_xlabel(co_labels[j], fontsize=24)
            ax.get_yaxis().set_visible(False)
            ax.tick_params(axis="x", labelsize=16)

    if conf_int or par_sigma:
        fig = plt.gcf()
        allaxes = fig.get_axes()
        for ax, splt_key in zip(allaxes[1:], ["M2", "COSI"]):
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
                    for com_par, com_title in zip(common_params_all, common_titles_all):
                        if splt_key == com_title:
                            low, up = core.get_param_confint(com_par, interval=conf_int)
                            ax.fill_between(
                                [low, up], ax.get_ylim()[1], color=f"C{i}", alpha=0.1,
                            )
    else:
        fig = plt.gcf()

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
    fig.legend(handles=patches, loc=legendloc, fontsize=legendfontsize)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    plt.suptitle(
        f"{psr_name} Mass Plots",
        fontsize=suptitlefontsize,
        x=suptitleloc[0],
        y=suptitleloc[1],
    )
