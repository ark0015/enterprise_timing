import os
import numpy as np

import enterprise_extensions as e_e
from enterprise_extensions.sampler import (
    JumpProposal,
    get_parameter_groups,
    get_timing_groups,
    group_from_params,
    save_runtime_info,
)

from enterprise_extensions.hypermodel import HyperModel
from PTMCMCSampler.PTMCMCSampler import PTSampler as ptmcmc


class TimingHyperModel(HyperModel):
    """Used to adapt Hypermodel class with timing changes"""

    def __init__(
        self,
        models,
        log_weights=None,
    ):
        super().__init__(
            models,
            log_weights=None,
        )
        self.tm_groups = []
        self.special_idxs = []
        for i, x in enumerate(self.params):
            if "timing_model" in str(x):
                self.tm_groups.append(i)
                if "Uniform" in str(x):
                    pmin = float(
                        str(x).split("Uniform")[-1].split("pmin=")[1].split(",")[0]
                    )
                    pmax = float(
                        str(x).split("Uniform")[-1].split("pmax=")[-1].split(")")[0]
                    )
                    if pmin + pmax != 0.0:
                        self.special_idxs.append(i)

    def get_parameter_groups(self):
        groups = []
        for p in super_model.models.values():
            pta_groups = []
            pta_groups.extend(h_t.get_parameter_groups(p))
            pta_groups.extend(h_t.get_timing_groups(p))
            pta_groups.append(
                h_t.group_from_params(
                    p,
                    [
                        x
                        for x in p.param_names
                        if any(y in x for y in ["timing_model", "ecorr"])
                    ],
                )
            )
            for grp in pta_groups:
                if not isinstance(grp, (int, np.int64)):
                    groups.append(
                        [
                            list(super_model.param_names).index(p.param_names[subgrp])
                            for subgrp in grp
                        ]
                    )
                else:
                    groups.append(super_model.param_names[grp])
        groups = list(np.unique(groups))
        groups.extend([[len(super_model.param_names) - 1]])  # nmodel

        return groups

    def initial_sample(self, tm_params_orig=None, tm_param_dict=None, zero_start=True):
        """
        Draw an initial sample from within the hyper-model prior space.
        :param tm_params_orig: dictionary of timing model parameter tuples, (val, err)
        :param tm_param_dict: a nested dictionary of parameters to vary in the model and their user defined values and priors
        :param zero_start: start all timing parameters at their parfile value (in tm_params_orig), or their refit values (tm_param_dict)
        """

        if zero_start and tm_params_orig:
            x0 = []
            for p in self.models[0].params:
                if "timing" in p.name:
                    if "DMX" in p.name:
                        p_name = ("_").join(p.name.split("_")[-2:])
                    else:
                        p_name = p.name.split("_")[-1]
                    if tm_params_orig[p_name][-1] == "normalized":
                        x0.append([np.double(0.0)])
                    else:
                        if p_name in tm_param_dict.keys():
                            x0.append([np.double(tm_param_dict[p_name]["prior_mu"])])
                        else:
                            x0.append([np.double(tm_params_orig[p_name][0])])
                else:
                    x0.append(np.array(p.sample()).ravel().tolist())
        else:
            x0 = [np.array(p.sample()).ravel().tolist() for p in self.models[0].params]
        uniq_params = [str(p) for p in self.models[0].params]

        for model in self.models.values():
            param_diffs = np.setdiff1d([str(p) for p in model.params], uniq_params)
            mask = np.array([str(p) in param_diffs for p in model.params])
            x0.extend(
                [
                    np.array(pp.sample()).ravel().tolist()
                    for pp in np.array(model.params)[mask]
                ]
            )
            uniq_params = np.union1d([str(p) for p in model.params], uniq_params)
        x0.extend([[0.1]])
        return np.array([p for sublist in x0 for p in sublist])

    def setup_sampler(
        self,
        outdir="chains",
        resume=False,
        sample_nmodel=True,
        empirical_distr=None,
        groups=None,
        timing=True,
        human=None,
        loglkwargs={},
        logpkwargs={},
    ):
        """
        Sets up an instance of PTMCMC sampler.

        We initialize the sampler the likelihood and prior function
        from the PTA object. We set up an initial jump covariance matrix
        with fairly small jumps as this will be adapted as the MCMC runs.

        We will setup an output directory in `outdir` that will contain
        the chain (first n columns are the samples for the n parameters
        and last 4 are log-posterior, log-likelihood, acceptance rate, and
        an indicator variable for parallel tempering but it doesn't matter
        because we aren't using parallel tempering).

        We then add several custom jump proposals to the mix based on
        whether or not certain parameters are in the model. These are
        all either draws from the prior distribution of parameters or
        draws from uniform distributions.
        """

        # dimension of parameter space
        ndim = len(self.param_names)

        # initial jump covariance matrix
        if os.path.exists(outdir + "/cov.npy"):
            try:
                cov = np.load(outdir + "/cov.npy")
            except (ValueError):
                cov = np.diag(np.ones(ndim) * 0.1**2)
        else:
            cov = np.diag(np.ones(ndim) * 1.0**2)  # used to be 0.1

        # parameter groupings
        if groups is None:
            groups = self.get_parameter_groups()

        sampler = ptmcmc(
            ndim,
            self.get_lnlikelihood,
            self.get_lnprior,
            cov,
            groups=groups,
            outDir=outdir,
            resume=resume,
            loglkwargs=loglkwargs,
            logpkwargs=logpkwargs,
        )

        save_runtime_info(self, sampler.outDir, human)

        # additional jump proposals
        jp = JumpProposal(
            self, self.snames, empirical_distr=empirical_distr, timing=timing
        )
        sampler.jp = jp
        # always add draw from prior
        sampler.addProposalToCycle(jp.draw_from_prior, 15)

        # try adding empirical proposals
        if empirical_distr is not None:
            print("Adding empirical proposals...\n")
            sampler.addProposalToCycle(jp.draw_from_empirical_distr, 25)

        # Red noise prior draw
        if "red noise" in self.snames:
            print("Adding red noise prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_red_prior, 10)

        # DM GP noise prior draw
        if "dm_gp" in self.snames:
            print("Adding DM GP noise prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dm_gp_prior, 10)

        # DM annual prior draw
        if "dm_s1yr" in jp.snames:
            print("Adding DM annual prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dm1yr_prior, 10)

        # DM dip prior draw
        if "dmexp" in "\t".join(jp.snames):
            print("Adding DM exponential dip prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dmexpdip_prior, 10)

        # DM cusp prior draw
        if "dm_cusp" in jp.snames:
            print("Adding DM exponential cusp prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dmexpcusp_prior, 10)

        # DMX prior draw
        if "dmx_signal" in jp.snames:
            print("Adding DMX prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dmx_prior, 10)

        # Chromatic GP noise prior draw
        if "chrom_gp" in self.snames:
            print("Adding Chromatic GP noise prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_chrom_gp_prior, 10)

        # SW prior draw
        if "gp_sw" in jp.snames:
            print("Adding Solar Wind DM GP prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dm_sw_prior, 10)

        # Chromatic GP noise prior draw
        if "chrom_gp" in self.snames:
            print("Adding Chromatic GP noise prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_chrom_gp_prior, 10)

        # SW prior draw
        if "gp_sw" in jp.snames:
            print("Adding Solar Wind DM GP prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dm_sw_prior, 10)

        # Chromatic GP noise prior draw
        if "chrom_gp" in self.snames:
            print("Adding Chromatic GP noise prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_chrom_gp_prior, 10)

        # Ephemeris prior draw
        if "d_jupiter_mass" in self.param_names:
            print("Adding ephemeris model prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_ephem_prior, 10)

        # GWB uniform distribution draw
        if np.any([("gw" in par and "log10_A" in par) for par in self.param_names]):
            print("Adding GWB uniform distribution draws...\n")
            sampler.addProposalToCycle(jp.draw_from_gwb_log_uniform_distribution, 10)

        # Dipole uniform distribution draw
        if "dipole_log10_A" in self.param_names:
            print("Adding dipole uniform distribution draws...\n")
            sampler.addProposalToCycle(jp.draw_from_dipole_log_uniform_distribution, 10)

        # Monopole uniform distribution draw
        if "monopole_log10_A" in self.param_names:
            print("Adding monopole uniform distribution draws...\n")
            sampler.addProposalToCycle(
                jp.draw_from_monopole_log_uniform_distribution, 10
            )

        # BWM prior draw
        if "bwm_log10_A" in self.param_names:
            print("Adding BWM prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_bwm_prior, 10)

        # FDM prior draw
        if "fdm_log10_A" in self.param_names:
            print("Adding FDM prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_fdm_prior, 10)

        # CW prior draw
        if "cw_log10_h" in self.param_names:
            print("Adding CW prior draws...\n")
            sampler.addProposalToCycle(jp.draw_from_cw_log_uniform_distribution, 10)

        # Prior distribution draw for parameters named GW
        if any([str(p).split(":")[0] for p in list(self.params) if "gw" in str(p)]):
            print("Adding gw param prior draws...\n")
            sampler.addProposalToCycle(
                jp.draw_from_par_prior(
                    par_names=[
                        str(p).split(":")[0]
                        for p in list(self.params)
                        if "gw" in str(p)
                    ]
                ),
                10,
            )

        # Non Linear Timing Draws
        if "timing_model" in jp.snames:
            print("Adding timing model jump proposal...\n")
            sampler.addProposalToCycle(jp.draw_from_timing_model, 30)
        if "timing_model" in jp.snames:
            print("Adding timing model prior draw...\n")
            sampler.addProposalToCycle(jp.draw_from_timing_model_prior, 10)

        # Model index distribution draw
        if sample_nmodel:
            if "nmodel" in self.param_names:
                print("Adding nmodel uniform distribution draws...\n")
                sampler.addProposalToCycle(self.draw_from_nmodel_prior, 25)

        return sampler
