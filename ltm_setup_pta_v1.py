import numpy as np
import os, sys
from collections import OrderedDict

current_path = os.getcwd()
splt_path = current_path.split("/")
top_path_idx = splt_path.index("nanograv")
# top_path_idx = splt_path.index("akaiser")
# top_path_idx = splt_path.index("ark0015")
top_dir = "/".join(splt_path[0 : top_path_idx + 1])

e_e_path = top_dir + "/enterprise_extensions/"
e_path = top_dir + "/enterprise/"
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


def pta_setup(psr, tnequad=False, coefficients=False):
    # create new attribute for enterprise pulsar object
    # UNSURE IF NECESSARY
    psr.tm_params_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
    for key in psr.tm_params_orig:
        psr.tm_params_orig[key] = (psr.t2pulsar[key].val, psr.t2pulsar[key].err)
    s = gp_signals.TimingModel(use_svd=False, normed=False, coefficients=coefficients)

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
    # white noise signals
    if tnequad:
        efeq = white_signals.MeasurementNoise(efac=efac,
                                              selection=backend, name=None)
        efeq += white_signals.TNEquadNoise(log10_tnequad=equad,
                                           selection=backend, name=None)
    else:
        efeq = white_signals.MeasurementNoise(efac=efac, log10_t2equad=equad,
                                              selection=backend, name=None)

    # ec = gp_signals.EcorrBasisModel(log10_ecorr=ecorr, selection=backend_ch,coefficients=coefficients)
    ec = white_signals.EcorrKernelNoise(log10_ecorr=ecorr, selection=backend_ch)

    # combine signals
    s += efeq + ec

    model = s(psr)

    # set up PTA
    pta = signal_base.PTA([model])

    return pta


def get_coeffs(pta, x0_dict):
    print(x0_dict)
    print("----------------------")
    print("")
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
    return psc
