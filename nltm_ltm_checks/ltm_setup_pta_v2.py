import numpy as np
import os, sys, inspect
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

from enterprise.signals import utils
from enterprise.signals import signal_base
from enterprise.signals import gp_signals

from enterprise_extensions import blocks
from enterprise_extensions import models


def pta_setup(psr, tnequad=False, coefficients=False, red_noise=True):
    # create new attribute for enterprise pulsar object
    # UNSURE IF NECESSARY
    psr.tm_params_orig = OrderedDict.fromkeys(psr.t2pulsar.pars())
    for key in psr.tm_params_orig:
        psr.tm_params_orig[key] = (psr.t2pulsar[key].val, psr.t2pulsar[key].err)

    model_args = inspect.getfullargspec(models.model_singlepsr_noise)
    model_keys = model_args[0][1:]
    model_vals = model_args[3]
    model_kwargs = dict(zip(model_keys, model_vals))
    model_kwargs.update(
        {
            "tm_var": False,
            "tm_linear": False,
            "red_var": red_noise,
            "noisedict": None,
            "white_vary": True,
            "coefficients": coefficients,
        }
    )

    pta = models.model_singlepsr_noise(psr, **model_kwargs)

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
