import pandas as pd

import argparse

import nltm_plot_utils_v5 as nltm

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "--parfile",
    default="",
    help="Location of parfile </PATH/TO/FILE/PARFILE.par>",
    required=True,
)
args = parser.parse_args()


nltm.make_dmx_file(args.parfile)
