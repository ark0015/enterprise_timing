# NLTM vs. LTM Marginalization Comparison
These are four files I used to compare the analytically marginalized timing model (LTM) and the numerically marginalized timing model (NLTM).

## ltm_simple_v1.py
This file is used to run the LTM model. 
- For J1640+2224, we ran with red noise and white noise.
- For J1600-3053, we ran one with red and white noise, and one with only white noise to match the NLTM runs.

## ltm_setup_pta_v2.py
This is used to reconstruct the pta with which the model originally ran. It is used in the plotting notebooks as utility functions.

## Plot_J1600-3053_nltm_vs_ltm_post.ipynb
Notebook used to examine and compare the NLTM and LTM runs for J1600-3053.
- Should have outputs pre-loaded, the relevant comparisons are in the last cells

## Plot_J1640+2224_nltm_vs_ltm_post.ipynb
Notebook used to examine and compare the NLTM and LTM runs for J1640+2224.
- Should have outputs pre-loaded, the relevant comparisons are in the last cells