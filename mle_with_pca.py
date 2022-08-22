# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

# +
mu1 = 1.
sigma1 = 2.
mu2 = 3.
sigma2 = 4.

n_pts = 500

x = sps.norm.rvs(loc=mu1, scale=sigma1, size=n_pts)
y = sps.norm.rvs(loc=mu2, scale=sigma2, size=n_pts)
# -

x2 = x+y
y2 = x-y

plt.plot(x2, y2, '.')


