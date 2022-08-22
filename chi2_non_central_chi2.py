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

# # Granger causality and hypothesis testing
#
# If H0 is true, test statistic is distributed according to chi2
#
# If H1 is true, test statistic is distributed according to non-central chi2

import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

# +
k = 5 # amt of history of x
l = 3 # amt of history of y
p = 1 # order of autoregression model

n = 100# sample size

x_min = -1.
x_max = 50.
x = np.linspace(x_min, x_max, n)

df = k*l*p # chi2 dof. H0 asymptotically chi2.
nc = n*x # non-central chi2 non-centrality parameter

# +
chi2 = sps.chi2.pdf(x, df)
ncchi2 = sps.ncx2.pdf(x, df, nc)

alpha = sps.chi2.sf(x, df)
beta = sps.ncx2.cdf(x, df, nc)

# +
plt.figure(figsize=(15, 10))
plt.subplot(211)
plt.plot(x, chi2, '.-', label='chi2')
plt.plot(x, ncchi2, '.-', label='non-central chi2')
plt.xlabel('Test statistic value')
plt.ylabel('PDF')
plt.legend()

plt.subplot(212)
plt.plot(x, alpha, '.-', label='chi2 sf (alpha)')
plt.plot(x, beta, '.-', label='non-central chi2 cdf (beta)')
plt.xlabel('Test statistic value')
plt.legend()

plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# -




