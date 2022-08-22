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

lamb = 31
x_range = np.arange(80)

poisson_pmf = sps.poisson.pmf(x_range, lamb)
gaussian_pdf = sps.norm.pdf(x_range, loc=lamb, scale=np.sqrt(lamb))

# +
plt.figure(figsize=(13, 10))
plt.subplot(211)
plt.plot(x_range, poisson_pmf, '.-', label='Poisson')
plt.plot(x_range, gaussian_pdf, '.-', label='Gaussian')
plt.axvline(31-4, lw=1, c='r')
plt.axvline(31+8, lw=1, c='r')
plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend()

plt.subplot(212)
plt.title('Gaussian pdf - Poisson pmf')
plt.plot(x_range, (gaussian_pdf-poisson_pmf))
# -


