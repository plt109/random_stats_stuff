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

# +
import numpy as np
import scipy.stats as sps

import matplotlib.pyplot as plt

# +
my_mu = 2.
my_sigma = 0.1

num_pts = 1000
# -

mc = sps.norm.rvs(loc=my_mu, scale=my_sigma, size=num_pts)

plt.hist(mc, bins=50)

# +
min_mu = -5.
max_mu = 5.
num_mu = 100
mu_arr = np.linspace(min_mu, max_mu, num_mu)
mu_step = np.unique(np.diff(mu_arr))[0]

min_sigma = 1e-3
max_sigma = 1.
num_sigma = 100
sigma_arr = np.linspace(min_sigma, max_sigma, num_sigma)
sigma_step = np.unique(np.diff(sigma_arr))[0]


# +
llh_bag = 0
llh_grid = np.zeros((num_mu, num_sigma))
for ind_mu, this_mu in enumerate(mu_arr):
    for ind_sigma, this_sigma in enumerate(sigma_arr):
        this_llh = np.sum(sps.norm.logpdf(mc, loc=this_mu, scale=this_sigma))
        
        if this_llh>0:
            print(this_llh, this_mu, this_sigma)
        llh_grid[ind_mu, ind_sigma] = this_llh
        
        llh_bag += (np.e**this_llh)*mu_step*sigma_step
        
        
print(f'{llh_bag:.3f}')
# -

f1, ax1 = plt.subplots()
cf = plt.pcolor(llh_grid)
f1.colorbar(cf, ax=ax1)

max(llh_grid.ravel()), min(llh_grid.ravel())

sps.norm.logpdf(my_mu, loc=my_mu, scale=1e-12)


