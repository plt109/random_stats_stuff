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
import matplotlib.pyplot as plt


def cal_global_corr(cov):
    inv_cov = np.linalg.inv(cov)
    diag_cov = np.diag(cov)
    diag_inv_cov = np.diag(inv_cov)
    return np.sqrt(1-1./(diag_cov*diag_inv_cov))


# ## Sampling time

# +
# totally independent case
mu, sigma = 0, 0.1 # mean and standard deviation
no_pts = 10

x = np.random.normal(mu, sigma, no_pts)
y = np.random.normal(mu, sigma, no_pts)

# +
# but this is not linear dependence, will not be captured
mu, sigma = 0, 0.1 # mean and standard deviation
no_pts = 10

x = np.random.normal(mu, sigma, no_pts)
y = x*x 

# +
# signal
no_pts = 10
mu, sigma = 0, 0.1 # mean and standard deviation

# add awgn
noise_mu, noise_sigma = 0., 0.001

x = np.random.normal(mu, sigma, no_pts)
y = x + np.random.normal(noise_mu, noise_sigma, no_pts)
# -

# ## Stack and compute

X = np.stack((x, y), axis=0)
type(X), np.shape(X)

cov = np.cov(X)
cal_global_corr(cov)

# ## Probing how good this thing is under null
# Like the random variables are indeed totally random but sometimes it seems like this global correlation coefficient still gives a rather large number.

# +
# totally independent case
mu, sigma = 0, 0.1 # mean and standard deviation
no_pts = 1000
no_rv = 12
no_trials = 1000

p_bag = []
for i_trials in range(no_trials):
    X = []
    for i_rv in range(no_rv):
        x = np.random.normal(mu, sigma, no_pts)
        X.append(x)
    X = np.asarray(X)
    
    cov = np.cov(X)
    p_bag.append(cal_global_corr(cov))

p_bag = np.asarray(p_bag)
# -

p_bag

np.shape(cov)

print(p_bag[0,:])

print(f'{p_bag[0,:]}')

# +
n_bins = 50

for i_rv in range(no_rv):
    plt.hist(p_bag[:,i_rv], bins=n_bins, histtype='step', label=f'rv {i_rv}')
plt.xlabel('Global Correlation Coefficient')
plt.ylabel(f'No. trials out of {no_trials} trials')
plt.title(f'{no_rv} random variables, {no_pts} measurements')
plt.legend()

# +
n_bins = 50

plt.figure(figsize=(13, 10), facecolor='w')
#plt.plot(p_bag[:,0], '.', label='variable 0')
#plt.plot(p_bag[:,1], '.', label='variable 1')
plt.plot(p_bag[:,0]-p_bag[:,1], '.', label='diff')
plt.legend()
# -

no_trials - np.sum(p_bag[:,0]==p_bag[:,1]) # number of times they're different

np.shape(p_bag[:,0])

p_bag

p_bag[:,0]

raise



np.diag(cov)

np.diag(inv_cov)

np.diag(cov)*np.diag(inv_cov)

1./(np.diag(cov)*np.diag(inv_cov))

1./3.726



aa = np.diag(cov)*np.diag(inv_cov)

aa[0]==aa[1]

np.sqrt(1-1./(np.diag(cov)*np.diag(inv_cov)))


