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
import scipy.optimize as spo

import matplotlib.pyplot as plt

# +
# standardizing number of samples to draw from h0
REF_N = 100_000 

# Pick a distribution to test (ER, NR band is defo not just a Gaussian)
def theta_to_distribution(theta):
    return sps.norm(loc=theta[0], scale=theta[1])

# Pick GOF
def cal_t(test_sample, h0):
    ref_samples = h0.rvs(REF_N)
    t, p = sps.ks_2samp(test_sample, ref_samples, alternative='two-sided')
    return t, p


# -

# Function for fitting sample for model parameters
def fit_sample(sample):
    # MLE for Gaussian has closed form
    mu_hat = np.mean(sample)
    sigma_hat = np.var(sample)**.5
    return [mu_hat, sigma_hat]


# Standard cycle
def draw_fit_cal_t(h0, sample_size):
    sample = h0.rvs(sample_size) # draw sample
    theta_hat = fit_sample(sample) # fit sample
    h0_refitted = theta_to_distribution(theta_hat) # new h0 from fitting
    t, p = cal_t(sample, h0_refitted) # compute test statistic (and p-value assuming asymptotics)
    return sample, theta_hat, t, p


# Standard cycle but bootstrapping instead of drawing from distribution
def bootstrap_fit_cal_t(input_sample, sample_size):    
    # draw sample
    ind_sel = np.random.randint(low=0, high=sample_size, size=(sample_size,))
    sample = input_sample[ind_sel]

    theta_hat = fit_sample(sample) # fit sample
    h0_refitted = theta_to_distribution(theta_hat) # new h0 from fitting
    t, p = cal_t(sample, h0_refitted) # compute test statistic (and p-value assuming asymptotics)
    return sample, theta_hat, t, p


# Current cycle without fitting
def draw_cal_t(h0, sample_size):
    sample = h0.rvs(sample_size) # draw sample
    t, p = cal_t(sample, h0) # compute test statistic (and p-value assuming asymptotics)
    return sample, t, p


# Current cycle without fitting
def bootstrap_cal_t(input_sample, h0, sample_size):
    # draw sample
    ind_sel = np.random.randint(low=0, high=sample_size, size=(sample_size,))
    sample = input_sample[ind_sel]
    
    t, p = cal_t(sample, h0) # compute test statistic (and p-value assuming asymptotics)
    return sample, t, p


# +
# Defining true distribution that only nature knows
true_mu = 3.
true_sigma = 0.1

true_theta = [true_mu, true_sigma]
true_distribution = theta_to_distribution(true_theta)

# +
# Just checking true distribution
nbins=50

true_check = true_distribution.rvs(REF_N)
x_arr = np.linspace(min(true_check), max(true_check), nbins)
y_arr = true_distribution.pdf(x_arr)

plt.hist(true_check, histtype='step', bins=nbins, density=True)
plt.plot(x_arr, y_arr, label='True pdf')
plt.axvline(true_mu, c='r', lw=1, label=f'True $\mu$ = {true_mu}')
plt.title(f'$\mu$ = {true_mu}, $\sigma$ = {true_sigma}')
plt.legend()
plt.show()

# +
# Calibration set
cali_size = 1000 #2000

cali_sample, cali_theta_hat, cali_t, _ = draw_fit_cal_t(true_distribution, cali_size)
cali_h0 = theta_to_distribution(cali_theta_hat)

# +
# Check how it looks
x_arr = np.linspace(min(cali_sample), max(cali_sample), nbins)

plt.hist(cali_sample, histtype='step', bins=nbins, density=True, label='Calibration set')
plt.plot(x_arr, true_distribution.pdf(x_arr), label=f'True pdf: $\mu$ = {true_mu:.2f}, $\sigma$ = {true_sigma:.2f}')
plt.plot(x_arr, cali_h0.pdf(x_arr), label=f'Cali H0: $\mu$ = {cali_theta_hat[0]:.2f}, $\sigma$ = {cali_theta_hat[1]:.2f}')

plt.title(f'$\mu$ = {true_mu}, $\sigma$ = {true_sigma}')
plt.legend()
plt.show()

# +
n_toys = 1000
    
# always draw from true h0 (proper procedure of simulating full experiment)
true_h0_t_bag = np.ones(n_toys)
for ind_toy in range(n_toys):
    this_sample, this_theta_hat, this_t, _ = draw_fit_cal_t(true_distribution, cali_size)
    true_h0_t_bag[ind_toy] = this_t
    
# bootstrapping from cali set and refitting
bootstrap_h0_t_bag = np.ones(n_toys)
for ind_toy in range(n_toys):
    this_sample, this_theta_hat, this_t, _ = bootstrap_fit_cal_t(cali_sample, cali_size)
    bootstrap_h0_t_bag[ind_toy] = this_t

# bootstrapping from cali set, no refitting
bootstrap_cheater_h0_t_bag = np.ones(n_toys)
for ind_toy in range(n_toys):
    _, this_t, _ = bootstrap_cal_t(cali_sample, cali_h0, cali_size)
    bootstrap_cheater_h0_t_bag[ind_toy] = this_t
    
# procedure in reality
refitted_h0_t_bag = np.ones(n_toys)

this_distribution = cali_h0 # start with h0 from fitting calibration sample
for ind_toy in range(n_toys):
    this_sample, this_theta_hat, this_t, _ = draw_fit_cal_t(this_distribution, cali_size)
    refitted_h0_t_bag[ind_toy] = this_t
    this_distribution = theta_to_distribution(this_theta_hat)

# always draw from cali h0 and compute t without refitting
cheater_h0_t_bag = np.ones(n_toys)
for ind_toy in range(n_toys):
    _, this_t, _ = draw_cal_t(cali_h0, cali_size)
    cheater_h0_t_bag[ind_toy] = this_t

# +
nbins=50

lb = min(min(cheater_h0_t_bag), min(true_h0_t_bag), min(refitted_h0_t_bag), min(bootstrap_h0_t_bag), min(bootstrap_cheater_h0_t_bag))
ub = max(max(cheater_h0_t_bag), max(true_h0_t_bag), max(refitted_h0_t_bag), max(bootstrap_h0_t_bag), max(bootstrap_cheater_h0_t_bag))
bin_edges = np.linspace(lb, ub, nbins)

plt.figure(figsize=(13, 10), facecolor='w')
plt.subplot(211)
plt.hist(cheater_h0_t_bag, label='No refitting', \
         histtype='step', bins=bin_edges, density=True)
plt.hist(true_h0_t_bag, label='Draw from first guy from true h0', \
         histtype='step', bins=bin_edges, density=True)
plt.hist(refitted_h0_t_bag, label='Draw from first guy from cali h0', \
         histtype='step', bins=bin_edges, density=True)
plt.hist(bootstrap_h0_t_bag, label='Bootstrap from cali sample and refit', \
         histtype='step', bins=bin_edges, density=True)
plt.hist(bootstrap_cheater_h0_t_bag, label='Bootstrap from cali sample and no refit', \
         histtype='step', bins=bin_edges, density=True)

plt.axvline(cali_t, c='k', lw=2, label='Test statistic from fitting calibration set')
plt.xlabel('T')
plt.ylabel('PDF')
plt.title(f'{n_toys} toys')
plt.legend()

plt.subplot(212)
plt.hist(cheater_h0_t_bag, label='No refitting', \
         histtype='step', bins=bin_edges, density=True, cumulative=-1)
plt.hist(true_h0_t_bag,  label='Draw from first guy from true h0', \
         histtype='step', bins=bin_edges, density=True, cumulative=-1)
plt.hist(refitted_h0_t_bag, label='Draw from first guy from cali h0', \
         histtype='step', bins=bin_edges, density=True, cumulative=-1)
plt.hist(bootstrap_h0_t_bag, label='Bootstrap from cali sample and refit', \
         histtype='step', bins=bin_edges, density=True, cumulative=-1)
plt.hist(bootstrap_cheater_h0_t_bag, label='Bootstrap from cali sample and no refit', \
         histtype='step', bins=bin_edges, density=True, cumulative=-1)

plt.axvline(cali_t, c='k', lw=2, label='Test statistic from fitting calibration set')

plt.minorticks_on()
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.4)
plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.1)
plt.xlabel('T')
plt.ylabel('p-value')
plt.legend()

plt.show()

# +
print(100-sps.percentileofscore(cheater_h0_t_bag, cali_t)) # no refitting
print(100-sps.percentileofscore(true_h0_t_bag, cali_t)) # draw first guy from h0
print(100-sps.percentileofscore(refitted_h0_t_bag, cali_t)) # draw first guy from cali h0


# -


