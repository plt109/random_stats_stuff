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

import pdb as pdb

# +
# standardizing number of samples to draw from h0
REF_N = 100_000 

# Pick a distribution to test (ER, NR band is defo not just a Gaussian)
def theta_to_distribution(theta):
    return sps.norm(loc=theta[0], scale=theta[1])

# Pick GOF

# 2-sample KS test
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


# Compute p-value from test statistic distribution
def cal_p_from_toys(t, t_distribution):
    return (100.-sps.percentileofscore(t_distribution, t))/100.


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
    
# draw from each refitted cali h0 (procedure in reality)
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

# -



test = true_distribution.rvs(REF_N)

aa, bb = np.histogram(test, bins=10, density=True)

aa

# +
distribution_bag = {'not_refitted': cheater_h0_t_bag,
                    'true_refitted': true_h0_t_bag,
                    'refitted': refitted_h0_t_bag,
                    'bootstrap_sample_refitted': bootstrap_h0_t_bag,
                    'bootstrap_sample_not_refit': bootstrap_cheater_h0_t_bag
}

legend_bag = {'not_refitted': 'Draw from same cali h0, no refitting',
                    'true_refitted': 'Draw from true h0, refitted',
                    'refitted': 'Draw from different cali h0, refitted',
                    'bootstrap_sample_refitted': 'Bootstrap from cali sample, refitted',
                    'bootstrap_sample_not_refit': 'Bootstrap from cali sample, not refitted'
}
# -

pval_bag = dict.fromkeys(distribution_bag, None)
for aa in distribution_bag.keys():
    pval_bag[aa] = cal_p_from_toys(cali_t, distribution_bag[aa])

pval_bag

# +
nbins=50

lb = min([min(distribution_bag[aa]) for aa in distribution_bag.keys()])
ub = max([max(distribution_bag[aa]) for aa in distribution_bag.keys()])
bin_edges = np.linspace(lb, ub, nbins)

plt.figure(figsize=(13, 10), facecolor='w')
plt.subplot(211)

for ind_aa, aa in enumerate(distribution_bag.keys()):
    plt.hist(distribution_bag[aa], label=f'{legend_bag[aa]}: {pval_bag[aa]:.2f}', \
             bins=bin_edges, density=True, \
             color=f'C{ind_aa}', histtype='step')

plt.axvline(cali_t, c='k', lw=2, label='Test statistic from fitting calibration set')
plt.xlabel('T')
plt.ylabel('PDF')
plt.title(f'{n_toys} toys')
plt.legend()

plt.subplot(212)

for ind_aa, aa in enumerate(distribution_bag.keys()):
    plt.hist(distribution_bag[aa], label=f'{legend_bag[aa]}: {pval_bag[aa]:.2f}', \
             bins=bin_edges, density=True, cumulative=-1, \
             color=f'C{ind_aa}', histtype='step')

plt.axvline(cali_t, c='k', lw=2, label='Test statistic from fitting calibration set')

plt.minorticks_on()
plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.4)
plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.1)
plt.xlabel('T')
plt.ylabel('p-value')
plt.legend()

plt.show()
# -

raise







# ## Knut's alternative method of getting ECDF
# but honestly, just toy MC many many and then `percentilescoreof` it

import scipy.interpolate as spi

aa = sps.norm().rvs(100) #just some data 
#np.linspace(0., 1., 20)

datapoints = aa.copy()
cdfval_fcn = spi.interp1d(sorted(datapoints), np.linspace(0,1,len(datapoints)), bounds_error=False, fill_value=(0.,1.))

T = 0.6
print(sps.percentileofscore(aa, T))
print(cdfval_fcn(T))

aa = np.append(aa, [0.1, 0.23])
cdfval_fcn2 = spi.interp1d(sorted(aa), np.linspace(0,1,len(aa)), bounds_error=False, fill_value=(0.,1.))

print(sps.percentileofscore(aa, T))
print(cdfval_fcn(T))
print(cdfval_fcn2(T))

# +
aa = np.random.random(50_000)
cdfval_fcn = spi.interp1d(sorted(aa), np.linspace(0,1,len(aa)), bounds_error=False, fill_value=(0.,1.))

T = 0.6

p = sps.percentileofscore(aa, T)/100.
print(f'before percentileofscore: {p:.3f}, err {p-T:.5f}')
p = cdfval_fcn(T)
print(f'before interpolated: {p:.3f}, err {p-T:.5f}')

print('***')

aa = np.append(aa, [0.1, 0.23])
cdfval_fcn2 = spi.interp1d(sorted(aa), np.linspace(0,1,len(aa)), bounds_error=False, fill_value=(0.,1.))

p = sps.percentileofscore(aa, T)/100.
print(f'after percentileofscore: {p:.3f}, err {p-T:.5f}')
p = cdfval_fcn(T)
print(f'before old interpolated: {p:.3f}, err {p-T:.5f}')
p = cdfval_fcn2(T)
print(f'before new interpolated: {p:.3f}, err {p-T:.5f}')
# -

np.random.random(100)

# number of toys required to get a particular accuracy for p-values
#print(f'{(((2.87e-7)**0.5)/1e-8)**2.:0.6e}')
print(f'{(((1e-3)**0.5)/1e-4)**2.:0.6e}')


1-sps.norm.cdf(3.) # p-value equivalent


