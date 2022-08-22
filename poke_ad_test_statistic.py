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

# # Poke test statistic of Anderson Darling test
# Just to see when we get negative values

#import flamedisx as fd
import numpy as np
import scipy.stats as sps
import matplotlib.pyplot as plt

# +
no_events = 2000
no_mc = 500

field = 's1'

stats_bag = []
p_bag = []
for imc in range(no_mc):
    '''
    aa = fd.ERSource(p_el_e0=6.).simulate(no_events)
    bb = fd.ERSource(p_el_e0=6.).simulate(no_events)
    gg = sps.anderson_ksamp([aa[field].values, bb[field].values])
    '''
    
    aa = np.random.normal(loc=0.0, scale=1.0, size=no_events)
    bb = np.random.normal(loc=0.0, scale=1.0, size=no_events)
    gg = sps.anderson_ksamp([aa, bb])
    
    stats_bag.append(gg.statistic)
    p_bag.append(gg.significance_level)
    
# -

plt.hist(stats_bag, histtype='step')
plt.xlabel('Test Statistic')

# +
plt.figure(figsize=(13, 10))
plt.plot(p_bag, stats_bag, '.')
plt.xlabel('p value')
plt.ylabel('test statistic')

plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# -

# Of all the MC trial that hit the max p-value, most(but not all) of them had negative values for the test statistic.
#
# For the 1-sample case, it is not possible to get a negative value by construction (like how $A^2$ is defined in https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test), but I did not think further about the 2-sample (or in this case it's called k-sample) case.
#
# I think it does not matter even if it is negative, as long as scipy.stats does not do anything funky with the value of the test statistic.


