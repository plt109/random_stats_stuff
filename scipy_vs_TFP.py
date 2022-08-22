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
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


def Binom(x,n,p):
    return tfp.distributions.Binomial(total_count=n, probs=p).prob(x).numpy()


# +
n = 1000
p = 0.01

num_mc = int(1e7)
# -

mc = stats.binom.rvs(n=n, p=p, size=num_mc)

mc_min = -0.5
mc_max = 25.5
mc_bins = np.arange(mc_min, mc_max)

centers = []
probs = []
for left, right in zip(mc_bins, mc_bins[1:]):
    centers.append((left + right) / 2.)
    probs.append(Binom((left + right) / 2., n, p))

weights = np.ones_like(mc) / len(mc)
plt.hist(mc, bins=mc_bins, weights=weights, label='scipy MC')
plt.plot(centers, probs, '.', label='TFP')
plt.legend()
plt.title('Binomial, n=1000, p=0.01')
plt.yscale('log')





# +
tf_rvs=tfp.distributions.Binomial(total_count=10000,probs=tf.cast(0.001, dtype=fd.float_type())).sample(sample_shape=(10000000)).numpy()
sps_rvs=stats.binom.rvs(10000,0.001, size=10000000)

plt.hist(sps_rvs,bins=31,label='scipy',range=[0,30],density=True)
plt.hist(tf_rvs,bins=31,alpha=0.5, label='tensorflow_probability',range=[0,30],density=True)
# -


