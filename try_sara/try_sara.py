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
from scipy import stats
import scipy.optimize as spo
import matplotlib.pyplot as plt
import seaborn as sns

from LPBkg.detc import BestM, dhatL2
# -

# ### Loading data

# +
data_path = '/home/peaelle42/analyses/random_stats_stuff/try_sara/data/'
ca=np.loadtxt(f'{data_path}/source_free.txt',dtype=str)[:,1:].astype(float) # calibration
bk=np.loadtxt(f'{data_path}/background.txt',dtype=str)[:,1:].astype(float) # bg
si=np.loadtxt(f'{data_path}/signal.txt',dtype=str)[:,1:].astype(float) # signal

cal=ca.reshape(1,len(ca))[0]
bkg=bk.reshape(1,len(bk))[0]
sig=si.reshape(1,len(si))[0]
# -

# ### Visualising data

print(len(cal), len(bkg), len(sig))
plt.figure(figsize=(13, 10))
sns.distplot(cal, kde=True, norm_hist= True, label='Calibration')
sns.distplot(bkg, kde=True, norm_hist= True, label='Background')
sns.distplot(sig, kde=True, norm_hist= True, label='Signal')
plt.legend()

np.shape(ca)


# ### Fitting background

#This is -loglikelihhod:
def mll(d, y=bkg): 
    #return -np.sum(np.log(stats.pareto.pdf(y,1,d)/stats.pareto.cdf(35,1,d)))
    return -np.sum(np.log(stats.pareto.pdf(y,d, scale=1.)/stats.pareto.cdf(35,d, scale=1.)))


res = spo.minimize(mll, 1.3)
bkg_shape_bf = res.x[0]


def powerlaw(y, d=bkg_shape_bf):
    # where the value 1.3590681192057597 is calculated by minimizing function mil 
    # with respect to the parameter ''d'' using ''Brent optimization''
    #return stats.pareto.pdf(y,1.3590681192057597,scale=1)/stats.pareto.cdf(35,1.3590681192057597,scale=1)
    return stats.pareto.pdf(y,d,scale=1)/stats.pareto.cdf(35,d,scale=1.)


# ### Checking fits

fig, ax = plt.subplots(figsize=(14, 7))
ax=sns.distplot(cal, kde=False, norm_hist= True)
uu=np.arange(min(cal),max(cal),0.05)
ax.plot(uu, powerlaw(uu, bkg_shape_bf),color='red',linestyle='dashed')
ax.set_xbound(lower=1,upper=15)
ax.set_ybound(lower=0, upper=0.8)

BestM(data=cal,g=powerlaw, Mmax=20,rg=[1,35])

for this_chunk in range(1):
    print(this_chunk)

type(this_chunk)


