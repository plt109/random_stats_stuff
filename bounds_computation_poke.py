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
k = 17
p = 0.6
prob_content = 0.95

n = k
bag = 0.
while bag<prob_content:
    print(f'before adding {n}: {bag:.3f}')
    bag += sps.binom.pmf(k, n, p)
    print(f'after adding {n}: {bag:.3f} \n***')
    n += 1
print(f'Should add until {n-1}')
# -



# ## blah

max_n = 50
n_arr = np.arange(max_n)
aa = sps.binom.pmf(k, n_arr, p)

# mle for N
# see https://math.stackexchange.com/questions/2628304/maximum-likelihood-estimator-for-n-in-binomial-with-known-p
mle = int(np.floor(k/p))

mle

dist_from_mle = abs(n_arr-mle)
zzz = np.argsort(dist_from_mle)

# +
n_sorted = n_arr[zzz]
prob_sorted = aa[zzz]
prob_cum = np.cumsum(prob_sorted)

is_enough = prob_cum>prob_content
# -

jackpot_ind = np.argmax(prob_cum>prob_content) # python quirk. argmax stops at first true condition.

n_sel = n_sorted[:jackpot_ind+1]
n_min = min(n_sel)
n_max = max(n_sel)
n_min, n_max, n_max-n_min+1, jackpot_ind+1

plt.figure(figsize=(13, 7))
plt.plot(prob_sorted, label='individual pmf')
plt.plot(prob_cum, label='accumulated prob content')
plt.plot(is_enough, '.-', label='is enough')
plt.axhline(prob_content, color='r', lw=1, label=f'target prob content: {prob_content}')
plt.legend()

# +


plt.figure(figsize=(13, 10))
plt.plot(n_arr, aa, '.-')
plt.axvline(mle, color='r', lw=1, label=f'MLE for N for binomial distribution: {mle}')
plt.axvline(n_min, color='r', lw=1, ls='-.')
plt.axvline(n_max, color='r', lw=1, ls='-.')
plt.xlabel('N')
plt.ylabel('PMF')
plt.legend()
plt.title(f'Prob success = {p}, observed {k}')

plt.minorticks_on()
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
# -



# ## stack exchange bayes bound code

from functools import reduce

# +
nmax = 200

def factorial(n):
    if n == 0:
        return 1
    return reduce(lambda a,b : a*b, range(1,n+1), 1)

def ncr(n,r):
    return factorial(n) / (factorial(r) * factorial(n-r))

def binomProbability(n, k, p):
    p1 = ncr(n,k)
    p2 = p**k
    p3 = (1-p)**(n-k)
    return p1*p2*p3

def posterior( n, k, p ):
    def p_k_given_n( n, k ):
        return binomProbability(n, k, p)
    def p_n( n ):
        return 1./nmax
    def p_k( k ):
        return sum( [ p_n(nd)*p_k_given_n(nd,k) for nd in range(k,nmax) ] )
    return (p_k_given_n(n,k) * p_n(n)) / p_k(k)

observed_k   = k
p_n_given_k  = [ posterior( n, observed_k, p ) for n in range(0,nmax) ]
cp_n_given_k = np.cumsum(p_n_given_k)
for n in range(0,nmax):
    print(n, p_n_given_k[n], cp_n_given_k[n])
# -








