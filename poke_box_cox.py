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


# ## See how the tranform acts on the uniform distribution

def my_box_cox(data, lamb):
    if lamb==0:
        output = np.log(data)
    else:
        output = (data**lamb-1.)/lamb
    return output


n_pts = 500
uniform_pool = np.random.uniform(0., 1., n_pts)

# ### Positive lambda

# +
n_lamb = 5
min_lamb = 0.
max_lamb = 6.
lamb_bag = np.linspace(min_lamb, max_lamb, n_lamb)

plt.figure(figsize=(13, 10), facecolor='w')
plt.hist(uniform_pool, histtype='step', label='Original')
for loc_lamb in lamb_bag:
    uniform_tranf = my_box_cox(uniform_pool, loc_lamb)
    plt.hist(uniform_tranf, histtype='step', label='lamb = %.2f' % loc_lamb)
plt.legend()
# -

# ### Negative lambda

# +
n_lamb = 5
min_lamb = -6.
max_lamb = 0.
lamb_bag = np.linspace(min_lamb, max_lamb, n_lamb)

plt.figure(figsize=(13, 10), facecolor='w')
plt.hist(uniform_pool, histtype='step', label='Original')
for loc_lamb in lamb_bag:
    uniform_tranf = my_box_cox(uniform_pool, loc_lamb)
    plt.hist(uniform_tranf, histtype='step', label='lamb = %.2f' % loc_lamb)
plt.legend()
# -

this_lamb = -1.
uniform_tranf = my_box_cox(uniform_pool, this_lamb)
plt.hist(uniform_tranf, bins=20, histtype='step')
plt.title('lambda = %.2f' % this_lamb)

this_lamb = -6.
uniform_tranf = my_box_cox(uniform_pool, this_lamb)
plt.hist(uniform_tranf, bins=20, histtype='step')
plt.title('lambda = %.2f' % this_lamb)


