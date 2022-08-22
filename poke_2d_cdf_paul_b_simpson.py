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

# # blah
# Trying to understand the conditions in the paper

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.stats as sps

# ### Poking around using a 2D Gaussian

no_pts = 100
x = np.linspace(0, 1, no_pts)
y = np.linspace(0, 1, no_pts)
x_grid, y_grid = np.meshgrid(x, y)
pos = np.dstack((x_grid, y_grid))

mu = np.asarray([.5, .5])
sigma = 0.02
cov = np.identity(len(mu))*sigma

dist = sps.multivariate_normal(mu, cov)
zz_pdf = dist.pdf(pos)
zz_cdf = dist.cdf(pos)


def plot_stuff(xx, yy, pdf, cdf, pdf_title, cdf_title):
    f1 = plt.figure(figsize=(13, 7))
    ax1 = f1.add_subplot(121)
    cs = ax1.contourf(xx, yy, pdf, cmap=cm.viridis)
    plt.xlabel('x')
    plt.ylabel('y')
    ax1.axis('equal')
    plt.title(pdf_title)
    plt.colorbar(cs)
    
    ax1 = f1.add_subplot(122)
    cs = ax1.contourf(xx, yy, cdf, cmap=cm.viridis)
    plt.xlabel('x')
    plt.ylabel('y')
    ax1.axis('equal')
    plt.title(cdf_title)
    plt.colorbar(cs)

    plt.tight_layout()


plot_stuff(x_grid, y_grid, zz_pdf, zz_cdf, '2d gauss pdf', '2d gauss cdf')


# ### Poking around with his function

def pauls_f(x, y, a):
    z = a*y*x**2 + (2-a)*x*y**2
    return z/2


gg_0 = pauls_f(x_grid, y_grid, 0.) 
gg_1 = pauls_f(x_grid, y_grid, 1.) 

plot_stuff(x_grid, y_grid, gg_0, gg_1, 'a=0.', 'a=1.')


