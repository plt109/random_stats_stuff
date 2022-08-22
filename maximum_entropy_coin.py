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

# # Maximum Entropy Principle
# Playing around with actually solving for $\lambda$ for the coin subjected to different constraints

import numpy as np
import scipy.optimize as spo
import math as math

# ## Learning how to use root solver

# +
a = 1
b = 0
c = -1

def func(x):
    return a*(x*x) + b*x + c

def lala(a, b, c):
    return (-b-np.sqrt(b*b-4*a*c))/(2*a), (-b+np.sqrt(b*b-4*a*c))/(2*a)

start_pt = 1
start_pt2 = start_pt+1e-11 # cause somehow need 2 starting points

res = spo.root_scalar(func, x0=start_pt, x1=start_pt2)

analytic_res = lala(a, b, c)
print('Root found = %.4f, Closed form = %.4f, %.4f' % (res.root, analytic_res[0], analytic_res[1]))
print(res)


# -

# ## Case II in notes
# 1 empirical constraint of E[X]=4
#
# $
# \begin{align}
# f(x) &= \frac{1}{Z} e^{-\lambda x}\\
# Z &= \sum_x e^{-\lambda x}
# \end{align}
# $

# Partition function
def z2(lamb):
    return np.sum(np.exp(-1.*lamb*np.linspace(1, 6, 6)))


def f2(lamb):
    x = np.linspace(1, 6, 6)
    f = np.sum(x*np.exp(-1.*lamb*x))/z2(lamb)
    return f


def fun2(lamb):
    observation = 4.
    return f2(lamb)-observation


def h2(lamb):
    x = np.linspace(1, 6, 6)
    p = np.exp(-1.*lamb*x)/z2(lamb)
    h = -1.*np.sum(p*np.log(p))
    return h


z2(lamb)

fun2(lamb)

res2 = spo.root_scalar(fun2, x0=start_pt, x1=start_pt2)
res_lamb2 = res2.root
print('lambda = %.4f, entropy = %.4f' % (res_lamb2, h2(res_lamb2)))
print('\nRoot finding status:\n%s' % res2)


# ## Case III in notes
# 1 empirical constraint of E[X]=6
#
# $
# \begin{align}
# f(x) &= \frac{1}{Z} e^{-\lambda x}\\
# Z &= \sum_x e^{-\lambda x}
# \end{align}
# $

# Cause it's the same model, just with different observation(constraint), so can recycle many things
def fun3(lamb):
    observation = 6.
    return f2(lamb)-observation


res3 = spo.root_scalar(fun3, x0=start_pt, x1=start_pt2)
res_lamb3 = res3.root
print('lambda = %.4f, entropy = %.4f' % (res_lamb3, h2(res_lamb3)))
print('\nRoot finding status:\n%s' % res3)



# ## Case IV in notes
# 2 empirical constraints of E[X]=4, E[X^2]=20
#
# $
# \begin{align}
# f(x) &= \frac{1}{Z} e^{-\lambda_1 x-\lambda_2 x^2}\\
# Z &= \sum_x e^{-\lambda_1 x-\lambda_2 x^2}
# \end{align}
# $

# Partition function
def z4(lamb):
    x = np.linspace(1, 6, 6)
    return np.sum(np.exp(-1.*lamb[0]*x-1.*lamb[1]*x*x))


def f4(lamb):
    x = np.linspace(1, 6, 6)
    return np.exp(-1.*lamb[0]*x-1.*lamb[1]*x*x)/z4(lamb)


def exp4(lamb):
    x = np.linspace(1, 6, 6)
    return np.sum(x*f4(lamb))


def var4(lamb):
    x = np.linspace(1, 6, 6)
    return np.sum(x*x*f4(lamb))


def h4(lamb):
    x = np.linspace(1, 6, 6)
    p = f4(lamb)
    h = -1.*np.sum(p*np.log(p))
    return h


# Finding roots using least squares cause how else you gonna solve 2 non-linear simultaneous equations?
def fun4(lamb):
    x_obs = 4.
    x2_obs = 20.
    return (exp4(lamb)-x_obs)**2+(var4(lamb)-x2_obs)**2


# +
#start_pt = [0.1, 0.1]
n_runs = 20
res_bag = []
lamb_bag = []
h_bag = []
for irun in range(n_runs):
    start_pt = np.random.random(2)
    res4 = spo.minimize(fun4, x0=start_pt)
    
    res_bag.append(res4)
    lamb_bag.append(res4.x)
    h_bag.append(h4(res4.x))

lamb_bag = np.asarray(lamb_bag)
zz = np.argmax(h_bag)
print('lambda = ', lamb_bag[zz])
print('H = %.4f' % np.max(h_bag))
print('\nRoot finding status:\n%s' % res_bag[zz])


# -

# ## Case V in notes
# 2 contradicting empirical constraints of E[X]=4, E[X^2]=1
#
# $
# \begin{align}
# f(x) &= \frac{1}{Z} e^{-\lambda_1 x-\lambda_2 x^2}\\
# Z &= \sum_x e^{-\lambda_1 x-\lambda_2 x^2}
# \end{align}
# $

# Finding roots using least squares cause how else you gonna solve 2 non-linear simultaneous equations?
def fun5(lamb):
    x_obs = 4.
    x2_obs = 1.
    return (exp4(lamb)-x_obs)**2+(var4(lamb)-x2_obs)**2


# +
#start_pt = [0.1, 0.1]
n_runs = 20
res_bag = []
lamb_bag = []
h_bag = []
for irun in range(n_runs):
    start_pt = np.random.random(2)
    res5 = spo.minimize(fun5, x0=start_pt)
    
    res_bag.append(res5)
    lamb_bag.append(res5.x)
    h_bag.append(h4(res5.x))

lamb_bag = np.asarray(lamb_bag)
zz = np.argmax(h_bag)
print('lambda = ', lamb_bag[zz])
print('H = %.4f' % np.max(h_bag))
print('\nRoot finding status:\n%s' % res_bag[zz])
# -

# It's supposed to die when you give 'contradicting empirical constraints', but this is numerical land. Your contradiction must be grotesque enough before it really dies.


