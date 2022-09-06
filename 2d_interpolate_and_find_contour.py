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
import scipy.interpolate as spi

import matplotlib.pyplot as plt


# -

def give_z(x, y):
    return (x**2+y**2)**0.5


# +
min_x = -2.
max_x = 2.
num_x = 200
x_arr = np.linspace(min_x, max_x, num_x)

min_y = -2.
max_y = 2.
num_y = 200
y_arr = np.linspace(min_y, max_y, num_y)
# -

np.shape(x_arr), np.shape(x)

# +
x = np.arange(-5.01, 5.01, 0.25) # (n,)
y = np.arange(-5.01, 5.01, 0.25) # (n,)
xx, yy = np.meshgrid(x, y) # (n, n)

# some sinc function
z = np.sin(xx**2+yy**2) # (n, n)
f = spi.interp2d(x, y, z, kind='cubic')

# some basic parabola
z2 = give_z(xx, yy)
f2 = spi.interp2d(x, y, z2, kind='cubic')

# +
z_interpolated = f(x_arr, y_arr)

plt.figure(figsize=(15, 10), facecolor='w')
plt.subplot(121)
plt.pcolor(x, y, z)
plt.axvline(x_arr[0], c='r')
plt.axvline(x_arr[-1], c='r')
plt.axhline(y_arr[0], c='r')
plt.axhline(y_arr[-1], c='r')
plt.title('original x, y, z')

plt.subplot(122)
CL = 90.
plot_level = np.percentile(z_interpolated.ravel(), CL)
plt.pcolor(x_arr, y_arr, z_interpolated)
contour_set = plt.contour(x_arr, y_arr, z_interpolated, levels=[plot_level])
plt.title('interpolated x, y, z')

# +
# Really just plotting contour lines on top of interpolated guy
plt.figure()
plt.pcolor(x_arr, y_arr, z_interpolated)

contour_lines = contour_set.allsegs[0][0]
plt.plot(contour_lines[:,0], contour_lines[:,1], c='w')

plt.title('interpolated x, y, z')
# -

np.shape(contour_set.allsegs)

np.size(contour_set.allsegs)

plt.plot(contour_lines[:,0], contour_lines[:,1], c='k')

# +
'''
x_arr: (200,)
y_arr: (200,)
z_interpolated: (200, 200)
'''
z_interpolated = f2(x_arr, y_arr)

plt.figure(figsize=(15, 10), facecolor='w')
plt.subplot(121)
plt.pcolor(x, y, z2)
plt.axvline(x_arr[0], c='r')
plt.axvline(x_arr[-1], c='r')
plt.axhline(y_arr[0], c='r')
plt.axhline(y_arr[-1], c='r')
plt.title('original x, y, z')

plt.subplot(122)
CL = 90.
plot_level = np.percentile(z_interpolated.ravel(), CL)
plt.pcolor(x_arr, y_arr, z_interpolated)
contour_set = plt.contour(x_arr, y_arr, z_interpolated, levels=[plot_level])
plt.title('interpolated x, y, z')

# +
# Really just plotting contour lines on top of interpolated guy
plt.figure()
plt.pcolor(x_arr, y_arr, z_interpolated)

contour_lines = contour_set.allsegs[0][0]
plt.plot(contour_lines[:,0], contour_lines[:,1], c='w')

plt.title('interpolated x, y, z')
# -


