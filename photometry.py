from glob import glob
from astropy.io import fits
import numpy as np
from numba import njit, prange
import sys
from astropy.convolution import Gaussian2DKernel
from photutils.segmentation import detect_sources
from photutils.segmentation import detect_threshold
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import convolve
from photutils.segmentation import deblend_sources
import math
import matplotlib.pyplot as plt
general_path = "NGC7465/stud"
filter = sys.argv[1]
filter_path = general_path+"/"+filter
fits_p = "*.fts"

def get_sub_images():
    files = glob(general_path+"/"+fits_p, recursive=True)
    result = {}
    result["bias"] = []
    result["flat"] = []
    result["dark"] = []
    result["obj"] = []
    for file in files:
        with fits.open(file) as hdu_list:
            if hdu_list[0].header["IMAGETYP"] in result.keys():
                result[hdu_list[0].header["IMAGETYP"]].append(file)
    result["flat"] = result.pop("obj")
    return result

#@njit(fastmath=True)
def is_in_bounds(i, j, data):
    result = (i>=0) and (i<np.shape(data)[0]) 
    result = result and (j>=0) and (j<np.shape(data)[1]) 
    return result

@njit(fastmath=True)
def subf(x,y):
        return (1-y)*(1-x)+x*y

#@njit(fastmath=True)
def I(x, y, data):
    i = math.floor(x)
    j = math.floor(y)
    f_x = x-i
    f_y = y-j
    result = 0
    for k in range(i, i+2):
        for l in range(j, j+2):
            if is_in_bounds(k, l, data):

                result+= subf(k-i,f_x)*subf(l-j, f_y)*data[k, l]
    return result


def do_plot_data(N, x_0, x_1, f):
    result_y = []
    result_x = []
    for i in range(N):
        x = x_0+(x_1-x_0)/(N-1)*i
        x = x
        result_x.append(x*0.357)
        result_y.append(f(x))
    return (result_x, result_y)

#@njit(fastmath=True)
def circle_I(R,x_c, y_c ,N_r, N_phi, data):
    result = 0
    dr = R/N_r
    dphi = 2*math.pi/N_phi
    phi = 0
    for _ in range(N_phi):
        x_0 = math.cos(phi)
        y_0 = math.sin(phi)
        phi += dphi
        r = 0
        for __ in range(1,N_r+1):
            r += dr
            x = r*x_0+x_c
            y = r*y_0+y_c
            result+= r*I(x, y, data)*dr*dphi
    return result

#@njit(fastmath=True)
def dcircle_I(R,x_c, y_c ,N_r, N_phi, data):
    result = 0
    dr = R/N_r
    dphi = 2*math.pi/N_phi
    phi = 0
    for _ in range(N_phi):
        x = R*math.cos(phi)+x_c
        y = R*math.sin(phi)+y_c
        phi += dphi
        result+= R*I(x, y, data)*dphi
    return result


with fits.open("./"+filter+"/result_sum.fts") as data_hdu:
    result_data = data_hdu[0].data


threshold = detect_threshold(result_data, nsigma = 10.)
sigma = 3.0*gaussian_fwhm_to_sigma
kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
kernel.normalize()
convolved_data = convolve(result_data, kernel)
segm = detect_sources(convolved_data, threshold, connectivity=8, npixels = 5)
segm_deblend = deblend_sources(convolved_data, segm, npixels=5, nlevels=32, contrast=0.001).data

index = segm_deblend[segm_deblend.shape[0]//2, segm_deblend.shape[1]//2]
center = np.mean(np.argwhere(segm_deblend == index), axis =0)

threshold = detect_threshold(result_data, nsigma = 200.)
sigma = 3.0*gaussian_fwhm_to_sigma
kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
kernel.normalize()
convolved_data = convolve(result_data, kernel)
segm = detect_sources(convolved_data, threshold, connectivity=8, npixels = 5)

segm_deblend = deblend_sources(convolved_data, segm, npixels=5, nlevels=32, contrast=0.001).data

index = segm_deblend[math.floor(center[0]), math.floor(center[1])]
center = np.mean(np.argwhere(segm_deblend == index), axis =0)
center = [math.floor(center[0]), math.floor(center[1])]

grad = np.array([math.cos(math.pi/4), -math.sin(math.pi/4)])
antigrad = np.array([grad[1], -grad[0]])

(x, y) = do_plot_data(100, 1, 150, lambda r  : circle_I(r, center[0], center[1], 10, 10, result_data))

plt.figure(figsize=(20, 10))
plt.plot(x, y)
plt.xlabel(r'$r$ in seconds')
plt.ylabel(r'$I(x)$')
plt.savefig(filter+'/slice_'+"I(R)"+ '.png')  

(x, y) = do_plot_data(100, 1, 150, lambda r  : dcircle_I(r, center[0], center[1], 10, 100, result_data))

plt.figure(figsize=(20, 10))
plt.plot(x, y)
plt.xlabel(r'$r$ in seconds')
plt.ylabel(r'$I(x)$')
plt.savefig(filter+'/slice_'+"dI(R)"+ '.png')  