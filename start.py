from glob import glob
from astropy.io import fits
import numpy as np
#from numba import njit, prange
import sys
from astropy.convolution import Gaussian2DKernel
from photutils.segmentation import detect_sources
from photutils.segmentation import detect_threshold
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.convolution import convolve
import math
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

def get_data(hdu : fits.hdu.hdulist.HDUList):
    return hdu[0].data*hdu[0].header["GAIN"]+hdu[0].header["BZERO"]

#@njit(fastmath=True)
def is_in_bounds(i, j, data):
    result = (i>=0) and (i<np.shape(data)[0]) 
    result = result and (j>=0) and (j<np.shape(data)[1]) 
    return result

#@njit(fastmath=True)
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

#@njit(fastmath=True, parallel = True)
def transform(dx, dy, data):
    new_data = np.zeros(data.shape)
    n = data.shape[0]
    m = data.shape[1]
    for i in range(n):
        for j in range(m):
            new_data[i, j] = I(i+dx,j+dy, data)
    return new_data


sorted_files = get_sub_images()
flats = sorted_files["flat"]
biases = sorted_files["bias"]
# суммируем и усредняем байасы 

biases_data = []
for _ in biases:
    with fits.open(_) as hdu_list:
        biases_data.append(get_data(hdu_list))
general_bias_data = np.mean(biases_data, axis =0)

# суммируем и усредняем флэты
flats_data = []
for _ in flats:
     with fits.open(_) as hdu_list:
        if hdu_list[0].header["FILTERS"] == filter:
            flats_data.append(get_data(hdu_list))
general_flat_data = np.median(flats_data, axis =0)
general_flat_data /= np.mean(general_flat_data)

hdu = fits.PrimaryHDU(data=general_flat_data)
hdu_list = fits.HDUList([hdu])
hdu_list.writeto(filter+'/result_flat'+'.fts', overwrite=True)

hdu = fits.PrimaryHDU(data=general_bias_data)
hdu_list = fits.HDUList([hdu])
hdu_list.writeto(filter+'/result_bias'+'.fts', overwrite=True)
#получаем итоговоые изображения обьекта

count = 1
expositions = []
exp_times=[]
sec_per_pixel = 0
for object_file in glob(filter_path+"/"+fits_p):
    with fits.open(object_file) as object_hdu:
        data = get_data(object_hdu)
        sec_per_pixel = float(object_hdu[0].header["IMSCALE"].split('x')[0])
        exp_times.append(object_hdu[0].header["EXPTIME"])
    #print(filter)
    print(data.shape)
    data -= general_bias_data
    data /= general_flat_data
    expositions.append(data)
    count+=1

#print(sec_per_pixel)
(x_n, y_n) = np.shape(expositions[0])
from photutils.segmentation import deblend_sources
import matplotlib.pyplot as plt
indexes = []
for i in range(len(expositions)):

    expositions[i] *= np.max(exp_times)/exp_times[i]
    hdu = fits.PrimaryHDU(data=expositions[i])
    hdu_list = fits.HDUList([hdu])
    hdu_list.writeto(filter+'/result'+str(i)+'.fts', overwrite=True)

    star_region = expositions[i][0: 300, y_n-300:y_n-1]
    
    indexes.append(np.unravel_index(np.argmax(star_region), star_region.shape)+np.array([0, y_n-300]))

sigma = 3.0*gaussian_fwhm_to_sigma
kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
kernel.normalize()
#складываем изображения
shift = indexes[1]-indexes[0]
#print(shift)
new_data = transform(*shift, expositions[1])
#hdu = fits.PrimaryHDU(data=new_data)
#hdu_list = fits.HDUList([hdu])
#hdu_list.writeto(filter+'/result_shifted2.fts', overwrite=True)

result_data = expositions[0]+new_data
result_data = result_data[100:x_n-100, 100:y_n-100]

threshold = detect_threshold(result_data, nsigma = 2.)
convolved_data = convolve(result_data, kernel)
segm = detect_sources(convolved_data, threshold, connectivity=8, npixels = 5)
segm_deblend = deblend_sources(convolved_data, segm, npixels=5, nlevels=32, contrast=0.001)
mask =np.sign(segm_deblend.data)
mask =np.ones(mask.shape) - mask
mask = (mask == 0)
#mask = np.ones(mask.shape)
#mask[mask.shape[0]//2-50:mask.shape[0]//2+50, mask.shape[1]//2-50: mask.shape[1]//2+50]= int(0)
#mask =np.sign(segm_deblend.data)
fig, ax = plt.subplots()
ax.imshow(segm_deblend.data)
fig.set_figwidth(segm_deblend.shape[0]/100)    #  ширина и
fig.set_figheight(segm_deblend.shape[1]/100)    #  высота "Figure"
fig.savefig(filter+'/segm_sum.png')

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = MedianBackground()
bkg = Background2D(result_data, (50, 50), filter_size=(3, 3),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator, mask = mask)


#print(np.var(bkg.background)/len(bkg.background))
result_data = result_data - bkg.background
#print(result_data[50,50])
hdu = fits.PrimaryHDU(data=bkg.background)
hdu_list = fits.HDUList([hdu])
hdu_list.writeto(filter+'/noise.fts', overwrite=True)

threshold = detect_threshold(result_data, nsigma = 10.)
sigma = 3.0*gaussian_fwhm_to_sigma
kernel = Gaussian2DKernel(sigma, x_size = 3, y_size = 3)
kernel.normalize()
convolved_data = convolve(result_data, kernel)
segm = detect_sources(convolved_data, threshold, connectivity=8, npixels = 5)
#from photutils.segmentation import deblend_sources
segm_deblend = deblend_sources(convolved_data, segm, npixels=5, nlevels=32, contrast=0.001).data


fig, ax = plt.subplots()
ax.imshow(segm_deblend)
fig.set_figwidth(segm_deblend.shape[0]/100)    #  ширина и
fig.set_figheight(segm_deblend.shape[1]/100)    #  высота "Figure"
fig.savefig(filter+'/result_segm_sum.png')


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

#строим срезы
#grad = np.array([result_data[center[0]+1, center[1]]-result_data[center[0]-1, center[1]], result_data[center[0], center[1]+1]-result_data[center[0], center[1]-1]])

#grad /= math.sqrt(np.sum(grad*grad))
grad = np.array([math.cos(math.pi/4), -math.sin(math.pi/4)])
#print(grad, math.sqrt(np.sum(grad*grad)))
antigrad = np.array([grad[1], -grad[0]])

titles = {}
titles["max"] = grad
titles["min"] = antigrad
for title in titles.keys():
    direction = titles[title]
    t = -100
    x = []
    y = []
    while is_in_bounds(*(center+t*direction), result_data) and t<=100:
        x.append(t*sec_per_pixel)
        y.append(I(*(center+t*direction), result_data))
        t+=0.5
    
    y=-2.5*np.log(y)
    y = y-26.74+26*2.5*math.log(3.828)
    plt.figure(figsize=(20, 10))
    plt.plot(x, y)
    plt.xlabel(r'$x$ in seconds')
    plt.ylabel(r'$I(x)$')
    plt.savefig(filter+'/slice_'+title+ '.png')    
    

hdu = fits.PrimaryHDU(data=result_data)
hdu_list = fits.HDUList([hdu])
hdu_list.writeto(filter+'/result_sum.fts', overwrite=True)

#3 посчитать цвет (берем одну апертуру и одну экспозицую для всех фильтрах, BVR = RGB )
