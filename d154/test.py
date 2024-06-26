import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from photutils.background import Background2D, SExtractorBackground
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit
from scipy.signal import convolve2d

def Histogram_image(Master_data, image_name, bins, xlim, info = False):
    #plots a histogram of image
    if info:
        print("Image median: ", np.median(Master_data))
        print("Image mean", np.mean(Master_data))
        print("Image standard deviation", np.std(Master_data))
    
    plt.figure()
    counts, bins = np.histogram(Master_data, bins)
    plt.stairs(counts, bins)
    plt.title("Histogram of " + image_name)
    plt.xlabel("counts")
    plt.ylabel("number of pixels")
    plt.xlim(xlim)
    plt.show()

def image_show(image, title, log = False):
    #plots an image
    plt.figure()
    plt.title(title)
    if log:
        plt.imshow(np.log(image), cmap = "gray")
        plt.colorbar()
    else:
        plt.imshow(image, cmap = "gray")
        plt.colorbar()
    plt.xlabel("x_coordinate")
    plt.ylabel("y_coordinate")

def background_romval(data, mesh, filter_size, name, save = False, info = False):
    #removes the background of an image, sigmaclips the negative pixels and plots the images
    
    sigma_clip = SigmaClip(sigma = 3.0)
    bkg_estimator = SExtractorBackground()
    bkg = Background2D(data, mesh, filter_size = filter_size, sigma_clip = sigma_clip, bkg_estimator = bkg_estimator)
    bkg_data = bkg.background
    background_removed_image = data - bkg_data
    
    median, sigma = np.median(background_removed_image), np.std(background_removed_image)
    Histogram_image(background_removed_image, "backgroundless_image", int((np.max(background_removed_image) - np.min(background_removed_image)) // 10), [median - 0.5 * sigma, median + 0.5 * sigma], info)
    
    non_subject_pixels = background_removed_image.copy()
    for j in range(len(background_removed_image)):
        for i in range(len(background_removed_image[0])):
            if abs(background_removed_image[j][i] - median) > sigma:
                non_subject_pixels[j][i] = np.nan
    sigma_bkg = np.nanstd(non_subject_pixels)

        
    for j in range(len(background_removed_image)):
        for i in range(len(background_removed_image[0])):
            if background_removed_image[j][i] < median - sigma_bkg and background_removed_image[j][i] < 0:
                background_removed_image[j][i] = 1
    min_image = np.min(background_removed_image)
    print(min_image)
    if min_image < 0:
        background_removed_image = background_removed_image - min_image + 1
    Histogram_image(background_removed_image, "positive_backgroundless_image", int((np.max(background_removed_image) - np.min(background_removed_image)) // 10), [median - 0.5 * sigma, median + 0.5 * sigma], info)    
    
    image_show(data, "original_data" + name, log = True)
    image_show(background_removed_image, "backgroundless_data" + name, log = True)
    
    if save:
        hdu = fits.PrimaryHDU(background_removed_image)
        hdu.writeto("backgroundless_" + name + ".fits", overwrite = True)
    return background_removed_image


light_file_v = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\Data\DDO 154\d154v.fits")
light_v = light_file_v[0].data
box_v = (170 , 170)
filter_size=(3, 3)
background_romval(light_v , box_v , filter_size , 'ddo154_v')
import astroalign as aa
