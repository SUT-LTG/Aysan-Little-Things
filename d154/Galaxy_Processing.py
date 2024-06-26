# importing Libraries

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from photutils.background import Background2D, MedianBackground
from astropy.stats import SigmaClip
from scipy.optimize import curve_fit
from scipy.signal import convolve2d

# background Reduction

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
    bkg_estimator = MedianBackground()
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


def open_image(name):
    #opens fits file
    filename = get_pkg_data_filename(name)
    return fits.getdata(filename, ext = 0)


# magnitude Map of Galaxies

def initial_mag_map(data, t, scale, constant):
    mag_map = data.copy()
    mag_map = mag_map / (t * (scale ** 2))
    mag_map = -2.5 * np.log(mag_map) / np.log(10) + constant
    return mag_map


def mag_correction(mag_map, mag_map_BV, correction_parameters, opacity, image_offset_V = (0, 0)):
    # offsetting images and correcting
    Dx, Dy = image_offset_V[0], image_offset_V[1]
    x1, y1 = len(mag_map_BV[0]), len(mag_map_BV)
    x2, y2 = len(mag_map[0]), len(mag_map)
    x3, y3 = x1 - Dx, y1 - Dy
    if x2 < x3:
        x3 = x2
    if y2 < y3:
        y3 = y2
    mag_map_corrected = np.zeros([y3, x3])
    for y in range(y3):
        for x in range(x3):
            mag_map_corrected[y][x] = mag_map[y][x] - mag_map_BV[y + Dy][x + Dx] * correction_parameters[2] - correction_parameters[0] - correction_parameters[1] * opacity
    return mag_map_corrected

    
def galaxy_cut(image, x_range, y_range):
    # cuts a section of an image
    galaxy_image = np.zeros((y_range[1] - y_range[0], x_range[1] - x_range[0]))
    for j in range(y_range[1] - y_range[0]):
        for i in range(x_range[1] - x_range[0]):
            galaxy_image[j][i] = image[y_range[0] + j][x_range[0] + i]
    return galaxy_image
    

# convolution

def fast_convolve(image, box_radius, sigma):
    # convolves an image with a guassian
    box = np.fromfunction(lambda i , j : np.exp(-((i-box_radius)**2 + (j-box_radius//2)**2)/2/sigma**2),(box_radius*2+1,box_radius*2+1))
    box = box/np.sum(box)
    return convolve2d(image,box,'valid')


# contour

def contour(data, name, lim_list, non_conv_data, plot = True, plot_int = True, path = False):
    # this function plots the contours given an intensity limit criteria.
    # if path is true and only one limit is given, it instead gives the coordinate for the vertices of a contour.
    intensity_list = []
    if plot:
        plt.figure()
    for k in lim_list:
        total_intensity = 0
        Z = np.zeros((len(data), len(data[0])))
        for i in range(len(data[0])):
            for j in range(len(data)):
                if data[j][i] > k:
                    Z[j][i] = 1
                    total_intensity += data[j][i]
        intensity_list.append(total_intensity)
        if plot:
            plt.contour(Z, 1)
    if plot:
        plt.title("contour of " + name)
        plt.imshow(non_conv_data, cmap = "gray")
        plt.colorbar()
        plt.xlabel("x_coordinate")
        plt.ylabel("y_coordinate")
    
    if plot_int:
        plt.figure()
        plt.plot(lim_list, intensity_list)
        plt.title("intensity againts intensity limit")
        plt.xlabel("intensity limit")
        plt.ylabel("intensity")
    
    if path and len(lim_list) == 1:
        cs = plt.contour(Z, 1)
        coord = cs.collections[0].get_paths()
        return coord
    
    return intensity_list


def line(x, a, b):
    return a * x + b


# ellipse fitting

def half_radius_ellipse(blurred_image, image, lim_list, name, line_limit = 0, plot = True):
    # this function gives the ellipse parameters for the half light radius in a galaxy.
    intensity_list = contour(blurred_image, name, lim_list, image, plot_int = False)
    popt, pcov = curve_fit(line, lim_list[line_limit:], intensity_list[line_limit:])
    if plot:
        plt.figure()
        plt.title("integrated intensity againts intensity detection limit for " + name)
        plt.xlabel("intensity limit")
        plt.ylabel("integrated intensity")
        plt.plot(lim_list, intensity_list, label = "intensity")
        plt.plot(lim_list, intensity_list - line(lim_list, popt[0], popt[1]), label = "intensity profile difference")
        plt.plot(lim_list, line(lim_list, 0, 0.1 * np.max(intensity_list)), label = "integrated intensity criteria")
    CI = intensity_list - line(lim_list, popt[0], popt[1])
    
    for i in range(len(lim_list)):
        if CI[i]  < 0.1 * np.max(intensity_list):
            radius_index = i
            break
        
    glx_intensity = intensity_list[radius_index]
    for k in range(len(lim_list)):
        if intensity_list[k] / glx_intensity < 0.5:
            half_light_lim = lim_list[k]
            break
    
    path = contour(blurred_image, name, [half_light_lim], image, plot = True, plot_int = False, path = True)
    contour_coordination = path[0]._vertices
    x = contour_coordination.transpose()[0]
    y = contour_coordination.transpose()[1]

    A = np.stack([x**2, x * y, y**2, x, y]).T
    b = np.ones_like(x)
    w = np.linalg.lstsq(A, b)[0].squeeze()

    xlin = np.linspace(0, len(image), len(image))
    ylin = np.linspace(0, len(image), len(image))
    X, Y = np.meshgrid(xlin, ylin)

    Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 + w[3]*X + w[4]*Y
    A, B, C, D, E = w[0], w[1], w[2], w[3], w[4]
    
    if plot:
        contour(blurred_image, name, [half_light_lim], image, plot = True, plot_int = False, path = False)
        plt.contour(X, Y, Z, [1], cmap = 'summer_r')
    
    F = (B ** 2 - 4 * A * C)
    x_center = (2 * C * D - B * E) / F
    y_center = (2 * A * E - B * D) / F
    theta = 1 / 2 * np.arctan(B / (A - C))
    a = (-1 / F) * np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E - F) * ((A + C) + np.sqrt((A - C) ** 2 + B ** 2)))
    b = (-1 / F) * np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E - F) * ((A + C) - np.sqrt((A - C) ** 2 + B ** 2)))
    
    return a, b, theta * 180 / np.pi, x_center, y_center

def mag_25_radius(blurred_image, image, name, plot = True):
    path = contour(blurred_image, name, [25], image, plot = True, plot_int = False, path = True)
    contour_coordination = path[0]._vertices
    x = contour_coordination.transpose()[0]
    y = contour_coordination.transpose()[1]

    A = np.stack([x**2, x * y, y**2, x, y]).T
    b = np.ones_like(x)
    w = np.linalg.lstsq(A, b)[0].squeeze()

    xlin = np.linspace(0, len(image), len(image))
    ylin = np.linspace(0, len(image), len(image))
    X, Y = np.meshgrid(xlin, ylin)

    Z = w[0]*X**2 + w[1]*X*Y + w[2]*Y**2 + w[3]*X + w[4]*Y
    A, B, C, D, E = w[0], w[1], w[2], w[3], w[4]
    
    if plot:
        contour(blurred_image, name, [25], image, plot = True, plot_int = False, path = False)
        plt.contour(X, Y, Z, [1], cmap = 'summer_r')
    
    F = (B ** 2 - 4 * A * C)
    x_center = (2 * C * D - B * E) / F
    y_center = (2 * A * E - B * D) / F
    theta = 1 / 2 * np.arctan(B / (A - C))
    a = (-1 / F) * np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E - F) * ((A + C) + np.sqrt((A - C) ** 2 + B ** 2)))
    b = (-1 / F) * np.sqrt(2 * (A * E ** 2 + C * D ** 2 - B * D * E - F) * ((A + C) - np.sqrt((A - C) ** 2 + B ** 2)))
    
    return a, b, theta * 180 / np.pi, x_center, y_center
    
