from astropy.io import fits
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D,SExtractorBackground
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import cv2
import astroalign as aa

def open_fits(path):
    fitsfile = fits.open(path)
    file = fitsfile[0].data
    return file 

#example: 
#open_fits(r"C:/Users\AYSAN\Desktop/project/Galaxy/Code\DDO69_V_background_subtracted.fits")

def log_scale_plot(image_data, plot_title, colorbar_title):
    norm = ImageNormalize(vmin=0., stretch=LogStretch())
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    im = ax.imshow(image_data, origin = "lower" , norm = norm)
    ax.set_title(plot_title)
    cbar = fig.colorbar(im)
    cbar.set_label(colorbar_title)
    plt.show()

#example:
#log_scale_plot(open_fits(r"C:/Users\AYSAN\Desktop/project/Galaxy/Code\DDO69_V_background_subtracted.fits"), "title" , "cbar title")

def log_scale_plot_2_images(image_data_1, image_data_2, image_1_title, image_2_title, plot_title, colorbar_title, wspace):
    norm = ImageNormalize(vmin=0., stretch=LogStretch())
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # Display the images
    im1 = axs[0].imshow(image_data_1, origin = "lower" , aspect='auto' , norm = norm)
    im2 = axs[1].imshow(image_data_2, origin = "lower" , aspect='auto' , norm = norm)
    axs[0].set_title(image_1_title)
    axs[1].set_title(image_2_title)
    fig.suptitle(plot_title)
    # Remove the space between the two images
    plt.subplots_adjust(wspace=wspace)
    # Create an axis for the colorbar on the right side of axs[1].
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    # Create a colorbar
    cbar = fig.colorbar(im1, cax=cax)
    cbar.set_label(colorbar_title)
    # Show the plot
    plt.show()

#example:
#log_scale_plot_2_images(open_fits(r"C:/Users\AYSAN\Desktop/project/Galaxy/Code\DDO69_V_background_subtracted.fits"), open_fits(r"C:/Users\AYSAN\Desktop/project/Galaxy/Code\DDO69_V_background_subtracted.fits"),"1", "2","all","cbar",0.1)

def align(images, source_indices , target_indice):
    target_fixed = images[target_indice].byteswap().newbyteorder('N')
    source_fixed_1 = images[source_indices[0]].byteswap().newbyteorder('N')
    source_fixed_2 = images[source_indices[0]].byteswap().newbyteorder('N')
    registered_image_1, footprint_1 = aa.register(source_fixed_1, target_fixed)
    registered_image_2,  footprint_2 = aa.register(source_fixed_2, target_fixed)
    list_of_aligned_images = [registered_image_1, registered_image_2, images[source_indices[2]]]
    return  list_of_aligned_images

#example:
#little_things_functions.align(lights, [0,1], 2)

def background_subtraction(light_images,sigma,boxes_list,filter_size):
    background_corrected_lights = []
    for j in range(0 , len(light_images)): 
    # create background------------------------------------------------------------------------------------------------------------------------------------------------
        sigma_clip = SigmaClip(sigma=sigma)
        bkg_estimator = SExtractorBackground()
        bkg = Background2D(light_images[j], boxes_list[j] , filter_size=filter_size,
        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        light_minus_bkg = light_images[j] - bkg.background
        min_value = int(np.min(light_minus_bkg))
        Acceptable_i = []
        for i in range(0,-min_value):
            newdata = light_minus_bkg + i
            num_negative_values = np.sum(newdata < 0)
            ratio = num_negative_values / newdata.size
            if ratio < 0.01:
                Acceptable_i.append(i)
            else:
                i = i+1
        min_i = np.min(Acceptable_i)
        corrected_light = light_minus_bkg + min_i
        corrected_light[corrected_light <= 0] = 1
        
        background_corrected_lights.append(corrected_light)
    return background_corrected_lights
#example:
#corrected = ltf.background_subtraction(lights,3,boxes,(3,3))

def get_boxes(images,center,box_size):
    # Slice the array
    box_size = int(box_size/2)
    image_boxes = []
    for i in range(0 , len(images)):
        box = images[i][int(center[1]) - box_size : int(center[1]) + box_size, int(center[0]) - box_size : int(center[0]) + box_size]
        image_boxes.append(box)
    return image_boxes
#example:
#galaxy_box = ltf.get_boxes(starless,[500,450],400)

def mag_table_correction(images, airmass_values, m_values,pixel_scale, exposures):
    #first step of mag correction (turning each pixel into a magnitude/arcsec value)
    magnitude_tables=[]
    for i in range(0,len(images)): 
        image = images[i]
        flux = (image/(exposures[i]*((pixel_scale)**2)))
        magnitude_table = (-2.5 * np.log10(flux) + 25) 
        magnitude_tables.append(magnitude_table)
 
    corrected_magnitude_tables = []
    for i in range(0,len(magnitude_tables)):
        s1 = m_values[i][0]
        s2 = airmass_values[1]*m_values[i][1]
        s3=m_values[i][2]*(magnitude_tables[1] - magnitude_tables[2])
        s4=airmass_values[1]*m_values[i][3]*(magnitude_tables[1] - magnitude_tables[2])
        corrected_magnitude_table = magnitude_tables[i] - s1 - s2 - s3 - s4
        corrected_magnitude_tables.append(corrected_magnitude_table)
      
    return corrected_magnitude_tables

#example:
#ltf.mag_table_correction(starless,airmass_values,m_values,pixel_scale)

def contour_lines_coordinates(box,sigma,level):
    #smoothing (gaussian convolution)
        smoothed = gaussian_filter(box, sigma)
        # Draw contour lines
        # Plot the smoothed array
        plt.imshow(smoothed, alpha = 0.75 , origin = "lower")
        # Add contour line on top of the smoothed array
        CS = plt.contour(smoothed, level)
        dat0 = CS.allsegs[0][0]
        x_coord = dat0[:, 0]
        y_coord = dat0[:, 1]
        return x_coord,y_coord

def ellipse(x, xc, yc, a, b, theta):
    return ((x[0] - xc) * np.cos(theta) + (x[1] - yc) * np.sin(theta))**2 / a**2 + ((x[0] - xc) * np.sin(theta) - (x[1] - yc) * np.cos(theta))**2 / b**2 - 1

def find_ellipse(image_box, center_of_mass_x, center_of_mass_y, x_points, y_points, ):
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import curve_fit
    from matplotlib.patches import Ellipse

    initial_guess = [center_of_mass_x, center_of_mass_y, (max(x_points) - min(x_points)) / 2 , (max(y_points) - min(y_points)) / 2 , 0]
    popt, pcov = curve_fit(ellipse, (x_points, y_points), np.zeros_like(x_points), p0=initial_guess)
    stdv=np.sqrt(np.diag(pcov))
    stdvx=stdv[0]
    stdvy=stdv[1]
    stdva=stdv[2]
    stdvb=stdv[3]
    stdvpa=stdv[4]

    xc, yc, a, b, theta = popt

    curve = ellipse(x_points,popt[0],popt[1],popt[2],popt[3],popt[4])

    # Assuming you have already defined xc, yc, a, b, and theta

    xc, yc, a, b, theta = popt
    print(popt)
    # Create a figure and axis
    plt.figure()
    ax = plt.gca()

    # Display the other image
    ax.imshow(image_box, cmap='gray' , origin = "lower")

    # Create the ellipse
    ellipse = Ellipse(xy=(xc, yc), width=2*a, height=2*b, angle=np.degrees(theta), edgecolor='r', facecolor='none', linewidth=2)

    # Add the ellipse to the axis
    ax.add_patch(ellipse)

    # Set axis limits (adjust as needed)
    ax.set_xlim(0,image_box.shape[1])
    ax.set_ylim(0,image_box.shape[0])  # Reverse y-axis for imshow
    plt.title("ellipse for mag = 25, DDO154 V")
    # Show the plot
    plt.show()


