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
    list_of_aligned_images = [registered_image_1, registered_image_2, images[source_indices[0]]]
    return  list_of_aligned_images , footprint_1, footprint_2

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
#plt.gray()
galaxy_name = 'DDO154'
box_size = 200
window_size = (40, 40)
U_exp = 1800
B_exp = 1200
V_exp = 600
exp = [U_exp,B_exp,V_exp]

# import files----------------------------------------------------------------------------------------------------------------------------------------------------
light_file_u = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\Data\DDO 154\d154u.fits")
light_u = light_file_u[0].data

light_file_b = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\Data\DDO 154\d154b.fits")
light_b = light_file_b[0].data

light_file_v = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\Data\DDO 154\d154v.fits")
light_v = light_file_v[0].data

box_u = (100 , 100)
box_b = (130 , 130)
box_v = (170 , 170)

lights = [light_u,light_b,light_v]
boxes = [box_u,box_b,box_v]
filters = ["U" , "B" , "V"]

import astroalign as aa
target_fixed = lights[2].byteswap().newbyteorder('N')
source_fixed_U = lights[0].byteswap().newbyteorder('N')
source_fixed_B = lights[1].byteswap().newbyteorder('N')
norm = ImageNormalize(vmin=0., stretch=LogStretch())
registered_image_B, footprint_B = aa.register(source_fixed_B, target_fixed)
registered_image_U, footprint_U = aa.register(source_fixed_U, target_fixed)

plt.imshow(registered_image_U/2 + target_fixed/2, origin = "lower" , alpha = 1 , norm = norm)
plt.colorbar()
plt.show()
plt.imshow(footprint_U,origin="lower")
lights[0] = registered_image_U
lights[1] = registered_image_B


corrected = background_subtraction(lights,3,boxes,(3,3))
print(np.shape(corrected))

