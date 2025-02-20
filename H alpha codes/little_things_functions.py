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
from skimage.transform import AffineTransform, warp

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
    source_fixed_2 = images[source_indices[1]].byteswap().newbyteorder('N')
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
        s3 = m_values[i][2]*(magnitude_tables[1] - magnitude_tables[2])
        s4 = airmass_values[1]*m_values[i][3]*(magnitude_tables[1] - magnitude_tables[2])
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
def contour_lines(box,sigma,level):
    #smoothing (gaussian convolution)
        smoothed = gaussian_filter(box, sigma)
        # Draw contour lines
        # Plot the smoothed array
        plt.imshow(smoothed, alpha = 0.75 , origin = "lower")
        # Add contour line on top of the smoothed array
        CS = plt.contour(smoothed, level)

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


def correct_reddening_B(image, E_B_V):
    A_B = (1+3.086)*E_B_V
    reddening_corrected_image = image + A_B
    return reddening_corrected_image


import os
import numpy as np
from astropy.io import fits
from skimage.transform import AffineTransform, warp
import matplotlib.pyplot as plt
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import LogStretch
from scipy.ndimage import gaussian_filter
from matplotlib.colors import Normalize
import matplotlib.cm as cm

def V_and_Halpha_alignment(galaxy_name, image1_path, image2_path, star_coords_V, star_coords_H, output_dir, alpha1=0.5, alpha2=0.5):
    norm = ImageNormalize(vmin=0., stretch=LogStretch())

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Function to open FITS file and handle endian issues
    def open_fits_file(file_path):
        with fits.open(file_path) as hdul:
            data = hdul[0].data
            if data.dtype.byteorder == '>':
                data = data.byteswap().newbyteorder()
            header = hdul[0].header
        return data, header

    # Open FITS files and get data
    V_image, header1 = open_fits_file(image1_path)
    H_image, header2 = open_fits_file(image2_path)

    # Calculate the affine transformation matrix using the star coordinates
    tform = AffineTransform()
    tform.estimate(star_coords_H, star_coords_V)

    # Apply the affine transformation to the H_image
    aligned_H_image = warp(H_image, inverse_map=tform.inverse, output_shape=V_image.shape)

    # Save the aligned image to a new FITS file
    aligned_H_image_path = os.path.join(output_dir, 'aligned_d101ha.fits')
    fits.writeto(aligned_H_image_path, aligned_H_image.astype(np.float32), overwrite=True)

    # Create a mask for the overlapping region
    overlap_mask = (aligned_H_image > 0) & (V_image > 0)

    # Find the bounding box of the overlapping region
    coords = np.argwhere(overlap_mask)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)

    # Crop the images to the bounding box
    aligned_H_image_cropped = aligned_H_image[x_min:x_max+1, y_min:y_max+1]
    V_image_cropped = V_image[x_min:x_max+1, y_min:y_max+1]

    # Update the headers to comply with FITS standard
    header1['OBSERVAT'] = 'Observatory 1'
    header2['OBSERVAT'] = 'Observatory 2'

    # Save the cropped images as new FITS files
    V_image_cropped_path = os.path.join(output_dir, f'cropped_{galaxy_name}_V.fits')
    aligned_H_image_cropped_path = os.path.join(output_dir, f'cropped_{galaxy_name}_H.fits')
    fits.writeto(V_image_cropped_path, V_image_cropped.astype(np.float32), header1, overwrite=True)
    fits.writeto(aligned_H_image_cropped_path, aligned_H_image_cropped.astype(np.float32), header2, overwrite=True)

    # Plot the cropped images on top of each other
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(V_image_cropped, cmap='Blues', alpha=alpha1, norm=norm, origin="lower")
    ax.imshow(aligned_H_image_cropped, cmap='Reds', alpha=alpha2, norm=norm, origin="lower")

    plt.title(f"Overlay and Contour lines for {galaxy_name}")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend([f"{galaxy_name} V filter", f'{galaxy_name} H-alpha filter'])
    plt.show()
    return V_image_cropped, aligned_H_image_cropped

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, ListedColormap
from scipy.ndimage import gaussian_filter
from astropy.visualization import AsinhStretch, ImageNormalize, PercentileInterval

def enhance_contrast(image, stretch=AsinhStretch(), interval=PercentileInterval(99.5)):
    norm = ImageNormalize(image, interval=interval, stretch=stretch)
    return norm

def plot_contours_V_and_Halpha(galaxy_name, V_image, H_image, folder_path, alpha=1, sigma=5, contour_levels=[4, 6, 8, 10]):
    # Enhance contrast
    norm_V = enhance_contrast(V_image)

    # Define colors for contours
    contour_colors = ['black', 'blue', 'magenta', 'red']

    # Plot the V filter image with enhanced contrast
    fig, ax = plt.subplots(figsize=(10, 10))
    im1 = ax.imshow(V_image, cmap='gray', alpha=alpha, norm=norm_V, origin="lower")

    # Create the Gaussian smoothed image for contours
    smoothed = gaussian_filter(H_image, sigma=sigma)

    # Add contours of the smoothed image with specific colors and line styles
    for i, level in enumerate(contour_levels):
        ax.contour(smoothed, levels=[level], colors=contour_colors[i], linestyles='solid', linewidths=2)

    # Create a custom color bar with lines
    cmap = ListedColormap(contour_colors)
    norm = Normalize(vmin=min(contour_levels), vmax=max(contour_levels))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the color bar with labels
    cbar_contour = fig.colorbar(sm, ax=ax, ticks=contour_levels, fraction=0.046, pad=0.04)
    

    cbar_contour.set_label('Contour Levels', labelpad=-30, fontsize=12)

    # Add colorbar for the V filter image
    cbar1 = fig.colorbar(im1, ax=ax, fraction=0.046, pad=0.05)
    cbar1.set_label("V filter, Asinh", labelpad=-40, fontsize=10)

    # Set larger font sizes for titles and labels
    ax.set_title(f"{galaxy_name} ", fontsize=14)
    ax.set_xlabel('X Coordinate (pixels)', fontsize=12)
    ax.set_ylabel('Y Coordinate (pixels)', fontsize=12)

    # Save the figure to the specified path
    output_path = f"{folder_path}\\{galaxy_name} H-alpha regions.png"
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, fftfreq
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

def compute_fft(image):
    fft_image = fft2(image)
    fft_image_shifted = fftshift(fft_image)
    return fft_image_shifted

def plot_image_and_fft(image, fft_image_shifted, name, filter, pixelscale):
    plt.figure(figsize=(10, 5))
    
    # Original image with axes in arcseconds
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray', origin='lower')
    plt.colorbar()
    plt.xlabel('X (arcseconds)')
    plt.ylabel('Y (arcseconds)')
    plt.title(f'Original Image {name}, {filter} filter')
    
    # FFT of the image with axes in inverse arcseconds
    plt.subplot(1, 2, 2)
    fft_shape = fft_image_shifted.shape
    freq_x = fftshift(fftfreq(fft_shape[1], d=pixelscale))
    freq_y = fftshift(fftfreq(fft_shape[0], d=pixelscale))
    extent = [freq_x.min(), freq_x.max(), freq_y.min(), freq_y.max()]
    plt.imshow(np.log(np.abs(fft_image_shifted)), cmap='gray', origin='lower', extent=extent)
    plt.colorbar()
    plt.xlabel('Frequency X (arcseconds⁻¹)')
    plt.ylabel('Frequency Y (arcseconds⁻¹)')
    plt.title(f'FFT of Image {name}, {filter} filter')
    
    plt.show()

def get_line_profile(image, start_point, angle, length):
    x0, y0 = start_point
    angle_rad = np.deg2rad(angle)
    profile = []

    for r in range(length):
        x = int(x0 + r * np.cos(angle_rad))
        y = int(y0 + r * np.sin(angle_rad))

        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            profile.append(np.abs(image[y, x]))  # Use the magnitude
        else:
            profile.append(np.nan)

    return profile
'''
def resolution(image, name, filter, num_theta, pixelscale):
    fft_image_shifted = compute_fft(image)
    plot_image_and_fft(image, fft_image_shifted, name, filter, pixelscale=pixelscale)
    length = int(np.min(fft_image_shifted.shape) / 2)
    center = np.array(fft_image_shifted.shape) // 2
    profiles = []
    
    for theta in np.linspace(0, 2 * np.pi, num=num_theta, endpoint=False):
        profile = get_line_profile(fft_image_shifted, center, np.rad2deg(theta), length)
        profiles.append(profile)
    
    mean_profile = np.mean(profiles, axis=0)
    
    return profiles, mean_profile, length
'''
import numpy as np
import matplotlib.pyplot as plt

def resolution(image, name, filter, num_theta, pixelscale):
    fft_image_shifted = compute_fft(image)
    plot_image_and_fft(image, fft_image_shifted, name, filter, pixelscale=pixelscale)
    length = int(np.min(fft_image_shifted.shape) / 2)
    center = np.array(fft_image_shifted.shape) // 2

    # Create a grid of coordinates
    y, x = np.ogrid[:fft_image_shifted.shape[0], :fft_image_shifted.shape[1]]
    y = y - center[1]
    x = x - center[0]
    
    # Calculate the radial distance of each point from the center
    distance = np.sqrt(x**2 + y**2)

    # Initialize an array to store the mean values for each radius
    radial_means = np.zeros(length)
    
    # Loop over each radius and compute the mean value
    for r in range(length):
        mask = (distance >= r) & (distance < r + 1)
        radial_means[r] = np.mean(np.abs(fft_image_shifted[mask]))

    return radial_means, length


def plot_profiles_on_image(image, profiles, num_theta):
    fft_image_shifted = compute_fft(image)
    fft_image_shifted = np.abs(fft_image_shifted)
    norm = ImageNormalize(vmin=0., stretch=LogStretch())
    plt.imshow(fft_image_shifted, cmap='gray', origin='lower', norm = norm)
    center = np.array(fft_image_shifted.shape) // 2
    length = profiles.shape[1]

    for i, theta in enumerate(np.linspace(0, 2 * np.pi, num=num_theta, endpoint=False)):
        x_offset = np.cos(theta)
        y_offset = np.sin(theta)

        x = [center[1] + r * x_offset for r in range(length)]
        y = [center[0] + r * y_offset for r in range(length)]

        plt.plot(x, y, label=f'Theta {np.rad2deg(theta):.1f}°', )

    plt.title('Profiles on Image')
    plt.show()

def calculate_pixelscale(star_coords_V, star_coords_H, V_pixelscale):
    def euclidean_distance(point1, point2):
        return np.linalg.norm(point1 - point2)

    # Calculate distances between stars in vertical setting
    dist_V_1_2 = euclidean_distance(star_coords_V[0], star_coords_V[1])
    dist_V_2_3 = euclidean_distance(star_coords_V[1], star_coords_V[2])
    dist_V_1_3 = euclidean_distance(star_coords_V[0], star_coords_V[2])
    
    # Calculate distances between stars in horizontal setting
    dist_H_1_2 = euclidean_distance(star_coords_H[0], star_coords_H[1])
    dist_H_2_3 = euclidean_distance(star_coords_H[1], star_coords_H[2])
    dist_H_1_3 = euclidean_distance(star_coords_H[0], star_coords_H[2])
    
    # Calculate ratios
    ratio_1_2 = dist_V_1_2/dist_H_1_2 
    ratio_2_3 = dist_V_2_3/dist_H_2_3
    ratio_1_3 = dist_V_1_3/dist_H_1_3
    
    # Calculate mean ratio
    mean_ratio = np.mean([ratio_1_2, ratio_2_3, ratio_1_3])
    
    return mean_ratio*V_pixelscale


def forward_fill(arr):
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr

def resolution_profiles(H_image, V_image, galaxy_name, pixelscale, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Compute profiles for H-alpha image
    profiles_H, mean_profile_H, length_H = resolution(H_image, galaxy_name, "H", num_theta=360, pixelscale=pixelscale)
    image_shape_H = H_image.shape
    x_ax_H = np.linspace(0, 1 / (2 * pixelscale), length_H)
    delta_nu_H = np.round(2 / (image_shape_H[1] * pixelscale), 5)

    # Compute profiles for V filter image
    profiles_V, mean_profile_V, length_V = resolution(V_image, galaxy_name, "V", num_theta=360, pixelscale=pixelscale)
    image_shape_V = V_image.shape
    x_ax_V = np.linspace(0, 1 / (2 * pixelscale), length_V)
    delta_nu_V = np.round(2 / (image_shape_V[1] * pixelscale), 5)

    # Replace NaNs with the last non-NaN value
    mean_profile_H = forward_fill(mean_profile_H)
    mean_profile_V = forward_fill(mean_profile_V)

    # Plot individual profile for H-alpha image
    plt.figure(figsize=(10, 5))
    plt.plot(x_ax_H, mean_profile_H, label=f'H-alpha filter, delta nu: {delta_nu_H} (arcseconds⁻¹)', color='red')
    plt.xlabel('Spatial Frequency (arcseconds⁻¹)')
    plt.ylabel('Mean Profile Intensity')
    plt.title(f'Mean Profile for {galaxy_name} - H-alpha filter')
    plt.legend()
    plt.show()

    # Plot individual profile for V filter image
    plt.figure(figsize=(10, 5))
    plt.plot(x_ax_V, mean_profile_V, label=f'V filter, delta nu: {delta_nu_V} (arcseconds⁻¹)', color='blue')
    plt.xlabel('Spatial Frequency (arcseconds⁻¹)')
    plt.ylabel('Mean Profile Intensity')
    plt.title(f'Mean Profile for {galaxy_name} - V filter')
    plt.legend()
    plt.show()

    # Plot both profiles on the same plot and save to specified folder path
    plt.figure(figsize=(10, 5))
    plt.plot(x_ax_H, mean_profile_H, label=f'H-alpha filter, delta nu: {delta_nu_H} (arcseconds⁻¹)', color='red')
    plt.plot(x_ax_V, mean_profile_V, label=f'V filter, delta nu: {delta_nu_V} (arcseconds⁻¹)', color='blue')
    plt.xlabel('Spatial Frequency (arcseconds⁻¹)')
    plt.ylabel('Mean Profile Intensity')
    plt.title(f'Mean Profile for {galaxy_name}')
    plt.legend()
    output_path = os.path.join(output_dir, f'combined_resolution_profiles_for_{galaxy_name}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show()

    # Normalize the profiles
    norm_mean_profile_H = mean_profile_H / np.max(mean_profile_H)
    norm_mean_profile_V = mean_profile_V / np.max(mean_profile_V)

    # Plot normalized profiles on top of each other
    plt.figure(figsize=(10, 5))
    plt.plot(x_ax_H, norm_mean_profile_H, label='Normalized H-alpha filter', color='red')
    plt.plot(x_ax_V, norm_mean_profile_V, label='Normalized V filter', color='blue')
    plt.xlabel('Spatial Frequency (arcseconds⁻¹)')
    plt.ylabel('Normalized Mean Profile Intensity')
    plt.title(f'Normalized Mean Profiles for {galaxy_name}')
    plt.legend()
    output_path_norm = os.path.join(output_dir, f'normalized_combined_resolution_profiles_for_{galaxy_name}.png')
    plt.savefig(output_path_norm, bbox_inches='tight', dpi=300)
    plt.show()
