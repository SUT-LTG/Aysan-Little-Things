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
galaxy_name = 'DDO69'
filter = "V"
#entered by hand ------------------------------------------------------------------------------------------------------------------------------------------------
center_v = [430 , 500]
center_b = [430 , 504]
center_u = [380 , 530]
center = center_v
box_size = 200
window_size = (40, 40)
B_exp = 2400
U_exp = 1800
V_exp = 1200
exp = V_exp
airmass_values=[1.22, 1.04, 1.1]
mU_values = [5.251, 0.459, -0.121, 0]
mV_values = [3.009, 0.2183592, 0.0415, 0.]
mB_values = [3.434, 0.3398903,-0.0157, 0.]
m_values
# import files----------------------------------------------------------------------------------------------------------------------------------------------------
light_file = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\Data\DDO69\d69_v.fits")
light = light_file[0].data

starless_file = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\starless fits\background first\starless_DDO69_v_background_subtracted.fit")
starless = starless_file[0].data

# create background------------------------------------------------------------------------------------------------------------------------------------------------
sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = SExtractorBackground()
bkg = Background2D(light, (250 , 300) , filter_size=(3, 3),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
data = light - bkg.background
min_value = int(np.min(data))
Acceptable_i = []
for i in range(0,-min_value):
    newdata = data + i
    num_negative_values = np.sum(newdata < 0)
    ratio = num_negative_values / newdata.size
    if ratio < 0.005:
        Acceptable_i.append(i)
    else:
        i = i+1
i = np.min(Acceptable_i)

#background correction--------------------------------------------------------------------------------------------------------------------------------------------

corrected_light = light - bkg.background + i
print(i)

plt.imshow(bkg.background, origin='lower', cmap='Greys_r',
           interpolation='nearest')
cbar = plt.colorbar()
cbar.set_label('pixel value')
plt.title("DDO69 background (V) , boxsize = 250x300")
plt.show()
norm = ImageNormalize(vmin=0., stretch=LogStretch())
image1 = light
image2 = corrected_light
norm = ImageNormalize(vmin=0., stretch=LogStretch())
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Display the images
im1 = axs[0].imshow(image1, origin = "lower" , aspect='auto' , norm = norm)
im2 = axs[1].imshow(image2, origin = "lower" , aspect='auto' , norm = norm)
axs[0].set_title('Light')
axs[1].set_title('Background subtracted')
fig.suptitle('DDO69 V-flter (scale = log)')
# Remove the space between the two images
plt.subplots_adjust(wspace=0.08)
# Create an axis for the colorbar on the right side of axs[1].
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
# Create a colorbar
cbar = fig.colorbar(im1, cax=cax)
cbar.set_label('log(pixel value)')
# Show the plot
plt.show()

#export background corrected:

output_filename = 'DDO69_V_background_subtracted.fits'
# Create a PrimaryHDU (header/data unit) from your array
primary_hdu = fits.PrimaryHDU(corrected_light)
# Create an HDUList and append the PrimaryHDU
hdul = fits.HDUList([primary_hdu])
# Write the HDUList to the FITS file
hdul.writeto(output_filename, overwrite=True)

#starless----------------------------------------------------------------------------------------------------------------------------------------------------------

image3 = starless
image2 = corrected_light
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# Display the images
im3 = axs[1].imshow(image3, origin = "lower" , aspect='auto' , norm = norm)
im2 = axs[0].imshow(image2, origin = "lower" , aspect='auto' , norm = norm)
axs[0].set_title('Background subtracted')
axs[1].set_title('starless image')
fig.suptitle('DDO69 V-filter (scale = log)')
# Remove the space between the two images
plt.subplots_adjust(wspace=0.08)
# Create an axis for the colorbar on the right side of axs[1].
divider = make_axes_locatable(axs[1])
cax = divider.append_axes("right", size="5%", pad=0.1)
# Create a colorbar
cbar = fig.colorbar(im1, cax=cax)
cbar.set_label('log(pixel value)')
# Show the plot
plt.show()


pixel_scale = 1.134 #(arcsec)
starless[starless <= 0] = 1
#starless---------------------------------------------------------------------------------------------------------------------------------------------------------
norm = ImageNormalize(vmin=0., stretch=LogStretch())
plt.imshow(starless, norm = norm , origin = "lower")
plt.title("starless %s %s"%(galaxy_name,filter))
plt.colorbar()
plt.show()

# Slice the array
galaxy_box = starless[center[1]-box_size : center[1]+box_size , center[0]-box_size : center[0]+box_size]
plt.imshow(galaxy_box, origin = "lower")
plt.title("galaxy box %s %s"%(galaxy_name,filter))
plt.colorbar()
plt.show()

#center of mass ---------------------------------------------------------------------------------------------------------------------------------------------------
norm = ImageNormalize(vmin=0., stretch=LogStretch())
image_center_of_mass = ndimage.center_of_mass(galaxy_box)
#coordinates
x, y = image_center_of_mass[1], image_center_of_mass[0]
# Create a figure and axes
fig, ax = plt.subplots()
# Display  image
ax.imshow(galaxy_box,norm=norm,origin="lower")
plt.title("center of mass %s %s"%(galaxy_name,filter))
# Mark the point with a red circle
circle = plt.Circle((x, y), radius=3, fill=False, color='red')
ax.add_patch(circle)
plt.show() 

#starless magnitude table------------------------------------------------------------------------------------------------------------------------------------------
flux = (starless/(exp*((pixel_scale)**2)))
magnitude_table = -2.5 * np.log10(flux) + 25
b1=mB_values[0]
b2=airmass_values[1]*mB_values[1]
b3=mB_values[2]*(inst_magB0[i][j]-inst_magV0[i][j])
b4=airmass_values[1]*mB_values[3]*(inst_magB0[i][j]-inst_magV0[i][j])
inst_mag_corrected_B[i][j]=inst_magB0[i][j]-b1-b2-b3-b4
plt.imshow(magnitude_table, origin = "lower")
plt.title("magnitude table %s %s"%(galaxy_name,filter))
plt.colorbar()
plt.show()

#smoothing (moving average)------------------------------------------------------------------------------------------------------------------------------------------
smoothed = gaussian_filter(galaxy_box, sigma = 5)
plt.imshow(smoothed, origin = "lower")
plt.title("smoothed galaxy box %s %s"%(galaxy_name,filter))
plt.colorbar()
plt.show()

# Draw contour lines at the levels specified------------------------------------------------------------------------------------------------------------------------
plt.figure()
plt.imshow(smoothed, origin='lower', cmap='gray', alpha=1)
CS = plt.contour(smoothed , levels=[30,40,50,60,70,80])
plt.clabel(CS, inline=True, fontsize=7)
plt.title("contour lines %s %s"%(galaxy_name,filter))
plt.colorbar(CS)
# Show the plot
plt.show()

contour_lines = CS.collections
ellipses = []
# Iterate through each contour line
for contour_line in contour_lines:
    # Get the path data for each contour line
    path_data = contour_line.get_paths()[0].vertices
    # Extract x and y coordinates
    x_coords, y_coords = path_data[:, 0], path_data[:, 1]
    # Create an array of points from x_coords and y_coords
    points = np.column_stack((x_coords, y_coords))
    # Fit an ellipse to the points
    ellipse = 
    ellipses.append(ellipse)



# Create a figure
plt.figure()

# Show the image
plt.imshow(smoothed, origin='lower', cmap='gray', alpha=1)

# Draw contour lines
CS = plt.contour(smoothed, levels=[30, 40, 50, 60, 70, 80])
plt.clabel(CS, inline=True, fontsize=7)
plt.title("contour lines %s %s" % (galaxy_name, filter))
plt.colorbar(CS)

contour_lines = CS.collections
for contour_line in contour_lines:
    path_data = contour_line.get_paths()[0].vertices
    x_coords, y_coords = path_data[:, 0], path_data[:, 1]
    # Now you have the x and y coordinates for each contour line

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Create a figure
plt.figure()

# Show the image
plt.imshow(smoothed, origin='lower', cmap='gray', alpha=1)

ellipse_x = []
ellipse_y = []

# Draw ellipses for each contour line
contour_lines = CS.collections
for contour_line in contour_lines:
    path_data = contour_line.get_paths()[0].vertices
    x_coords, y_coords = path_data[:, 0], path_data[:, 1]
    ellipse_x.append(x_coords)
    ellipse_y.append(y_coords)

points = list(zip(ellipse_x, ellipse_y))

import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt

def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = linalg.eig(np.dot(linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a

def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])

def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))

def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])

def find_ellipse(x, y):
    xmean = x.mean()
    ymean = y.mean()
    x = x - xmean
    y = y - ymean
    a = fitEllipse(x, y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    return center, phi, axes

# Example data points
fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
a_points = np.array(points)
x = a_points[:, 0]
y = a_points[:, 1]
axs[0].plot(x, y)
center, phi, axes = find_ellipse(x, y)
print("Center:", center)
print("Angle of rotation:", phi)
print("Axes lengths:", axes)
axs[1].plot(x, y)
plt.show()




