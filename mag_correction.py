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
from scipy.optimize import curve_fit

plt.gray()
galaxy_name = 'DDO69'
filter = ["U" , "B"  , "V"]
center_v = [430 , 500]
center_b = [430 , 500]
center_u = [380 , 530]
centers = [center_u,center_b,center_v]
box_size = 200
B_exp = 2400
U_exp = 1800
V_exp = 1200
exp = [U_exp,B_exp,V_exp]
airmass_values=[1.22, 1.04, 1.1]
mU_values = [5.251, 0.459    , -0.121, 0.]
mV_values = [3.009, 0.2183592, 0.0415, 0.]
mB_values = [3.434, 0.3398903,-0.0157, 0.]
m_values = [mU_values,mB_values,mV_values]

pixel_scale = 1.134 #(arcsec)

starless_file_u = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\starless fits\background first\starless_DDO69_u_background_subtracted.fit")
starless_u = starless_file_u[0].data
#print(np.shape(starless_u))
starless_file_b = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\starless fits\background first\starless_DDO69_b_background_subtracted.fit")
starless_b = starless_file_b[0].data
#print(np.shape(starless_b))
starless_file_v = fits.open(r"C:/Users\AYSAN\Desktop\ddo69_starless_V.fit")
starless_v = starless_file_v[0].data
#print(np.shape(starless_v))
starless_u[starless_u <= 0] = 1
starless_b[starless_b <= 0] = 1
starless_v[starless_v <= 0] = 1
images =  [starless_u,starless_b,starless_v]
def mag_table_correction(images, airmass_values, m_values):
    #first step of mag correction (turning each pixel into a magnitude/arcsec value)
    magnitude_tables=[]
    for i in range(0,len(images)): 
        image = images[i]
        flux = (image/(exp[i]*((pixel_scale)**2)))
        magnitude_table = (-2.5 * np.log10(flux) + 25) 
        magnitude_tables.append(magnitude_table)
 
    corrected_magnitude_tables = []
    B_V_magnitude_table = [magnitude_table[1],magnitude_table[2]]
    for j in range(0,len(B_V_magnitude_table)):
        i = j+1
        s1 = m_values[i][0]
        s2 = airmass_values[1]*m_values[i][1]
        s3=m_values[i][2]*(magnitude_tables[1] - magnitude_tables[2])
        s4=airmass_values[1]*m_values[i][3]*(magnitude_tables[1] - magnitude_tables[2])
        corrected_magnitude_table = magnitude_tables[i] - s1 - s2 - s3 - s4
        corrected_magnitude_tables.append(corrected_magnitude_table)
        '''
        plt.imshow( corrected_magnitude_table, origin = "lower")
        plt.title("corrected %s %s"%(galaxy_name,filter[i]))
        plt.colorbar()
        plt.show()
        '''
    return corrected_magnitude_tables


#------------------------------------------------------------------------------------------------------------------------------
images = [starless_u,starless_b,starless_v]
galaxy_boxes = []
magnitude_tables=[]
for i in range(0,len(images)): 
    image = images[i]
    norm = ImageNormalize(vmin=0., stretch=LogStretch())
    galaxy_box = image[centers[i][1]-box_size : centers[i][1]+box_size , centers[i][0]-box_size : centers[i][0]+box_size]
    galaxy_boxes.append(galaxy_box)
    '''
    plt.imshow(galaxy_box, origin = "lower")
    plt.title("galaxy box %s %s"%(galaxy_name,filter[i]))
    plt.colorbar()
    plt.show()
    '''
    image_center_of_mass = ndimage.center_of_mass(galaxy_box)
    #coordinates
    x, y = image_center_of_mass[1], image_center_of_mass[0]
    
    # Create a figure and axes
    fig, ax = plt.subplots()
    # Display  image
    ax.imshow(galaxy_box,norm=norm,origin="lower")
    plt.title("center of mass %s %s"%(galaxy_name,filter[i]))
    # Mark the point with a red circle
    circle = plt.Circle((x, y), radius=1, fill=False, color='red')
    ax.add_patch(circle)
    plt.show() 
    


def contour_lines(box,sigma,level):
    #smoothing (gaussian convolution)----------------------------------------------------------------------------------------------
        smoothed = gaussian_filter(box, sigma)
        '''
        plt.imshow(smoothed, origin = "lower")
        plt.title("smoothed galaxy box %s %s"%(galaxy_name,filter[i]))
        plt.colorbar()
        plt.show()
        '''
        # Draw contour lines-----------------------------------------------------------------------------------------------------------
        # Plot the smoothed array
        plt.imshow(smoothed, alpha = 0.75)
        # Add contour line on top of the smoothed array
        CS = plt.contour(smoothed, level)
        dat0 = CS.allsegs[0][0]
        x_coord = dat0[:, 0]
        y_coord = dat0[:, 1]
        '''
        plt.plot(dat0[:, 0], dat0[:, 1])
        plt.show()
        # Set the title
        ax.set_title("Contours for %s V for contour %s" % (galaxy_name,level))
        # Display the plot
        plt.show()
        '''

        return x_coord,y_coord

V_mag_table = mag_table_correction(images,airmass_values,m_values)[1]
mag_box = V_mag_table[centers[2][1]-box_size : centers[2][1]+box_size , centers[2][0]-box_size : centers[2][0]+box_size]
plt.imshow(mag_box)
plt.title("My code, my starless")
plt.colorbar()
plt.show()

contour = contour_lines(mag_box,sigma=5,level=[25])
x_points = contour[0]
y_points = contour[1]

def ellipse(x, xc, yc, a, b, theta):
    return ((x[0] - xc) * np.cos(theta) + (x[1] - yc) * np.sin(theta))**2 / a**2 + ((x[0] - xc) * np.sin(theta) - (x[1] - yc) * np.cos(theta))**2 / b**2 - 1

initial_guess = [(max(x_points) - min(x_points)) / 2, 200., 200., (max(y_points) - min(y_points)) / 2, 0]

popt, pcov = curve_fit(ellipse, (x_points, y_points), np.zeros_like(x_points), p0=initial_guess)
stdv=np.sqrt(np.diag(pcov))
stdvx=stdv[0]
stdvy=stdv[1]
stdva=stdv[2]
stdvb=stdv[3]
stdvpa=stdv[4]

xc, yc, a, b, theta = popt

curve = ellipse(x_points,popt[0],popt[1],popt[2],popt[3],popt[4])

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Assuming you have already defined xc, yc, a, b, and theta
xc, yc, a, b, theta = popt
print(popt)
# Create a figure and axis
plt.figure()
ax = plt.gca()

# Display the other image
plt.imshow(mag_box, cmap='gray')

# Create the ellipse
ellipse = Ellipse(xy=(xc, yc), width=2*a, height=2*b, angle=np.degrees(theta), edgecolor='r', facecolor='none', linewidth=2)

# Add the ellipse to the axis
ax.add_patch(ellipse)

# Set axis limits (adjust as needed)
ax.set_xlim(0, mag_box.shape[1])
ax.set_ylim(mag_box.shape[0], 0)  # Reverse y-axis for imshow
plt.title("My code, my starless")
# Show the plot
plt.show()


    
     


