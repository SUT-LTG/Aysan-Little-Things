from astropy.io import fits
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D,SExtractorBackground,MedianBackground
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import reza
plt.gray()


light_file = fits.open(r"C:/Users\AYSAN\Desktop/project/Galaxy\Data\DDO69\d69_V.fits")
light = light_file[0].data

starless_file = fits.open(r"c:/Users\AYSAN\Desktop/project/Galaxy\starless fits\background first\starless_DDO69_B_background_subtracted.fit")
starless = starless_file[0].data

sigma_clip = SigmaClip(sigma=3.0)
bkg_estimator = SExtractorBackground()
bkg = Background2D(light, (250 , 300) , filter_size=(3, 3),
                   sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

data = light - bkg.background + 25.


# the problem: 

print('light =' , reza.statist(light) , 'background =' , reza.statist(bkg.background) , 'corrected =' , reza.statist(data))

# Calculate the number of negative values
num_negative_values = np.sum(data < 0)

# Calculate the ratio
ratio = num_negative_values / data.size

print(f"The ratio of negative values to the total number of values is: {ratio}")
#-----------------------------------------------------------------------------------------------------------------------------#
# no correction: 

import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

# Generate a test image
image = data

# Create normalizer object with LogStretch
norm = ImageNormalize(vmin=0., stretch=LogStretch())

# Make the figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(image, origin='lower', norm=norm)
plt.title("light-bkg (no correction)")
# Add colorbar
fig.colorbar(im)

plt.show()
#-----------------------------------------------------------------------------------------------------------------------------#
# solution one: 
positive_data = data + np.abs(np.min(data))

# Create normalizer object with LogStretch
norm = ImageNormalize(vmin=0., vmax=65536, stretch=LogStretch())

# Make the figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(positive_data, origin='lower', norm=norm)
plt.title("light-bkg (plus min vlue)")
# Add colorbar
fig.colorbar(im)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------#
# solution two:
arr = data

# Find negative values and create a new array
neg_arr = arr[arr < 0]

print(np.mean(neg_arr)) #-8 for B
print(np.median(neg_arr)) #-7 for B

arr = arr + np.median(neg_arr)
# Create normalizer object with LogStretch
norm = ImageNormalize(vmin=0., vmax=65536, stretch=LogStretch())

# Make the figure
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(positive_data, origin='lower', norm=norm)
plt.title("light-bkg (plus median of negative values)")
# Add colorbar
fig.colorbar(im)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------#
 #solution 3:
from scipy.stats import norm
'''
# Generate some data for this demonstration.

# Fit a normal distribution to the data:
mu, std = norm.fit(data)

# Plot the histogram.
plt.hist(data, bins=25, density=True, alpha=0.6)
print(np.min(data))
# Plot the PDF.

x = np.linspace(np.min(data), np.max(data), 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)

plt.show()
'''