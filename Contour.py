import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from photutils.aperture import EllipticalAperture
from skimage.measure import find_contours
from scipy.optimize import curve_fit
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

hdulist = fits.open('ddo69_starless_V_1.fit')
gal = hdulist[0].data
#data = gal[395:615,255:590]
data = gal[370:635,230:615]
pixel_scale = 1.134

plt.figure(figsize=(7, 7))

contours = plt.contour(data, origin='lower', alpha=0.05, levels=np.arange(data.min(), data.max(), 15))
#contours = plt.contour(data, origin='lower', alpha=0.05)

len_of_ver=[]
path_arr=[]
for item in contours.collections:
    for path in item.get_paths():
        len_of_ver.append(len(path.vertices[:, 0]))
        path_arr.append(path)
    
path_element=np.argmax(len_of_ver)
vertices=path_arr[path_element].vertices
#plt.plot(vertices[:, 0]*pixel_scale, vertices[:, 1]*pixel_scale, color='r', linewidth=1)

x, y = vertices[:, 0], vertices[:, 1]
def ellipse(x, xc, yc, a, b, theta):
    return ((x[0] - xc) * np.cos(theta) + (x[1] - yc) * np.sin(theta))**2 / a**2 + ((x[0] - xc) * np.sin(theta) - (x[1] - yc) * np.cos(theta))**2 / b**2 - 1

initial_guess = [423.641476335231-230., 502.547546561502-370., (max(x) - min(x)) / 2, (max(y) - min(y)) / 2, 0]

popt, pcov = curve_fit(ellipse, (x, y), np.zeros_like(x), p0=initial_guess)
stdv=np.sqrt(np.diag(pcov))
stdvx=stdv[0]
stdvy=stdv[1]
stdva=stdv[2]
stdvb=stdv[3]
stdvpa=stdv[4]

xc, yc, a, b, theta = popt
#print(theta*180/np.pi)
print(xc+230,yc+370,a,b,theta)
print(stdvx,stdvy,stdva, stdvb, stdvpa)

t = np.linspace(0, 2*np.pi, 100)
ellipse_x = xc + a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta)
ellipse_y = yc + a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta)
plt.plot(ellipse_x, ellipse_y, color='r')

#ellipse_x2 = xc + a * np.cos(t) * np.cos(0) - b * np.sin(t) * np.sin(0)
#ellipse_y2 = yc + a * np.cos(t) * np.sin(0) + b * np.sin(t) * np.cos(0)
#plt.plot(ellipse_x2, ellipse_y2, color='r')
height, width = data.shape[:2]

x_pixels = range(width)
y_pixels = range(height)
x_arcseconds = [x * pixel_scale for x in x_pixels]
y_arcseconds = [y * pixel_scale for y in y_pixels]

plt.imshow(data,  extent=(min(x_arcseconds), max(x_arcseconds), min(y_arcseconds), max(y_arcseconds)), origin='lower',vmin=0, vmax=400)
plt.xlabel('Arcsecond',fontname="Times New Roman")
plt.ylabel('Arcsecond',fontname="Times New Roman")
plt.title('CONTOUR PLOT OF DDO69 IN THE V FILTER',fontname="Times New Roman")
#plt.imshow(data, origin='lower',vmin=0, vmax=400)
plt.colorbar()
plt.show()














