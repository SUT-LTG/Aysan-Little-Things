import re
import numpy as np
import math
import astropy
import glob2
from astropy.io import fits
import matplotlib.pyplot as plt

filename = 'ubv_correction.txt'
airmass_values = []
integration_time_values = []
mU_values = []
mB_values = []
mV_values = []
pixel_scale=[]

current_reference = None
values_4_indexes = []

with open(filename, 'r') as file:
    lines = file.readlines()

for i, line in enumerate(lines):

    if "Pixel scale (arcsec)" in line:
        pixel_scale_text = line.split('---')[1]
        pixel_scale_values = [float(value.strip()) for value in pixel_scale_text.split(',')]
        
    if "Airmass" in line:
        airmass_text = line.split('---')[1]
        airmass_values = [float(value) for value in re.findall(r"[-+]?\d*\.\d+|\d+", airmass_text)]

    if "Integration time (s)" in line:
        integration_time_text = line.split('---')[1]
        integration_time_values = [int(value.strip()) for value in integration_time_text.split(',')]

    if "reference mU" in line:
        current_reference = "mU"
    elif "reference m1" in line:
        current_reference = "m1"
    elif "reference m2" in line:
        current_reference = "m2"

    if "values 4" in line and current_reference:
        values_text = lines[i + 1:i + 5]
        values = [float(value) for line in values_text for value in line.strip().split()]
        
        if current_reference == "mU":
                mU_values = values
        elif current_reference == "m1":
                mB_values = values
        elif current_reference == "m2":
                mV_values = values
                
print("Airmass Values:", airmass_values)
print("Pixel Scale Values:", pixel_scale_values)
print("Integration Time Values:", integration_time_values)
print("mU Values:", mU_values)
print("mB Values:", mB_values)
print("mV Values:", mV_values)
'''
#####################

img_U = glob2.glob('*_U_1.fit')
img_B= glob2.glob('*_B_1.fit')
img_V = glob2.glob('*_V_1.fit')
opening_imgU = [ fits.getdata(image) for image in img_U ]
opening_imgB = [ fits.getdata(image) for image in img_B ]
opening_imgV = [ fits.getdata(image) for image in img_V ]

lenxU=np.shape(opening_imgU[0])[0]
lenyU=np.shape(opening_imgU[0])[1]
lenxB=np.shape(opening_imgB[0])[0]
lenyB=np.shape(opening_imgB[0])[1]
lenxV=np.shape(opening_imgV[0])[0]
lenyV=np.shape(opening_imgV[0])[1]

inst_magU0=np.zeros([lenxU,lenyU])
inst_magB0=np.zeros([lenxB,lenyB])
inst_magV0=np.zeros([lenxV,lenyV])

inst_mag_corrected_U=np.zeros([lenxU,lenyU])
inst_mag_corrected_B=np.zeros([lenxB,lenyB])
inst_mag_corrected_V=np.zeros([lenxV,lenyV])      

for i in range (lenxB):
    for j in range (lenyB):
        inst_magB0[i][j]=-2.5*np.log10(opening_imgB[0][i][j]/(integration_time_values[1]*(pixel_scale_values[1]**2)))+25
        inst_magV0[i][j]=-2.5*np.log10(opening_imgV[0][i][j]/(integration_time_values[2]*(pixel_scale_values[2]**2)))+25
        
        b1=mB_values[0]
        b2=airmass_values[1]*mB_values[1]
        b3=mB_values[2]*(inst_magB0[i][j]-inst_magV0[i][j])
        b4=airmass_values[1]*mB_values[3]*(inst_magB0[i][j]-inst_magV0[i][j])
        inst_mag_corrected_B[i][j]=inst_magB0[i][j]-b1-b2-b3-b4

        v1=mV_values[0]
        v2=airmass_values[2]*mV_values[1]
        v3=mV_values[2]*(inst_magB0[i][j]-inst_magV0[i][j])
        v4=airmass_values[2]*mV_values[3]*(inst_magB0[i][j]-inst_magV0[i][j])
        inst_mag_corrected_V[i][j]=inst_magV0[i][j]-v1-v2-v3-v4

for i in range (lenxU):
    for j in range (lenyU):
        inst_magU0[i][j]=-2.5*np.log10(opening_imgU[0][i][j]/(integration_time_values[0]*(pixel_scale_values[0]**2)))+25
        u1=mU_values[0]
        u2=airmass_values[0]*mU_values[1]
        inst_mag_corrected_U[i][j]=inst_magU0[i][j]-u1-u2
       

plt.imshow(inst_mag_corrected_V, origin='lower')
plt.colorbar()
plt.show()


hdu = fits.PrimaryHDU(inst_mag_corrected_U)
#hdu.header['OBJECT']='inst_mag'
#hdu.header['DATE']='2023-05-25'
hdu.writeto('inst_mag_U.fits',overwrite=True)

hdu = fits.PrimaryHDU(inst_mag_corrected_B)
#hdu.header['OBJECT']='inst_mag'
#hdu.header['DATE']='2023-05-25'
hdu.writeto('inst_mag_B.fits',overwrite=True)

hdu = fits.PrimaryHDU(inst_mag_corrected_V)
#hdu.header['OBJECT']='inst_mag'
#hdu.header['DATE']='2023-05-25'
hdu.writeto('inst_mag_V.fits',overwrite=True)
'''










