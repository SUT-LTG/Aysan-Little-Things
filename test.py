from astropy.io import fits
import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
from astropy.stats import SigmaClip
from photutils.background import Background2D,SExtractorBackground
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
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
mB_values = [3.434, 0.3398903,-0.0157, 0.]
mV_values = [3.009, 0.2183592, 0.0415, 0.]

m_values = [mU_values,mB_values,mV_values]
pixel_scale = 1.134 #(arcsec)
plt.gray()
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

lenxU=np.shape(starless_u)[0]
lenyU=np.shape(starless_u)[1]
lenxB=np.shape(starless_b)[0]
lenyB=np.shape(starless_b)[1]
lenxV=np.shape(starless_v)[0]
lenyV=np.shape(starless_v)[1]

inst_magU0=np.zeros([lenxU,lenyU])
inst_magB0=np.zeros([lenxB,lenyB])
inst_magV0=np.zeros([lenxV,lenyV])

inst_mag_corrected_U=np.zeros([lenxU,lenyU])
inst_mag_corrected_B=np.zeros([lenxB,lenyB])
inst_mag_corrected_V=np.zeros([lenxV,lenyV])      

for i in range (lenxB):
    for j in range (lenyB):
        inst_magB0[i][j]=-2.5*np.log10(starless_b[i][j]/(exp[1]*(pixel_scale**2)))+25
        inst_magV0[i][j]=-2.5*np.log10(starless_v[i][j]/(exp[2]*(pixel_scale**2)))+25
        
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

mag_box = inst_mag_corrected_V[centers[2][1]-box_size : centers[2][1]+box_size , centers[2][0]-box_size : centers[2][0]+box_size]

plt.imshow(mag_box, origin='lower')
plt.title("Golta's mag correction, golta starless")
plt.colorbar()
plt.show()
