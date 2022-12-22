#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 20:13:10 2022

Code for preforming the Kaiser-Squires 
Copies from M. Meneghetti 

@author: bmetcalf
"""
import scipy.fftpack as fftpack
import numpy as np
 
def KS93(g1,g2) :
    """
    Function for doing the Kaiser & Squires mass reconstruction from shear data
    
    Parameters
    ----------
    g1 : TYPE
        map of shear component 1 
    g1 : TYPE
        map of shear component 2

    Returns kappa map
    -------
    None.

    """

    # zero-padding
    g1_pad=gpad(g1)
    g2_pad=gpad(g2)
    
    D1,D2 = kernel(g1_pad.shape[0])
    
    g1ft = fftpack.fftn(g1_pad)
    g2ft = fftpack.fftn(g2_pad)
    
    kappaft = D1*g1ft + D2*g2ft
    
    kappa = fftpack.ifftn(kappaft)
    
    return mapcrop(kappa.real,g1.shape[0])

def kernel(n):
    
    kx,ky = np.meshgrid(fftpack.fftfreq(n),fftpack.fftfreq(n))
    norm=(kx**2 + ky**2 + 1e-12)
    D1=-(kx**2-ky**2)/norm
    D2=-2*kx*ky/norm
    return(D1,D2)

def gpad(gmap):
    
    def padwithzeros(vector,pad_width,iaxis,kwargs) :
        vector[:pad_width[0]] = 0
        vector[-pad_width[1]:] = 0
        return vector
    
    return np.lib.pad(gmap, 2*gmap.shape[0],padwithzeros)

def mapcrop(inmap,n) :
    
    xmin=int(inmap.shape[0]/2-n/2)
    ymin=int(inmap.shape[1]/2-n/2)
    xmax=int(xmin+n)
    ymax=int(ymin+n)
    outmap=inmap[xmin:xmax,ymin:ymax]
    return(outmap)


## read fits file
from astropy.io import fits
import matplotlib.pyplot as plt
## pip install scikit-image
from skimage.measure import block_reduce

plt.rc('axes', labelsize=3*20)
plt.rc('legend', fontsize=3*5) 

range_o = 20  # range of field in arcmin
g_sigma = 0.0 # shear error for one galaxy 
n = 500        # number of galaxies per arcmin^2
#pix_size = 0.1  # reconstruction pixel size in arcmin
pix_size = 3/np.sqrt(n)  # reconstruction pixel size in arcmin


hdul = fits.open('snap_058.sph1000x1000S30Zl0.506868Zs2.000000prj3.g1.fits')
g1 = hdul[0].data
hdul.close()

hdul = fits.open('snap_058.sph1000x1000S30Zl0.506868Zs2.000000prj3.g2.fits')
g2 = hdul[0].data
hdul.close()

# down size to 1 arcmin pixels
Ndwnsize = int( pix_size * g1.shape[0] / range_o)
sigma = g_sigma/np.sqrt(n)


g1 = block_reduce(g1, block_size=(Ndwnsize,Ndwnsize), func=np.mean)
g2 = block_reduce(g2, block_size=(Ndwnsize,Ndwnsize), func=np.mean)

g1 = g1 + sigma * np.random.normal(size=g1.shape)
g2 = g2 + sigma * np.random.normal(size=g1.shape)

plt.imshow(g1,origin='lower')
plt.show()
input("Press Enter to continue...")

levels = [0.10,0.15,0.20]
kappa = np.zeros_like(g1)
#fig,ax = plt.subplots(4,1,figsize=(60,30),sharey=True,gridspec_kw={'wspace':0})
for i in range(4) :
    kappa = KS93(g1*(1-kappa),g2*(1-kappa))
    kappa = kappa - kappa.min()
    #ax[i].imshow(kappa)
    plt.imshow(kappa,origin='lower',cmap='jet',vmax=0.6)#,cmap='cubehelix')
    plt.contour(kappa, levels, colors='w')
    plt.title('iteration ' + str(i))
    plt.show()
    input("Press Enter to continue...")
    
    #plt.plot(kappa[500,:],label='reconstructed')
    #plt.plot(kappa_o[500,:],label='original')
    
    #plt.legend()
    #plt.show()


hdul = fits.open('snap_058.sph1000x1000S30Zl0.506868Zs2.000000prj3.kappa.fits')
kappa_o = hdul[0].data
hdul.close()
kappa_o = kappa_o - kappa_o.min()

kappa_reduced = block_reduce(kappa_o, block_size=(int(Ndwnsize/2), int(Ndwnsize/2)), func=np.mean)
plt.imshow(kappa_reduced,origin='lower',cmap='jet',vmax=0.5)#,cmap='cubehelix')
plt.contour(kappa_reduced, levels, colors='w')#, extent=extent)
plt.title(r'down sampled true $\kappa$')
plt.show()
input("Press Enter to continue...")

plt.imshow(kappa_o,origin='lower',cmap='jet',vmax=0.5)#cmap='cubehelix')
#plt.contour(kappa_o, levels, colors='w')#, extent=extent)
plt.title(r'full resolution true $\kappa$')
plt.show()



