#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 08:55:38 2022

@author: bmetcalf
"""

from scipy import special as sy 
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants as const 
from astropy import units as u
from scipy.special import erfi
from scipy.integrate import trapezoid
import sys

def Gamma_dte(te,b) :
    
    a = b/te/te
    res = np.zeros(len(te))
    a_split = 100
    ## this part is here because of numerical problems for te <1
    index = np.asarray(a > a_split).nonzero()
    n = (-(6+a_split)/(8*a_split*a_split) + np.sqrt(np.pi)*(12+4*a_split+a_split*a_split)*np.exp(-a_split/4)*erfi(np.sqrt(a_split)/2) / 16 / a_split**2.5)*a_split*a_split
    res[index] = n*(te[index]*np.sqrt(a_split/b) )**2.05
    
    index = np.asarray(a <= a_split).nonzero()
    
    res[index] = (-(6+a[index])/(8*a[index]*a[index]) + np.sqrt(np.pi)*(12+4*a[index]+a[index]*a[index])*np.exp(-a[index]/4)*erfi(np.sqrt(a[index])/2) / 16 / a[index]**2.5)*a[index]*a[index]

    #print(te)
    #sys.exit()
    return res

G=const.G
c=const.c

Ds = 8*u.kpc
sigma = 120*u.km/u.s

te = 10**np.arange(-1,2,0.01)
mass = 1*const.M_sun
to = np.sqrt(4*(G*mass/c/c).to('km')*Ds.to('km')/sigma/sigma)
b = (0.5*to.to('day')*to.to('day')).value

G_dlnte = Gamma_dte(te,b)/te
norm = trapezoid(G_dlnte,np.log(te))
plt.plot(te,G_dlnte/norm,label=r'1.0 $M_{sun}$')
#plt.plot(to.to('day').value/5.25,0.7,'o')

#te = 10**np.arange(-0.3,2,0.01)
mass = 0.25*const.M_sun
to = np.sqrt(4*(G*mass/c/c).to('km')*Ds.to('km')/sigma/sigma)
b = (0.5*to.to('day')*to.to('day')).value

G_dlnte = Gamma_dte(te,b)/te
norm = trapezoid(G_dlnte,np.log(te))
plt.plot(te,G_dlnte/norm,label=r'0.25 $M_{sun}$')
#plt.plot(to.to('day').value/5,0.7,'o')

#te = 10**np.arange(-0.5,2,0.01)
mass = 0.1*const.M_sun
to = np.sqrt(4*(G*mass/c/c).to('km')*Ds.to('km')/sigma/sigma)
b = (0.5*to.to('day')*to.to('day')).value

G_dlnte = Gamma_dte(te,b)/te
norm = trapezoid(G_dlnte,np.log(te))
plt.plot(te,G_dlnte/norm,label=r'0.1 $M_{sun}$')
plt.plot(to.to('day').value/5.25,0.7,'o')

#plt.plot(te,0.04*te**1.05)
#plt.plot(te,0.09*te**1.05)

plt.xscale('log')
#plt.yscale('log')

plt.xlabel(r'$t_E$ (days)')
plt.ylabel(r'$\Gamma^{-1} \times d \Gamma / d\ln t_E$ ')
plt.legend()
plt.ylim(0.0,0.8)
plt.xlim(0.1,100)

#plt.savefig('dGammadt.png')
plt.show()