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

def mag(t,yo,te) :
    y = np.sqrt(yo*yo + t*t/te/te)
    
    return (y*y + 2)/y/np.sqrt(y*y+4)


te = 1
yo = 0.5
t = np.arange(-1.25,1.25,0.01)
m = mag(t,yo,te)

fig1, ax = plt.subplots()



for yo in np.arange(0.1,1.0,0.1) :
    m = mag(t,yo,te)
    ax.plot(t,m,label=r'$y_o=$'+f'{yo:.2f}')

ax.plot([-1.25,1.25],[1.34,1.34],linestyle=':')
ax.set_box_aspect(1)
plt.xlabel(r'$(t-t_o) ~ / ~t_E$')
plt.legend()
plt.xlim(-1.25,1.25)
plt.savefig('microlensing_lightcurves.png')
plt.show()
