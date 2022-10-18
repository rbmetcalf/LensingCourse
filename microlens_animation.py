#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 18:12:15 2022

@author: bmetcalf
"""

# need special functions for incomplete elliptic integrals of the first kind
from scipy import special as sy 
import numpy as np
import matplotlib.pyplot as plt


## function that takes source position returns image position for image 1
def image1(y1,y2):
    # magnitude of source position
    m = np.sqrt(y1*y1 + y2*y2)
    x1 = y1 * 0.5 * (m+np.sqrt(m*m+4)) /m
    x2 = y2 * 0.5 * (m+np.sqrt(m*m+4)) /m
    return x1,x2

## function that takes source position returns image position for image 2
def image2(y1,y2):
    # magnitude of source position
    m = np.sqrt(y1*y1 + y2*y2)
    x1 = y1 * 0.5 * (m-np.sqrt(m*m+4)) /m
    x2 = y2 * 0.5 * (m-np.sqrt(m*m+4)) /m
    return x1,x2

from matplotlib.animation import FuncAnimation
t=np.arange(0,1,0.0005)*2*np.pi # angles in radians for a full circle
xx = np.cos(t)
yy = np.sin(t)


yo = np.array([-3.0,-0.25])  # initial position of source
dy = 6./60*np.array([1,0])  # change in source postion
r = 0.3
  
#source position
fig, ax = plt.subplots(figsize=(10,10))

def frame(i) :    

    y = yo + i*dy

    ax.clear()
    ax.set_aspect(1)
    # draw Einstein radius
    ax.plot(xx,yy,linestyle='--',label='Einstein radius')
    ax.plot(0.0,0.0,'o')

    # outline of a circular source
    y1 = np.array(r*np.cos(t)+y[0])
    y2 = np.array(r*np.sin(t)+y[1])

    ax.plot(y1,y2,label='source')

    x1,x2 = image1(y1,y2)
    ax.plot(x1,x2,label='image1')

    x1,x2 = image2(y1,y2)
    ax.plot(x1,x2,label='image2')
  
    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(-2.5,2.5)
    ax.legend()

plt.xlabel(r'$x ~/~ R_{\rm Einstein}$',fontsize=23)
plt.ylabel(r'$y ~/~ R_{\rm Einstein}$',fontsize=23)


# run the animation
ani = FuncAnimation(fig,frame, frames=60, interval=200, repeat=False)

plt.show()