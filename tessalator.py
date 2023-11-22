#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:40:23 2022

@author: bmetcalf
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=5) 
from scipy.optimize import minimize,fsolve,brentq
from scipy.spatial import Delaunay
import sys

class triangle_image_finder2 :
    
    def __init__(self,lens,image_range,n) :
        
        self.lens = lens
        
        x1 = np.arange(0,n)*(image_range[0][0]-image_range[0][1])/(n-1) - image_range[0][0]
        x2 = np.arange(0,n)*(image_range[1][0]-image_range[1][1])/(n-1) - image_range[1][0]
        
        self.points = np.empty([n*n,2])
        
        for i in range(n) :
            for j in range(n) :
                self.points[i*n+j][0] = x1[i]
                self.points[i*n+j][1] = x2[j]
        
        self.tri = Delaunay(self.points)
        self.ys = np.empty_like(self.points)
        
        self.set_lens(lens)
 

    def set_lens(self,lens) :
        for i in range(0,len(self.points)) :
            self.ys[i] = self.points[i] - lens.alpha(self.points[i])
            
         
    def find_images(self,y) :
  
        ## find triangles that the source is inside
        found = self.find_triangle(self.tri,self.ys,y)
    
        print('found ',found)
        images = []
        for i in found :
            t = self.tri.simplices[i]
            ## image in center of triangle
            x = self.points[t[0]]
            x = x + self.points[t[1]]
            x = x + self.points[t[2]]
            x = x/3.
        
            images.append(x)
       
        return np.array(images)
    
    def cross(self,p1,p2) :
        return p1[0]*p2[1]-p1[1]*p2[0]

    def find_triangle(self,tri,points,x) :
        j=0
        found = []
        for t in tri.simplices :
            p0 = points[t[0]]
            p2 = points[t[1]]
            p1 = points[t[2]]
    
            s = np.sign(self.cross(x-p0,p1-p0))
            s += np.sign(self.cross(x-p1,p2-p1))
            s += np.sign(self.cross(x-p2,p0-p2))
    
            if(abs(s)==3) :
                found.append(j)
            j = j + 1
        
        return found

    
    
    