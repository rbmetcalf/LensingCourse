#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 18:35:27 2022

@author: bmetcalf
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=5) 
from scipy.optimize import minimize,fsolve,brentq

import astropy.cosmology as cosmo

class sis_lens:
    
    def __init__(self,xo,b):
        self.xo = xo
        self.b = b
        
    def alpha(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        return self.b*dx/r

    def kappa(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        return self.b/r
    
    def reset(self,p) :
        self.b = p[0]

    def resetXo(self,xo) :
        self.xo = xo
        
    def number_of_params(self) :
        return 1
    
    def lambda_r(self,x) :
        return 1

    def lambda_t(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        return 1-self.b/r
    
    def mu(self,x) :
        return 1.0/self.lambda_t(x)
    def images(self,yy1,yy2,verbose=False):
       
        y = np.array([yy1 - self.xo[0],yy2 - self.xo[1]])
        y_length = np.sqrt(y[0]*y[0] + y[1]*y[1])
        v = y/y_length
        
        imag = [y +  self.b*v + self.xo]
        if y_length < self.b :
            imag.append( y -  self.b*v + self.xo )
        
        return imag
    
    def xt(self) :
       theta = np.arange(0,2*np.pi,0.01)
       x = np.empty([2,len(theta)])
       x[0] = self.b*np.cos(theta) + self.xo[0]
       x[1] = self.b*np.sin(theta) + self.xo[1]
       
       return x

class sis_shear_lens:
    
    def __init__(self,xo,b,gamma1,gamma2):
        self.xo = xo
        self.b = b
        self.gamma = np.array([gamma1,gamma2])
        self.v = np.empty(2)
        
    def alpha(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        
        return self.b*dx/r + np.array([self.gamma[0]*dx[0] + self.gamma[1]*dx[1]  , self.gamma[1]*dx[0] - self.gamma[0]*dx[1]])

    def kappa(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        return self.b/r
    
    
    def reset(self,p) :
        self.b = p[0]
        self.gamma[0] = p[1]
        self.gamma[1] = p[2]

    def resetXo(self,xo) :
        self.xo = xo
        
    def number_of_params(self) :
        return 3
    
    def gamma_func(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        c=dx[0]/r
        s=dx[1]/r
        kappa = self.kappa(x)
        
        gamma1 = self.gamma[0] - (c*c - s*s) * kappa
        gamma2 = self.gamma[1] - 2 * c*s * kappa
        
        return gamma1,gamma2,kappa
    
    def lambda_r(self,x) :
        g1,g2,k = self.gamma_func(x)
        g = np.sqrt(g1*g1 + g2*g2)
        return 1 - k + g

    def lambda_t(self,x) :
        g1,g2,k = self.gamma_func(x)
        g = np.sqrt(g1*g1 + g2*g2)
        return 1 - k - g 

    def mu(self,x) :
        return 1.0/self.lambda_t(x)/self.lambda_r(x)
   
    def xt(self) :
       theta = np.arange(0,2*np.pi,0.1)
       x = np.empty([2,len(theta)])
       for i in range(0,len(theta)) :
           self.v[0] = np.cos(theta[i])
           self.v[1] = np.sin(theta[i])
           def func(r):
               return self.lambda_t(r*self.v + self.xo)
           rt = fsolve(func,self.b)
           
           x[:,i] = rt * self.v + self.xo
       
       return x
   
    def yt(self) :
        
        xt = self.xt()
        yt = np.empty_like(xt)
        for i in range(0,len(yt[0,:])) : 
            yt[:,i] = xt[:,i] - self.alpha(xt[:,i])
        
        return yt

class nsis_lens:
    
    def __init__(self,xo,b,xc):
        self.xo = xo
        self.b = b
        self.xc = xc
        
    def alpha(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        return self.b* dx / r * (np.sqrt( 1+ self.xc * self.xc/r/r) - self.xc/r)

    def kappa(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1] + self.xc*self.xc)
        return self.b/r
    
    def reset(self,p) :
        self.b = p[0]
        self.xc = p[1]

    def resetXo(self,xo) :
        self.xo = xo

    def number_of_params(self) :
       return 2

    def lambda_r(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        return 1 + self.b*( (np.sqrt(r*r + self.xc*self.xc) - self.xc)/r/r - 1.0/np.sqrt(r*r + self.xc*self.xc) )

    def lambda_t(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        return 1 - self.b*(np.sqrt(r*r + self.xc*self.xc) - self.xc)/r/r
    
    def mu(self,x) :
        return 1.0/self.lambda_t(x)/self.lambda_r(x)

    def xt(self) :
        
        def func(r):
            return 1 - self.b*(np.sqrt(r*r + self.xc*self.xc) - self.xc)/r/r
        rt = fsolve(func,self.b)
        
        theta = np.arange(0,2*np.pi,0.01)
        x = np.empty([2,len(theta)])
        x[0] = rt * np.cos(theta) + self.xo[0]
        x[1] = rt * np.sin(theta) + self.xo[1]
        
        return x
    
    def xr(self) :
        
        def func(r):
            return 1 + self.b*( (np.sqrt(r*r + self.xc*self.xc) - self.xc)/r/r - 1.0/np.sqrt(r*r + self.xc*self.xc) )
        rc = fsolve(func,self.b)
        
        theta = np.arange(0,2*np.pi,0.01)
        x = np.empty([2,len(theta)])
        x[0] = rc * np.cos(theta) + self.xo[0]
        x[1] = rc * np.sin(theta) + self.xo[1]
        
        return x
    
    # def images(self,y1,y2) :
    #     y = np.sqrt( (y1-self.xo[0])**2 + (y1-self.xo[1])**2) / self.b
    #     xc = self.xc/self.b
        
    #     def func(x) :
    #         return x*x*x - 2*y*x*x + (y*y + 2*xc -1)*x - 2*y*xc
        
    #     U=np.linspace(-1,1,100)
    #     c = func(U)
    #     s = np.sign(c)
    #     imag=[]
        
    #     for i in range(100):
    #         if s[i] + s[i+1] == 0: 
        
class sie_lens:
    
    def __init__(self,xo,b,f,phi):
        self.xo = xo
        self.b = b
        self.f = f
        self.phi = phi
        self.fp = np.sqrt(1-f*f)
    
    def rotate(self,x,phi) :
        tmp = x[0]
        x[0] = x[0]*np.cos(phi) - x[1]*np.sin(phi)
        x[1] = x[1]*np.cos(phi) + tmp*np.sin(phi)
    
    def alpha(self,x) :
        dx = x - self.xo
        
        self.rotate(dx,-self.phi)
        
        r = np.sqrt(dx[0]*dx[0] + dx[1]*dx[1])
        a = np.array([np.arcsinh(self.fp*dx[0]/r/self.f),np.arcsin(self.fp*dx[1]/r)]) * np.sqrt(self.f)/self.fp
        
        self.rotate(a,self.phi)
        
        return self.b*a

    def kappa(self,x) :
        dx = x - self.xo
        r = np.sqrt(dx[0]*dx[0] + self.f*self.f*dx[1]*dx[1])
        
        return 0.5*self.b*np.sqrt(self.f)/r
    
    def reset(self,p) :
        self.b = p[0]
        self.f = p[1]
        self.fp = np.sqrt(1-p[1]*p[1])
        self.phi = p[2]
        
    def resetXo(self,xo) :
        self.xo = xo
        
    def number_of_params(self) :
        return 3
    
    def lambda_r(self,x) :
        return 1
    
    def lambda_t(self,x) :
        return 1 - 2*self.kappa(x)

    def mu(self,x) :
        return 1.0/self.lambda_t(x)

    def xt(self) :
        theta = np.arange(0,2*np.pi,0.01)
        x = np.empty([2,len(theta)])
        delta = np.sqrt(np.cos(theta - self.phi)**2 + (self.f*np.sin(theta - self.phi))**2)
        x[0] = self.b*np.sqrt(self.f)*np.cos(theta) / delta + self.xo[0]
        x[1] = self.b*np.sqrt(self.f)*np.sin(theta) / delta + self.xo[1]
        
        return x
    
    def yt(self) :
        
        xt = self.xt()
        yt = np.empty_like(xt)
        for i in range(0,len(yt[0,:])) : 
            yt[:,i] = xt[:,i] - self.alpha(xt[:,i])
        
        return yt

    def yr(self) :
        theta = np.arange(0,2*np.pi,0.01)
        x = np.empty([2,len(theta)])
 
        x[0] = -np.sqrt(self.f) / self.fp * np.arcsinh(  self.fp/ self.f * np.cos(theta) ) / self.fp
        x[1] = -np.sqrt(self.f) / self.fp * np.arcsin(  self.fp * np.sin(theta) ) / self.fp

        x[0] = self.b * x[0] + self.xo[0]
        x[1] = self.b * x[1] + self.xo[1]
        
        return x

    def images(self,yy1,yy2,verbose=False):
       
        y = np.array([yy1 - self.xo[0],yy2 - self.xo[1]])
        self.rotate(y,-self.phi)
        
        def phi_func(phi):
            a1=self.b*np.sqrt(self.f)/self.fp*np.arcsinh(self.fp/self.f*np.cos(phi))
            a2=self.b*np.sqrt(self.f)/self.fp*np.arcsin(self.fp*np.sin(phi))
            return (y[0]+a1)*np.sin(phi)-(y[1]+a2)*np.cos(phi)

        U=np.linspace(0.,2.0*np.pi+0.001,100)
        c = phi_func(U)
        s = np.sign(c)
        phi=[]
        xphi=[]
        imag=[]
        
        for i in range(100-1):
            if s[i] + s[i+1] == 0: # opposite signs
                u = brentq(phi_func, U[i], U[i+1])
                z = phi_func(u) ## angle of a possible image 
                if np.isnan(z) or abs(z) > 1e-3:
                    continue
                x = y[0]*np.cos(u)+y[1]*np.sin(u)+self.b*np.sqrt(self.f)/self.fp*(np.sin(u)*np.arcsin(self.fp*np.sin(u)) + np.cos(u)*np.arcsinh(self.fp/self.f*np.cos(u)))
                if (x>0):
                    phi.append(u)
                    xphi.append(x)
                    t = x*np.array([np.cos(u),np.sin(u)])
                    self.rotate(t,self.phi)
                    imag.append(t + self.xo)
                if (verbose):
                    print('found zero at {}'.format(u))
                    if (x<0):
                        print ('discarded because x is negative ({})'.format(x))
                    else:
                        print ('accepted because x is positive ({})'.format(x))
                        
        return imag
