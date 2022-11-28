#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 09:53:26 2022

@author: bmetcalf
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=5) 

class CHI_SQUARED_SOURCE :
    
    def __init__(self,lens,x_images):
        self.lens = lens
        self.x = x_images
        self.n_images = len(x_images)
        self.y=np.zeros((self.n_images,2))

    def __call__(self,p) :
        
        self.lens.reset(p)
        self.y[0] = self.x[0]-self.lens.alpha(self.x[0])
        sig = np.sign(self.lens.lambda_r(self.x[0])*self.lens.lambda_t(self.x[0]))
        for i in range(1,self.n_images) :
            self.y[i] = self.x[i]-self.lens.alpha(self.x[i])
            sig += np.sign(self.lens.lambda_r(self.x[i])*self.lens.lambda_t(self.x[i]))
    
        return np.sum((self.y-self.y[0])**2) + 100 * abs(sig) * np.sum((self.y-self.lens.xo)**2)
    
    def plot(self,result,title='') :
        
        self.lens.reset(result.x)
       
        fig1, ax = plt.subplots()
        ax.set_box_aspect(1)

        ax.scatter(self.x[:,0],self.x[:,1],label='observed image positions')
        ax.scatter(self.lens.xo[0],self.lens.xo[1],label='ob. lens positions',s=6)


        y = np.empty_like(self.x)
        center = np.zeros(2)
        for i in range(0,len(self.x)) :
             y[i] = self.x[i]-self.lens.alpha(self.x[i])
             ax.scatter(y[i,0],y[i,1],label='source positions '+str(i),s=3.)
             
             center += self.x[i]

        center = center/len(self.x)
        boxsize = 1.5*(np.max(self.x) - np.min(self.x))
        
        plt.xlim(center[0]-boxsize/2,center[0]+boxsize/2)
        plt.ylim(center[1]-boxsize/2,center[1]+boxsize/2)
        
        if hasattr(self.lens,'xt') :
            xt = self.lens.xt()
            ax.plot(xt[0,:],xt[1,:],linestyle='--',label='tang. crit.')
                
            yt = np.empty_like(xt)
            for i in range(0,len(yt[0,:])) : 
                yt[:,i] = xt[:,i] - self.lens.alpha(xt[:,i]) 
                   
            ax.plot(yt[0,:],yt[1,:],label='tang. caustic',linewidth=0.5)
            
        if hasattr(self.lens,'xr') :
            xr = self.lens.xr()
            ax.plot(xr[0,:],xr[1,:],linestyle='--',label='radial crit.')
          
            yr = np.empty_like(xr)
            for i in range(0,len(yt[0,:])) : 
                yr[:,i] = xr[:,i] - self.lens.alpha(xr[:,i]) 
             
            ax.plot(yr[0,:],yr[1,:],label='radial caustic',linewidth=0.5)

        plt.legend()
        plt.title(title)
        plt.show()
        
class CHI_SQUARED_IMAGE :
    
    def __init__(self,lens,x_images):
        self.lens = lens
        self.x = x_images
        self.n_images = len(x_images)
        self.y=np.zeros((self.n_images,2))

    def chi2_xo(self,xo,p) :
        self.lens.resetXo(xo)
        return self.__call__(p)
    
    def __call__(self,p) :
        
        self.lens.reset(p)
        self.lens.lambda_t(self.x[1])
        y = self.x[0]-self.lens.alpha(self.x[0])
        for i in range(1,self.n_images) :
            y += self.x[i]-self.lens.alpha(self.x[i])
            
        y = y/self.n_images
        
        x_images = np.array(self.lens.images(y[0],y[1]))
               
        #print('n_images = ',len(x_images),x_images[0,:])
        n_images = len(x_images)
        
        parity_ob = np.empty(self.n_images)
        for i in range(0,self.n_images) :
            parity_ob[i] = np.sign(self.lens.mu(self.x[i]))
        
        parity_mod = np.empty(n_images)
        for i in range(0,n_images) :
            parity_mod[i] = np.sign(self.lens.mu(x_images[i]))
        
        chi2_test = 0
        matrix = np.zeros([n_images,self.n_images])
        for j in range(0,n_images) :
            dchi2 = 1.0e100
            for i in range(0,self.n_images) :
                
                matrix[j,i] = np.sum( (x_images[j]-self.x[i])**2 )
                if(parity_ob[i] == parity_mod[j]) :
                    tmp = matrix[j,i]
                    if(tmp < dchi2 ) :
                        dchi2 = tmp
                        
            chi2_test += dchi2
        
        sort_index = np.argsort(matrix.flatten())
        
        elim_ob = []
        elim_mod = []
        chi2 = 0
        for n in sort_index :

            i = n % n_images
            j = int(n / n_images)
            
            if( i not in elim_ob) :
                if( j not in elim_mod) :
                    elim_ob.append(i)
                    elim_mod.append(j)
                    #print(i,j)
                    chi2 += matrix[i,j]
        
        #print(elim_ob,elim_mod)
        #for j in range(0,n_images) :
        #    chi2 += np.min(matrix[j,:])
 
        #for j in range(n_images,self.n_images) :
        #    chi2 += np.sum( (y-self.lens.xo)**2 )/0.01
        if(n_images != self.n_images) :
            chi2 += 1.0e6
            
        return chi2
    
    def plot(self,result,title='') :
        
        self.lens.reset(result.x)
       
        fig1, ax = plt.subplots()
        ax.set_box_aspect(1)

        ax.scatter(self.x[:,0],self.x[:,1],label='observed image positions')
        ax.scatter(self.lens.xo[0],self.lens.xo[1],label='ob. lens positions',s=6)

        y = self.x[0]-self.lens.alpha(self.x[0])
        for i in range(1,self.n_images) :
            y += self.x[i]-self.lens.alpha(self.x[i])
    
        y = y/self.n_images

        x_images = np.array(self.lens.images(y[0],y[1]))

        ax.scatter(x_images[:,0],x_images[:,1],label='model image positions')
        ax.plot(y[0],y[1],'x',label='model source position')
        
        center = np.zeros(2)
        for i in range(0,len(self.x)) :  
             center += self.x[i]

        center = center/len(self.x)
        boxsize = 1.5*(np.max(self.x) - np.min(self.x))
        
        plt.xlim(center[0]-boxsize/2,center[0]+boxsize/2)
        plt.ylim(center[1]-boxsize/2,center[1]+boxsize/2)
        
        if hasattr(self.lens,'xt') :
            xt = self.lens.xt()
            ax.plot(xt[0,:],xt[1,:],linestyle='--',label='tang. crit.')
                
            yt = np.empty_like(xt)
            for i in range(0,len(yt[0,:])) : 
                yt[:,i] = xt[:,i] - self.lens.alpha(xt[:,i]) 
                   
            ax.plot(yt[0,:],yt[1,:],label='tang. caustic',linewidth=0.5)
            
        if hasattr(self.lens,'xr') :
            xr = self.lens.xr()
            ax.plot(xr[0,:],xr[1,:],linestyle='--',label='radial crit.')
          
            yr = np.empty_like(xr)
            for i in range(0,len(yt[0,:])) : 
                yr[:,i] = xr[:,i] - self.lens.alpha(xr[:,i]) 
             
            ax.plot(yr[0,:],yr[1,:],label='radial caustic',linewidth=0.5)

        plt.legend()
        plt.title(title)
        plt.show()
  
class CHI_SQUARED2 :
    
    def __init__(self,lens,x_images,x_lens,sigma_image,sigma_g):
        self.lens = lens
        self.x = x_images
        self.n_images = len(x_images)
        self.y=np.zeros((self.n_images,2))
        self.xg = x_lens
        self.sigma_images = sigma_image
        self.sigma_g = sigma_g
        
        self.np = lens.number_of_params()
        

    def __call__(self,p) :
        
        assert self.np+2*self.n_images + 2 == len(p)
        
        self.lens.reset(p)
        self.lens.resetXo(p[-2:])
        x = p[self.np : self.np + 1]
        self.y[0] = x - self.lens.alpha(x)
        image_position_chi2 = np.sum( (x-self.x[0])**2 )
          
        for i in range(1,self.n_images) :
            x = p[self.np + 2*i : self.np + 2*i + 1]
            self.y[i] = x - self.lens.alpha(x)
            image_position_chi2 += np.sum( (x-self.x[i])**2 )

        image_position_chi2 = image_position_chi2/self.sigma_images**2

        lens_postion_chi2 = (p[-2] - self.xg[0])**2 + (p[-1] - self.xg[1])**2
        lens_postion_chi2 = lens_postion_chi2/self.sigma_g**2
        
        self.y = self.y-self.y[0]
    
        return np.sum(self.y**2) + image_position_chi2 + lens_postion_chi2
 
    def plot(self,result,title='') :
          
        fig1, ax = plt.subplots()
        ax.set_box_aspect(1)
        
        # plot original positions of images and lens
        ax.scatter(self.x[:,0],self.x[:,1],label='ob. image positions')
        ax.scatter(self.xg[0],self.xg[1],label='ob. lens positions',s=6)
        
        # plot model positions of images and lens
        ax.plot(result.x[self.np],result.x[self.np + 1],'+',label='model image positions',color='red')
        for i in range(1,self.n_images) :
            ax.plot(result.x[self.np + 2*i],result.x[self.np + 2*i + 1],'+',color='red')
  
        ax.plot(result.x[-2],result.x[-1],'x',label='model lens positions',color='red')
   
        self.lens.reset(result.x)
        self.lens.resetXo(result.x[-2:])
       
        y = np.empty_like(self.x)
        center = np.zeros(2)
        for i in range(0,len(self.x)) :
            y[i] = self.x[i]-self.lens.alpha(self.x[i])
            ax.scatter(y[i,0],y[i,1],label='source positions '+str(i),s=3.)
            
            center += self.x[i]

        center = center/len(self.x)
        boxsize = 1.5*(np.max(self.x) - np.min(self.x))
       
        plt.xlim(center[0]-boxsize/2,center[0]+boxsize/2)
        plt.ylim(center[1]-boxsize/2,center[1]+boxsize/2)
       
        if hasattr(self.lens,'xt') :
           xt = self.lens.xt()
           ax.plot(xt[0,:],xt[1,:],linestyle='--',label='tang. crit.')
        
           yt = np.empty_like(xt)
           for i in range(0,len(yt[0,:])) : 
               yt[:,i] = xt[:,i] - self.lens.alpha(xt[:,i]) 
           
           ax.plot(yt[0,:],yt[1,:],label='tang. caustic',linewidth=0.5)

        plt.legend()
        plt.title(title)
        plt.show()
 