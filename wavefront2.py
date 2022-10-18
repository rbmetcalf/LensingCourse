#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 18:53:21 2022

@author: bmetcalf
"""

import math
import numpy as np
import matplotlib.pyplot as plt

class Vector2D:
    """A two-dimensional vector with Cartesian coordinates."""

    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        """Human-readable string representation of the vector."""
        return '{:g}i + {:g}j'.format(self.x, self.y)

    def __repr__(self):
        """Unambiguous string representation of the vector."""
        return repr((self.x, self.y))

    def dot(self, other):
        """The scalar (dot) product of self and other. Both must be vectors."""

        if not isinstance(other, Vector2D):
            raise TypeError('Can only take dot product of two Vector2D objects')
        return self.x * other.x + self.y * other.y
    # Alias the __matmul__ method to dot so we can use a @ b as well as a.dot(b).
    __matmul__ = dot

    def __sub__(self, other):
        """Vector subtraction."""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __add__(self, other):
        """Vector addition."""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar):
        """Multiplication of a vector by a scalar."""

        if isinstance(scalar, int) or isinstance(scalar, float):
            return Vector2D(self.x*scalar, self.y*scalar)
        raise NotImplementedError('Can only multiply Vector2D by a scalar')

    def __rmul__(self, scalar):
        """Reflected multiplication so vector * scalar also works."""
        return self.__mul__(scalar)

    def __neg__(self):
        """Negation of the vector (invert through origin.)"""
        return Vector2D(-self.x, -self.y)

    def __truediv__(self, scalar):
        """True division of the vector by a scalar."""
        return Vector2D(self.x / scalar, self.y / scalar)

    def __mod__(self, scalar):
        """One way to implement modulus operation: for each component."""
        return Vector2D(self.x % scalar, self.y % scalar)

    def __abs__(self):
        """Absolute value (magnitude) of the vector."""
        return math.sqrt(self.x**2 + self.y**2)

    def distance_to(self, other):
        """The distance between vectors self and other."""
        return abs(self - other)

    def to_polar(self):
        """Return the vector's components in polar coordinates."""
        return self.__abs__(), math.atan2(self.y, self.x)


class Potential :
    def __init__(self, m, a , rmax):
        self.m=m
        self.a = a 
        self.rmax = rmax

    def __call__(self,r) :
        if r>self.rmax :
            return 0,0
        return -self.m*self.a/(r+self.a),self.m*self.a/(r+self.a)**2

class Potential2 :
    def __init__(self, m, a , rmax):
        self.m=m
        self.a = a 
        self.rmax = rmax

    def __call__(self,r) :
        
        p=-self.m*self.a*np.exp(-r/rmax)/(r+self.a)
        
        return p,-p/(r+self.a) - p/rmax

class Stepper :

    def __init__(self,c,xm,potential) :
        self.c = c
        self.xm = xm
        self.pot = potential
        self.vec_f = np.vectorize(self.f) 
        
    def __call__(self,x,v) :
        return self.vec_f(x,v)
        
    def f(self,x,v) :
        rv = (x-self.xm)
        r = abs(rv)
        phi,dphi = self.pot(r)
        
        x = x + (1 + 2*phi)*v*self.c
        #x = x + v*self.c
        v = v - 2*dphi*( rv - v.dot(rv)*v )/r*self.c
        v = v / abs(v)
        return x,v

# def step(x,v,xm,pot) :
#     rv = (x-xm)
#     r = abs(rv)
#     phi,dphi = pot(r)
    
#     x = x + 2*phi*v
#     v = v + 2*dphi*( rv - v.dot(rv)*rv )/r
    
#     return x,v
# step_v = np.vectorize(step)
    
    
def x_comp(v):
    return v.x
genx = np.vectorize(x_comp)

def y_comp(v):
    return v.y
geny = np.vectorize(y_comp)

def plotx(x,style='-') :
    #plt.plot(genx(x),geny(x),color='red')
    plt.plot(genx(x),geny(x),linestyle=style)

#### set initial conditions
x = []
v = []
for t in np.arange(np.pi*(0.5-0.05),np.pi*(0.5+0.05),np.pi/8000.) :
    y=Vector2D(np.cos(t),np.sin(t))
    x.append(y)
    v.append( y )

plotx(x)

m=0.3  # mass of lens
a=0.02   # softening length for lens
rmax=0.05
x_lens = Vector2D(0,1.3) # position of lens
c = 0.002 # sleed of light

pot = Potential2(m,a,rmax)
step = Stepper(c,x_lens,pot)

N=350
index1 = 325
ray1 = []
index2 = 240
ray2 = []
#index3 = int(3*99.9)
index3 = 399
ray3 = []
index4 = 100
ray4 = []

while( x[300].y < 1.15) :
    x,v = step(x,v)
    
for i in range(1,N) :
    x,v = step(x,v)
    ray1.append(x[index1])
    ray2.append(x[index2])
    ray3.append(x[index3])
    ray4.append(x[index4])

    if( (i % 30) == 0) :
        plotx(x)
        
#plt.plot(xray,yray,linestyle=':')
plotx(x)
plotx(ray1,style='--')
plotx(ray2,style='--')
plotx(ray3,style='--')
plotx(ray4,style='--')

plt.plot(x_lens.x,x_lens.y,'o')
plt.xlim(-0.2,0.2)
plt.ylim(1.15,1.6)


plt.savefig('wavefronts.png')

plt.show()