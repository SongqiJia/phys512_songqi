import numpy as np
from matplotlib import pyplot as plt

logdx=np.linspace(-5,1,1001)
dx=10**logdx

def fun1(x):
    return np.exp(x) #define exp^(x)

def fun2(x):
    return np.exp(10*x) #define exp^(0.01x)

def fun3(x):
    return np.sin(x)

x0=1

def twosidederiv(fun,x0,dx): #define 2-sided derivative
    y1=fun(x0+dx)
    ym=fun(x0-dx)
    d2=(y1-ym)/(2*dx) #calculate the 2-sided derivative.
    return d2

def foursidederiv(fun,x0,dx): #define 4-sided derivative
    dd = twosidederiv(fun,x0,dx)
    d2d = twosidederiv(fun,x0,2*dx)
    return (4*dd - d2d)/3 #from what we calculated earlier

def thirdderiv(fun,x0,dx):
    dd = twosidederiv(fun,x0,dx)
    d2d = twosidederiv(fun,x0,2*dx)
    return (d2d - dd)/(dx**2) #from what we calculated earlier

d3 = thirdderiv(fun3,x0,dx)

# plt.ion()
plt.clf()
#make a log plot of our errors in the derivatives
plt.loglog(dx,np.abs(d3-np.cos(x0)))
plt.show()


