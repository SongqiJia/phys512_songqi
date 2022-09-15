import numpy as np
from matplotlib import pyplot as plt

logdx=np.linspace(-15,0,1001)
dx=10**logdx

def fun1(x):
    return np.exp(x) #define exp^(x)

def fun2(x):
    return np.exp(0.01*x) #define exp^(0.01x)

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

d4_fun1 = foursidederiv(fun1, x0, dx)
d4_fun2 = foursidederiv(fun2, x0, dx)

# plt.ion()
plt.clf()
#make a log plot of our errors in the derivatives
plt.loglog(dx,np.abs(d4_fun1-fun1(x0)))
plt.savefig('deriv_errors1.png')
plt.show()


plt.clf()
#make a log plot of our errors in the derivatives
plt.loglog(dx,np.abs(d4_fun2-0.01*fun2(x0))) # derive of exp^0.01*x = 0.01*exp^0.01*x
plt.savefig('deriv_errors2.png')
plt.show()
