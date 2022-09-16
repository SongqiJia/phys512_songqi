import numpy as np
from matplotlib import pyplot as plt

def fun1(x):
    return np.exp(x) #define exp^(x)

def fun2(x):
    return np.exp(10*x) #define exp^(10*x)

def fun3(x):
    return np.sin(x)

def twosidederiv(fun,x0,dx): #define 2-sided derivative
    y1=fun(x0+dx)
    ym=fun(x0-dx)
    d2=(y1-ym)/(2*dx) #calculate the 2-sided derivative.
    return d2

def thirdderiv(fun,x0):
    dd = twosidederiv(fun,x0,10**-3)
    d2d = twosidederiv(fun,x0,2*10**-3)
    return (d2d - dd)/((10**-6)) # from what we calculated earlier, and by setting dx to be 10^-3

def optimal_dx(fun,x0,d3):
    return 10**-5.3*((fun(x0)/d3)**1/3) # from what we learned in class, factored 10^-16 out so that it won't reach lower roundoff of double

def estimated_error(fun,x0,d3,dx):
    return fun(x0)*(10**-16)/dx + d3*dx**2

def ndiff(fun,x,full=False):
    d3 = thirdderiv(fun,x) #calculated estimated thrid deriv
    dx = optimal_dx(fun,x,d3) #calculated optimal dx
    d2 = twosidederiv(fun,x,dx) #calculated numerical deriv
    if full == False:
        return d2
    else:
        err = estimated_error(fun,x,d3,dx) #if required calculate estimated error
        return (d2,err)

tup1 = ndiff(fun1,1,full=True)
print ("numerical deriv of f(x)=exp(x) at x = 1 is: ", tup1[0], ", the estimated error is: ", tup1[1], 
    ", the real error is: ", np.abs(tup1[0]-fun1(1)))

tup2 = ndiff(fun2,10,full=True)
print ("numerical deriv of f(x)=exp(10*x) at x = 10 is: ", tup2[0], ", the estimated error is: ", tup2[1], 
    ", the real error is: ", np.abs(tup2[0]-10*fun2(10)))

tup3 = ndiff(fun3,1,full=True)
print ("numerical deriv of f(x)=sin(x) at x = 1 is: ", tup3[0], ", the estimated error is: ", tup3[1], 
    ", the real error is: ", np.abs(tup3[0]-np.cos(1)))
