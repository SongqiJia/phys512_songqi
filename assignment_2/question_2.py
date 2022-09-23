import numpy as np

function_call = 0 #global variable to see how many function calls we did

def exp_int(x):
    global function_call
    #update the total number of function calls
    function_call = function_call+np.size(x)
    return np.exp(x)

def integrate(fun,a,b,tol,extra=None):
    if extra is None:
        x=np.linspace(a,b,5)
        dx=x[1]-x[0]
        y=fun(x)
        #do the 3-point integral
        i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
        i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
        myerr=np.abs(i1-i2)
        if myerr<tol:
            return i2
        else:
            mid=(a+b)/2
            #pass the already calculated function value into recursive call
            int1=integrate(fun,a,mid,tol/2,extra=np.array([y[0], y[1], y[2]]))
            int2=integrate(fun,mid,b,tol/2,extra=np.array([y[2], y[3], y[4]]))
            return int1+int2
    else:
        x=np.linspace(a,b,5)
        #we already have the 1st, middle and last function value from extra
        #need only to calculate 2nd and 4th value
        y=np.array([extra[0],0,extra[1],0,extra[2]])
        y[1] = fun(x[1])
        y[3] = fun(x[3])
        dx=x[1]-x[0]
        i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
        i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
        myerr=np.abs(i1-i2)
        if myerr<tol:
            return i2
        else:
            mid=(a+b)/2
            int1=integrate(fun,a,mid,tol/2,extra=np.array([y[0], y[1], y[2]]))
            int2=integrate(fun,mid,b,tol/2,extra=np.array([y[2], y[3], y[4]]))
            return int1+int2
        

def integrate_old(fun,a,b,tol):
    x=np.linspace(a,b,5)
    dx=x[1]-x[0]
    y=fun(x)
    #do the 3-point integral
    i1=(y[0]+4*y[2]+y[4])/3*(2*dx)
    i2=(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/3*dx
    myerr=np.abs(i1-i2)
    if myerr<tol:
        return i2
    else:
        mid=(a+b)/2
        int1=integrate_old(fun,a,mid,tol/2)
        int2=integrate_old(fun,mid,b,tol/2)
        return int1+int2

ans_new = integrate(exp_int,1,2,1e-6)
print('The integrated value is: ', ans_new)
# ans_old = integrate_old(exp_int,1,2,1e-6)
# print('The integrated value is: ', ans_old)
print('The number of function calls is: ', function_call)

