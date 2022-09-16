import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate

def lorentz(x):
    return 1/(1+x**2)

def rat_eval(p,q,x): #stolen from ratfit_exact.py from course github
    top=0
    for i in range(len(p)): #calculate nominator of rational fit
        top=top+p[i]*x**i
    bot=1
    for i in range(len(q)): #calculate denominator of rational fit
        bot=bot+q[i]*x**(i+1)
    return top/bot

def rat_fit(x,y,n,m):
    assert(len(x)==n+m-1)
    assert(len(y)==len(x)) #assert if number of fit points are bad
    mat=np.zeros([n+m-1,n+m-1]) #create zero square matrix
    #create matrix accroding to lecture ppt page 14
    for i in range(n):
        mat[:,i]=x**i 
    for i in range(1,m):
        mat[:,i-1+n]=-y*x**i
    # pars=np.dot(np.linalg.inv(mat),y) #inverse matrix
    pars=np.dot(np.linalg.pinv(mat),y) #for second part of the question
    p=pars[:n] #p coeff is first n
    q=pars[n:] #q ceoff is after first n
    return p,q



# fun=np.cos;x0=-np.pi/2;x1=np.pi/2; #comment this out to have cos
fun=lorentz;x0=-1;x1=1 #comment this out to have lorentzian

# n=3
# m=3

n=4 #for second part of the question
m=5

#again, the following is stolen from ratfit.py from course github
x=np.linspace(x0,x1,n+m-1) #create linspace with right amount of points
y=fun(x)
p,q=rat_fit(x,y,n,m) #calculate rational fit coeff
xx=np.linspace(x[0],x[-1],1001)
y_true=fun(xx) #true value of cos
pred=rat_eval(p,q,xx) #rational fit value

fitp=np.polyfit(x,y,n+m-1) #np.polyfit returns vector of coefficients with degree of n+m-1
pred_poly=np.polyval(fitp,xx) #polynomial fit value

myfun=interpolate.interp1d(x,y,'cubic') #scipy.interpolate.interp1d returns a function which cubic spine fit of the data
pred_spline=myfun(xx)

#print the standard deviation of the fitted values with true values
print('rat err ',np.std(pred-y_true))
print('poly err ',np.std(pred_poly-y_true))
print('spline err ',np.std(pred_spline-y_true))
