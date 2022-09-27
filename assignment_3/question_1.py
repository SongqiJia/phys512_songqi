import numpy as np
from matplotlib import pyplot as plt

#define dydx = y/(1+x^2)
def myfun(x,y):
    return y/(1+x**2)

#copied from rk4.py in class repo
#equations can be found in class ppt page 5
def rk4_step(fun,x,y,h):
    k1=fun(x,y)*h
    k2=h*fun(x+h/2,y+k1/2)
    k3=h*fun(x+h/2,y+k2/2)
    k4=h*fun(x+h,y+k3)
    dy=(k1+2*k2+2*k3+k4)/6
    return y+dy

def rk4_stepd(fun,x,y,h):
    rk4_h = rk4_step(fun,x,y,h)
    #run rk4 twice with step h/2
    #note we called the function 3 times per step
    rk4_h2 = rk4_step(fun,x+h/2,rk4_step(fun,x,y,h/2),h/2)
    #from our previous calculation
    return (rk4_h2*16-rk4_h)/15


npt=201 #use 200 steps
x0=-20
x1=20
y0=1
x=np.linspace(x0,x1,npt)
y1=np.zeros(npt) #y1 uses rk4_step
y2=np.zeros(npt) #y2 uses rk4_stepd
y1[0]=y0 #starting conditions
y2[0]=y0
for i in range(npt-1):
    h=x[i+1]-x[i]
    y1[i+1]=rk4_step(myfun,x[i],y1[i],h)
    y2[i+1]=rk4_stepd(myfun,x[i],y2[i],h)

#calculate the constant term in real solution
c0 = 1/(np.exp(np.arctan(-20)))
#real solution
truth=c0*np.exp(np.arctan(x))
print('Error for rk4_step is: ', np.std(truth-y1))
print('Error for rk4_stepd is: ',np.std(truth-y2))

plt.clf()
plt.plot(x,np.abs(y1-truth))
plt.plot(x,np.abs(y2-truth))
plt.legend(['rk4_step','rk4_stepd'])
plt.title('rk4 error, ' + repr(npt)+ ' points')
plt.savefig('rk4_err.png')
plt.show()

