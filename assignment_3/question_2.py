import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt

#get time translation coeffs
day_to_year = 1/365.25 #radioactivity uses Julian year
hour_to_day = 1/24
min_to_hour = 1/60
sec_to_min = 1/60
hour_to_year = hour_to_day * day_to_year
min_to_year = min_to_hour * hour_to_year
sec_to_year = sec_to_min * min_to_year

#create life time of years
lifetime=np.zeros(14)
lifetime[0] = 4.468e+9
lifetime[1] = 24.10*day_to_year
lifetime[2] = 6.70*hour_to_year
lifetime[3] = 245500
lifetime[4] = 75380
lifetime[5] = 1600
lifetime[6] = 3.8235*day_to_year
lifetime[7] = 3.10*min_to_year
lifetime[8] = 26.8*min_to_year
lifetime[9] = 19.9*min_to_year
lifetime[10] = (164.3e-6)*sec_to_year
lifetime[11] = 22.3
lifetime[12] = 5.015
lifetime[13] = 138.376*day_to_year

def fun(x,y):
    #set up the ODE system
    global lifetime
    dydx=np.zeros(len(lifetime)+1)
    #U238 decay
    dydx[0]=-y[0]/lifetime[0]
    #decay into middle decay products, then themselves decay
    for i in np.arange(1,len(lifetime)):
        dydx[i]=y[i-1]/lifetime[i-1]-y[i]/lifetime[i]
    #decay into Pb206
    dydx[14]=y[13]/lifetime[13]
    return dydx

#use integrate.solve_ivp with Radau method to solve the ODE system implicitly
def solve_decay(x0,x1):
    y0 = np.zeros(15)
    y0[0] = 1 #start with pure U238
    ans_stiff=integrate.solve_ivp(fun,[x0,x1],y0,method='Radau')
    return ans_stiff

ans_utopb = solve_decay(1,1e+10)
t = ans_utopb.t
y_U238 = ans_utopb.y[0,:]
y_Pb206 = ans_utopb.y[-1,:]
y_U234 = ans_utopb.y[3,:]
y_Th230 = ans_utopb.y[4,:]

#true value if U238 decays instantly into Pb206
y_U238_true = 0.5**(t/lifetime[0])
y_Pb206_true = 1-y_U238_true


plt.clf()
plt.loglog(t,y_Pb206/y_U238)
plt.plot(t,y_Pb206_true/y_U238_true)
plt.legend(['scipy.integrate.solve_ivp','true value'])
plt.savefig('U238_to_Pb206.png')
plt.show()

plt.clf()
plt.loglog(t,y_U234/y_Th230)
plt.savefig('U234_to_Th230.png')
plt.show()



    
