import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate

#define function to integrate according to the solution to Griffith 2.7
def fun_int(x,z):
    return (z-x)/((1+(z**2)-(2*z*x))**(3/2))


#define integrator that uses Simpsons rule
def integrator(z):
    x = np.linspace(-1,1,1001) #integrate from -1 to 1
    y = fun_int(x,z)
    dx=np.median(np.diff(x))
    #integrate according to Simpson rules
    ints_simp=dx/3.0*(y[0]+y[-1]+4*np.sum(y[1::2])+2*np.sum(y[2:-1:2]))
    return ints_simp


simp_val = []
quad_val = []
z_arr = np.linspace(0,5,1001)
for z in z_arr: #note z = 1 is included in this linspace
    simp_temp = integrator(z)
    quad_temp,err = integrate.quad(fun_int,-1,1,args=(z,))
    simp_val = np.append(simp_val,simp_temp)
    quad_val = np.append(quad_val,quad_temp)


plt.clf()
plt.plot(z_arr,simp_val)
plt.plot(z_arr,quad_val)
plt.ylim(-2, 5) #set y-axis limits so that the singularity does not make the plot look too bad
plt.legend(['Simpsons rule','scipy quad'])
plt.savefig('question_1.png')
plt.show()