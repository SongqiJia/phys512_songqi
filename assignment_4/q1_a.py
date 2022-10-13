import numpy as np
from matplotlib import pyplot as plt

#copied from newton.py in class repo. changed to be same as problem param
def calc_lorentz(p,t):
    y=p[0]/(1.0+((t-p[2])**2)/(p[1]**2))
    grad=np.zeros([t.size,p.size])
    #now differentiate w.r.t. all the parameters
    grad[:,0]=1.0/(1.0+((t-p[2])**2)/(p[1]**2))
    grad[:,1]=(2*p[0]*p[1]*((t-p[2])**2))/(((t-p[2])**2+p[1]**2)**2)
    grad[:,2]=(2*p[0]*p[1]**2*(t-p[2]))/(((t-p[2])**2+p[1]**2)**2)
    return y,grad

#Take the derivative of a function. The function fun must take 
# an array of parameters and a t array as input. param is the parameter values 
# we want to take derivative at. param should have the same length
# as the parameter array that fun takes.
def fun_deriv(fun,param,t):
    grad=np.zeros([t.size,param.size])
    n_param = np.size(param)
    dx = 1e-6
    for i in range(n_param):
        param_new = np.copy(param)
        #calculate the two-sided derivative of the ith parameter
        param_new[i] = param[i]-dx
        fun_minus = fun(param_new,t)
        param_new[i] = param[i]+dx
        fun_plus = fun(param_new,t)
        grad[:,i] = (fun_plus - fun_minus)/(2*dx)
    return grad

def lorentz(p,t):
    y=p[0]/(1.0+((t-p[2])**2)/(p[1]**2))
    return y

def calc_lorentz_2(p,t):
    y=lorentz(p,t)
    grad = fun_deriv(lorentz,p,t)
    return y,grad

#p[0] = a, p[1] = w, p[2] = t0, p[3] = b
#p[4] = c, p[5] = dt
def tri_lorentz(p,t):
    y=p[0]/(1.0+((t-p[2])**2)/(p[1]**2)) + \
    p[3]/(1.0+((t-p[2]+p[5])**2)/(p[1]**2)) + \
    p[4]/(1.0+((t-p[2]-p[5])**2)/(p[1]**2))
    return y

#use numerical derivation to calculate grad matrix
def calc_lorentz_3(p,t):
    y=tri_lorentz(p,t)
    grad=fun_deriv(tri_lorentz,p,t)
    return y,grad

stuff=np.load('sidebands.npz')
t=stuff['time']
x=stuff['signal']

# plt.clf()
# plt.plot(t,x,'.')
# plt.savefig('raw_data.png')
# plt.show()

#from the raw data graph guess initial values
# p0=np.array([1.4,0.0001,0.0002]) #starting guess, close but not exact
p0=np.array([1.42,1.79e-5,1.92e-4,0.3,0.3,5e-5])

#copied from newton.py
p=p0.copy()
for j in range(5):
    # pred,grad=calc_lorentz(p,t)
    # pred,grad=calc_lorentz_2(p,t)
    pred,grad=calc_lorentz_3(p,t)
    r=x-pred
    err=(r**2).sum()
    r=np.matrix(r).transpose()
    grad=np.matrix(grad)

    lhs=grad.transpose()*grad
    rhs=grad.transpose()*r
    dp=np.linalg.inv(lhs)*(rhs)
    for jj in range(p.size):
        p[jj]=p[jj]+dp[jj]
    print(p,err)

plt.clf()
plt.plot(t,x,'.')
plt.plot(t,pred)
plt.savefig('tri_lorentz_fit.png')
plt.show()

#same as in linear case as discussed above
residual = x-pred
N=np.mean((residual)**2)
par_errs=np.sqrt(N*np.diag(np.linalg.inv(lhs)))
print(N, par_errs)

plt.clf()
plt.plot(t,residual)
plt.savefig('residual.png')
plt.show()
