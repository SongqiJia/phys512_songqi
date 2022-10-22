import numpy as np
from matplotlib import pyplot as plt
import camb

#copied from planck_likelihood.py
def get_spectrum(pars,t,lmax=3000):
    #print('pars are ',pars)
    H0=pars[0]
    ombh2=pars[1]
    omch2=pars[2]
    tau=pars[3]
    As=pars[4]
    ns=pars[5]
    pars=camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=0.06,omk=0,tau=tau)
    pars.InitPower.set_params(As=As,ns=ns,r=0)
    pars.set_for_lmax(lmax,lens_potential_accuracy=0)
    results=camb.get_results(pars)
    powers=results.get_cmb_power_spectra(pars,CMB_unit='muK')
    cmb=powers['total']
    tt=cmb[:,0]    #you could return the full power spectrum here if you wanted to do say EE
    tt= tt[2:]
    tt=tt[:len(t)]
    return tt

#Take the derivative of a function. The function fun must take 
# an array of parameters and a t array as input. param is the parameter values 
# we want to take derivative at. param should have the same length
# as the parameter array that fun takes.
def fun_deriv(fun,param,t):
    grad=np.zeros([t.size,param.size])
    n_param = np.size(param)
    dx = param*((1e-16)**(1/3))
    for i in range(n_param):
        param_new = np.copy(param)
        #calculate the two-sided derivative of the ith parameter
        param_new[i] = param[i]-dx[i]
        fun_minus = fun(param_new,t)
        param_new[i] = param[i]+dx[i]
        fun_plus = fun(param_new,t)
        grad[:,i] = (fun_plus - fun_minus)/(2*dx[i])
    return grad



#use numerical derivation to calculate grad matrix
def calc_camb(p,t):
    y=get_spectrum(p,t)
    grad=fun_deriv(get_spectrum,p,t)
    return y,grad


planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])

#use part 1 for initial guess
p0=np.asarray([60,0.02,0.1,0.05,2.00e-9,1.0])
# p0=np.asarray( [69, 0.022, 0.12, 0.06, 2.1e-9, 0.95]) #another initial guess

#copied from newton.py
p=p0.copy()
for j in range(5):
    pred,grad=calc_camb(p,ell)
    resid=spec-pred
    r=np.matrix(resid).transpose()
    grad=np.matrix(grad)

    noise_inverse = np.matrix(np.diag(1/errs))
    lhs=grad.transpose()*noise_inverse*grad
    rhs=grad.transpose()*noise_inverse*r
    dp=np.linalg.pinv(lhs)*(rhs)

    # u,s,v=np.linalg.svd(grad,0)
    # unu = u.T@noise_inverse@u
    # dp=v.T@np.matrix(np.diag(1/s))@np.linalg.inv(unu)@u.T@noise_inverse@r

    for jj in range(p.size):
        p[jj]=p[jj]+dp[jj]
    chisq=np.sum((resid/errs)**2)
    print(p,chisq)


#same as in linear case as discussed above
residual = spec-pred
par_errs=np.sqrt(np.diag(np.linalg.inv(lhs)))
print(par_errs)
f = open("planck_fit_params.txt", "a")
f.write(f"The best fit params are: {p}")
f.write(f"The erros are: {par_errs}")
f.write('\n')
f.close()


cov = np.linalg.inv(lhs)
chisq=np.sum( (residual/errs)**2)
print("chisq is ",chisq," for ",len(spec)-len(p0)," degrees of freedom.")
np.save('cov_matrix',cov)