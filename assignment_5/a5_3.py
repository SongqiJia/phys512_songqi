from tracemalloc import start
import numpy as np
import camb
from matplotlib import pyplot as plt
import time

#use the covariance matrix and noise from previous calculation

planck=np.loadtxt('COM_PowerSpect_CMB-TT-full_R3.01.txt',skiprows=1)
ell=planck[:,0]
spec=planck[:,1]
errs=0.5*(planck[:,2]+planck[:,3])
tt_lenth = len(spec)
# start_pos = np.load("best_fit_pars.npy")
start_pos = np.load("final_chain.npy")[-1,:]
start_pos = start_pos[1:]
cov = np.load('cov_matrix.npy')
nparam = len(start_pos)

print(cov)
print(start_pos)
print(nparam)

#copied from planck_likelihood.py
def get_spectrum(pars,lmax=3000):
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
    tt=tt[:tt_lenth]
    return tt

#get chisq for the three lorentzian fit
def get_chisq(p):
    model = get_spectrum(p)
    resid=spec-model
    chisq=np.sum( (resid/errs)**2)
    return chisq

#get the step size using the covariance matrix. Scaled by step_size
def get_step(step_size):
    zero_mean = np.zeros(nparam)
    step = np.random.multivariate_normal(zero_mean,cov)*step_size
    return step

#mostly copied from class PPT
def run_mcmc(start_pos,nstep,step_size):
    nparam = start_pos.size
    params=np.zeros([nstep,nparam+1])
    params[0,1:] = start_pos #save initial params
    cur_chisq = get_chisq(start_pos)
    params[0,0] = cur_chisq #save initial chisq
    cur_pos = start_pos.copy()
    for i in range(1,nstep): #loop through nstep
        new_pos=cur_pos+get_step(step_size)
        new_chisq=get_chisq(new_pos)
        if new_chisq<cur_chisq:
            accept=True
        else:
            delt=new_chisq-cur_chisq
            prob=np.exp(-0.5*delt) #accept probability
            if np.random.rand()<prob:
                accept=True
            else:
                accept=False
        if accept:
            cur_pos=new_pos
            cur_chisq=new_chisq
        params[i,0]=cur_chisq #save current chisq
        params[i,1:]=cur_pos #save current params
        if i%100 ==0:
            print("finished step ",i,"/",nstep)
    return params

#I tried and this scaling converges fastest
step_size = np.array([0.4,0.4,0.4,0.4,0.4,0.4])
#using an initial guess close to best fit params
pars_start = start_pos
nstep=10000
start_time = time.time()
chain=run_mcmc(pars_start,nstep,step_size)
end_time = time.time()
print("time for running is", end_time-start_time)

#print mean and std of params
for i in range(pars_start.size):
    val=np.mean(chain[:,i])
    scat=np.std(chain[:,i])
    print([val,scat])



# for i in range(nparam+1):
#     plt.clf()
#     plt.plot(chain[:,i])
#     plt.savefig(f'conv_{i}.png')

# for i in range(1,nparam+1):
#     plt.clf()
#     plt.loglog(np.abs(np.fft.rfft(chain[:,i])))
#     plt.savefig(f'fft_{i}_param.png')

np.save("chain_cont_2",chain)
old_chain = np.load("final_chain.npy")[:-1,:]
new_chain = np.vstack((old_chain,chain))
# new_chain = chain
np.save("final_chain_2",new_chain)

np.savetxt("planck_chain.txt", new_chain)