import numpy as np
from matplotlib import pyplot as plt

#use the covariance matrix and noise from previous calculation
nparam = 6
cov = np.load('cov_matrix.npy')
N = np.load('sigma_squared.npy')
stuff=np.load('sidebands.npz')
t=stuff['time']
x=stuff['signal']

#get chisq for the three lorentzian fit
def get_chisq(t,x,p):
    y=p[0]/(1.0+((t-p[2])**2)/(p[1]**2)) + \
    p[3]/(1.0+((t-p[2]+p[5])**2)/(p[1]**2)) + \
    p[4]/(1.0+((t-p[2]-p[5])**2)/(p[1]**2))
    chisq = np.sum((x-y)**2/N)
    return chisq

#get the step size using the covariance matrix. Scaled by step_size
def get_step(step_size):
    zero_mean = np.zeros(nparam)
    step = np.random.multivariate_normal(zero_mean,cov)*step_size
    return step

#mostly copied from class PPT
def run_mcmc(t,x,start_pos,nstep,step_size):
    nparam = start_pos.size
    params=np.zeros([nstep,nparam+1])
    params[0,0:-1] = start_pos #save initial params
    cur_chisq = get_chisq(t,x,start_pos)
    params[0,-1] = cur_chisq #save initial chisq
    cur_pos = start_pos.copy()
    for i in range(1,nstep): #loop through nstep
        new_pos=cur_pos+get_step(step_size)
        new_chisq=get_chisq(t,x,new_pos)
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
        params[i,0:-1]=cur_pos #save current params
        params[i,-1]=cur_chisq #save current chisq
    return params

#I tried and this scaling converges fastest
step_size = np.array([5,5,5,5,5,5])
#using an initial guess close to best fit params
pars_start = np.array([1.4,1.7e-5,2e-4,0.1,0.1,5e-5])
nstep=30000
chain=run_mcmc(t,x,pars_start,nstep,step_size)

#print mean and std of params
for i in range(pars_start.size):
    val=np.mean(chain[:,i])
    scat=np.std(chain[:,i])
    print([val,scat])

#plot chisq
plt.plot(chain[:,-1])
plt.savefig('chisq_converge.png')
plt.show()

#plot param
plt.clf()
plt.plot(chain[:,0])
plt.savefig('first_param_conv.png')
plt.show()

plt.clf()
plt.plot(chain[:,1])
plt.savefig('second_param_conv.png')
plt.show()

#plot param fft
plt.clf()
plt.loglog(np.abs(np.fft.rfft(chain[:,0])))
plt.savefig('fft_first_param.png')
plt.show()