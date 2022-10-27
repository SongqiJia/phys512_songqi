import numpy as np
from matplotlib import pyplot as plt

# old_chain = np.load("chain_old.npy")
# new_chain = np.load("chain_cont.npy")
# old_chain = old_chain[:-1,:]
# final_chain = np.vstack((old_chain,new_chain))
# np.save("final_chain",final_chain)



chain = np.load("final_chain_tau.npy")

#print mean and std of params
for i in range(1,7):
    val=np.mean(chain[:,i])
    scat=np.std(chain[:,i])
    print([val,scat])

#I learned about this kind of plot from David Cai
fig = plt.figure(figsize = (15,5),constrained_layout = True)
axes = fig.subplots(2,3).flatten()
for i in np.arange(1,7):
    axes[i-1].plot(chain[:, i])
    axes[i-1].set_title(f"{i}")
fig.savefig("param_conv_tau.png")

fig = plt.figure(figsize = (15,5),constrained_layout = True)
axes = fig.subplots(2,3).flatten()
for i in np.arange(1,7):
    axes[i-1].loglog(np.abs(np.fft.rfft(chain[:, i])))
    axes[i-1].set_title(f"{i}")
fig.savefig("param_fft_tau.png")


val = np.zeros(6)
scat = np. zeros(6)
for i in range(1,7):
    val[i-1]=np.mean(chain[:,i])
    scat[i-1]=np.std(chain[:,i])


h = val[0]/100
d_h = scat[0]/100
baryon = val[1]
d_baryon = scat[1]
dmd = val[2]
d_dmd = scat[2]

h_array = chain[:,1]/100
h_sq_array = h_array**2
baryon_array = chain[:,2]
d_h2_baryon = np.cov(h_sq_array,baryon_array)[0,1]
dmd_array = chain[:,3]
d_h2_dmd = np.cov(h_sq_array,dmd_array)[0,1]

omega_b = baryon/h**2
omega_c = dmd/h**2

d_omega_b_sq = d_baryon**2/h**4 + 4*baryon**2*d_h**2/h**8 - 2*d_h2_baryon*baryon/h**6
d_omega_c_sq = d_dmd**2/h**4 + 4*dmd**2*d_h**2/h**8 - 2*d_h2_dmd*dmd/h**6

omega_numda = 1 - omega_b - omega_c
d_omega_numda = d_omega_b_sq + d_omega_c_sq

print("Dark Energy is",omega_numda,"+-",np.sqrt(d_omega_numda))