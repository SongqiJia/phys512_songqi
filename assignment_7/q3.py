import numpy as np
from matplotlib import pyplot as plt

u=np.linspace(0,1,2001)
u=u[1:]

v=-2*u*np.log(u)
print('max v is ',v.max())

plt.clf()
plt.plot(u,v,'k')
plt.savefig('exp_region.png')
plt.show()


N=1000000
u=np.random.rand(N)
#.86 seems to be max value of v
v=np.random.rand(N)*0.75
r=v/u
accept=u<np.exp(-0.5*r)
exp=r[accept]

accept_rate = len(exp)/len(r)
print('accept rate is:', accept_rate)

bins=np.linspace(1,50,5001)
cents=0.5*(bins[1:]+bins[:-1])
aa,bb=np.histogram(exp,bins)
aa=aa/aa.sum()
pred=np.exp(-cents)
pred=pred/pred.sum()
plt.clf()
plt.plot(cents,aa,'*')
plt.plot(cents,pred,'r')
plt.savefig('exp_ratio_uniform.png')
plt.show()
