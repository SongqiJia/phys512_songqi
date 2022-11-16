import numpy as np
from matplotlib import pyplot as plt

def power_law(n):
    k=2
    q = np.random.rand(n)**(1/(1-k))
    return q


n=10000000
t=power_law(n)


bins=np.linspace(1,50,5001)
aa,bb=np.histogram(t,bins)
aa=aa/aa.sum()

cents=0.5*(bins[1:]+bins[:-1])
pred=cents**-2
pred=pred/pred.sum()
plt.clf()
plt.plot(cents,aa,'*')
plt.plot(cents,pred,'r')
plt.savefig('power_law.png')
plt.show()

mypower=cents**-2
myexp=np.exp(-cents)
y=t**-2*np.random.rand(n)

plt.clf()
plt.plot(t,y,'.')
plt.plot(cents,mypower,'b')
plt.plot(cents,myexp,'r')
plt.xlim(1,50)
plt.savefig('power_vs_exp.png')
plt.show()

accept=y<np.exp(-t)
t_use=t[accept]
accept_rate = len(t_use)/len(t)
print('accept rate is:', accept_rate)

aa,bb=np.histogram(t_use,bins)
aa=aa/aa.sum()
pred=np.exp(-cents)
pred=pred/pred.sum()
plt.clf()
plt.plot(cents,aa,'*')
plt.plot(cents,pred,'r')
plt.savefig('exp_from_power.png')
plt.show()