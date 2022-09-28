import numpy as np
from matplotlib import pyplot as plt

#load data from txt file
dat = np.loadtxt('dish_zenith.txt')

x = dat[:,0]
y = dat[:,1]
z = dat[:,2]

ndata = len(x)

#set up our A matrix
A=np.empty([ndata,4])
A[:,0] = ((x**2)+(y**2))
A[:,1] = x
A[:,2] = y
A[:,3] = 1

#same as in class. Ignore noise for now. 
#m = (A^T*A)^{-1}*(A^T*z)
lhs=A.T@A
rhs=A.T@z
m=np.linalg.inv(lhs)@rhs
pred=A@m

#estimate parameter error
#assume noise is naively the mean between real z and predicted z
#variance = (A^T*N^{-1}*A)^{-1}
N=np.mean((z-pred)**2)
par_errs=np.sqrt(N*np.diag(np.linalg.inv(lhs)))

a = m[0]
x0 = -m[1]/(2*a)
y0 = -m[2]/(2*a)
z0 = m[3]-a*(x0**2)-a*(y0**2)

param = [a,x0,y0,z0]

print('Predicted parameters of the new model are: ', m)
print('Error bars are: ', par_errs)
print('Predicted parameters of the old model are: ', param)

f = 1/(4*a)
print("Focal length is: ", f)
print('Error of focal length is: ',f*(1/4)*par_errs[0]/m[0])


