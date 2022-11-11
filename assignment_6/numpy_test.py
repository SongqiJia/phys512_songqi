import numpy as np
from matplotlib import pyplot as plt
n=100000
x=np.arange(n)
x[n//2:]=x[n//2:]-n

kernel=np.exp(-0.5*x**2/10**2) #make a Gaussian kernel
plt.plot(kernel)
plt.show()

# y = np.exp(-0.5*((x[50000]/10)**2))
# print(y)
print('value at midpoint is:',x[50000])
print('type of the array is:',type(x[50000]))
print('squared value is:',x[50000]**2)
print('true square value is:',(-50000)**2)

# x=-50000
# print(np.exp(-0.5*((x**2)/(10**2))))
# print(x**2)