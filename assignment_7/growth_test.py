import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(0,5.501)
y1=np.exp(-x)
y2= x**-2

# plt.plot(x,y1)
# plt.plot(x,y2)
# plt.legend(['exp','power'])
# plt.show()

print(np.min(y2-y1))
# plt.plot(x,y2-y1)
# plt.show()