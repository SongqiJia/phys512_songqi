import numpy as np
from matplotlib import pyplot as plt

def shift_fun(y,dx):
    N=len(y)
    k=np.arange(len(y))
    ramp=np.exp(2*np.pi*1J*k*dx/N)
    yft=np.fft.fft(y)
    yft_shift=yft*ramp
    y_shift=np.fft.ifft(yft_shift)
    return y_shift

N=1000
x=np.linspace(-5,5,N)
y=np.exp(-0.5*x**2)

dx=N/2

y_shift=shift_fun(y,dx)

plt.plot(x,y)
plt.plot(x,y_shift)
plt.show()
