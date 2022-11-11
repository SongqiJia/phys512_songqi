import numpy as np
from matplotlib import pyplot as plt

def correlation(f,g):
    F=np.fft.fft(f)
    G=np.fft.fft(g)
    H=F*np.conj(G)
    h=np.fft.ifft(H)
    return h

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

def corr_gaussian(dx):
    y_shift = shift_fun(y,dx)
    cor = correlation(y,y_shift)
    return cor
    
for i in range(0,500,50):
    h=corr_gaussian(i)
    cut = int(N/2)
    h = np.concatenate((h[cut+1:],h[:cut]))
    plt.plot(h)
    plt.savefig(f"corr_gaussian_{i}.png")
    plt.show()
