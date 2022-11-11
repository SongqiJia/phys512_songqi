import numpy as np
from matplotlib import pyplot as plt

def convolution(f,g):
    f_new = add_zeros(f)
    g_new = add_zeros(g)
    F=np.fft.fft(f_new)
    G=np.fft.fft(g_new)
    H=F*G
    h=np.fft.ifft(H)
    return h

def add_zeros(array):
    N=len(array)
    zeros = np.zeros(N-1)
    new_array = np.concatenate((array,zeros))
    # cut = int(len(new_array)/2)
    # new_array = np.concatenate((new_array[cut+1:],new_array[:cut]))
    return new_array

N=1000
cut=int(N/2)
x=np.linspace(-5,5,N)
y1=np.exp(-0.5*x**2)

arr = convolution(y1,y1)
plt.plot(arr)
plt.show()
