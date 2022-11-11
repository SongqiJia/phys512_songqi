import numpy as np
from matplotlib import pyplot as plt


N=1000
cut=int(N/2)
x = np.asarray(range(N))
k0=1.5
y1=np.sin(2*np.pi*k0*x/N)

k = np.asarray(range(N))

dft_ana = 1/(2J)*((1-np.exp(-2*np.pi*1J*(k-k0)))/(1-np.exp(-2*np.pi*1J*(k-k0)/N))
-(1-np.exp(-2*np.pi*1J*(k+k0)))/(1-np.exp(-2*np.pi*1J*(k+k0)/N)))

dft_fft = np.fft.fft(y1)

# plt.plot(dft_ana)
# plt.show
# plt.plot(dft_ana-dft_fft)
# plt.show()

win = 0.5 - 0.5*np.cos(2*np.pi*x/N)

win_y = win*y1
# plt.plot(np.fft.fft(win_y))
# plt.show()
print(np.fft.fft(win))
