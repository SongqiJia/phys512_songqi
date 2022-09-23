import numpy as np
from matplotlib import pyplot as plt

x_true = np.linspace(0.5,1,10001) #x array from 0.5 to 1
#renormalize x array to be from -1 to 1 for cheb fit
x_renorm = x_true*4-3
y = np.log2(x_true)
cheb_coeff = np.polynomial.chebyshev.chebfit(x_renorm,y,30)
print(cheb_coeff)
#we only need 8 coeff to have error less than 1e-6
nuse = 8
cheb_coeff_cut = cheb_coeff[:nuse]

def mylog2(x):
    #stop calculation if any of the x is smaller or equal to zero
    if np.any(x<=0):
        print('natural log only accept positive number')
    else:
        #use frexp to get mantissa and exponent
        mantissa, exponent = np.frexp(x)
        #use the previously calculated cheb coefficients
        global cheb_coeff_cut
        #renomalize the mantissa for chebval
        mantissa_renorm = mantissa*4-3
        mantissa_log = np.polynomial.chebyshev.chebval(mantissa_renorm,cheb_coeff_cut)
        #from our previous calculation
        log2_val = mantissa_log + exponent
        #convert to natural log
        natural_log = log2_val/np.log2(np.exp(1))
        return natural_log

#test the error of mylog2
y_test_val_true = np.linspace(1,10,1001)
x_test_val = np.exp(y_test_val_true)
y_test_val_cheb = mylog2(x_test_val)
err = np.abs(y_test_val_true-y_test_val_cheb)
print('the std of error is: ', np.std(err))
plt.clf()
plt.xscale('log')
plt.plot(x_test_val,err)
plt.savefig('question_3.png')
plt.show()

