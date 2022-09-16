import numpy as np
import scipy.interpolate as intp
from matplotlib import pyplot as plt

dat = np.loadtxt('lakeshore.txt')

def lakeshore(V,data):
    temp_ys = data[:,0] #first column of data is temperature values
    v_xs = data[:,1] #second column of data is voltage values
    #using scipy.interpolate.interp1d to do a cubic spline interpolation 
    #interp1d returns a function which can take in new points and return new values based on the interpolation
    spline = intp.interp1d(v_xs,temp_ys,'cubic') 
    interp_temp = spline(V)
    return interp_temp

#test the interpolation scheme
plot_xs = np.linspace(0.1,1.64,1000)
plot_ys = lakeshore(plot_xs,dat)

plt.clf()
plt.plot(dat[:,1],dat[:,0],'*')
plt.plot(plot_xs,plot_ys)
plt.savefig('spine_interp.png')
plt.show()

#Bootstrapping to find the estimated error. Mostly stolen from tutorial 2's bootstrap_interp.py
rng = np.random.default_rng(seed=12345) # Create a random number generator object
N_resamples = 20 # How many times to resamples
N_samples = 100 # How many points to resample with

temp_y = dat[:,0]
v_x = dat[:,1]
arr_size = temp_y.size
gen_pts = [] # To hold the interpolated y values for each run.
for i in range(N_resamples):
    # Making a list of all the indices for our x interpolation points.
    # Note!!!! This is different from tutorial code. I made sure that we keep the first and last point so we are indeed 
    # interpolating, rather than possible extrapolating. 
    indices = list(range(1,arr_size-1))  
    to_interp = rng.choice(indices,size=N_samples-2,replace=False) # Choosing N_samples-2 (98) indices of values to use for new interp
    to_interp.sort() # Make sure x is increasing
    to_interp = np.insert(to_interp,0,0) # Add index to the first point
    to_interp = np.append(to_interp,arr_size-1) # Add index to the last point
    new_interpolation = intp.interp1d(v_x[to_interp],temp_y[to_interp]) # Choosing N_samples (100) indices of values to use for new interp
    interpolated_ys = new_interpolation(plot_xs) # Interpolate y-values at plot points and save.
    gen_pts.append(interpolated_ys)

gen_pts = np.array(gen_pts) # Convert list of lists to 2D array
stds = np.std(gen_pts,axis=0) # Calculate std dev at each x across resamplings

plt.clf()
plt.plot(plot_xs,stds)
plt.show()

error2 = np.mean(stds) # Take mean of std for overall error
error2_std = np.std(stds) # Get std dev of that error.
print(f"{error2 = :.3e} +/- {error2_std:.3e}")