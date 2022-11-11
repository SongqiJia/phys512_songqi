import numpy as np
from matplotlib import pyplot as plt
import h5py



def read_template(filename):
    dataFile=h5py.File(filename,'r')
    template=dataFile['template']
    tp=template[0]
    tx=template[1]
    return tp,tx
def read_file(filename):
    dataFile=h5py.File(filename,'r')
    dqInfo = dataFile['quality']['simple']
    qmask=dqInfo['DQmask'][...]

    meta=dataFile['meta']
    #gpsStart=meta['GPSstart'].value
    gpsStart=meta['GPSstart'][()]
    #print meta.keys()
    #utc=meta['UTCstart'].value
    utc=meta['UTCstart'][()]
    #duration=meta['Duration'].value
    duration=meta['Duration'][()]
    #strain=dataFile['strain']['Strain'].value
    strain=dataFile['strain']['Strain'][()]
    dt=(1.0*duration)/len(strain)

    dataFile.close()
    return strain,dt,utc

fname_h = []
fname_l = []
fname_temp = []
event_name = []

fname_h.append('./LOSC_Event/H-H1_LOSC_4_V2-1126259446-32.hdf5')
fname_l.append('./LOSC_Event/L-L1_LOSC_4_V2-1126259446-32.hdf5')
fname_temp.append('./LOSC_Event/GW150914_4_template.hdf5')
event_name.append('GW150914')

fname_h.append('./LOSC_Event/H-H1_LOSC_4_V2-1128678884-32.hdf5')
fname_l.append('./LOSC_Event/L-L1_LOSC_4_V2-1128678884-32.hdf5')
fname_temp.append('./LOSC_Event/LVT151012_4_template.hdf5')
event_name.append('LVT151012')

fname_h.append('./LOSC_Event/H-H1_LOSC_4_V2-1135136334-32.hdf5')
fname_l.append('./LOSC_Event/L-L1_LOSC_4_V2-1135136334-32.hdf5')
fname_temp.append('./LOSC_Event/GW151226_4_template.hdf5')
event_name.append('GW151226')

fname_h.append('./LOSC_Event/H-H1_LOSC_4_V1-1167559920-32.hdf5')
fname_l.append('./LOSC_Event/L-L1_LOSC_4_V1-1167559920-32.hdf5')
fname_temp.append('./LOSC_Event/GW170104_4_template.hdf5')
event_name.append('GW170104')


def smooth_vector(vec,sig):
    n=len(vec)
    x=np.arange(0,n,1.0)
    x[n//2:]=x[n//2:]-n
    kernel=np.exp(-0.5*((x/sig)**2)) #make a Gaussian kernel
    kernel=kernel/kernel.sum()
    vecft=np.fft.rfft(vec)
    kernelft=np.fft.rfft(kernel)
    vec_smooth=np.fft.irfft(vecft*kernelft) #convolve the data with the kernel
    return vec_smooth

def window_fun(n):
    y=np.ones(n)
    cos_n = n//5
    cos_x = np.linspace(-np.pi/2,np.pi/2,cos_n)
    cos_y = np.cos(cos_x)
    cut = cos_n//2
    y[:cut]=cos_y[:cut]
    y[-cut:]=cos_y[-cut:]
    return y

# plt.plot(window_fun(1000))
# plt.show()

def ligo_analysis(n):

    hname=fname_h[n]
    lname=fname_l[n]
    print('reading file ',hname)
    strain_h,dt_h,utc_h=read_file(hname)
    strain_l,dt_l,utc_l=read_file(lname)
    ename = event_name[n]

    template_name=fname_temp[n]
    tp,tx=read_template(template_name)

    length = len(strain_h)

    win = window_fun(length)


    noise_ft_h=np.fft.fft(win*strain_h)
    noise_smooth_h=smooth_vector(np.abs(noise_ft_h)**2,10)
    noise_smooth_h=noise_smooth_h[:len(noise_ft_h)//2+1] #will give us same length


    noise_ft_l=np.fft.fft(win*strain_l)
    noise_smooth_l=smooth_vector(np.abs(noise_ft_l)**2,10)
    noise_smooth_l=noise_smooth_l[:len(noise_ft_l)//2+1] #will give us same length

    tobs=dt_h*length
    dnu=1/tobs
    nu=np.arange(len(noise_smooth_h))*dnu
    nu[0]=0.5*nu[1]

    Ninv_h=1/noise_smooth_h
    Ninv_h[nu>1500]=0
    Ninv_h[nu<20]=0

    Ninv_l=1/noise_smooth_l
    Ninv_l[nu>1500]=0
    Ninv_l[nu<20]=0

    template_ft_white_h=np.fft.rfft(tp*win)*np.sqrt(Ninv_h)
    data_ft_white_h=np.fft.rfft(strain_h*win)*np.sqrt(Ninv_h)
    template_white_h = np.fft.irfft(template_ft_white_h)
    data_white_h = np.fft.irfft(data_ft_white_h)
    rhs_h = np.fft.irfft(data_ft_white_h*np.conj(template_ft_white_h))
    lhs_h = np.dot(template_white_h,template_white_h)
    m_h = rhs_h/lhs_h
    m_h_shifted = np.fft.fftshift(m_h)

    template_ft_white_l=np.fft.rfft(tp*win)*np.sqrt(Ninv_l)
    data_ft_white_l=np.fft.rfft(strain_l*win)*np.sqrt(Ninv_l)
    template_white_l = np.fft.irfft(template_ft_white_l)
    data_white_l = np.fft.irfft(data_ft_white_l)
    rhs_l = np.fft.irfft(data_ft_white_l*np.conj(template_ft_white_l))
    lhs_l = np.sum(template_white_l**2)
    m_l = rhs_l/lhs_l
    m_l_shifted = np.fft.fftshift(m_l)

    signal_height_h = np.max(np.abs(m_h))
    num_noise_h = np.std(m_h)
    ana_noise_h = 1/np.sqrt(length*lhs_h)
    num_SNR_h = signal_height_h/num_noise_h
    ana_SNR_h = signal_height_h/ana_noise_h

    signal_height_l = np.max(np.abs(m_l))
    num_noise_l = np.std(m_l)
    ana_noise_l = 1/np.sqrt(length*lhs_l)
    num_SNR_l = signal_height_l/num_noise_l
    ana_SNR_l = signal_height_l/ana_noise_l

    num_SNR_total = 0.5*(np.sqrt(num_SNR_h)+ np.sqrt(num_SNR_l))**2
    ana_SNR_total = 0.5*(np.sqrt(ana_SNR_h)+ np.sqrt(ana_SNR_l))**2

    weight_h = np.abs(template_ft_white_h)**2
    weight_l = np.abs(template_ft_white_l)**2
    half_v_h = np.average(nu,weights = weight_h)
    half_v_l = np.average(nu,weights = weight_l)

    t_diff = np.abs(np.argmax(np.abs(m_h))-np.argmax(np.abs(m_l)))*dt_h
    t_diff_var = np.sqrt(2)*10*dt_h
    angle = 3e5*t_diff/1e3 + np.pi
    angle_var = 3e5*t_diff_var/1e3


    fig = plt.figure(figsize = (12,10),constrained_layout = True)
    axes = fig.subplots(2,2).flatten()
    axes[0].loglog(nu,noise_smooth_h)
    axes[0].set_title('Hanford smoothed noise')
    axes[0].set_xlim(20,1500)
    axes[1].loglog(nu,noise_smooth_l)
    axes[1].set_title('Livingston smoothed noise')
    axes[1].set_xlim(20,1500)
    axes[2].plot(m_h_shifted)
    axes[2].set_title('Hanford matched filer')
    axes[3].plot(m_l_shifted)
    axes[3].set_title('Livingston matched filer')
    plt.suptitle(f'Event is {ename}')
    plt.savefig(f'ana_{ename}.png')
    # plt.show()
    # plt.plot(m_h)
    # plt.xlim([np.argmax(np.abs(m_h))-50,np.argmax(np.abs(m_h))+50])
    # plt.show()
    print('\nEvent is ',ename)
    print('Hanford Numerical SNR =',num_SNR_h)
    print('Hanford Analytic SNR =',ana_SNR_h)
    print('Livingston Numerical SNR =',num_SNR_l)
    print('Livingston Analytic SNR =',ana_SNR_l)
    print('Total Numerical SNR =',num_SNR_total)
    print('Total Analytic SNR =',ana_SNR_total)
    print('Hanford Half-weight frequency is: ', half_v_h)
    print('Livingston Half-weight frequency is: ', half_v_l)
    print('Time difference is: ',t_diff,'+-',t_diff_var)
    print('Angle of the source is: ',angle,'+-',angle_var)


for i in range(4):
    ligo_analysis(i)
