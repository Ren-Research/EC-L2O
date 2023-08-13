import numpy as np 

# Data preparation
def smooth(x,window_len=15,window='hanning'):
    
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    
    traject_length = x.shape[0]
    
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    si = int(window_len/2)
    return y[si:si+traject_length]

def sliding_windows(price_data, seq_length):
    x = []
    for i in range(len(price_data)-seq_length+1):
        x.append(price_data[i:(i+seq_length),:])
    
    return np.array(x)

def load_power_shortage(df, eta_w = 0.3, eta_s = 0.1, c_dr = 0.2,
            rho_air = 1.23, S_array = 100, A_swept = 5000,
            P_short = 20000, time_step = 2, smooth_traject=True):
    v_wind = df["Wind Speed"].values
    p_wind = 0.5*eta_w*rho_air*A_swept*(v_wind**3)

    temperature = df["Temperature"].values
    radiation = df["GHI"].values
    p_solar = eta_s*S_array*radiation
    solar_correction = 1 - 0.05*(temperature-25)
    p_solar_corrected = p_solar*solar_correction

    p_total = p_wind + p_solar_corrected
    
    # p_total = p_total
    P_short_new = np.maximum(P_short - p_total, 0)
    if smooth_traject: 
        P_short_new = smooth(P_short_new)

    data_preprocessed=0.8*P_short_new/np.max(P_short_new)
    
    return data_preprocessed[::time_step]


