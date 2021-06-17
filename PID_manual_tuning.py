# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 09:57:16 2021

@author: alboa
"""

import matplotlib.pyplot as plt
import expanded_ivp_funcs as e_ivp
import numpy as np
from tqdm import tqdm
import itertools
import pandas as pd
import datetime as dt
import math
import Data_prep_4_expanded as dp4e
# Import added vaccine data
# Specify period, overshoot and non-estimating parameters                        

start_day = '2020-12-01'  # start day
simdays = 100
overshoot = 0
beta,phi1,phi2 = [0.17, 0.02, 0.09]
gamma1 = 1/9
gamma2 = 1/7
gamma3 = 1/16
theta = 0.09


t0 = pd.to_datetime(start_day)
overshoot = dt.timedelta(days=overshoot)

# Load data
data = dp4e.Create_dataframe(
    Gamma1=gamma1,
    Gamma2=gamma2,
    forecast=False
)

mp = [gamma1, gamma2, gamma3, theta, phi1, phi2]

T = np.zeros(len(data['R2'])*10)
for i in range(len(data['R2'])):
   T[(i*10):(i*10)+10] = data['R2'][i]/10





#%%
t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
                    mp=mp,
                    T = T,
                    #K = opt_params,
                    K = [0.100,-0.0100,-0.01000],
                    beta_initial = beta,
                    simtime=simdays,
                    stepsize=0.1,
                    method=e_ivp.RK4   
                )
# generate ICU data vector
ICU = []
for i in range(len(t)):
    ICU.append(State_vec[i][3])
    
plt.plot(t,ICU,t,np.ones(len(State_vec))*322)
plt.show
#plt.plot(range(len(beta_vals)),beta_vals)
plt.show