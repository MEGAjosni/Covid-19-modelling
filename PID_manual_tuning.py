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
import data_prep_S3I3R as dp4e
import tikzplotlib
# Import added vaccine data
# Specify period, overshoot and non-estimating parameters                        

start_day = '2020-12-01'  # start day
simdays = 100
overshoot = 0
beta,phi1,phi2 = [0.13, 0.005, 0.02]
gamma1 = 1/9
gamma2 = 1/7
gamma3 = 1/21
theta = 0.166


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
#opt_params = [round(-0.0004979999999999999,8),round( -6.7999999999999976e-06,8),round( -0.0068839999999999995,8)]
opt_params = [-0.0001, -0.0003, 0.1] 
t, State_vec_PID ,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
                    mp=mp,
                    T = T,
                    #K = opt_params,
                    #K = [-0.00005,-0.000001,-0.000007],
                    #K = [-0.0000008, -0.00000003, -0.0003],
                    #K = [-0.0006,-0.00002,-0.009],
                    K =  opt_params,
                    beta_initial = beta,
                    simtime=simdays,
                    stepsize=0.1,
                    method=e_ivp.RK4   
                )

t,SIR=e_ivp.simulateSIR(
    X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
    mp=[beta]+mp,
    T = T,
    simtime=simdays,
    stepsize=0.1,
    method=e_ivp.RK4 
    
    )

# generate ICU data vector
ICU_PID = []
ICU = []
for i in range(len(t)):
    ICU_PID.append(State_vec_PID[i][3])
    ICU.append(SIR[i][3])
    
custom_tics = pd.date_range(start = t0,periods=100+1).strftime('%d/%m-%Y')    
fig, ax = plt.subplots()
ax2 = ax.twinx()    
ax.plot(t,ICU_PID,t,ICU)
ax.plot(t,np.ones(len(State_vec_PID))*322,linestyle = 'dashed')
ax2.plot(np.linspace(0,100,len(beta_vals)),beta_vals,color = 'r')
ax.set_ylim(0,500)
ax2.set_ylim(0,0.4) 
avg = round(sum(beta_vals)/len(beta_vals),5)
title = "PID 2.0, average beta: " + str(avg)
plt.title(title)
ax.set_xlabel("Days since start,    Parameters: " + r'Kp= $' + "{:.6f}".format(
    opt_params[0]) + ", " + r'Ki = $' + "{:.6f}".format(opt_params[1])+ ", " + r'kd = $' + "{:.6f}".format(opt_params[2]))
ax.set_ylabel("Number of people in ICU ")
ax2.set_ylabel("Beta values")
ax.legend(['With PID', 'Without PID', "Threshold"],loc="upper left")
ax2.legend(['Beta'],loc="upper right")
ax.set_xticklabels(custom_tics)
ax.tick_params(axis='x', rotation=15)
#tikzplotlib.save('PID_realistisk    .tex')
plt.pause(0.5)
plt.show
sum(beta_vals)/len(beta_vals)
