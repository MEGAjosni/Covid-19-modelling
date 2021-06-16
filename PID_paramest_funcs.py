# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:10:43 2021

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



def estimate_params_expanded_PID(
        X_0,
        data,
        mp, # Known model parameters [gamma1, gamma2, gamma3, theta, phi1, phi2]
        beta_initial,
        simdays,
        precision,
):
    best_beta_avg = -math.inf 
    best_params=[0,0,0]
    T = np.zeros(len(data['R2'])*10)
    for i in range(len(data['R2'])):
        T[(i*10):(i*10)+10] = data['R2'][i]/10
    for k in tqdm(range(precision)):
            Kp = np.linspace(best_params[0] - 100 / (10 ** k), best_params[0] + 100 / (10 ** k), 41)
            Ki = np.linspace(best_params[1] - 100 / (10 ** k), best_params[1] + 100 / (10 ** k), 41)
            Kd = np.linspace(best_params[2] - 100 / (10 ** k), best_params[2] + 100 / (10 ** k), 41)
    
            for comb in itertools.product(Kp,Ki,Kd):
                params = list(comb)
                t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=X_0,
                    mp=mp,
                    T = T,
                    beta_initial = beta_initial,
                    K = params,
                    simtime=simdays,
                    stepsize=0.1,
                    method=e_ivp.RK4
                    
                )

                if max(error_vals) <= 0:
                    if sum(beta_vals)/len(beta_vals) > best_beta_avg:
                        best_params = params
                        best_beta = sum(beta_vals)/len(beta_vals)
            print("\n Current best params: ", best_params, "with lowest beta of: ", best_beta)  
    return best_beta, best_params
                    
                    
                    
                    
# Specify period, overshoot and non-estimating parameters                        

start_day = '2020-12-01'  # start day
simdays = 70
overshoot = 0
beta,phi1,phi2 = [0.17, 0.02, 0.09]
gamma1 = 1/9
gamma2 = 1/7
gamma3 = 1/16
theta = 1/30


t0 = pd.to_datetime(start_day)
overshoot = dt.timedelta(days=overshoot)

# Load data
data = dp4e.Create_dataframe(
    Gamma1=gamma1,
    Gamma2=gamma2,
    s2=t0,
    sim_days=simdays,
    forecast=False
)

mp = [gamma1, gamma2, gamma3, theta, phi1, phi2]

T = np.zeros(len(data['R2'])*10)
for i in range(len(data['R2'])):
   T[(i*10):(i*10)+10] = data['R2'][i]/10




worst_case_beta , opt_params = estimate_params_expanded_PID(
    X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
    data = data,
    simdays = simdays,
    beta_initial = beta,
    mp=mp,
    precision=3
    )
#%%
t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
                    mp=mp,
                    T = T,
                    K = opt_params,
                    beta_initial = beta,
                    simtime=simdays,
                    stepsize=0.1,
                    method=e_ivp.RK4
                    
                )
# generate ICU data vector
ICU = []
for i in range(len(t)):
    ICU.append(State_vec[i][3])
    
plt.plot(t,ICU,t,np.ones(len(State_vec))*322/2)
plt.show
plt.plot(range(len(beta_vals)),beta_vals)
plt.show