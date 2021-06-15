# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:10:43 2021

@author: alboa

"""

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


def estimate_params_expanded_PID(
        X_0,
        data,
        mp,  # Known model parameters [beta,gamma1, gamma2, gamma3, theta, phi1, phi2]
        simdays,
        precision=2
):
    best_beta = -math.inf 
    best_params=[0,0,0]
    T = np.zeros(len(data['R2'])*10)
    for i in range(len(data['R2'])):
        T[(i*10):(i*10)+10] = data['R2'][i]/10
    for k in tqdm(range(precision)):
            Kp = np.linspace(best_params[0] - 1 / (10 ** k), best_params[0] + 1 / (10 ** k), 21)
            Ki = np.linspace(best_params[1] - 1 / (10 ** k), best_params[1] + 1 / (10 ** k), 21)
            Kd = np.linspace(best_params[2] - 1 / (10 ** k), best_params[2] + 1 / (10 ** k), 21)
    
            for comb in itertools.product(Kp,Ki,Kd):
                params = list(comb)
                t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=X_0,
                    mp=mp,
                    T = T,
                    K = params,
                    simtime=simdays,
                    stepsize=0.1,
                    method=e_ivp.RK4
                    
                )
                print('hey')
                if max(error_vals) <= 0:
                    
                    if min(beta_vals) > best_beta:
                        best_params = params
                        best_beta = min(beta_vals)
                        
    return best_beta, best_params
                    
                    
                    
                    
                    
                                

start_day = '2020-12-01'  # start day
simdays = 21
overshoot = 7
beta,phi1,phi2 = [0.195738, 0.010765, 0.002307]
gamma1 = 1/7
gamma2 = 1/14
gamma3 = 1/21
theta = 1/15


t0 = pd.to_datetime(start_day)
overshoot = dt.timedelta(days=overshoot)

# Load data
data = dp4e.Create_dataframe(
    Gamma1=gamma1,
    Gamma2=gamma2,
    s2=t0,
    sim_days=100,
    forecast=False
)

mp = [beta,gamma1, gamma2, gamma3, theta, phi1, phi2]

worst_case_beta , opt_params = estimate_params_expanded_PID(
    X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
    data = data,
    simdays = simdays,
    mp=mp,
    precision=4
)

