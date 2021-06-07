# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 15:10:43 2021

@author: alboa
"""
import expanded_ivp_funcs as e_ivp
import numpy as np
# Import added vaccine data
Activated_vaccines = np.loadtxt('vac_data_kalender_14_04_2021.csv') # 1st observation is january 4th

simdays = 100
# X : State vector
# S : susceptible at risk
# I1 : regular infected
# I2 : ICU infected
# I3 : respirator infected
# R1 : regular recovered
# R2 : vaccinated
# R3 : dead

X_0 = [(6 * 10 ** 6 - 800000),
       30000,
       400,
       300,
       300000,
       500000,
       700]
# mp : model parameters
# beta : Infection rate parameter
# gamma1 : Rate of recovery for infected
# gamma2 : Rate of recovery for ICU
# gamma3 : Rate of recovery for respirator
# theta1 : Death rate at ICU
# theta2 : Death rate in respirator
# phi1 : Rate to ICU from infected
# phi2 : Rate to respirator from ICU
# N : Population
mp = [0.22,
      (1 / 7),
      (1 / 20),
      (1 / 20),
      0.2 * (1 / 20),
      0.2 * (1 / 20),
      0.001,
      0.5,
      (6 * 10 ** 6)]

Min_betas = []
count = 0
for i in np.linspace(-2,0,50):
        for j in np.linspace(-2/1000,0,50):
            for k in np.linspace(-2*80,0,50):
                mp[0] = 0.22
                t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=X_0,
                    mp=mp,
                    T = Activated_vaccines[30:130],
                    K = [i,j,k],
                    simtime=simdays,
                    stepsize=1,
                    method=e_ivp.RK4
                    
                )
                
                count += 1 
                if  count % 1000 == 0:
                    print("Completed %: ", count * 100/(50**3))
                if max(error_vals) <= 0:
                    Min_betas.append([min(beta_vals),i,j,k])
                    
opt_parameters = []
best_beta = 0
for i in range(len(Min_betas)):
    if Min_betas[i][0] >= best_beta:
        best_beta = Min_betas[i][0]
        opt_parameters = [Min_betas[i][1],Min_betas[i][2],Min_betas[i][3]]
                    
                    
                    
                    
                                