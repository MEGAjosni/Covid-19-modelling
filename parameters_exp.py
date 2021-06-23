# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:33:45 2021

@author: alboa
"""
import scipy.io
import get_data as gd
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import numpy as np
import Data_prep_4_expanded as Data


N = 5800000
Gamma1 = 1/9
Gamma2 = 1/7
Gamma3 = 1/16
t0 = pd.to_datetime('2021-02-20')
sim_days = 21
t1 = t0 + dt.timedelta(days=sim_days)

X = Data.Create_dataframe(Gamma1 = Gamma1,
                     Gamma2 = Gamma2,
                     forecast = False)


#estimate theta
dR3 = np.array(X['R3'][t0+dt.timedelta(days=1):t1])-np.array(X['R3'][t0:t0 + dt.timedelta(days=sim_days-1)])
I3 = np.array(X['I3'][t0:t0 + dt.timedelta(days=sim_days-1)])

thetas = dR3/I3
theta = thetas.mean()



"""
#estimate phi2
dI3 = np.array(X['I3'][t0+dt.timedelta(days=1):t1])-np.array(X['I3'][t0:t0 + dt.timedelta(days=sim_days-2)])
I2 = np.array(X['I2'][t0:t0 + dt.timedelta(days=sim_days-2)])

phi2s = (dI3+(Gamma3+theta)*I3)/I2
phi2 = phi2s.mean()

#estimate phi1
dI2 = np.array(X['I2'][t0+dt.timedelta(days=1):t1])-np.array(X['I2'][t0:t0 + dt.timedelta(days=sim_days-2)])
I1 = np.array(X['I1'][t0:t0 + dt.timedelta(days=sim_days-2)])

phi1s = (dI2 + (Gamma2+phi2)*I2)/I1
phi1 = phi1s.mean()

#estimate tau
taus = dI2 = np.array(X['R2'][t0+dt.timedelta(days=1):t1])-np.array(X['R2'][t0:t0 + dt.timedelta(days=sim_days-2)])
tau = taus.mean()

#estimate betas
dS = np.array(X['S'][t0+dt.timedelta(days=1):t1])-np.array(X['S'][t0:t0 + dt.timedelta(days=sim_days-2)])
S = np.array(X['S'][t0:t0 + dt.timedelta(days=sim_days-2)])
R1 = np.array(X['R1'][t0:t0 + dt.timedelta(days=sim_days-2)])
dI1 = np.array(X['I1'][t0+dt.timedelta(days=1):t1])-np.array(X['I1'][t0:t0 + dt.timedelta(days=sim_days-2)])

betas = -(dS/S+tau/(S+I1+R1))*N/I1

#parameters expanded model w.o beta
mp_nobeta = [Gamma1,Gamma2,Gamma3,tau, theta, phi1, phi2]
"""


# =============================================================================
# 
# # Get data
# s1 = pd.to_datetime('2020-01-27') #start of data
# s2 = pd.to_datetime('2021-01-01') #start of simulation
# sim_days = 21
# 
# # Forecast vaccine data or actual vaccine data
# forecast = True
# 
# #Load data
# mat = scipy.io.loadmat('data/Inter_data.mat') #1st observation march 11 2020
# Activated_vaccines = np.loadtxt('vac_data_kalender_14_04_2021.csv')# 1st observation jan 4th 2021
# Data_Infected = gd.infect_dict['Test_pos_over_time'][s1 : s2 + dt.timedelta(days=sim_days)]
# Data_Dead = gd.infect_dict['Deaths_over_time'][s1 : s2 + dt.timedelta(days=sim_days)]
# Data_Hospitalised = gd.infect_dict['Newly_admitted_over_time']
# 
# 
# # Offsets:
# ICU_RESP_Offset = np.zeros(int((pd.to_datetime('2020-03-11')-s1).days))
# VAC_Offset = np.zeros(int((pd.to_datetime('2021-01-04')-s1).days))
# DEAD_Offset = np.zeros(int((pd.to_datetime('2020-03-11')-s1).days))
# HOSPITAL_Offset = np.zeros(int((pd.to_datetime('2020-03-01')-s1).days))
# 
# #Vaccination data (R2)
# if forecast == True:
#     R2 = np.concatenate((VAC_Offset,Activated_vaccines),axis = 0)
#     R2 = R2[0:((s2-s1).days+sim_days)]
# 
# #ICU/RESP data (I3)
# I3 = np.concatenate((ICU_RESP_Offset,mat['All_Data'][:,0]),axis = 0)
# I3 = I3[0:((s2-s1).days+sim_days)]
# 
# 
# 
# #Construct data matrix 
# N = 5813298 
# 
# R3 = DEAD_Offset
# I2 = HOSPITAL_Offset
# 
# S = [N]
# 
# for i in range((s2-s1).days+sim_days):
#     print(i)
#     
#     # create R3
#     if i > len(DEAD_Offset):
#         np.concatenate((R3,R3[i-1]+Data_Dead['Antal_døde']))
#     
#     # create I2
#     if i > len(HOSPITAL_Offset):
#         np.concatenate((I2,Data_Dead['Antal_døde'][i-int((Gamma2)**(-1)):i]))
#     
#     # create S # first let S be equal to N, then subsequently let S be equal to s_-1 plus
#     if i != 0:
#         #First determine how big S is out of vaccinatable population. 
#         v_pop = N-I2[i-1]-I3[i-1]-R2[i-1]-R3[i-1] 
#         # NOTE MAYBE ADJUST HERE
#         temp = S[i-1]-Data_Infected['NewPositive'][i]-(R2[i-1]*(S[i-1]/v_pop))
#         S.append(temp)
#     
#     
# =============================================================================
