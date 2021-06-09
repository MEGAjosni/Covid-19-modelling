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
from tqdm import tqdm

Gamma2 = 1 / 9

# Get data
s1 = pd.to_datetime('2020-01-27')  # start of data
s2 = pd.to_datetime('2021-01-01')  # start of simulation
sim_days = 21

# Forecast vaccine data or actual vaccine data
forecast = True

# Load data
mat = scipy.io.loadmat('data/Inter_data.mat')  # 1st observation march 11 2020
Activated_vaccines = np.loadtxt('vac_data_kalender_14_04_2021.csv')  # 1st observation jan 4th 2021
Data_Infected = gd.infect_dict['Test_pos_over_time'][s1: s2 + dt.timedelta(days=sim_days)]
Data_Dead = gd.infect_dict['Deaths_over_time']  # [s1 : s2 + dt.timedelta(days=sim_days)]
Data_Hospitalized = gd.infect_dict['Newly_admitted_over_time'][s1: s2 + dt.timedelta(days=sim_days)]

# Offsets:
ICU_RESP_Offset = np.zeros(int((pd.to_datetime('2020-03-11') - s1).days))
VAC_Offset = np.zeros(int((pd.to_datetime('2021-01-04') - s1).days))
DEAD_Offset = np.zeros(int((pd.to_datetime('2020-03-11') - s1).days))
HOSPITAL_Offset = np.zeros(int((pd.to_datetime('2020-03-01') - s1).days))

# Vaccination data (R2)
if forecast:
    R2 = np.concatenate((VAC_Offset, Activated_vaccines), axis=0)
    R2 = R2[0:((s2 - s1).days + sim_days)]

# ICU/RESP data (I3)
I3 = np.concatenate((ICU_RESP_Offset, mat['All_Data'][:, 0]), axis=0)
I3 = I3[0:((s2 - s1).days + sim_days)]

# Construct data matrix
N = 5813298

R3 = list(DEAD_Offset)
I2 = list(HOSPITAL_Offset)

S = [N]

for i in tqdm(range((s2 - s1).days + sim_days)):

    # create R3
    if i > len(DEAD_Offset) - 1:
        key = Data_Dead.keys()[0]
        R3.append(R3[i - 1] + Data_Dead[key][i])

    # create I2
    if i > len(HOSPITAL_Offset) - 1:
        I2.append(Data_Hospitalized['Total'][i - int(Gamma2 ** (-1)):i])

    # create S # first let S be equal to N, then subsequently let S be equal to s_-1 plus
    if i != 0:
        # First determine how big S is out of vaccinatable population.
        v_pop = N - I2[i - 1] - I3[i - 1] - R2[i - 1] - R3[i - 1]
        # NOTE MAYBE ADJUST HERE
        temp = S[i - 1] - Data_Infected['NewPositive'][i] - (R2[i - 1] * (S[i - 1] / v_pop))
        S.append(temp)
