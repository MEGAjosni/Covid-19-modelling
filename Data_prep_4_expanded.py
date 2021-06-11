# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:33:45 2021

@author: alboa
"""
import scipy.io
import os
import get_data as gd
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import numpy as np
from tqdm import tqdm

def Create_dataframe(Gamma1,Gamma2,s2,sim_days,forecast):
    
    # ***** Description *****
#
#   Constructs time indexed dataframe of the 7 variables of the expanded SIR 
#   model
#   
#   Inputs : 
#       Gamma1 : fracton, rate of recovery from infection
#       Gamma2 : fraction, rate of recovery from hospitalization
#       s2: date, Start of simulation
#       Sim_days : int, number of days of simulation
#       forecast : boolean,
#   Output : 
#       X : dataframe, Containing [S,I1,I2,I3,R1,R2,R3], indexed by date. 
#
# ***** End *****

    
    #Load data
    s1 = pd.to_datetime('2020-01-27')  # start of data
    
    data_dir = os.getcwd() + '\\data\\vaccinationsdata-dashboard-covid19-10062021-fii5\\Vaccine_DB\\FaerdigVacc_daekning_DK_prdag.csv'
    
    
    mat = scipy.io.loadmat('data/Inter_data.mat') #1st observation march 11 2020
    Activated_vaccines = np.loadtxt('vac_data_kalender_14_04_2021.csv')# 1st observation jan 4th 2021
    Data_Infected = gd.infect_dict['Test_pos_over_time'][s1 : s2 + dt.timedelta(days=sim_days)]
    Data_Dead = gd.infect_dict['Deaths_over_time'][s1 : s2 + dt.timedelta(days=sim_days)]
    Data_Hospitalized = gd.infect_dict['Newly_admitted_over_time'][s1 : s2 + dt.timedelta(days=sim_days)]
    # manual "get_data" for updated vaccination data
    Data_Vaccinated =  pd.read_csv(filepath_or_buffer=data_dir,sep=',',thousands='.',decimal=',',engine='python') #[s1 : s2 + dt.timedelta(days=sim_days)]
    date_name = Data_Vaccinated.columns[0]

    # If data is dateindexed convert to datetime64[ns]

    format1 = sum([str(Data_Vaccinated[date_name][0])[i] == 'yyyy-mm-dd'[i] for i in range(10)]) == 2
   

    

        # Remove totals from bottom
    j = len(Data_Vaccinated[date_name]) - 1
    while True:
        if len(str(Data_Vaccinated[date_name][j])) == 10:

            format1 = sum([str(Data_Vaccinated[date_name][j])[i] == 'yyyy-mm-dd'[i] for i in range(10)]) == 2
            

            if format1:
                break
        else:
            Data_Vaccinated = Data_Vaccinated.drop(index=[j])
        j += -1

    Data_Vaccinated[date_name] = pd.to_datetime(Data_Vaccinated[date_name], dayfirst=True)
    Data_Vaccinated = Data_Vaccinated.set_index(pd.DatetimeIndex(Data_Vaccinated[date_name]))

    # As list is now indexed with dates, get rid of the datecolumn
    Data_Vaccinated = Data_Vaccinated.drop(columns=[date_name])

    
    # Offsets:
    ICU_RESP_Offset = np.zeros(int((pd.to_datetime('2020-03-11') - s1).days))
    VAC_Offset_Forecast = np.zeros(int((pd.to_datetime('2021-01-04') - s1).days))
    VAC_Offset_Empirical = np.zeros(int((pd.to_datetime('2021-01-15') - s1).days))
    DEAD_Offset = np.zeros(int((pd.to_datetime('2020-03-11') - s1).days))
    HOSPITAL_Offset = np.zeros(int((pd.to_datetime('2020-03-01') - s1).days))
    
    
    
    # Initialize data lists
    N = 5813298
    S = [N]
    I1 = []
    I2 = list(HOSPITAL_Offset)
    I3 = np.concatenate((ICU_RESP_Offset, mat['All_Data'][:, 0]), axis=0)
    I3 = list(I3[0:((s2 - s1).days + sim_days)])
    R1 = []
    X = []
    if forecast:
        R2 = np.concatenate((VAC_Offset_Forecast, Activated_vaccines), axis=0)
        R2 = list(R2[0:((s2 - s1).days + sim_days)])
    if not forecast:
        R2 = list(np.concatenate((VAC_Offset_Empirical, Data_Vaccinated['Antal færdigvacc. personer']), axis=0))
    R3 = list(DEAD_Offset)
    
    
    # Some variables need transformation/calculation
    for i in tqdm(range((s2 - s1).days + sim_days)):
        if i < int(Gamma1 ** (-1)):
            I1.append(sum(Data_Infected['NewPositive'][0:i]))
        else : 
            I1.append(sum(Data_Infected['NewPositive'][i-int(Gamma1 ** (-1)):i]))
        # create R3
        if i > len(DEAD_Offset) - 1:
            key = Data_Dead.keys()[0]
            R3.append(R3[i - 1] + Data_Dead[key][i-len(DEAD_Offset)])
    
        # create I2
        if i > len(HOSPITAL_Offset) - 1:
            I2.append(sum(Data_Hospitalized['Total'][i - int(Gamma2 ** (-1)):i]))
    
        # create S # first let S be equal to N, then subsequently let S be equal to s_-1 plus
        if i != 0:
            # First determine how big S is out of vaccinatable population.
            v_pop = N - I2[i - 1] - I3[i - 1] - R2[i - 1] - R3[i - 1]
            # NOTE MAYBE ADJUST HERE
            temp = S[i - 1] - Data_Infected['NewPositive'][i] - (R2[i - 1] * (S[i - 1] / v_pop))
            S.append(temp)
            
        R1.append(N-S[i]-R2[i]-R3[i]-I1[i]-I2[i]-I3[i])   
        X.append([S[i],I1[i],I2[i],I3[i],R1[i],R2[i],R3[i]])
    datelist = pd.date_range(s1, periods=(s2 - s1).days + sim_days).tolist()
    X = pd.DataFrame(X,columns=['S','I1','I2','I3','R1','R2','R3'],index = datelist)
    return X
        
            