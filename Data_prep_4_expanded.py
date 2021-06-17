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


def Create_dataframe(
        Gamma1: float = 1/9,  # Fraction, rate of recovery from infection
        Gamma2: float = 1/7,  # Fraction, rate of recovery from hospitalization
        forecast: bool = False
) -> pd.core.frame.DataFrame:
    # ***** Description *****
    #
    #   Constructs time indexed dataframe of the 7 variables of the expanded SIR
    #   model
    #
    #   Inputs :
    #       Gamma1 : fracton, rate of recovery from infection
    #       Gamma2 : fraction, rate of recovery from hospitalization
    #       t0: date, Start of simulation
    #       sim_days : int, number of days of simulation
    #       forecast : boolean,
    #   Output :
    #       X : dataframe, Containing [S,I1,I2,I3,R1,R2,R3], indexed by date.
    #
    # ***** End *****

    # Load data

    # Define dates
    t0 = pd.to_datetime('2020-01-27')  # Pandemic start
    t1 = pd.to_datetime('2021-05-31')  # Newest data

    # *****************
    # >>> Load data <<<
    # *****************

    # *** This has wierd data ***
    Activated_vaccines = pd.read_csv('vac_data_kalender_14_04_2021.csv', engine='python')  # 1st observation jan 4th 2021

    infect_keys = list(gd.infect_dict.keys())
    # >>>  infect_keys indices  <<<
    # [0]  Antigentests_pr_dag,
    # [1]  Cases_by_age,
    # [2]  Cases_by_sex,
    # [3]  Deaths_over_time,
    # [4]  Municipality_cases_time_series,
    # [5]  Municipality_tested_persons_time_series,
    # [6]  Municipality_test_pos,
    # [7]  Newly_admitted_over_time,
    # [8]  plejehjem_ugeoversigt,
    # [9]  Region_summary,
    # [10] Rt_cases_2021_03_02,
    # [11] Rt_cases_2021_06_01,
    # [12] Test_pos_over_time,
    # [13] Test_pos_over_time_antigen,
    # [14] Test_regioner

    vaccine_keys = list(gd.vaccine_dict.keys())
    # >>> vaccine_keys indices <<<
    # [0]  FaerdigVacc_daekning_DK_prdag
    # [1]  FaerdigVacc_kommune_dag
    # [2]  FaerdigVacc_region_dag
    # [3]  FoersteVacc_kommune_dag
    # [4]  FoersteVacc_region_dag
    # [5]  Noegletal_kommunale_daily
    # [6]  Noegletal_regionale_daily
    # [7]  PaabegVacc_daek_DK_prdag
    # [8]  Vaccinationer_kommuner_befolk
    # [9]  Vaccinationer_regioner_befolk
    # [10] Vaccinationer_region_aldgrp_koen
    # [11] Vaccinationsdaekning_kommune
    # [12] Vaccinationsdaekning_nationalt
    # [13] Vaccinationsdaekning_region
    # [14] Vaccinationstyper_regioner
    # [15] Vaccinations_Daekning_region_pr_dag

    # Load data

    Data_Infected = (gd.infect_dict[infect_keys[12]]['NewPositive'][t0: t1] +
                    gd.infect_dict[infect_keys[13]]['NewPositive'][t0: t1]).fillna(0)
    Data_Dead = gd.infect_dict[infect_keys[3]][t0: t1]
    Data_Hospitalized = gd.infect_dict[infect_keys[7]][t0: t1]
    Data_Vaccinated = gd.vaccine_dict[vaccine_keys[0]]

    DK_vaccine_keys = list(Data_Vaccinated.keys())
    # >>>  DK_vaccine_keys indices  <<<
    # [0]  geo,
    # [1]  Antal færdigvacc. personer,
    # [2]  Antal borgere,
    # [3]  Færdigvacc. (%),
    # [4]  Kumuleret antal færdigvacc.

    # Offsets:

    mat = scipy.io.loadmat('data/Inter_data.mat')  # 1st observation march 11 2020
    ICU_RESP = pd.DataFrame(data=mat['All_Data'][:, 0], index=pd.date_range(start='2020-03-11', periods=len(mat['All_Data'][:, 0])),
               columns=['Resp']).astype(int)


    # Define a timedelta of 1 day
    td1 = dt.timedelta(days=1)

    # ******************************************
    # >>> Create dataframe with state values <<<
    # ******************************************

    # Initialize DataFrame
    dates = pd.date_range(start=t0, end=t1)
    X = pd.DataFrame(data=0, index=dates, columns=['S', 'I1', 'I2', 'I3', 'R1', 'R2', 'R3'])
    N = 5800000  # DK population
    X['S'][t0] = N

    # Fill in state values
    for day in dates:
        if forecast:
            pass
            # # *** This does not work atm ***
            # R2 = np.concatenate((VAC_Offset_Forecast, Activated_vaccines), axis=0)
            # R2 = list(R2[0:((t0 - t1).days + sim_days)])
        else:
            X['R2'][day] = sum(Data_Vaccinated[DK_vaccine_keys[1]][day: day])

        X['I3'][day] = sum(ICU_RESP['Resp'][day: day])

        v_pop = N - X['I2'][day] - X['I3'][day] - X['R2'][day] - X['R3'][day]
        X['S'][day + td1] = X['S'][day] - \
                            sum(Data_Infected[day + td1: day + td1]) - \
                            X['R2'][day] * X['S'][day] / v_pop

        X['I2'][day] = sum(Data_Hospitalized['Total'][day - dt.timedelta(days=int(1 / Gamma2)): day])
        X['I1'][day] = sum(Data_Infected[day - dt.timedelta(days=int(1 / Gamma1)): day]) - X['I2'][day]
        X['R3'][day] = sum(Data_Dead[Data_Dead.keys()[0]][t0: day])

    X['R1'] = N - (X['S'] + X['R2'] + X['R3'] + X['I1'] + X['I2'] + X['I3'])

    return X

print(Create_dataframe())