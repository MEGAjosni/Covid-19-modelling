# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:05:20 2021

@author: alboa
"""

import pandas as pd
import Data_prep_4_expanded as dp4e


# Get data

s2 = pd.to_datetime('2021-01-01')  # start of simulation
Gamma2 = 1/9
Gamma1 = 1/9
sim_days = 100
forecast = False



X = dp4e.Create_dataframe(Gamma1,Gamma2 ,s2, sim_days, forecast)