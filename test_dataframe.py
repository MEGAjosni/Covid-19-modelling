# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 11:05:20 2021

@author: alboa
"""

import pandas as pd
import Data_prep_4_expanded as dp4e
import numpy as np


# Get data

s2 = pd.to_datetime('2021-01-01')  # start of simulation
Gamma2 = 1/9
Gamma1 = 1/7






data1 = dp4e.Create_dataframe(Gamma1,Gamma2,forecast = True, early = True)
data2 = dp4e.Create_dataframe(Gamma1,Gamma2,forecast = True, early = False)
data3 = dp4e.Create_dataframe(Gamma1,Gamma2)