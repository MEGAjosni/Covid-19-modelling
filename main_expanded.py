# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 09:33:45 2021

@author: alboa
"""

import get_data as gd
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import numpy as np

# Get data
s1 = pd.to_datetime('2020-01-27') #start of data
s2 = pd.to_datetime('2021-01-01') #start of simulation
sim_days = 21

I = gd.infect_dict['Test_pos_over_time'][s1 : s2 + dt.timedelta(days=sim_days)]
