# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 11:12:16 2021

@author: alboa


"""
import tikzplotlib
import pandas as pd
import numpy as np
from get_data import*
import matplotlib.pyplot as plt





# ***** Description *****
#
# Computes an estimate of daily activated vaccines, and compares with actual data
# From ssi using get_data.py.
#
# ***** End *****


# Read constant data from excel file.
location = "C:/Users/alboa/Documents/GitHub/covid-19-modelling/data/Vaccine prognoser (jan-juli).xlsx"
df = pd.read_excel(location,header=1)

# Format relevant columns to np.arrays 
# We also divide by two for the vaccines that require two doses. 
# 
daily_phizer = np.array([df[df.columns[8]]]).transpose()/2
daily_moderna = np.array([df[df.columns[9]]]).transpose()/2
daily_astra = np.array([df[df.columns[10]]]).transpose()/2
daily_johnson = np.array([df[df.columns[11]]]).transpose()
daily_andre = np.array([df[df.columns[12]]]).transpose()/2

# Adds offset to the effect of vaccine, corresponding tothe number of weeks from delivery to fully vaccinated plus one week for full effect of vaccine. 
# value for 'others' category is not known, so its set to 5. 
offsets = [5+1,5+1,int((4+9)/2)+1,1, 5+1] #value for others category is not determined, set to 5. 

# Adds zeros corresponding to the offset, so the effect of each vaccine is consistent with
# vacination time (first data point is still 4th of january )
phizer_active = np.append(np.zeros(offsets[0]*7),daily_phizer)
moderna_active = np.append(np.zeros(offsets[1]*7),daily_moderna)
astra_active = np.append(np.zeros(offsets[2]*7),daily_astra)
johnson_active = np.append(np.zeros(offsets[3]*7),daily_johnson)
andre_active = np.append(np.zeros(offsets[4]*7),daily_andre)



max_points=max(len(phizer_active),len(moderna_active),len(astra_active),len(johnson_active),len(andre_active))


# Extend with zeros so sizes are compatible
max_points=max(len(phizer_active),len(moderna_active),len(astra_active),len(johnson_active),len(andre_active))
phizer_active = np.append(phizer_active,np.zeros(max_points-len(phizer_active)))
moderna_active = np.append(moderna_active,np.zeros(max_points-len(moderna_active)))
astra_active = np.append(astra_active,np.zeros(max_points-len(astra_active)))
johnson_active = np.append(johnson_active,np.zeros(max_points-len(johnson_active)))
andre_active = np.append(andre_active,np.zeros(max_points-len(andre_active)))

# total active vaccines (with individual offsets)
Total_active = phizer_active + moderna_active + astra_active + johnson_active + andre_active



# save Total_active to file for use in model.
np.savetxt('vac_data_kalender_14_04_2021.csv',Total_active,delimiter = ',')



# import vaccination data from ssi using get_data.py
vac_df = vaccine_dict['FaerdigVacc_daekning_DK_prdag']
faerdig_vac_daglig = np.array([vac_df[vac_df.columns[1]]]).transpose()
# first observation is 15th of january

# (Optional) remove zeros for clearer plot.
remove = True
if remove == True:
    phizer_active[phizer_active==0] = 'nan'
    moderna_active[moderna_active==0] = 'nan'
    astra_active[astra_active==0] = 'nan'
    johnson_active[johnson_active==0] = 'nan'
    andre_active[andre_active==0] = 'nan'

######### Generate plots ########

t = np.linspace(0,max_points,max_points)

plt.figure(0)
plt.plot(t[0:len(phizer_active)],phizer_active)
plt.plot(t[0:len(moderna_active)],moderna_active)
plt.plot(t[0:len(astra_active)],astra_active)
#plt.plot(t[0:len(johnson_active)],johnson_active)
plt.plot(t[0:len(andre_active)],andre_active)
plt.plot(t[0:len(Total_active)],Total_active,':',linewidth=2,color='b')


plt.title("Daily activated vaccines")
plt.legend(["Phizer","Moderna","AstraZeneca","Others","Total"])
plt.xlabel("days since 04/01/2021")
plt.ylabel("Active vaccinations")
tikzplotlib.save('Vaccine_plot.tex',axis_height='6cm', axis_width='15cm')
# plots from 4th of january to last known observation of march 


#%%
#vac_data_points = len(faerdig_vac_daglig)
#t_vac_data = np.linspace(10,vac_data_points+9,vac_data_points)
#plt.figure(1)
#plt.plot(t,Total_active)
#plt.plot(t_vac_data,faerdig_vac_daglig)
#plt.title("Daily activated vaccines")
#plt.legend(["Planned","Actual"])
#plt.xlabel("days since 04/01/2021")
#plt.ylabel("Active vaccinations")
#sns.despine()
















