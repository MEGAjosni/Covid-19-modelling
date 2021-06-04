# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:22:09 2021

@author: Marcus
"""
import matplotlib.pyplot as plt
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

t, State_vec = e_ivp.simulateSIR(
    X_0=X_0,
    mp=mp,
    T = Activated_vaccines[30:130],
    simtime=simdays,
    stepsize=1,
    method=e_ivp.RK4
)

# format output from e_ivp.simulateSIR to be stackplotted (might be unnessecary :D)
S = [] 
I1 = []
I2 = []
I3 = []
R1 = []
R2 = []
R3 = []

I2_hat = 1000
I3_hat = 1000
error_vals1 = []
for i in range(0,len(t)):
    temp = State_vec[i]
    S.append(temp[0])
    I1.append(temp[1])
    I2.append(temp[2])
    I3.append(temp[3])
    R1.append(temp[4])
    R2.append(temp[5])
    R3.append(temp[6])

    e = max(temp[2]-I2_hat,0)+max(temp[3]-I3_hat,0)
    error_vals1.append(e)

plt.stackplot(t,S,I1,I2,I3,R1,R2,R3,labels = ["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"])
plt.title("Stacked area")
plt.legend(["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"],loc = 'lower left')

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 6])
plt.show()

plt.plot(t,State_vec)
plt.title(" All bins ")
plt.legend(["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"])

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 6])
plt.show()


plt.plot(t,State_vec)
plt.title(" infected and dead ")
plt.legend(["I1", "I2", "I3", "R3"])

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 5])
plt.show()

plt.plot(t,I2,I3)
plt.title("Intensive and respirator")
plt.legend([ "I2: intensive", "I3: Respirator"])

plt.show()
#%% PID control simulation 


t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
    X_0=X_0,
    mp=mp,
    T = Activated_vaccines[30:130],
    K = [-1,-1/1000,-10],
    simtime=simdays,
    stepsize=1,
    method=e_ivp.RK4
)
S_PID = [] 
I1_PID = []
I2_PID = []
I3_PID = []
R1_PID = []
R2_PID = []
R3_PID = []

for i in range(0,len(t)):
    temp = State_vec[i]
    S_PID.append(temp[0])
    I1_PID.append(temp[1])
    I2_PID.append(temp[2])
    I3_PID.append(temp[3])
    R1_PID.append(temp[4])
    R2_PID.append(temp[5])
    R3_PID.append(temp[6])


plt.plot(t,error_vals1,error_vals)
plt.title("Errors (number of people above threshold)")
plt.legend([ "Error with out PID", ""])
plt.show()

I2_threshold = [1000 for i in range(101)]
I3_threshold = [1000 for i in range(101)]

fig, axs = plt.subplots(2)
axs[0].plot(t,I2,t,I3,t,I2_PID,t,I3_PID,t,I2_threshold,t,I3_threshold)
axs[0].legend(["I2","I3","I2 with PID", "I3 with PID" ,"I2 threshold","I3 threshold"])
beta_vals.append(beta_vals[-1])
axs[1].plot(t,beta_vals)
axs[1].legend("Beta Values")
