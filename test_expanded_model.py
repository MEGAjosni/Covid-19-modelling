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
      0.3,
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
for i in range(0,len(t)):
    temp = State_vec[i]
    S.append(temp[0])
    I1.append(temp[1])
    I2.append(temp[2])
    I3.append(temp[3])
    R1.append(temp[4])
    R2.append(temp[5])
    R3.append(temp[6])

plt.stackplot(t,S,I1,I2,I3,R1,R2,R3,labels = ["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"])
plt.title("Out of butt parameters - Stacked area")
plt.legend(["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"],loc = 'lower left')

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 6])
plt.show()

plt.plot(t,State_vec)
plt.title("Out of butt parameters")
plt.legend(["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"])

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 6])
plt.show()
