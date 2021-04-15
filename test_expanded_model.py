# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:22:09 2021

@author: Marcu
"""
import matplotlib.pyplot as plt
import Extended_model_fun as emf
simdays = 100
    # X : State vector
        # S1 : susceptible at risk
        # S2 : susceptible non risk
        # I1 : regular infected
        # I2 : ICU infected
        # I3 : respirator infected
        # R1 : regular recovered
        # R2 : vacinated
        # R3 : dead

X_0 = [(6*10**6-800000)*0.2,
       (6*10**6-800000)*0.8,
       30000,
       400,
       300,
       300000,
       500000,
       700]
    # mp : model parameters
        # beta1 : Infection rate parameter in group at risk
        # beta2 : Infection rate parameter in group not at risk
        # gamma1 : Rate of recovery for infected
        # gamma2 : Rate of recovery for ICU
        # gamma3 : Rate of recovery for respirator
        # t : Vaccination rate
        # theta1 : Death rate at ICU
        # theta2 : Death rate in respirator
        # phi1 : Rate to ICU from infected
        # phi2 : Rate to respirator from ICU
        # N : Population
mp = [0.15,
      0.22,
      (1/7),
      (1/20),
      (1/20),
      15000,
      0.2*(1/20),
      0.2*(1/20),
      0.001,
      0.3,
      (6*10**6)]

t, State_vec = emf.simulateSIR(
    X_0=X_0,
    mp=mp,
    simtime=simdays,
    method=emf.RK4
)

plt.plot(t, State_vec)
plt.title("ud af røven parametre")
plt.legend(["S1","S2","I1","I2","I3","R1","R2 (vac)", "R3"])

plt.ylabel("Number of people")
plt.ylim([0, 6*10**5])