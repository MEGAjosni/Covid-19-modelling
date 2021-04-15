# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:22:09 2021

@author: Marcu
"""
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

X_0 = [6*10**7*0.8,6*10**7*0.2,]
    # mp : model parameters
        # beta1 : Infection rate parameter in group at risk
        # beta2 : Infection rate parameter in group not at risk
        # gamma1 : Rate of recovery for infected
        # gamma2 : Rate of recovery for ICU
        # gamma3 : Rate of recovery for respirator
        # t1 : Vaccination rate in group at risk
        # t2 : Vaccination rate in group not at risk
        # t3 : Vaccination rate of infected group
        # t4 : Vaccination rate of recovered group
        # theta1 : Death rate at ICU
        # theta2 : Death rate in respirator
        # phi1 : Rate of ICU from infected
        # phi2 : Rate of respirator from ICU
        # N : Population
mp = []

t, SIR = ivp.simulateSIR(
    X_0=X_0,
    mp=mp,
    simtime=simdays,
    method=ivp.RK4
)