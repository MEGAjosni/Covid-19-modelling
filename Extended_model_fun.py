# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:31:45 2021

@author: alboa
"""
import math

def derivative_expanded(X, mp):
    # *** Description ***
    # Computes the derivative of X using model parameters

    # ************* Input *************
    #
    # X : State vector
        # S1 : susceptible at risk
        # S2 : susceptible non risk
        # I1 : regular infected
        # I2 : ICU infected
        # I3 : respirator infected
        # R1 : regular recovered
        # R2 : vacinated
        # R3 : dead
        
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
    # ************* Output *************
    #
    # dX : derivative of state vector X. 
 
    # Extract data
    beta1, beta2, gamma, t1, t2, t3, t4, theta1, theta2, phi1, phi2, N = mp
    S1, S2, I1, I2, I3, R1, R2, R3 = X

    dX = [
        -(beta1 * (I1 / N) + t1)*S1, #dS1/dt
        -(beta2 * (I1 / N) + t2)*S2, #dS2/dt
        beta1 *(I1 / N) * S1 + beta2 * (I1 / N) * S2 - (gamma + t3 + phi1) * I1, #dI1/dt
        phi1 * I1 - (gamma + theta1 + phi2) * I2, #dI2/dt 
        phi2 * I2 - (gamma + theta2)* I3, #dI3/dt
        gamma * (I1 + I2 + I3) - t4 * R1, #dR1/dt
        t1 * S1 + t2 * S2 + t3 * I1 + t4 * R1, #dR2/dt
        theta1 * I1 + theta2 * I2, #dR3/dt
        
    ]

    return dX
