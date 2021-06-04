# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:31:45 2021

@author: alboa
"""
import math


def derivative_expanded(X, mp, t):
    # *** Description ***
    # Computes the derivative of X using model parameters

    # ************* Input *************
    #
    # t : added vaccinations 
    #
    # X : State vector
    # S : susceptible
    # I1 : regular infected
    # I2 : ICU infected
    # I3 : respirator infected
    # R1 : regular recovered
    # R2 : vaccinated
    # R3 : dead

    # mp : model parameters
    # beta : Infection rate parameter 
    # gamma1 : Rate of recovery for infected
    # gamma2 : Rate of recovery for ICU
    # gamma3 : Rate of recovery for respirator
    # theta1 : Death rate at ICU
    # theta2 : Death rate in respirator
    # phi1 : Rate of ICU from infected
    # phi2 : Rate of respirator from ICU
    # N : Population
    # ************* Output *************
    #
    # dX : derivative of state vector X. 

    # Extract data
    beta, gamma1, gamma2, gamma3, theta1, theta2, phi1, phi2, N = mp
    S, I1, I2, I3, R1, R2, R3 = X

    dX = [
        -((beta * I1)/ N + (t / (S + I1 + R1))) * S,  # dS/dt
        (beta * I1 / N) * S - (gamma1 + phi1 +(t / (S + I1 + R1))) * I1,# dI1/dt
        phi1 * I1 - (gamma2 + theta1 + phi2) * I2,  # dI2/dt
        phi2 * I2 - (gamma3 + theta2) * I3,  # dI3/dt
        gamma1 * I1 + gamma2 * I2 + gamma3 * I3 - (t /(S + I1 + R1)) * R2,  # dR1/dt
        t, # dR2/dt
        theta1 * I1 + theta2 * I2,  # dR3/dt

    ]

    return dX


def RK4(
        X_k: list,  # Values of the expanded SIR at time t_k
        mp: list,  # Model parameters [beta, gamma, N]
        t: int, # added vacinations
        stepsize: float = 1# t_kp1 - t_k
        
):
    # *** Description ***
    # Uses Rung Kutta 4 to solve ODE system numerically.

    # *** Output ***
    # X_kp1 [list]:         Values of SIR at time t_kp1

    N = range(len(X_k))

    K_1 = derivative_expanded(X_k, mp,t)
    K_2 = derivative_expanded([X_k[i] + 1 / 2 * stepsize * K_1[i] for i in N], mp,t)
    K_3 = derivative_expanded([X_k[i] + 1 / 2 * stepsize * K_2[i] for i in N], mp,t)
    K_4 = derivative_expanded([X_k[i] + stepsize * K_3[i] for i in N], mp,t)

    X_kp1 = [X_k[i] + stepsize / 6 * (K_1[i] + 2 * (K_2[i] + K_3[i]) + K_4[i]) for i in N]

    return X_kp1

def PID_cont(X,mp,d,e_total,e_prev,K):
    I2_hat = 1000
    I3_hat = 1000
    e = max(X[2]-I2_hat , X[3]-I3_hat)
    e_total = e_total + e
    
    if d % 7 == 0:
        try:
            mp[0] = mp[0] * (0.8+0.4/(1+math.exp(-((K[0]* e  ) + K[1]*e_total + K[2]*(e-e_prev)))))
        except: OverflowError

    if mp[0] < 0:
        mp[0] = 0
    if mp[0] > 0.24:
        mp[0] = 0.24

    return mp[0],e,e_total
        
   
def simulateSIR(
        X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
        mp: list,  # Model parameters [beta, gamma, N]
        T: list, # Total added vaccinations
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 1,  # t_kp1 - t_k
        method=RK4  # Numerical method to be used [function]
):
    # *** Description ***
    # Simulate SIR-model.

    # *** Output ***
    # t [list]:             All points in time simulated
    # SIR [nested list]:    Values of SIR at all points in time t

    SIR = [X_0]

    t = [i * stepsize for i in range(int(simtime / stepsize) + 1)]

    for i in range(int(simtime / stepsize)):
        SIR.append(method(SIR[i], mp,T[i], stepsize))
        

    return t, SIR

def simulateSIR_PID(
        X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
        mp: list,  # Model parameters [beta, gamma, N]
        T: list, # Total added vaccinations
        K: list, # parameters for penalty function
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 1,  # t_kp1 - t_k
        method=RK4  # Numerical method to be used [function]
       
):
    # *** Description ***
    # Simulate SIR-model.

    # *** Output ***
    # t [list]:             All points in time simulated
    # SIR [nested list]:    Values of SIR at all points in time t

    SIR = [X_0]

    t = [i * stepsize for i in range(int(simtime / stepsize) + 1)]
    e_total = 0
    e_prev = 0
    error_vals = []
    beta_vals = []
    for i in range(int(simtime / stepsize)):
        mp[0],e_prev,e_total = PID_cont(SIR[i],mp,i,e_total,e_prev,K)
        error_vals.append(e_prev)
        beta_vals.append(mp[0])
        SIR.append(method(SIR[i], mp,T[i], stepsize))
        

    return t, SIR, beta_vals,error_vals

