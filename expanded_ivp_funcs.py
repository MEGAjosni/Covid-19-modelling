# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 13:31:45 2021

@author: alboa
"""


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
    # R2 : vaccinated
    # R3 : dead

    # mp : model parameters
    # beta1 : Infection rate parameter in group at risk
    # beta2 : Infection rate parameter in group not at risk
    # gamma1 : Rate of recovery for infected
    # gamma2 : Rate of recovery for ICU
    # gamma3 : Rate of recovery for respirator
    # t : Vaccination rate
    # theta1 : Death rate at ICU
    # theta2 : Death rate in respirator
    # phi1 : Rate of ICU from infected
    # phi2 : Rate of respirator from ICU
    # N : Population
    # ************* Output *************
    #
    # dX : derivative of state vector X. 

    # Extract data
    beta1, beta2, gamma1, gamma2, gamma3, t, theta1, theta2, phi1, phi2, N = mp
    S1, S2, I1, I2, I3, R1, R2, R3 = X

    dX = [
        -(beta1 * (I1 / N) + t * (1 / (S1 + S2 + I1 + R2))) * S1,  # dS1/dt
        -(beta2 * (I1 / N) + t * (1 / (S1 + S2 + I1 + R2))) * S2,  # dS2/dt
        beta1 * (I1 / N) * S1 + beta2 * (I1 / N) * S2 - (gamma1 + phi1) * I1 - t * (1 / (S1 + S2 + I1 + R2)) * I1,
        # dI1/dt
        phi1 * I1 - (gamma2 + theta1 + phi2) * I2,  # dI2/dt
        phi2 * I2 - (gamma3 + theta2) * I3,  # dI3/dt
        gamma1 * I1 + gamma2 * I2 + gamma3 * I3 - t * (1 / (S1 + S2 + I1 + R2)) * R2,  # dR1/dt
        t,  # dR2/dt
        theta1 * I1 + theta2 * I2,  # dR3/dt

    ]

    return dX


def RK4(
        X_k: list,  # Values of the expanded SIR at time t_k
        mp: list,  # Model parameters [beta, gamma, N]
        stepsize: float = 0.1  # t_kp1 - t_k
):
    # *** Description ***
    # Uses Rung Kutta 4 to solve ODE system numerically.

    # *** Output ***
    # X_kp1 [list]:         Values of SIR at time t_kp1

    N = range(len(X_k))

    K_1 = derivative_expanded(X_k, mp)
    K_2 = derivative_expanded([X_k[i] + 1 / 2 * stepsize * K_1[i] for i in N], mp)
    K_3 = derivative_expanded([X_k[i] + 1 / 2 * stepsize * K_2[i] for i in N], mp)
    K_4 = derivative_expanded([X_k[i] + stepsize * K_3[i] for i in N], mp)

    X_kp1 = [X_k[i] + stepsize / 6 * (K_1[i] + 2 * (K_2[i] + K_3[i]) + K_4[i]) for i in N]

    return X_kp1


def simulateSIR(
        X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
        mp: list,  # Model parameters [beta, gamma, N]
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 0.1,  # t_kp1 - t_k
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
        SIR.append(method(SIR[i], mp, stepsize))

    return t, SIR
