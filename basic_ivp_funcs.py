import math
from tqdm import tqdm
import numpy as np


def derivative(
        X: list,  # Vector to compute derivative of
        mp: list  # Model parameters [beta, gamma, N]
):

    # *** Description ***
    # Computes the derivative of X using model parameters

    # *** Output ***
    # dX [list]:            Derivative of X

    beta, gamma = mp
    N = 5800000
    S, I, R = X

    dX = [
        - beta * S * I / N,
        I * (beta * S / N - gamma),
        gamma * I
    ]

    return np.array(dX)


"""
def simulateSV(
<<<<<<< Updated upstream:basic_ivp_funcs.py
        V_0: int,  # Initial values of V_0
=======
        V_start: float,  # Initial values in simulation of V
>>>>>>> Stashed changes: inivalfunctions.py
        mp: list,  # Model parameters [beta, gamma, N]
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 0.1,  # t_kp1 - t_k
        method=RK4V  # Numerical method to be used [function]
            ):
<<<<<<< Updated upstream:basic_ivp_funcs.py
    SV = [V_0]
    t = [i * stepsize for i in range(int(simtime / stepsize) + 1)]
    for i in range(int(simtime / stepsize)):
        SV.append(method(SV[i], mp=mp, stepsize=stepsize))

    return t, SV

=======
    V = [V_start]
    t = [i * stepsize for i in range(int(simtime / stepsize) + 1)]
    for i in range(int(simtime / stepsize)):
        V.append(method(1, V[i], mp, stepsize))

    return t, V
    
>>>>>>> Stashed changes:inivalfunctions.py
"""


def ExplicitEuler(
        X_k: np.array,  # Values of SIR at time t_k
        mp: list,  # Model parameters [beta, gamma]
        T: np.array,
        stepsize=0.1  # t_kp1 - t_k
):
    # *** Description ***
    # Uses Explicit Euler to solve ODE system numerically.

    # *** Output ***
    # X_kp1 [list]:         Values of SIR at time t_kp1

    dX_k = derivative(X_k, mp)

    X_kp1 = X_k + stepsize * dX_k

    return X_kp1


def RK4(
        X_k: np.array,  # Values of SIR at time t_k
        mp: list,  # Model parameters [beta, gamma]
        T: np.array,
        stepsize: float = 0.1  # t_kp1 - t_k
):
    # *** Description ***
    # Uses Rung Ketta 4 to solve ODE system numerically.

    # *** Output ***
    # X_kp1 [list]:         Values of SIR at time t_kp1

    K_1 = derivative(X_k, mp)
    K_2 = derivative(X_k + 1/2 * stepsize * K_1, mp)
    K_3 = derivative(X_k + 1/2 * stepsize * K_2, mp)
    K_4 = derivative(X_k + stepsize * K_3, mp)

    X_kp1 = X_k + stepsize/6 * (K_1 + 2 * (K_2 + K_3) + K_4)

    return X_kp1


def simulateSIR(
        X_0: np.array,  # Initial values of SIR [S_0, I_0, R_0]
        mp: list,  # Model parameters [beta, gamma]
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 0.1,  # t_kp1 - t_k
        method=RK4  # Numerical method to be used [function]
):
    # *** Description ***
    # Simulate SIR-model.

    # *** Output ***
    # t [list]:             All points in time simulated
    # SIR [nested list]:    Values of SIR at all points in time t

    n_steps = int(simtime / stepsize)

    SIR = np.zeros((3, n_steps + 1))
    SIR[:, 0] = X_0

    t = np.arange(start=0, stop=simtime+stepsize/2, step=stepsize)

    for k in range(n_steps):
        SIR[:, k+1] = method(SIR[:, k], mp, 0, stepsize)

    return t, SIR
