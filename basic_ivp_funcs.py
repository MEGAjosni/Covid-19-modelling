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
    N = sum(X)
    S, I, R = X

    dX = [
        - beta * S * I / N,
        I * (beta * S / N - gamma),
        gamma * I
    ]

    return np.array(dX)


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
        betas = None, #Array where each entry is the beta value of the corresponding day
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 0.1,  # t_kp1 - t_k
        noise_var = 0, #noise type 1. Added to beta
        noise_var2 = 0, #noise type 2. Added to compartments
        method=RK4,  # Numerical method to be used [function]
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
    gen_noise = np.random.normal(0,noise_var,n_steps)
    for k in range(n_steps):
        if betas is None:
            SIR[:, k+1] = method(SIR[:, k], [max([mp[0]+gen_noise[k],0]),mp[1]], 0, stepsize)
            d1 = np.random.normal(0,noise_var2*SIR[0,k+1])
            d2 = np.random.normal(0,noise_var2*SIR[1,k+1])
            SIR[:, k+1] += [d1, d2, -d1-d2]
            print(mp[0]+gen_noise[k])
            
        else:
            SIR[:, k+1] = method(SIR[:, k], [betas[int(np.floor(k*stepsize))],mp[1]], 0, stepsize)
    return t, SIR.transpose()
