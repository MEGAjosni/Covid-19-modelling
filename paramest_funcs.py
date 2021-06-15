# >>> Scripts <<<
import basic_ivp_funcs as b_ivp
import expanded_ivp_funcs as e_ivp
import get_data as gd
import Data_prep_4_expanded as dp4e

# >>> Packages <<<
import numpy as np
import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
import itertools
from numpy import linalg as LA


def estimate_beta_simple(
        X_0,
        t1,  # Start
        t2,  # Stop
        real_data,
        gamma=1 / 9,
        precision=5
):
    err_min = math.inf
    best_beta = 0

    for k in range(precision):
        for beta in np.linspace(best_beta - 1 / (10 ** k), best_beta + 1 / (10 ** k), 21):
            if beta >= 0:
                _, SIR = b_ivp.simulateSIR(
                    X_0=X_0,
                    mp=[beta, gamma],
                    method=b_ivp.RK4,
                    simtime=(t2 - t1).days
                )
                S_rel_err = LA.norm(SIR[0::10, 0] - real_data["S"][t1:t2]) ** 2 / LA.norm(real_data["S"][t1:t2])
                I_rel_err = LA.norm(SIR[0::10, 1] - real_data["I"][t1:t2]) ** 2 / LA.norm(real_data["I"][t1:t2])
                R_rel_err = LA.norm(SIR[0::10, 2] - real_data["R"][t1:t2]) ** 2 / LA.norm(real_data["R"][t1:t2])
                err = S_rel_err + I_rel_err + R_rel_err
                # err = (np.square(sim_data[0::10] - real_data['I'][t1:t2].to_numpy())).mean()
                if err < err_min:
                    err_min = err
                    best_beta = beta

    return best_beta


def estimate_params_expanded(
        X_0,
        t1,  # Start
        t2,  # Stop
        data,
        mp,  # Known model parameters [gamma1, gamma2, gamma3, theta]
        precision=2
):

    gamma1, gamma2, gamma3, theta = mp
    err_min = math.inf
    best_params = [0, 0, 0]
    T = np.zeros(len(data['R2'])*10)
    for i in range(len(data['R2'])):
        T[(i*10):(i*10)+10] = data['R2'][i]/10
    # Normalize data
    real_data = np.transpose(data[t1:t2].to_numpy())
    norm_real_data = np.nan_to_num(real_data / np.linalg.norm(real_data, axis=1, keepdims=True), nan=0)

    for k in tqdm(range(precision)):
        beta_vals = np.linspace(best_params[0] - 1 / (10 ** k), best_params[0] + 1 / (10 ** k), 21)
        phi1_vals = np.linspace(best_params[1] - 1 / (10 ** k), best_params[1] + 1 / (10 ** k), 21)
        phi2_vals = np.linspace(best_params[2] - 1 / (10 ** k), best_params[2] + 1 / (10 ** k), 21)

        for comb in itertools.product(beta_vals, phi1_vals, phi2_vals):
            params = list(comb)
            if all(i >= 0 for i in params):
                _, SIR = e_ivp.simulateSIR(
                    X_0=X_0,
                    mp=[params[0], gamma1, gamma2, gamma3, theta, params[1], params[2]],
                    method=e_ivp.RK4,
                    T=T,
                    simtime=(t2 - t1).days
                )

                # Normalize simulation
                SIR = np.transpose(SIR[0::10, :])
                norm_SIR = np.nan_to_num(SIR / np.linalg.norm(SIR, axis=1, keepdims=True), nan=0)

                # Find and compare error
                err = np.sum(np.square(norm_real_data - norm_SIR))
                if err < err_min:
                    err_min = err
                    best_params = params

    t, SIR = e_ivp.simulateSIR(
        X_0=X_0,
        mp=[best_params[0], gamma1, gamma2, gamma3, theta, best_params[1], best_params[2]],
        method=e_ivp.RK4,
        T=T,
        simtime=(t2 - t1).days
    )
    
    plt.plot(t, SIR[:, 1:])
    plt.legend(['I1', 'I2', 'I3', 'R1', 'R2', 'R3'])
    plt.show()

    return [round(k,precision) for k in best_params]


# Specify period, overshoot and non-estimating parameters
start_day = '2020-12-01'  # start day
simdays = 100
overshoot = 7
gamma1 = 1/7
gamma2 = 1/14
gamma3 = 1/21
theta = 1/15

t0 = pd.to_datetime(start_day)
overshoot = dt.timedelta(days=overshoot)

# Load data
data = dp4e.Create_dataframe(
    Gamma1=gamma1,
    Gamma2=gamma2,
    s2=t0,
    sim_days=100,
    forecast=False
)

mp = [gamma1, gamma2, gamma3, theta]

# Search for best values of beta, phi1 and phi2
opt_params = estimate_params_expanded(
    X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
    t1=t0 - overshoot,
    t2=t0 + overshoot,
    data=data,
    mp=mp,
    precision=6
)

print(opt_params)

