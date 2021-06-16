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


def estimate_beta_simple(
        X_0,
        t1,  # Start
        t2,  # Stop
        data,
        gamma=1 / 9,
        precision=3
):
    # Initialize local variables
    err_min = math.inf
    best_beta = 0

    # Get relevant data
    real_data = np.transpose(data.loc[t1:t2].to_numpy())

    for k in range(precision):
        for beta in np.linspace(best_beta - 1 / (10 ** k), best_beta + 1 / (10 ** k), 21):
            if beta > 0:
                _, SIR = b_ivp.simulateSIR(
                    X_0=X_0,
                    mp=[beta, gamma],
                    method=b_ivp.RK4,
                    simtime=(t2 - t1).days
                )

                # Get simulation points corresponding to real data
                SIR = np.array(SIR)[0::10,:]
                # Find and compare error
                rel_err = np.sum(np.nan_to_num(np.linalg.norm(real_data - SIR, axis=1) / np.linalg.norm(real_data, axis=1), nan=0))
                if rel_err < err_min:
                    err_min = rel_err
                    best_beta = beta

    return round(best_beta, precision)


def beta_over_time_simple(
        t1: dt.date,
        t2: dt.date,
        overshoot: dt.timedelta,
        data: pd.core.frame.DataFrame,
        gamma: float = 1/9
):

    simdays = (t2 - t1).days + 1
    betas = np.zeros(simdays)

    for k in range(simdays):
        betas[k] = estimate_beta_simple(
            X_0=data.loc[t1 - overshoot + dt.timedelta(days=k)].to_numpy(copy=True),
            t1=t1 + dt.timedelta(days=k) - overshoot,
            t2=t2 + dt.timedelta(days=k) + overshoot,
            data=data,
            gamma=gamma,
            precision=3
        )

    return betas


def estimate_params_expanded(
        X_0: np.array,
        t1: dt.date,  # Start
        t2: dt.date,  # Stop
        data: pd.core.frame.DataFrame,
        mp: list,  # Known model parameters [gamma1, gamma2, gamma3, theta]
        precision: int = 5
) -> np.array:

    gamma1, gamma2, gamma3, theta = mp
    err_min = math.inf
    best_params = [0, 0, 0]

    # Get relevant data
    real_data = np.transpose(data.loc[t1:t2].to_numpy())

    for k in tqdm(range(precision)):
        beta_vals = np.linspace(best_params[0] - 1 / (10 ** k), best_params[0] + 1 / (10 ** k), 21)
        phi1_vals = np.linspace(best_params[1] - 1 / (10 ** k), best_params[1] + 1 / (10 ** k), 21)
        phi2_vals = np.linspace(best_params[2] - 1 / (10 ** k), best_params[2] + 1 / (10 ** k), 21)

        for par in itertools.product(beta_vals, phi1_vals, phi2_vals):
            params = list(par)
            if all(i >= 0 for i in params):
                _, SIR = e_ivp.simulateSIR(
                    X_0=X_0,
                    mp=[params[0], gamma1, gamma2, gamma3, theta, params[1], params[2]],
                    method=e_ivp.RK4,
                    T=np.zeros((t2 - t1).days * 10 + 1),
                    simtime=(t2 - t1).days
                )

                # Get simulation points corresponding to real data
                SIR = SIR[:, 0::10]

                # Find and compare error
                rel_err = np.sum(np.nan_to_num(np.linalg.norm(real_data - SIR, axis=1) / np.linalg.norm(real_data, axis=1), nan=0))
                if rel_err < err_min:
                    err_min = rel_err
                    best_params = params

    return np.round(best_params, decimals=precision)


def params_over_time_expanded(
        t1: dt.date,
        t2: dt.date,
        overshoot: dt.timedelta,
        data: pd.core.frame.DataFrame,
        mp: list
):

    simdays = (t2 - t1).days + 1
    params = np.zeros((3, simdays))

    for k in tqdm(range(simdays)):
        params[:, k] = estimate_params_expanded(
            X_0=data.loc[t1 - overshoot + dt.timedelta(days=k)].to_numpy(copy=True),
            t1=t1 + dt.timedelta(days=k) - overshoot,
            t2=t1 + dt.timedelta(days=k) + overshoot,
            data=data,
            mp=mp,
            precision=3
        )
        print(params[:, 0:k+1])

    return params


# Specify period and overshoot
start_day = '2021-01-31'  # start day
simdays = 45
overshoot = 10

t0 = pd.to_datetime(start_day)
overshoot = dt.timedelta(days=overshoot)

# Load data
data = dp4e.Create_dataframe(
    Gamma1=1/9,
    Gamma2=1/14,
    t0=t0,
    sim_days=100,
    forecast=False
)

mp = [1/9, 1/7, 1/16, 1/5]

# opt_params = estimate_params_expanded(
#         X_0=data.loc[t0 - overshoot].to_numpy(copy=True),
#         t1=t0 - overshoot,  # Start
#         t2=t0 + overshoot,  # Stop
#         data=data,
#         mp=mp,  # Known model parameters [gamma1, gamma2, gamma3, theta]
#         precision=5
# )

# Search for best values of beta, phi1 and phi2
opt_params = params_over_time_expanded(
    t1=t0,
    t2=t0 + dt.timedelta(days=simdays),
    overshoot=overshoot,
    data=data,
    mp=mp
)

print(opt_params)
