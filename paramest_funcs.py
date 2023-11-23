# >>> Scripts <<<
import basic_ivp_funcs as b_ivp
import expanded_ivp_funcs as e_ivp
import get_data as gd
import data_prep_S3I3R as dp4e

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
    real_data = data.loc[t1:t2].to_numpy()

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

def estimate_beta_simple_LA(
        X_0,
        t1,  # Start
        t2,  # Stop
        data,
        gamma = None,
        precision=3
):
    N  =sum(X_0)
    simdays = (t2 - t1).days + 1

    dX = data[['S','I']][t1+dt.timedelta(days=1):t2].to_numpy() - data[['S','I']][t1: t2+dt.timedelta(days=-1)].to_numpy()
    dX_norms = np.linalg.norm(dX,axis=0)
    
    
    for k in range(simdays):
        dxt = (np.array(data.loc[t1+dt.timedelta(days=k):t1+dt.timedelta(days=k)])-np.array(data.loc[t1+dt.timedelta(days=k-1):t1+dt.timedelta(days=k-1)]))[0]
        xt = (np.array(data.loc[t1+dt.timedelta(days=k):t1+dt.timedelta(days=k)]))[0]
        
        A = np.array(
            #[xt[0]*xt[1]/N])/dX_norms
            [-xt[0]*xt[1]/N,xt[0]*xt[1]/N])/dX_norms
        b = np.array(
            #[dxt[1]+gamma*xt[1]])/dX_norms
            [dxt[0], dxt[1]+gamma*xt[1]])/dX_norms
        if k == 0:
            As = A
            bs = b
        else:
            As = np.concatenate([As,A],axis = 0)
            bs = np.concatenate([bs,b])
    
    if gamma is not None:
        params = (np.dot(As,As)**(-1))*np.dot(As,bs)
    else:
        params = np.linalg.lstsq(As, bs, rcond=None)[0]
    return params


def beta_over_time_simple(
        t1: dt.date,
        t2: dt.date,
        overshoot: dt.timedelta,
        data: pd.core.frame.DataFrame,
        gamma: float = 1/9
):

    simdays = (t2 - t1).days + 1
    betas = np.zeros(simdays)

    for k in tqdm(range(simdays)):
        betas[k] = estimate_beta_simple(
            X_0=data.loc[t1 - overshoot + dt.timedelta(days=k)].to_numpy(copy=True),
            t1=t1 + dt.timedelta(days=k) - overshoot,
            t2=t2 + dt.timedelta(days=k) + overshoot,
            data=data,
            gamma=gamma,
            precision=3
        )

    return betas

def beta_over_time_simple_LA(
        t1: dt.date,
        t2: dt.date,
        overshoot: dt.timedelta,
        data,# pd.core.frame.DataFrame,
        gamma: float = 1/9
):

    simdays = (t2 - t1).days + 1
    betas = np.zeros(simdays)

    for k in tqdm(range(simdays)): 
        betas[k] = estimate_beta_simple_LA(
            X_0=data.loc[t1 - overshoot + dt.timedelta(days=k)].to_numpy(copy=True),
            t1=t1 + dt.timedelta(days=k) - overshoot,
            t2=t1 + dt.timedelta(days=k),
            data=data,
            gamma=gamma,
            precision=3
        )

    return betas


def estimate_params_expanded_LA(
        X_0: np.array,
        t1: dt.date,  # Start
        t2: dt.date,  # Stop
        data: pd.core.frame.DataFrame,
        mp: list,  # Known model parameters [gamma1, gamma2, gamma3]
        precision: int = 5
):
    N = sum(X_0)
    simdays = (t2 - t1).days + 1

    dX = data[['S', 'I1', 'I2', 'I3', 'R3']][t1:t2].to_numpy() - data[['S', 'I1', 'I2', 'I3', 'R3']][t1-dt.timedelta(days=1): t2-dt.timedelta(days=1)].to_numpy()
    dX_norms = np.linalg.norm(dX, axis=0)

    for k in range(simdays):
        dxt = (np.array(data.loc[t1+dt.timedelta(days=k):t1+dt.timedelta(days=k)])-np.array(data.loc[t1+dt.timedelta(days=k-1):t1+dt.timedelta(days=k-1)]))[0]
        xt = (np.array(data.loc[t1+dt.timedelta(days=k):t1+dt.timedelta(days=k)]))[0]
        # xt = [S I1 I2 I3 R1 R2 R3]

        A = np.array(
            [
                [-xt[1]*xt[0]/N,    0,      0,      0],
                [xt[0]*xt[1]/N,     -xt[1], 0,      0],
                [0,                 xt[1],  -xt[2], 0],
                [0,                 0,      xt[2],  -xt[3]],
                [0,                 0,      0,      xt[3]]
            ]
        ) / np.transpose(dX_norms)[:, None]

        b = np.array(
            [
                dxt[0] + dxt[5]*xt[0]/(xt[0]+xt[1]+xt[4]),
                dxt[1] + xt[1]*mp[0] + xt[1]*dxt[5]/(xt[0]+xt[1]+xt[4]),
                dxt[2] + mp[1]*xt[2],
                dxt[3] + mp[2]*xt[3],
                dxt[6]
            ]
        ) / dX_norms

        if k == 0:
            As = A
            bs = b
        else:
            As = np.concatenate([As,A],axis = 0)
            bs = np.concatenate([bs,b])

    beta,phi1,phi2,theta = np.linalg.lstsq(As, bs, rcond=None)[0]
    mp2 = [beta, phi1, phi2, theta]
    return np.array(mp2)

def params_over_time_expanded_LA(
        t1: dt.date,
        t2: dt.date,
        overshoot: dt.timedelta,
        data: pd.core.frame.DataFrame,
        mp: list
):

    simdays = (t2 - t1).days + 1
    params = np.zeros((4, simdays))

    for k in tqdm(range(simdays)):
        params[:, k] = estimate_params_expanded_LA(
            X_0=data.loc[t1 - overshoot + dt.timedelta(days=k)].to_numpy(copy=True),
            t1=t1 + dt.timedelta(days=k) - overshoot,
            t2=t1 + dt.timedelta(days=k) + overshoot,
            data=data,
            mp=mp,
            precision=3
        )
        # print(params[:, 0:k+1])

    return params
