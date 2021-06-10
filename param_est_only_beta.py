# >>> Files <<<
import basic_ivp_funcs as b_ivp
import get_data as gd

# >>> Packages <<<
import numpy as np
import time
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm


def estimate_beta(
    X_0,
    t1, # Start
    t2, # Stop
    data,
    gamma=1/9,
    precision=5
):

    print(data)

    mse_min = math.inf
    beta_opt = 0

    for k in range(precision):
        for beta in tqdm(np.linspace(beta_opt - 1/(10**k), beta_opt + 1/(10**k), 21)):
            if beta >= 0:
                _, SIR = b_ivp.simulateSIR(
                    X_0=X_0,
                    mp=[beta, gamma],
                    method=b_ivp.RK4,
                    simtime=(t2-t1).days
                )
                sim_data = SIR[:, 1]
                mse = (np.square(sim_data[0::10] - data[t1:t2].to_numpy())).mean()
                # print(mse)
                if mse < mse_min:
                    print('triggered')
                    mse_min = mse
                    beta_opt = beta

    return beta_opt


t1 = pd.to_datetime('2020-12-01')  # start day
simdays = dt.timedelta(days=100)
overshoot = dt.timedelta(days=7)

data_pcr = gd.infect_dict['Test_pos_over_time']['NewPositive'][t1-overshoot:t1+simdays+overshoot]
data_antigen = gd.infect_dict['Test_pos_over_time_antigen']['NewPositive'][t1-overshoot:t1+simdays+overshoot]

data_total = data_pcr + data_antigen


opt_beta = np.empty(shape=100, dtype=float)

beta = estimate_beta(np.array([5800000, 30000, 0]) , t1, t1+simdays, data_total)
print(beta)











#
#
# def simulateSIR_betafun(
#         X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
#         X: list,  # Nested numpy array of [S, I, R] for each simulated day
#         gamma: int,  # Model parameter gamma
#         N: int,  # Population size
#         simtime: int = 100,  # How many timeunits into the future that should be simulated
#         stepsize: float = 0.1,  # t_kp1 - t_k
#         method=ExplicitEuler  # Numerical method to be used [function]
# ):
#     # *** Description ***
#     # Simulate modified SIR-model, with beta as a continuous
#     # function instead of a constant
#
#     # *** Output ***
#     # t [list]:             All points in time simulated
#     # SIR [nested list]:    Values of SIR at all points in time t
#     # betas [list]:         Values of beta in all points in time t
#
#     SIR = [X_0]
#     betas = []
#     beta_time = 7  # half the amount of days needed to compute one beta value
#
#     t = [i * stepsize for i in range(int(simtime / stepsize) + 1)]
#
#     for i in tqdm(range(int(simtime / stepsize))):
#         if i < int(beta_time / stepsize):
#             test_data = X[0:i, :]
#         elif i > int((simtime - beta_time) / stepsize):
#             test_data = X[i:-1, :]
#         else:
#             test_data = X[i - beta_time:i + beta_time, :]
#
#         beta, errs = pestbeta.estimate_beta(
#             X_0=X_0,
#             data=test_data,
#             gamma=gamma,
#             n_points=10,
#             layers=5)
#
#         betas.append(beta)
#         SIR.append(method(SIR[i], [beta, gamma, N], stepsize))
#
#     return t, SIR, betas
