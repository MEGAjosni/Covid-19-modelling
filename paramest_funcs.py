# >>> Scripts <<<
import basic_ivp_funcs as b_ivp
import expanded_ivp_funcs as e_ivp
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
import itertools


def estimate_beta_simple(
        X_0,
        t1,  # Start
        t2,  # Stop
        real_data,
        gamma=1/9,
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
                sim_data = SIR[:, 1]
                err = (np.square(sim_data[0::10] - real_data['I'][t1:t2].to_numpy())).mean()
                if err < err_min:
                    err_min = err
                    best_beta = beta

    return best_beta


def estimate_params_expanded(
        X_0,
        t1,  # Start
        t2,  # Stop
        real_data,
        mp, # Known model parameters
        precision=5
):
    err_min = math.inf
    best_params = [0, 0, 0]

    for k in range(precision):
        np.linspace(best_params[0] - 1 / (10 ** k), best_beta + 1 / (10 ** k), 21)
        np.linspace(best_params[1] - 1 / (10 ** k), best_beta + 1 / (10 ** k), 21)
        np.linspace(best_params[2] - 1 / (10 ** k), best_beta + 1 / (10 ** k), 21)
        
        for beta in :
            if beta >= 0:
                _, SIR = b_ivp.simulateSIR(
                    X_0=X_0,
                    mp=[beta, gamma],
                    method=b_ivp.RK4,
                    simtime=(t2 - t1).days
                )
                sim_data = SIR[:, 1]
                err = (np.square(sim_data[0::10] - real_data[t1:t2].to_numpy())).mean()
                if err < err_min:
                    err_min = err
                    best_beta = beta

    return best_beta


for x in itertools.product([1, 2], [3, 4], [5, 6]):
    print(x)



#
# # Specify period and overshoot
# start_day = '2020-12-01'  # start day
# simdays = 100
# overshoot = 7
#
# t0 = pd.to_datetime(start_day)
# overshoot = dt.timedelta(days=overshoot)
#
# # Load data
# data = pd.read_csv('data/X_basic.csv', index_col=0, parse_dates=True)
#
# # Search for best values of beta
# opt_beta = np.empty(shape=simdays, dtype=float)
#
# for i in tqdm(range(simdays)):
#     opt_beta[i] = estimate_beta_simple(
#         X_0=data.loc[t0 + dt.timedelta(days=i) - overshoot].to_numpy(copy=True),
#         t1=t0 - overshoot + dt.timedelta(days=i),
#         t2=t0 + overshoot + dt.timedelta(days=i),
#         real_data=data
#     )
#
# # Make plot of results
# t = pd.date_range(t0, periods=simdays).strftime('%d/%m-%Y')
#
# fig, ax1 = plt.subplots()
#
# color = 'tab:blue'
# ax1.set_xlabel('Date')
# ax1.set_ylabel(r'$\beta$')
# ax1.plot(t, opt_beta, color=color)
# ax1.tick_params(axis='x', rotation=45)
# ax1.legend([r'$\beta$'], loc="upper center")
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:orange'
# ax2.set_ylabel('Infected')  # we already handled the x-label with ax1
# ax2.plot(t, data['I'][t0:t0+dt.timedelta(days=simdays-1)], color=color)
# ax2.legend(['Infected'], loc="upper right")
#
# plt.xticks(ticks=t[0::7], labels=t[0::7])
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()


# data_pcr = gd.infect_dict['Test_pos_over_time']['NewPositive'][t0 - overshoot:t0 + simdays + overshoot]
# data_antigen = gd.infect_dict['Test_pos_over_time_antigen']['NewPositive'][t0 - overshoot:t0 + simdays + overshoot]
# data_total = data_pcr + data_antigen



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
