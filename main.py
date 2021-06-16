import basic_ivp_funcs as b_ivp
import paramest_funcs as pestbeta
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import tikzplotlib
import numpy as np
from SIR_basic_data import X
import paramest_funcs as paramest

gamma = 1 / 9  # predifed gamma

# start of simulation
t1 = pd.to_datetime('2020-12-01')
# number of days to simulate over
sim_days = 21
# end of simulation
t2 = t1 + dt.timedelta(days=sim_days)

# find optimal beta
c1 = time.process_time()

beta_opt = paramest.estimate_beta_simple(
    X_0=X.loc[t1],
    t1=t1,
    t2=t2,
    data=X,
    gamma=gamma,
    precision=5
)

beta_opt = 0.17
# Simulate optimal solution
mp = [beta_opt, gamma]

t, SIR = b_ivp.simulateSIR(
    X_0=X.loc[t1],
    mp=mp,
    simtime=sim_days,
    method=b_ivp.RK4
)

c2 = time.process_time()

print("Simulation completed in", c2 - c1, "seconds.")
t = pd.date_range(t1, periods=sim_days + 1).strftime('%d/%m-%Y')
alpha = 0.5
# Plot optimal solution
fig, ax = plt.subplots()
ax2 = ax.twinx()
# plot simulations
ax.plot(t, np.array(SIR)[0::10, 0], c="g", label="S est.")
ax2.plot(t, np.array(SIR)[0::10, 1], c="tab:orange", label="I est.")
ax2.plot(t, np.array(SIR)[0::10, 2], c="b", label="R est.")
plt.title("Simulation using optimal parameters")
ax.set_xlabel("Days since start,    Parameters: " + r'$\beta = $' + "{:.6f}".format(
    beta_opt) + ", " + r'$\gamma = $' + "{:.6f}".format(gamma))
ax.set_ylabel("Number of susceptible people ")
ax2.set_ylabel("Number of infected or recovered people")
T = list(range(sim_days + 1))

# Data points
ax.scatter(T, X['S'][t1:t2], c="g", alpha=alpha, label="S")
ax2.scatter(T, X['I'][t1:t2], c="tab:orange", alpha=alpha, label="I")
ax2.scatter(T, X['R'][t1:t2], c="b", alpha=alpha, label="R")

ax.tick_params(axis='x', rotation=45)
ax.legend(loc="center left")
ax2.legend(loc="center right")
tikzplotlib.save('test.tex')
plt.show()

# %% Variying beta
from SIR_basic_data import X

# start of simulation
t1 = pd.to_datetime('2020-12-01')
# number of days to simulate over
sim_days = 80
# end of simulation
t2 = t1 + dt.timedelta(days=sim_days)


gamma = 1/9
N = 5800000

betas = pestbeta.beta_over_time_simple(
        t1 = t1,
        t2 = t2,
        overshoot = dt.timedelta(days = 7),
        data = X,
        gamma = gamma
)



dS = np.array(X['S'][t1+dt.timedelta(days=1):t1+dt.timedelta(days=sim_days-1)])-np.array(X['S'][t1:t1 + dt.timedelta(days=sim_days-2)])
S = np.array(X['S'][t1:t1 + dt.timedelta(days=sim_days-2)])
I = np.array(X['I'][t1:t1 + dt.timedelta(days=sim_days-2)])


betas_calc = -dS*N/(S*I)
betas_calc_avg = []
betas_calc_ls = []
for i in range(len(betas_calc)):
    if i > 7 and i < len(betas_calc)-7:
        betas_calc_avg.append((betas_calc[i-7:i+7]).mean())
        A = (-S*I/N)[i-7:i+7]
        b = dS[i-7:i+7]
        betas_calc_ls.append(np.linalg.inv(np.dot(A.transpose,A))*np.dot(A.transpose(),b))
    elif i < 7:
        betas_calc_avg.append((betas_calc[0:i+7]).mean())
        A = (-S*I/N)[0:i+7]
        b = dS[0:i+7]
        betas_calc_ls.append(np.linalg.inv(np.dot(A.transpose,A))*np.dot(A.transpose(),b))
    elif i > len(betas_calc)-7:
        betas_calc_avg.append((betas_calc[i-7:-1]).mean())
        A = (-S*I/N)[i-7:-1]
        b = dS[i-7:-1]
        betas_calc_ls.append(np.linalg.inv(np.dot(A.transpose,A))*np.dot(A.transpose(),b))
        

T = list(range(sim_days + 1))
# Plot optimal solution
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(T, betas.transpose(), c = "b")
T2 = list(range(sim_days -3))
ax.plot(T2, np.array(betas_calc_avg).transpose(), c = "g")
ax2.plot(T, np.asarray(X["I"][t1:t2]), c = "tab:orange")

plt.title("Simulation varying beta")
ax.legend(["Beta"])
ax.set_xlabel("time")
ax.set_ylabel("Number of people")
ax2.set_ylabel("Beta")
ax2.legend("Infected")
plt.show()

#%% 
