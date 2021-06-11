import basic_ivp_funcs as b_ivp
import estimate_beta as pestbeta
import get_data as gd
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
#import tikzplotlib
import numpy as np
from SIR_basic_data import X
import paramest_funcs as paramest

gamma = 1/9 #predifed gamma


#start of simulation
t1 = pd.to_datetime('2021-01-05')
#number of days to simulate over
sim_days = 21
#end of simulation
t2 = t1 + dt.timedelta(days = sim_days)

#find optimal beta
c1 = time.process_time()
beta_opt = paramest.estimate_beta_simple(
    X_0=X.loc[t1],
    t1 = t1,
    t2 = t2,
    real_data=X,
    gamma = gamma,
    precision = 5
)

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

# Plot optimal solution
plt.plot(t[0:-1-8], np.array(SIR)[:,1])
plt.title("Simulation using optimal parameters")
plt.legend(["Susceptible", "Infected", "Removed"])
plt.xlabel("Days since start,    Parameters: " + r'$\beta = $' + "{:.6f}".format(
    beta_opt) + ", " + r'$\gamma = $' + "{:.6f}".format(gamma))
plt.ylabel("Number of people")
#plt.ylim([0, max(X[:,1])+1000])
#T = list(range(sim_days))
#plt.bar(T,X['I'][t1:t2])
plt.show()
#tikzplotlib.save('test.tex')

#%% Variying beta

"""
gamma = 1/9

X_0 = [N - I_0 - R_0, I_0, R_0]
mp = [beta_opt, gamma, N]

t, SIR, betas = b_ivp.simulateSIR_betafun(
    X_0 = X_0,
    X = test_data,
    gamma = gamma,
    N = N,
    simtime = 100,
    stepsize = 1,
    method = b_ivp.RK4)

# Plot optimal solution
plt.plot(t, np.asarray(SIR)[:,1])
plt.title("Simulation using optimal parameters")
plt.legend(["Susceptible", "Infected", "Removed"])
plt.xlabel("Days since start,    Parameters: " + r'$\beta = $' + "{:.6f}".format(
    beta_opt) + ", " + r'$\gamma = $' + "{:.6f}".format(gamma_opt))
plt.ylabel("Number of people")
plt.ylim([0, max(I)+1000])
T = list(range(simdays))
plt.bar(T,I[days:days+simdays])
plt.show()
"""