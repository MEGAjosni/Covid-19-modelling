import basic_ivp_funcs as b_ivp
import paramest_funcs as pest
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import tikzplotlib
import numpy as np
from SIR_basic_data import X

### MAIN SCRIPT - GROUP 8.3 ###################
#   Authors: Marcus, Albert, Jonas
#   Date: 23/06 - 2021
#   
#   The purpose of this script is to produce
#   plots created for the report. Notice that
#   scripts for PID plots are not included here,
#   but found in PID_manual_tuning.py
###############################################


#Parameter estimation: simple SIR
gamma = 1 / 9  # predifed gamma

# start of simulation
t1 = pd.to_datetime('2020-12-01')
# number of days to simulate over
sim_days = 21
# end of simulation
t2 = t1 + dt.timedelta(days=sim_days)

# find optimal beta
c1 = time.process_time()

beta_opt = pest.estimate_beta_simple(
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

#Create plots
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

# %% basic SIR: Variying beta
from SIR_basic_data import X

# start of simulation
t1 = pd.to_datetime('2020-12-01')
# number of days to simulate over
sim_days = 80
# end of simulation
t2 = t1 + dt.timedelta(days=sim_days)


gamma = 1/9
N = 5800000

betas = pest.beta_over_time_simple(
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
        betas_calc_ls.append((np.dot(A.transpose(),A))**(-1)*np.dot(A.transpose(),b))
    elif i < 7:
        betas_calc_avg.append((betas_calc[0:i+7]).mean())
        A = (-S*I/N)[0:i+7]
        b = dS[0:i+7]
        betas_calc_ls.append((np.dot(A.transpose(),A))**(-1)*np.dot(A.transpose(),b))
    elif i > len(betas_calc)-7:
        betas_calc_avg.append((betas_calc[i-7:-1]).mean())
        A = (-S*I/N)[i-7:-1]
        b = dS[i-7:-1]
        betas_calc_ls.append((np.dot(A.transpose(),A))**(-1)*np.dot(A.transpose(),b))



T = list(range(sim_days + 1))
# Plot optimal solution
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.plot(T, betas.transpose(), c = "b")
T2 = list(range(sim_days -3))
ax.plot(T2, np.array(betas_calc_avg).transpose(), c = "g")
ax.plot(T2, np.array(betas_calc_ls).transpose(), c = "r")

ax2.scatter(T, np.asarray(X["I"][t1:t2]), c = "tab:orange")
#ax2.plot(t, np.array(SIR)[:, 1], c="r", label="I est.")
plt.title("Simulation varying beta")
ax.legend(["Beta"])
ax.set_xlabel("time")
ax.set_ylabel("Number of people")
ax2.set_ylabel("Beta")
ax2.legend("Infected")
plt.show()



#%% Expanded model: parameter estimation
import data_prep_S3I3R as dp4e
import basic_ivp_funcs as b_ivp
import paramest_funcs as pest
import expanded_ivp_funcs as e_ivp
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import tikzplotlib
import numpy as np


# Specify period and overshoot
start_day = '2020-12-01'  # start day
sim_days = 14

t1 = pd.to_datetime(start_day)

gammas = [1/9, 1/7, 1/21] #gamma1 gamma2 gamma3

# Load data
data = dp4e.Create_dataframe(
    Gamma1=gammas[0],
    Gamma2=gammas[1],
    Gamma3 = gammas[2],
    forecast=False,
)

# find optimal parameters
opt_params = pest.estimate_params_expanded_LA(
    X_0 = data.loc[t1],
    t1=t1,
    t2=t1 + dt.timedelta(days=14),
    data=data,
    mp=gammas
)

# include predefined gammas and optimal parameters
mp = np.array([opt_params[0],gammas[0],gammas[1],gammas[2],opt_params[3],opt_params[1],opt_params[2]])

T = data["R2"][t1:t1 + dt.timedelta(days=sim_days)]


#Normal simulation
t, SIR = e_ivp.simulateSIR(
    X_0=data.loc[t1],
    mp=mp,
    T = T,
    simtime=sim_days,
    method=e_ivp.RK4
)



#Create plot
t = pd.date_range(t1, periods=sim_days + 1).strftime('%d/%m-%Y')
alpha = 0.5
# Plot optimal solution
fig, ax = plt.subplots()
ax2 = ax.twinx()
# plot simulations
ax.plot(t, np.array(SIR)[0::10, 3], c="orange", label="I3 est")
ax2.plot(t, np.array(SIR)[0::10, 6], c="tab:blue", label="R3 est")
ax2.plot(t, np.array(SIR)[0::10, 2], c="tab:orange", label="I2 est.")
#ax2.plot(t, np.array(SIR)[0::10, 2], c="b", label="R est.")
ax.set_xlabel("Date")
#ax.set_ylabel("Number of susce"ptible people ")
ax2.set_ylabel("Number of dead people")
ax.set_ylabel("Number of people in ICU")

n = 5
T = list(range(sim_days + 1))[0::n]
threshold = 300*np.ones(len(T))

# Data points
ax.plot(T,threshold, "--", label = "I3 threshold")
ax.scatter(T, data['I3'][t1:t1 + dt.timedelta(days=sim_days)][0::n], c="tab:orange", alpha=alpha, label="I3")
ax2.scatter(T, data['R3'][t1:t1 + dt.timedelta(days=sim_days)][0::n], c="b", alpha=alpha, label="R3")
ax2.scatter(T, data['I2'][t1:t1 + dt.timedelta(days=sim_days)][0::n], c="tab:orange", alpha=alpha, label="I2")
#ax2.scatter(T, data['R1'][t1:t2], c="b", alpha=alpha, label="R1")

ax.tick_params(axis='x', rotation=45)
ax.legend(loc="upper left")
ax2.legend(loc="center right")
tikzplotlib.save('test_expand_2.tex')
plt.show()


#%% Expanded model: simulation of lockdown with december parameters
import data_prep_S3I3R as dp4e
import basic_ivp_funcs as b_ivp
import paramest_funcs as pest
import expanded_ivp_funcs as e_ivp
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import tikzplotlib
import numpy as np


# Specify period and overshoot
start_day = '2020-12-01'  # start day
sim_days = 180

t1 = pd.to_datetime(start_day)

gammas = [1/9, 1/7, 1/21] #gamma1 gamma2 gamma3

# Load data
data = dp4e.Create_dataframe(
    Gamma1=gammas[0],
    Gamma2=gammas[1],
    Gamma3 = gammas[2],
    forecast=False,
)

# find optimal parameters
opt_params = pest.estimate_params_expanded_LA(
    X_0 = data.loc[t1],
    t1=t1,
    t2=t1 + dt.timedelta(days=14),
    data=data,
    mp=gammas
)

# include predefined gammas and optimal parameters
mp = np.array([opt_params[0],gammas[0],gammas[1],gammas[2],opt_params[3],opt_params[1],opt_params[2]])

T = data["R2"][t1:t1 + dt.timedelta(days=sim_days)]


#Normal simulation
t, SIR = e_ivp.simulateSIR(
    X_0=data.loc[t1],
    mp=mp,
    T = T,
    simtime=sim_days,
    method=e_ivp.RK4
)

T = np.array(T)*0

#No vaccination simulation
t, SIR2 = e_ivp.simulateSIR(
    X_0=data.loc[t1],
    mp=mp,
    T = T,
    simtime=sim_days,
    method=e_ivp.RK4
)


# Load Forecast data
data = dp4e.Create_dataframe(
    Gamma1=gammas[0],
    Gamma2=gammas[1],
    Gamma3 = gammas[2],
    forecast=True,
    early = True,
)


T = data["R2"][t1:t1 + dt.timedelta(days=sim_days)]

#Forecast vaccination simulation
t, SIR3 = e_ivp.simulateSIR(
    X_0=data.loc[t1],
    mp=mp,
    T = T,
    simtime=sim_days,
    method=e_ivp.RK4
)

#Create plot
t = pd.date_range(t1, periods=sim_days + 1).strftime('%d/%m-%Y')
alpha = 0.5
# Plot optimal solution
fig, ax = plt.subplots()
ax2 = ax.twinx()
# plot simulations
ax.plot(t, np.array(SIR)[0::10, 3], c="orange", label="I3 est. +")
ax.plot(t, np.array(SIR2)[0::10, 3], c="black", label="I3 est. F", alpha = alpha)
ax.plot(t, np.array(SIR3)[0::10, 3], c="black", label="I3 est. -", alpha = alpha)
ax2.plot(t, np.array(SIR)[0::10, 6], c="tab:blue", label="R3 est. +")
ax2.plot(t, np.array(SIR2)[0::10, 6], c="b", label="R3 est.F", alpha = alpha)
ax2.plot(t, np.array(SIR3)[0::10, 6], c="b", label="R3 est.-", alpha = alpha)
#ax2.plot(t, np.array(SIR)[0::10, 2], c="tab:orange", label="I2 est.")
#ax2.plot(t, np.array(SIR)[0::10, 2], c="b", label="R est.")
ax.set_xlabel("Date")
#ax.set_ylabel("Number of susce"ptible people ")
ax2.set_ylabel("Number of dead people")
ax.set_ylabel("Number of people in ICU")

n = 5
T = list(range(sim_days + 1))[0::n]
threshold = 300*np.ones(len(T))

# Data points
ax.plot(T,threshold, "--", label = "I3 threshold")
ax.scatter(T, data['I3'][t1:t1 + dt.timedelta(days=sim_days)][0::n], c="tab:orange", alpha=alpha, label="I3")
ax2.scatter(T, data['R3'][t1:t1 + dt.timedelta(days=sim_days)][0::n], c="b", alpha=alpha, label="R3")
#ax2.scatter(T, data['I2'][t1:t1 + dt.timedelta(days=sim_days)][0::n], c="tab:orange", alpha=alpha, label="I2")
#ax2.scatter(T, data['R1'][t1:t2], c="b", alpha=alpha, label="R1")

ax.tick_params(axis='x', rotation=45)
ax.legend(loc="upper left")
ax2.legend(loc="center right")
tikzplotlib.save('test_expand_2.tex')
plt.show()

#%% Parameters over time: Expanded model
# Specify period and overshoot
start_day = '2020-12-01'  # start day
sim_days = 119
overshoot = dt.timedelta(days=7)
t1 = pd.to_datetime(start_day)

#load data
data = dp4e.Create_dataframe(
    Gamma1=mp[0],
    Gamma2=mp[1],
    forecast=False,
)

#compute optimal parameters overtime
opt_params = pest.params_over_time_expanded_LA(
    t1=t1,
    t2=t1 + dt.timedelta(days=sim_days),
    overshoot=overshoot,
    data=data,
    mp=mp
)


#Create plot
fig, ax = plt.subplots()
ax2 = ax.twinx()
new_opt = np.delete(opt_params, (1), axis=0)
ax.plot(new_opt.transpose())
#plt.plot(thetas)
ax.legend([r"$\beta$", r"$\phi_2$", r"$\theta$"], loc="center left")

ax2.plot(opt_params[1,:].transpose(), c = "gold")
ax2.legend([r"$\phi_1$"])
tikzplotlib.save('test_expand_3.tex')
plt.show()
