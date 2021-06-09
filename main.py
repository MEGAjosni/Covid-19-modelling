import basic_ivp_funcs as b_ivp
import params_est_funcs as pest
import get_data as gd
import matplotlib.pyplot as plt
import param_est_only_beta as pestbeta
import time
import datetime as dt
import pandas as pd
#import tikzplotlib
import numpy as np


# Get data
<<<<<<< Updated upstream
#startdata
#start of pandemic
s1 = pd.to_datetime('2020-01-27')
#start of simulation
s2 = pd.to_datetime('2021-01-05')+dt.timedelta(days = 0)
simdays = 21
b = 8
#All DeltaI data
datatemp =  gd.infect_dict['Test_pos_over_time']
days = (s2-s1).days
DI = datatemp['NewPositive']
N = 5800000
S = []
I = []
R = []
X = []
for i in range(days+simdays):
    if i < 9:
        I.append(sum(DI[0:i]))
    else:
        I.append(sum(DI[i-9:i]))
    if i == 0:
        S.append(N-DI[i])
    else:
        S.append(S[i-1] - DI[i])

    R.append(N-S[i]-I[i])

    X.append([S[i],I[i],R[i]])

X = np.asarray(X)
=======
#start date
s = pd.to_datetime('2020-12-01')
#start of pandemic
s_p = pd.to_datetime('2020-02-25')
b = 8
num_days = 21


print(gd.vaccine_dict['FaerdigVacc_daekning_DK_prdag']['Kumuleret antal færdigvacc.'])
>>>>>>> Stashed changes

data = gd.infect_dict['Test_pos_over_time'][s - dt.timedelta(days=b): s + dt.timedelta(days=num_days)]

test_data = [None] * num_days
V_data = [None] * num_days
for i in range(num_days):
    test_data[i] = sum(data['NewPositive'][i:i+10]) * 19
    V_data[i] = sum(data['NewPositive'][0:i+10]) * 19
# Initial values
<<<<<<< Updated upstream
S_0 = S[days]
I_0 = I[days]
R_0 = R[days]
X_0 = [S_0,I_0,R_0]
test_data = X[days:days+simdays,:]
=======
N = 5800000
I_0 = test_data[0]
R_0 = 230000
V_0 = V_data[0]

X_0 = [N - I_0 - R_0, I_0, R_0]
>>>>>>> Stashed changes


# Find optimal parameters
#c1 = time.process_time()
#beta_opt, gamma_opt, errs = pest.estimate(
#    X_0=X_0,
#    data=test_data,
#    n_points=10,
#    layers=5
#)
#c2 = time.process_time()

#optimal parameter beta using frobenius norm
# using 
gamma = 1/9

c1 = time.process_time()
beta_opt, gamma_opt, errs = pestbeta.estimate_beta(
    X_0=X_0,
    data=X,
    gamma = gamma,
    n_points=100,
    layers=5)
c2 = time.process_time()
# Simulate optimal solution

X_0 = [N - I_0 - R_0, I_0, R_0]
mp = [beta_opt, gamma_opt, N]

t, SIR = b_ivp.simulateSIR(
    X_0=X_0,
    mp=mp,
    simtime=simdays,
    method=b_ivp.RK4
)

print("Simulation completed in", c2 - c1, "seconds.")

# Plot optimal solution
plt.plot(t, np.asarray(SIR)[:,1])
plt.title("Simulation using optimal parameters")
plt.legend(["Susceptible", "Infected", "Removed"])
plt.xlabel("Days since start,    Parameters: " + r'$\beta = $' + "{:.6f}".format(
    beta_opt) + ", " + r'$\gamma = $' + "{:.6f}".format(gamma_opt))
plt.ylabel("Number of people")
plt.ylim([0, N])
<<<<<<< Updated upstream
T = list(range(simdays))
plt.bar(T,I[days:days+simdays])
=======
T = list(range(num_days))
plt.bar(T, test_data)
>>>>>>> Stashed changes
plt.show()
#tikzplotlib.save('test.tex')


# data = np.array(errs)
# length = data.shape[0]
# width = data.shape[1]
# x, y = np.meshgrid(np.linspace(0, 1, length, endpoint=True), np.linspace(0, 1, width, endpoint=True))

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# ax.plot_surface(x, y, data.T)
# ax.set(xlabel=r'$\beta$', ylabel=r'$\gamma$', zlabel='Error')
# plt.show()
<<<<<<< Updated upstream


=======
t, SV = ivp.simulateSV(
    V_start=V_0,
    mp=mp,
    simtime=simdays,
    method=ivp.RK4V
)

# Plot optimal solution - kummuleret
plt.plot(t, SV)
plt.title("Simulation using optimal parameters")
plt.legend(["Acummulated"])
plt.xlabel("Days since start,    Parameters: " + r'$\beta = $' + "{:.6f}".format(
    beta_opt) + ", " + r'$\gamma = $' + "{:.6f}".format(gamma_opt))
plt.ylabel("Number of people")
plt.ylim([0, N])
T = list(range(num_days))
plt.bar(T, V_data)
plt.show()
>>>>>>> Stashed changes
