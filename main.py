import basic_ivp_funcs as b_ivp
import params_est_funcs as pest
import get_data as gd
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd
import tikzplotlib


# Get data
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


# Initial values
S_0 = S[days]
I_0 = I[days]
R_0 = R[days]
X_0 = [S_0,I_0,R_0]
test_data = X[days:days+simdays,:]


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
# using gamma = 1/9

c1 = time.process_time()
beta_opt, gamma_opt, errs = pest.estimate(
    X_0=X_0,
    data=test_data,
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
T = list(range(total_days))
plt.bar(T, test_data)
tikzplotlib.save('test.tex')
plt.show()

# data = np.array(errs)
# length = data.shape[0]
# width = data.shape[1]
# x, y = np.meshgrid(np.linspace(0, 1, length, endpoint=True), np.linspace(0, 1, width, endpoint=True))

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1, projection='3d')
# ax.plot_surface(x, y, data.T)
# ax.set(xlabel=r'$\beta$', ylabel=r'$\gamma$', zlabel='Error')
# plt.show()
