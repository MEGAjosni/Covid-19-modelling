import inivalfunctions as ivp
import parestfunctions as pest
import get_data as gd
import matplotlib.pyplot as plt
import time
import datetime as dt
import pandas as pd


# Get data
s = pd.to_datetime('2020-12-01')
b = 8

print(gd.vaccine_dict['FaerdigVacc_daekning_DK_prdag']['Kumuleret antal f�rdigvacc.'])

data = gd.infect_dict['Test_pos_over_time'][s - dt.timedelta(days=b): s + dt.timedelta(days=21)]

test_data = [None] * 21
for i in range(21):
    test_data[i] = sum(data['NewPositive'][i:i+10]) * 19


# Initial values
N = 5800000
I_0 = test_data[0]
R_0 = 230000

X_0 = [N - I_0 - R_0, I_0, R_0]


# Find optimal parameters
c1 = time.process_time()
beta_opt, gamma_opt, errs = pest.estimate(
    X_0=X_0,
    data=test_data,
    n_points=10,
    layers=5
)
c2 = time.process_time()

# Simulate optimal solution
simdays = 100

X_0 = [N - I_0 - R_0, I_0, R_0]
mp = [beta_opt, gamma_opt, N]

t, SIR = ivp.simulateSIR(
    X_0=X_0,
    mp=mp,
    simtime=simdays,
    method=ivp.RK4
)

print("Simulation completed in", c2 - c1, "seconds.")

# Plot optimal solution
plt.plot(t, SIR)
plt.title("Simulation using optimal parameters")
plt.legend(["Susceptible", "Infected", "Removed"])
plt.xlabel("Days since start,    Parameters: " + r'$\beta = $' + "{:.6f}".format(
    beta_opt) + ", " + r'$\gamma = $' + "{:.6f}".format(gamma_opt))
plt.ylabel("Number of people")
plt.ylim([0, N])
T = list(range(21))
plt.bar(T, test_data)
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
