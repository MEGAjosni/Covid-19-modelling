# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:22:09 2021

@author: Marcus
"""
import matplotlib.pyplot as plt
import expanded_ivp_funcs as e_ivp
import numpy as np
import time
import sys

# Import added vaccine data
Activated_vaccines = np.loadtxt('vac_data_kalender_14_04_2021.csv') # 1st observation is january 4th

simdays = 100
# X : State vector
# S : susceptible at risk
# I1 : regular infected
# I2 : ICU infected
# I3 : respirator infected
# R1 : regular recovered
# R2 : vaccinated
# R3 : dead

X_0 = np.array([
    5.2 * 10 ** 6,
    30000,
    400,
    300,
    300000,
    500000,
    700
])
# mp : model parameters
# beta : Infection rate parameter
# gamma1 : Rate of recovery for infected
# gamma2 : Rate of recovery for ICU
# gamma3 : Rate of recovery for respirator
# theta1 : Death rate at ICU
# theta2 : Death rate in respirator
# phi1 : Rate to ICU from infected
# phi2 : Rate to respirator from ICU
# N : Population
mp = [0.22,
      (1 / 7),
      (1 / 20),
      (1 / 20),
      0.2 * (1 / 20),
      0.2 * (1 / 20),
      0.001,
      0.5,
      (6 * 10 ** 6)]

t0 = time.time()
for i in range(1000):
    t, State_vec = e_ivp.simulateSIR(
        X_0=X_0,
        mp=mp,
        T=Activated_vaccines[30:130],
        simtime=simdays,
        stepsize=1,
        method=e_ivp.RK4
    )
t1 = time.time()
print('Total time:', (t1-t0)/1000, 's')


# format output from e_ivp.simulateSIR to be stackplotted (might be unnessecary :D)
S = []
I1 = []
I2 = []
I3 = []
R1 = []
R2 = []
R3 = []

I2_hat = 1000
I3_hat = 1000
error_vals1 = []
for i in range(0,len(t)):
    temp = State_vec[i]
    S.append(temp[0])
    I1.append(temp[1])
    I2.append(temp[2])
    I3.append(temp[3])
    R1.append(temp[4])
    R2.append(temp[5])
    R3.append(temp[6])

    e = max(temp[2]-I2_hat,0)+max(temp[3]-I3_hat,0)
    error_vals1.append(e)

plt.stackplot(t,S,I1,I2,I3,R1,R2,R3,labels = ["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"])
plt.title("Stacked area")
plt.legend(["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"],loc = 'lower left')

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 6])
plt.show()

plt.plot(t,State_vec)
plt.title(" All bins ")
plt.legend(["S", "I1", "I2", "I3", "R1", "R2 (vac)", "R3"])

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 6])
plt.show()


plt.plot(t,State_vec)
plt.title(" infected and dead ")
plt.legend(["I1", "I2", "I3", "R3"])

plt.ylabel("Number of people")
plt.ylim([0, 6 * 10 ** 5])
plt.show()

plt.plot(t,I2,I3)
plt.title("Intensive and respirator")
plt.legend([ "I2: intensive", "I3: Respirator"])

plt.show()


#%% PID control simulation
import math

K0 = [-1,-1/1000,-80]

t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
    X_0=X_0,
    mp=mp,
    T = Activated_vaccines[30:130],
    K = K0,
    simtime=simdays,
    stepsize=1,
    method=e_ivp.RK4
)
max_error = max(error_vals)
min_beta = min(beta_vals)

S_PID = []
I1_PID = []
I2_PID = []
I3_PID = []
R1_PID = []
R2_PID = []
R3_PID = []

for i in range(0,len(t)):
    temp = State_vec[i]
    S_PID.append(temp[0])
    I1_PID.append(temp[1])
    I2_PID.append(temp[2])
    I3_PID.append(temp[3])
    R1_PID.append(temp[4])
    R2_PID.append(temp[5])
    R3_PID.append(temp[6])


plt.plot(t,error_vals1,error_vals)
plt.title("Errors (number of people above threshold)")
plt.legend([ "Error with out PID", ""])
plt.show()

I3_threshold = [1000 for i in range(101)]
zeros = [0 for i in range(101)]

x = np.arange(0.0, 2, 0.01)
fig, axs = plt.subplots(2)
axs[0].plot(t,I3,t,I3_PID,t,I3_threshold)
axs[0].legend(["I3", "I3 with PID","I3 threshold"])
beta_vals.append(beta_vals[-1])
axs[1].plot(t,beta_vals)
axs[1].legend("Beta Values")

#%% PID control with "gradient descent"

K0 = [-1,-1/1000,-80]
n = 1000

t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
    X_0=X_0,
    mp=mp,
    T = Activated_vaccines[30:130],
    K = K0,
    simtime=simdays,
    stepsize=1,
    method=e_ivp.RK4
)
max_error = max(error_vals)
min_beta = min(beta_vals)


nb_min_beta = min_beta
for i in range(3):
    k1 = (i-1)/n
    for j in range(3):
        k2= (j-1)/n
        for k in range(3):
            k3 = (k-1)/n
            if i != 1 and j != 1 and k != 1:
                K_temp = [K0[0]+k1,K0[1]+k2,K0[2]+k3]
                t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=X_0,
                    mp=mp,
                    T = Activated_vaccines[30:130],
                    K = K_temp,
                    simtime=simdays,
                    stepsize=1,
                    method=e_ivp.RK4
                )
                max_error = max(error_vals)
                if max_error < 0 and min(beta_vals) > nb_min_beta:
                    nb_min_beta = min(beta_vals)
                    Kn = K_temp
K0 = Kn

while nb_min_beta > min_beta:
    min_beta = nb_min_beta
    for i in range(9):
        k1 = (i-4)/n
        for j in range(9):
            k2= (j-4)/n
            for k in range(9):
                k3 = (k-4)/n
                K_temp = [K0[0]+k1,K0[1]+k2,K0[2]+k3]

                t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
                    X_0=X_0,
                    mp=mp,
                    T = Activated_vaccines[30:130],
                    K = K_temp,
                    simtime=simdays,
                    stepsize=1,
                    method=e_ivp.RK4
                )
                max_error = max(error_vals)
                if max_error < 0 and min(beta_vals) > nb_min_beta:
                    nb_min_beta = min(beta_vals)
                    Kn = K_temp
    K0 = Kn
    print(K0)
    print(nb_min_beta)

t, State_vec,beta_vals,error_vals = e_ivp.simulateSIR_PID(
    X_0=X_0,
    mp=mp,
    T = Activated_vaccines[30:130],
    K = K0,
    simtime=simdays,
    stepsize=1,
    method=e_ivp.RK4
)

plt.plot(t,error_vals1,error_vals)
plt.title("Errors (number of people above threshold)")
plt.legend([ "Error with out PID", ""])
plt.show()

I3_threshold = [1000 for i in range(101)]
zeros = [0 for i in range(101)]

x = np.arange(0.0, 2, 0.01)
fig, axs = plt.subplots(2)
axs[0].plot(t,I3,t,I3_PID,t,I3_threshold)
axs[0].legend(["I3", "I3 with PID","I3 threshold"])
beta_vals.append(beta_vals[-1])
axs[1].plot(t,beta_vals)
axs[1].legend("Beta Values")

#%% Optimal parameters of the expanded model
import scipy.io
mat = scipy.io.loadmat('Inter_data.mat')
T = len(mat["All_Data"])

I2 = [0 for i in range(T)]
I3 = [0 for i in range(T)]
for i in range(T):
    [I2[i],I3[i]] = mat["All_Data"][i]
