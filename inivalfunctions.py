import math

def derivative(
        X: list,  # Vector to compute derivative of
        mp: list  # Model parameters [beta, gamma, N]
):
    # *** Description ***
    # Computes the derivative of X using model parameters

    # *** Output ***
    # dX [list]:            Derivative of X

    beta, gamma, N = mp
    S, I, R = X

    dX = [
        - beta * S * I / N,
        I * (beta * S / N - gamma),
        gamma * I
    ]

    return dX

def derivativeV(
        V_0: int, #Number of accumulated infected at time t_0
        V: int,  # Number to compute derivative of
        mp: list  # Model parameters [beta, gamma, N]
            ):
    beta, gamma, N = mp
    beta = beta/N
    v = gamma/beta
    I = V + v*math.log(N-V)-V_0-v*math.log(N-V_0);
    S = N-V
    dV = beta *S*I
    return dV

def RK4V()

def ExplicitEuler(
        X_k: list,  # Values of SIR at time t_k
        mp: list,  # Model parameters [beta, gamma, N]
        stepsize=0.1  # t_kp1 - t_k
):
    # *** Description ***
    # Uses Explicit Euler to solve ODE system numerically.

    # *** Output ***
    # X_kp1 [list]:         Values of SIR at time t_kp1

    N = range(len(X_k))
    dX_k = derivative(X_k, mp)

    X_kp1 = [X_k[i] + stepsize * dX_k[i] for i in N]

    return X_kp1


def RK4(
        X_k: list,  # Values of SIR at time t_k
        mp: list,  # Model parameters [beta, gamma, N]
        stepsize: float = 0.1  # t_kp1 - t_k
):
    # *** Description ***
    # Uses Rung Ketta 4 to solve ODE system numerically.

    # *** Output ***
    # X_kp1 [list]:         Values of SIR at time t_kp1

    N = range(len(X_k))

    K_1 = derivative(X_k, mp)
    K_2 = derivative([X_k[i] + 1 / 2 * stepsize * K_1[i] for i in N], mp)
    K_3 = derivative([X_k[i] + 1 / 2 * stepsize * K_2[i] for i in N], mp)
    K_4 = derivative([X_k[i] + stepsize * K_3[i] for i in N], mp)

    X_kp1 = [X_k[i] + stepsize / 6 * (K_1[i] + 2 * (K_2[i] + K_3[i]) + K_4[i]) for i in N]

    return X_kp1


def simulateSIR(
        X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
        mp: list,  # Model parameters [beta, gamma, N]
        simtime: int = 100,  # How many timeunits into the future that should be simulated
        stepsize: float = 0.1,  # t_kp1 - t_k
        method=ExplicitEuler  # Numerical method to be used [function]
):
    # *** Description ***
    # Simulate SIR-model.

    # *** Output ***
    # t [list]:             All points in time simulated
    # SIR [nested list]:    Values of SIR at all points in time t

    SIR = [X_0]

    t = [i * stepsize for i in range(int(simtime / stepsize) + 1)]

    for i in range(int(simtime / stepsize)):
        SIR.append(method(SIR[i], mp, stepsize))

    return t, SIR
