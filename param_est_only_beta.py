import basic_ivp_funcs as b_ivp
import numpy as np


def estimate_beta(
        X_0: list,  # Initial values of SIR [S_0, I_0, R_0]
        data: list, # Data to fit model to
        gamma: float, # fixed gamma value
        n_points: int = 10,  # Number parameter values to test in intervals during each iteration
        layers: int = 3,  # Accuracy, number of iterations

):
    # *** Description ***
    # Computes the values of beta and gamma that gives the best modelfit on data using Mean Squared Error.

    # *** Output ***
    # beta_opt [scalar]:    Optimal value of beta
    # gamma_opt [scalar]:   Optimal value of gamma

    simdays = len(data)
    stepsize = 1

    N = sum(X_0)

    # Specify first search area
    beta = [i / n_points for i in range(n_points + 1)]


    # Iterate layers times
    errs = []
    beta_opt = 0
    for k in range(layers):
        min_err = -1
        bg_opt_index = [0, 0]
        errs = []
        for i in beta:
                mp = [i,gamma, N]
                t, SIR = b_ivp.simulateSIR(X_0, mp, simdays, stepsize, b_ivp.RK4)
                data_est = [SIR[int(i / stepsize)] for i in range(simdays)]
                data_est = np.asarray(data_est)
                err = (np.linalg.norm(data-data_est))**2
                if err < min_err or min_err == -1:
                    bg_opt_index = [i]
                    min_err = err
                errs.append(err)
                

        beta_opt = bg_opt_index[0]

        if k < layers - 1:
            beta = [i / (n_points ** (k + 2)) + beta_opt - 1 / (2 * n_points ** (k + 1)) for i in range(n_points + 1)]
            beta = [beta[i] if beta[i] >= 0 else 0 for i in range(len(beta))]
            
    return beta_opt, errs
