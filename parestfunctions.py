import inivalfunctions as ivp

def MeanSquaredError(
        data: list, # Data
        data_est: list # Estimate of data
):
    # *** Description ***

    n = len(data)

    if n == len(data_est):
        mse = sum([(data[i] - data_est[i]) ** 2 for i in range(n)]) / n
    else:
        print("Error in function MeanSquaredError: List don't have the same length")
        quit()

    return mse

def estimate(
        X_0: list, # Initial values of SIR [S_0, I_0, R_0]
        data: list, # Data to fit model to
        n_points: int=10, # Number parameter values to test in intervals during each iteration
        layers: int=3, # Accuracy, number of iterations
        method=MeanSquaredError # Method to assess fit
):
    # *** Description ***
    # Computes the values of beta and gamma that gives the best modelfit on data using Mean Squared Error.

    # *** Output ***
    # beta_opt [scalar]:    Optimal value of beta
    # gamma_opt [scalar]:   Optimal value of gamma

    simdays = len(data)
    stepsize = 0.1

    N = sum(X_0)

    # Specify first search area
    beta = [i / n_points for i in range(n_points + 1)]
    gamma = [i / n_points for i in range(n_points + 1)]

    # Iterate layers times
    for k in range(layers):
        min_err = -1
        bg_opt_index = [0, 0]
        errs = []
        for i in beta:
            gammaerrs = []
            for j in gamma:
                mp = [i, j, N]
                t, SIR = ivp.simulateSIR(X_0, mp, simdays, stepsize, ivp.RK4)
                data_est = [SIR[int(i / stepsize)][1] for i in range(simdays)]
                err = method(data, data_est)
                if err < min_err or min_err == -1:
                    bg_opt_index = [i, j]
                    min_err = err
                gammaerrs.append(err)
            errs.append(gammaerrs)

        beta_opt, gamma_opt = bg_opt_index[0], bg_opt_index[1]

        if k < layers - 1:
            beta = [i / (n_points ** (k + 2)) + beta_opt - 1 / (2 * n_points ** (k + 1)) for i in range(n_points + 1)]
            beta = [beta[i] if beta[i] >= 0 else 0 for i in range(len(beta))]
            gamma = [i / (n_points ** (k + 2)) + gamma_opt - 1 / (2 * n_points ** (k + 1)) for i in range(n_points + 1)]
            gamma = [gamma[i] if gamma[i] >= 0 else 0 for i in range(len(gamma))]

    return beta_opt, gamma_opt, errs
