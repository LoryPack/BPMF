import numpy as np
import random
import pandas as pd
from numpy.random import multivariate_normal
from scipy.stats import wishart

from utilities import Normal_Wishart  # this gives a symmetric matrix


def BPMF(R, R_test, U_in, V_in, T, D, initial_cutoff, lowest_rating, highest_rating, output_file,
         mu_0=None, Beta_0=None, W_0=None, nu_0=None, save_file=True):
    """
    R is the ranking matrix (NxM, N=#users, M=#movies); we are assuming that R[i,j]=0 means that user i has not ranked movie j
    R_test is the ranking matrix that contains test values. Same assumption as above. 
    U_in, V_in are the initial values for the MCMC procedure. 
    T is the number of steps. 
    D is the number of hidden features that are assumed in the model.    
    
    mu_0 is the average vector used in sampling the multivariate normal variable
    Beta_0 is a coefficient (?)
    W_0 is the DxD scale matrix in the Wishart sampling 
    nu_0 is the number of degrees of freedom used in the Wishart sampling. 
    
    U matrices are DxN, while V matrices are DxM.
    
    If save_file=True, this function internally saves the file at each iteration; this results in a different file for each value 
    of D and is useful when the algorithm may stop during the execution. 
    """

    def ranked(i, j):  # function telling if user i ranked movie j in the train dataset.
        if R[i, j] != 0:
            return True
        else:
            return False

    def ranked_test(i, j):  # function telling if user i ranked movie j in the test dataset.
        if R_test[i, j] != 0:
            return True
        else:
            return False

    N = R.shape[0]
    M = R.shape[1]

    R_predict = np.zeros((N, M))
    U_old = np.array(U_in)
    V_old = np.array(V_in)

    train_err_list = []
    test_err_list = []
    train_epoch_list = []

    # initialize now the hierarchical priors:
    alpha = 2  # observation noise, they put it = 2 in the paper
    mu_u = np.zeros((D, 1))
    mu_v = np.zeros((D, 1))
    Lambda_U = np.eye(D)
    Lambda_V = np.eye(D)

    # COUNT HOW MAY PAIRS ARE IN THE TEST AND TRAIN SET:
    pairs_test = 0
    pairs_train = 0
    for i in range(N):
        for j in range(M):
            if ranked(i, j):
                pairs_train = pairs_train + 1
            if ranked_test(i, j):
                pairs_test = pairs_test + 1

    # print(pairs_test, pairs_train)

    # SET THE DEFAULT VALUES for Wishart distribution
    # we assume that parameters for both U and V are the same.

    if mu_0 is None:
        mu_0 = np.zeros(D)
    if nu_0 is None:
        nu_0 = D
    if Beta_0 is None:
        Beta_0 = 2
    if W_0 is None:
        W_0 = np.eye(D)
        
    # results = pd.DataFrame(columns=['step', 'train_err', 'test_err'])

    for t in range(T):
        # print("Step ", t)
        # FIRST SAMPLE THE HYPERPARAMETERS, conditioned on the present step user and movie feature matrices U_t and V_t:

        # parameters common to both distributions:
        Beta_0_star = Beta_0 + N
        nu_0_star = nu_0 + N
        W_0_inv = np.linalg.inv(W_0)  # compute the inverse once and for all

        # movie hyperparameters:
        V_average = np.sum(V_old, axis=1) / N  # in this way it is a 1d array!!
        # print (V_average.shape)
        S_bar_V = np.dot(V_old, np.transpose(V_old)) / N  # CHECK IF THIS IS RIGHT!
        mu_0_star_V = (Beta_0 * mu_0 + N * V_average) / (Beta_0 + N)
        W_0_star_V_inv = W_0_inv + N * S_bar_V + Beta_0 * N / (Beta_0 + N) * np.dot(
            np.transpose(np.array(mu_0 - V_average, ndmin=2)), np.array((mu_0 - V_average), ndmin=2))
        W_0_star_V = np.linalg.inv(W_0_star_V_inv)
        mu_V, Lambda_V, cov_V = Normal_Wishart(mu_0_star_V, Beta_0_star, W_0_star_V, nu_0_star, seed=None)

        # user hyperparameters
        # U_average=np.transpose(np.array(np.sum(U_old, axis=1)/N, ndmin=2)) #the np.array and np.transpose are needed for it to be a column vector
        U_average = np.sum(U_old, axis=1) / N  # in this way it is a 1d array!!  #D-long
        # print (U_average.shape)
        S_bar_U = np.dot(U_old, np.transpose(U_old)) / N  # CHECK IF THIS IS RIGHT! #it is DxD
        mu_0_star_U = (Beta_0 * mu_0 + N * U_average) / (Beta_0 + N)
        W_0_star_U_inv = W_0_inv + N * S_bar_U + Beta_0 * N / (Beta_0 + N) * np.dot(
            np.transpose(np.array(mu_0 - U_average, ndmin=2)), np.array((mu_0 - U_average), ndmin=2))
        W_0_star_U = np.linalg.inv(W_0_star_U_inv)
        mu_U, Lambda_U, cov_U = Normal_Wishart(mu_0_star_U, Beta_0_star, W_0_star_U, nu_0_star, seed=None)

        # print (S_bar_U.shape, S_bar_V.shape)
        # print (np.dot(np.transpose(np.array(mu_0-U_average, ndmin=2)),np.array((mu_0-U_average), ndmin=2).shape))

        # UP TO HERE IT PROBABLY WORKS, FROM HERE ON IT HAS TO BE CHECKED!!!

        """SAMPLE THEN USER FEATURES (possibly in parallel):"""

        U_new = np.array([])  # define the new stuff.
        V_new = np.array([])

        for i in range(N):  # loop over the users
            # first compute the parameters of the distribution
            Lambda_U_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            for j in range(M):  # loop over the movies
                if ranked(i, j):  # only if movie j has been ranked by user i!
                    Lambda_U_2 = Lambda_U_2 + np.dot(np.transpose(np.array(V_old[:, j], ndmin=2)),
                                                     np.array((V_old[:, j]), ndmin=2))  # CHECK
                    mu_i_star_1 = V_old[:, j] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                    # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

            Lambda_i_star_U = Lambda_U + alpha * Lambda_U_2
            Lambda_i_star_U_inv = np.linalg.inv(Lambda_i_star_U)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_U,
                                                          mu_U)  ###CAREFUL!! Multiplication matrix times a row vector!! It should give as an output a row vector as for how it works
            mu_i_star = np.dot(Lambda_i_star_U_inv, mu_i_star_part)
            # extract now the U values!
            U_new = np.append(U_new, multivariate_normal(mu_i_star, Lambda_i_star_U_inv))

        # you need to reshape U_new and transpose it!!
        U_new = np.transpose(np.reshape(U_new, (N, D)))
        # print (U_new.shape)

        """SAMPLE THEN MOVIE FEATURES (possibly in parallel):"""

        for j in range(M):
            Lambda_V_2 = np.zeros((D, D))  # second term in the construction of Lambda_U
            mu_i_star_1 = np.zeros(D)  # first piece of mu_i_star
            for i in range(N):  # loop over the movies
                if ranked(i, j):
                    Lambda_V_2 = Lambda_V_2 + np.dot(np.transpose(np.array(U_new[:, i], ndmin=2)),
                                                     np.array((U_new[:, i]), ndmin=2))
                    mu_i_star_1 = U_new[:, i] * R[i, j] + mu_i_star_1  # CHECK DIMENSIONALITY!!!!!!!!!!!!
                    # coeff=np.transpose(np.array(V_old[j]*R[i,j], ndmin=2))+coeff  #CHECK DIMENSIONALITY!!!!!!!!!!!!

            Lambda_j_star_V = Lambda_V + alpha * Lambda_V_2
            Lambda_j_star_V_inv = np.linalg.inv(Lambda_j_star_V)

            mu_i_star_part = alpha * mu_i_star_1 + np.dot(Lambda_V, mu_V)
            mu_j_star = np.dot(Lambda_j_star_V_inv, mu_i_star_part)
            V_new = np.append(V_new, multivariate_normal(mu_j_star, Lambda_j_star_V_inv))

        # you need to reshape U_new and transpose it!!
        V_new = np.transpose(np.reshape(V_new, (M, D)))

        # save U_new and V_new in U_old and V_old for next iteration:         
        U_old = np.array(U_new)
        V_old = np.array(V_new)

        # print (V_new.shape)
        # print (V_new.shape, U_new.shape)

        if t > initial_cutoff:  # initial_cutoff is needed to discard the initial transient
            R_step = np.dot(np.transpose(U_new), V_new)
            for i in range(N):  # reduce all the predictions to the correct ratings range.
                for j in range(M):
                    if R_step[i, j] > highest_rating:
                        R_step[i, j] = highest_rating
                    elif R_step[i, j] < lowest_rating:
                        R_step[i, j] = lowest_rating

            R_predict = (R_predict * (t - initial_cutoff - 1) + R_step) / (t - initial_cutoff)
            train_err = 0  # initialize the errors.
            test_err = 0

            # compute now the RMSE on the train dataset:
            for i in range(N):
                for j in range(M):
                    if ranked(i, j):
                        train_err = train_err + (R_predict[i, j] - R[i, j]) ** 2
            train_err_list.append(np.sqrt(train_err / pairs_train))
            print("Training RMSE at iteration ", t - initial_cutoff, " :   ", "{:.4}".format(train_err_list[-1]))
            # compute now the RMSE on the test dataset:
            for i in range(N):
                for j in range(M):
                    if ranked_test(i, j):
                        test_err = test_err + (R_predict[i, j] - R_test[i, j]) ** 2
            test_err_list.append(np.sqrt(test_err / pairs_test))
            print("Test RMSE at iteration ", t - initial_cutoff, " :   ", "{:.4}".format(test_err_list[-1]))
            
            train_epoch_list.append(t)

            # row = pd.DataFrame.from_items([('step', t), ('train_err', train_err), ('test_err', test_err)])
            # results = results.append(row)  # save results at every iteration:
            results = pd.DataFrame.from_items([('step', train_epoch_list), ('train_err', train_err_list), ('test_err', test_err_list)])
            if save_file: 
                results.to_csv(output_file)   
    
    return R_predict, train_err_list, test_err_list, train_epoch_list
