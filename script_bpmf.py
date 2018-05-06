import numpy as np
import random
from numpy.random import multivariate_normal
from scipy.stats import wishart
from bpmf import BPMF
import pandas as pd
from utilities import ranking_matrix, reduce_matrix, read_correspondence_list

# BPMF is having some convergence problems!!!!!!  I SHOULD MANUALLY CUT THE RATING VALUES!!

# THE ONLY WAY RMSE CAN INCREASE A LOT IS SINCE YOU MAY GET EXPLODING RATINGS IN THE PREDICTED
# MATRIX

"""File "/home/lorenzo/BPMF/my_code/bpmf.py", line 160, in BPMF
    V_new = np.append(V_new, multivariate_normal(mu_j_star, Lambda_j_star_V_inv))
  File "mtrand.pyx", line 4524, in mtrand.RandomState.multivariate_normal
  File "/home/lorenzo/miniconda3/lib/python3.6/site-packages/scipy/linalg/decomp_svd.py", line 132, in svd
    raise LinAlgError("SVD did not converge")
numpy.linalg.linalg.LinAlgError: SVD did not converge
"""

# WARNING:
"""/home/lorenzo/BPMF/my_code/bpmf.py:137: RuntimeWarning: covariance is not positive-semidefinite.
"""

# define the number of users and movies and load the matrices

N = 943  # these are for the very small example
M = 1682

# boundary for the possible ratings:
lowest_rating = 1
highest_rating = 5  # 10 for latest-small

N_max = 671

print('Loading data...')
"""
datapath = 'data/ml-latest-small/'

movies_corr_list = read_correspondence_list(datapath + "correspondance_list_movies.mtx")
users_corr_list = np.arange(671)
M_max = movies_corr_list[-1]  # M_max is the last element of this

R = reduce_matrix(N_max, M_max, datapath + "ml-latest-small-train.mtx", users_corr_list, movies_corr_list)
R_test = reduce_matrix(N_max, M_max, datapath + "ml-latest-small-test.mtx", users_corr_list, movies_corr_list)
"""
datapath= 'data/ml-very-small/'
R = ranking_matrix(N, M, datapath + "ml-train.mtx")
R_test = ranking_matrix(N, M, datapath + "ml-test.mtx")
print('End')


(N, M) = R.shape  # extract the actual number of users and movies
print("There are {} users and {} movies".format(N, M))

# EXPERIMENT PARAMETERS:
T = 100  # number of iterations
initial_cutoff = 0  # number of burn-in iterations (that are discarded)
# D_list = [10, 20, 30, 40, 50, 60]
D_list = [70, 80, 90, 100]  # number of hidden features
# D_list = [30, 40, 50, 60, 70, 80, 90, 100]
save_file_in_function = False  # set this to False if you want to save a single file for all values of D
R_pred_list = []
train_err_list = []
test_err_list = []

print(len(D_list), 'simulations have to be done, each with', T, 'iterations.')

results = pd.DataFrame(columns=['D', 'train_err', 'test_err', 'train_step'])

for D in D_list:
    print('D ', D)

    # starting values for the U, V matrices
    U_in = np.zeros((D, N))  # how should we get the starting values of them??  MAYBE THIS MAY GIVE NOT GOOD VALUES OF COVARIANCE MATRIX
    V_in = np.zeros((D, M))
    
    output_filename = "results/N{}_M{}_T{}_cutoff{}_D{}.csv".format(N, M, T, initial_cutoff, D)

    R_pred, train_err, test_err, train_epochs = BPMF(R, R_test, U_in, V_in, T, D, initial_cutoff, lowest_rating,
                                                     highest_rating, output_filename, save_file=save_file_in_function)

    row = pd.DataFrame.from_items([('D', D), ('train_err', train_err), ('test_err', test_err), ('train_step', train_epochs)])

    if not save_file_in_function: 
        results = results.append(row)  # save results at every iteration:
        results.to_csv("results/N{}_M{}_T{}_cutoff{}_D{}.csv".format(N, M, T, initial_cutoff, D_list))

    R_pred_list.append(R_pred)
    train_err_list.append(train_err)  # train_err is a list itself, so this will be a list of lists
    test_err_list.append(test_err)

print('Finished simulations; now saving results')
# results = pd.DataFrame.from_items([('D', D_list), ('train_err', train_err_list), ('test_err', test_err_list)])
# results.to_csv("results/N{}_M{}_T{}_cutoff{}_D{}.csv".format(N, M, T, initial_cutoff, D_list))
print('End')
