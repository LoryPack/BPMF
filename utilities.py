import numpy as np
import random
from numpy.random import multivariate_normal
from scipy.stats import wishart


def Normal_Wishart(mu_0, lamb, W, nu, seed=None):
    """Function extracting a Normal_Wishart random variable"""
    # first draw a Wishart distribution:
    Lambda = wishart(df=nu, scale=W, seed=seed).rvs()  # NB: Lambda is a matrix.
    # then draw a Gaussian multivariate RV with mean mu_0 and(lambda*Lambda)^{-1} as covariance matrix.
    cov = np.linalg.inv(lamb * Lambda)  # this is the bottleneck!!
    mu = multivariate_normal(mu_0, cov)
    return mu, Lambda, cov


def reduce_matrix(N_max, M_max, filename, correspondence_list_users, correspondence_list_movies, sep=" "):
    """In some datasets, the movies and users have a certain identifier that corresponds to one
       of a larger dataset; this means not all the user/movie identifier are used. Then it is better to
       reduce the matrix, in order to have a smaller representation.  We assume to have a correspondence list
       both for users and movies, i.e. a list where element i indicates the i-th used identifier; ex:
       correspondence_list_users = [1,3,7] means that the users 1,3,7 are respectively the 1st, 2nd and 3rd. Then
       they could be renamed in this way, saving a lot of space.
       """

    # first call ranking_matrix on the filename, generating a big matrix (many rows/columns will be empty)
    R = ranking_matrix(N_max, M_max, filename, sep)

    N_actual = len(correspondence_list_users)
    M_actual = len(correspondence_list_movies)

    R_reduced = np.zeros((N_actual, M_actual))

    for i, user in enumerate(correspondence_list_users):
        for j, movie in enumerate(correspondence_list_movies):
            R_reduced[i, j] = R[correspondence_list_users[i] - 1, correspondence_list_movies[j] - 1]

    return R_reduced


def ranking_matrix(N, M, filename, sep=" "):
    """Function creating the NxM rating matrix from filename.
    It assumes that the file contains on every line a triple (user, movie, ranking).
    Moreover, users' and movies are numbered starting from 1.
    """
    R = np.zeros((N, M))
    f = open(filename, "r")
    for line in f:
        if line[0] == '%':
            # this is a comment
            continue
        (user, movie, ranking) = line.split(sep)
        R[np.int(user) - 1, np.int(movie) - 1] = np.int(ranking)
    return R


def read_correspondence_list(filename):
    """Function reading the correspondence list from a -mtx file "filename"""
    corr_list = []
    f = open(filename, "r")
    for line in f:
        if line[0] == '%':
            continue
        corr_list.append(np.int(line))
    return corr_list
