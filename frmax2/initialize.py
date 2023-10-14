import copy

import numpy as np


def bisection_search(func, e_init, i, eps):
    assert func(e_init) < 0.0, "initial sample must be negative"

    len(e_init)
    e_zero = copy.copy(e_init)
    e_init[i] = 0.0

    right = e_init  # positive side
    left = e_zero  # negative side

    eps += 1e-8  # to accept floating point error
    while True:
        precision = np.linalg.norm(left - right)
        if precision < eps:
            break
        mid = (right + left) * 0.5
        if func(mid) < 0:
            left = mid
        else:
            right = mid
    return left


def natural_basis(n_dim, i):
    basis = np.zeros(n_dim)
    basis[i] = 1.0
    return basis


def initialize(func, param_init, e_length, eps=1e-3):
    n_dim = len(param_init)
    m_dim = len(e_length)

    sample_list = []
    label_list = []
    for i in range(m_dim):
        e = natural_basis(m_dim, i)

        e_init_upper = e_length * e
        x_init_upper = np.hstack((param_init, e_init_upper))

        e_init_lower = -e_length * e
        x_init_lower = np.hstack((param_init, e_init_lower))

        x_surf_upper = bisection_search(func, x_init_upper, n_dim + i, eps)
        x_surf_lower = bisection_search(func, x_init_lower, n_dim + i, eps)

        sample_list.append(x_surf_upper)
        label_list.append(False)
        sample_list.append(x_surf_lower)
        label_list.append(False)

    # add positive sample
    sample_list.append(np.hstack((param_init, np.zeros(m_dim))))
    label_list.append(True)

    X = np.vstack(sample_list)
    Y = np.array(label_list)

    e_maxes = np.max(X[:, n_dim:], axis=0)
    e_mines = np.min(X[:, n_dim:], axis=0)
    ls_fr = (e_maxes - e_mines) * 0.25
    return X, Y, ls_fr
