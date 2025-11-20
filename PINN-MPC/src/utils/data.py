import os

import numpy as np
from pyDOE import lhs
from scipy.io import loadmat

# === FK 및 Jacobian 함수 정의 ===
def fk_2dof(alpha, beta, l0=0.2236, l1=0.18, l2=0.5586):
    x = -l1 * np.sin(alpha) - l2 * np.sin(alpha + beta)
    y = l0 + l1 * np.cos(alpha) + l2 * np.cos(alpha + beta)
    return x, y

def jacobian_2dof(alpha, beta, l1=0.18, l2=0.5586):
    J11 = -l1 * np.cos(alpha) - l2 * np.cos(alpha + beta)
    J12 = -l2 * np.cos(alpha + beta)
    J21 = -l1 * np.sin(alpha) - l2 * np.sin(alpha + beta)
    J22 = -l2 * np.sin(alpha + beta)
    return J11, J12, J21, J22



def load_data(path):
    """
    Loads reference data and input bounds.

    :param path: path of the reference data, stored in 'pendulum.npz'
    :return
    np.ndarray lb: lower bounds of the inputs of the training data,
    np.ndarray ub: upper bounds of the inputs of the training data,
    int input_dim: dimension of the inputs,
    int output_dim: dimension of the outputs,
    np.ndarray X_test: input tensor of the testing data,
    np.ndarray Y_test output tensor of the testing data,
    np.ndarray X_test: input tensor of the testing data,
    np.ndarray Y_test output tensor of the testing data,
    """

    npzfile = np.load(path)

    # Lower and upper bound
    lb = npzfile['lb']
    ub = npzfile['ub']

    # All data
    X_star = npzfile['X']
    Y_star = npzfile['Y']

    X_test = npzfile['X_test']
    Y_test = npzfile['Y_test']

    input_dim = X_star.shape[1]
    output_dim = Y_star.shape[1]

    return lb, ub, input_dim, output_dim, X_test, Y_test, X_star, Y_star


def generate_data_points(N_z, lb, ub):
    X_data = np.hstack((np.zeros((N_z, 1)), lb[1:] + (ub[1:] - lb[1:]) * lhs(len(ub) - 1, N_z))) # lhs는 latin hypercube sampling 0~1사이에 균등하게 샘플되는듯
    Y_data = X_data[:, 1:5]
    return X_data, Y_data


def generate_collocation_points(N_phys, lb, ub):
    X_phys = lb + (ub - lb) * lhs(len(ub), N_phys)
    return X_phys


def generate_data_ee_points(N_z, lb, ub):
    sampled = lb[1:] + (ub[1:] - lb[1:]) * lhs(len(ub) - 1, N_z)
    alpha, beta = sampled[:, 0], sampled[:, 1]
    alphadot, betadot = sampled[:, 2], sampled[:, 3]
    u1, u2 = sampled[:, 4], sampled[:, 5]

    # task space로 변환
    x, y = fk_2dof(alpha, beta)
    J11, J12, J21, J22 = jacobian_2dof(alpha, beta)
    xdot = J11 * alphadot + J12 * betadot
    ydot = J21 * alphadot + J22 * betadot

    # t=0 추가하고 구성
    t_col = np.zeros((N_z, 1))
    x_data = np.hstack((t_col, x[:, None], y[:, None], xdot[:, None], ydot[:, None], u1[:, None], u2[:, None]))
    y_data = x_data[:, 1:5]  # [x, y, xdot, ydot]

    return x_data, y_data


def generate_collocation_ee_points(N_phys, lb, ub):
    q_samples = lb + (ub - lb) * lhs(len(ub), N_phys)  # shape: (N, 7)

    # 해석: [t, α, β, α̇, β̇, u₁, u₂]
    t = q_samples[:, 0]
    alpha = q_samples[:, 1]
    beta = q_samples[:, 2]
    alphadot = q_samples[:, 3]
    betadot = q_samples[:, 4]
    u1 = q_samples[:, 5]
    u2 = q_samples[:, 6]

    # FK + 자코비안
    x, y = fk_2dof(alpha, beta)
    J11, J12, J21, J22 = jacobian_2dof(alpha, beta)
    xdot = J11 * alphadot + J12 * betadot
    ydot = J21 * alphadot + J22 * betadot

    # 새로운 7차원 구성: [t, x, y, xdot, ydot, u1, u2]
    X_phys = np.stack([t, x, y, xdot, ydot, u1, u2], axis=1)
    return X_phys




def load_ref_trajectory(path):
    X_12_ref = loadmat(os.path.join(path, 'y_soll.mat'))['y_soll'].T
    X_34_ref = loadmat(os.path.join(path, 'Dy_soll.mat'))['Dy_soll'].T
    X_ref = np.hstack((X_12_ref, X_34_ref))

    T_ref = loadmat(os.path.join(path, 't_soll.mat'))['t_soll'].T

    freq = 10
    X_ref = X_ref[::freq]
    T_ref = T_ref[::freq]

    return X_ref, T_ref