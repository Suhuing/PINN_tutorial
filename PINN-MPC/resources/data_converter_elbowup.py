import numpy as np

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

# === 파일 로드 ===
data = np.load("data.npz")
converted_data = {}


def estimate_xy_bounds(lb, ub, num_samples=10000):
    alpha_range = np.random.uniform(lb[1], ub[1], num_samples)
    beta_range = np.random.uniform(lb[2], ub[2], num_samples)
    alphadot_range = np.random.uniform(lb[3], ub[3], num_samples)
    betadot_range = np.random.uniform(lb[4], ub[4], num_samples)

    x, y = fk_2dof(alpha_range, beta_range)
    J11, J12, J21, J22 = jacobian_2dof(alpha_range, beta_range)
    xdot = J11 * alphadot_range + J12 * betadot_range
    ydot = J21 * alphadot_range + J22 * betadot_range

    ee_lb = np.min(np.stack([x, y, xdot, ydot], axis=1), axis=0)
    ee_ub = np.max(np.stack([x, y, xdot, ydot], axis=1), axis=0)

    return ee_lb, ee_ub

def is_elbow_up(X):
    """Returns a boolean mask for elbow-up samples (beta < 0)"""
    if X.shape[1] == 7:
        beta = X[:, 2]
    elif X.shape[1] == 4:
        beta = X[:, 1]
    else:
        raise ValueError("X must have 4 or 7 columns.")
    return beta < 0



# === 변환 함수 ===
def convert_to_ee(X, U=None):
    if X.shape[1] == 7:
        t = X[:, 0]
        alpha = X[:, 1]
        beta = X[:, 2]
        alphadot = X[:, 3]
        betadot = X[:, 4]
        u1 = X[:, 5] if U is None else U[:, 0]
        u2 = X[:, 6] if U is None else U[:, 1]
    elif X.shape[1] == 4:
        t = np.zeros((X.shape[0],))
        alpha = X[:, 0]
        beta = X[:, 1]
        alphadot = X[:, 2]
        betadot = X[:, 3]
        u1 = np.zeros((X.shape[0],))
        u2 = np.zeros((X.shape[0],))
    else:
        raise ValueError("X must have 4 or 7 columns.")

    x, y = fk_2dof(alpha, beta)
    J11, J12, J21, J22 = jacobian_2dof(alpha, beta)
    xdot = J11 * alphadot + J12 * betadot
    ydot = J21 * alphadot + J22 * betadot

    return np.stack([t, x, y, xdot, ydot, u1, u2], axis=1)

mask_X      = is_elbow_up(data['X'])
elbow_up_indices = np.where(mask_X)[0]  # elbow-up인 X의 인덱스

converted_data['X'] = convert_to_ee(data['X'][elbow_up_indices], data['X'][elbow_up_indices, 5:7])
converted_data['Y'] = convert_to_ee(data['Y'][elbow_up_indices], np.zeros((len(elbow_up_indices), 2)))[:, 1:5]
converted_data['X_test'] = convert_to_ee(data['X_test'], data['X_test'][:, 5:7])
converted_data['Y_test'] = convert_to_ee(data['Y_test'], np.zeros((data['Y_test'].shape[0], 2)))[:, 1:5]
converted_data['X0'] = convert_to_ee(data['X0'], np.zeros((data['X0'].shape[0], 2)))[:, 1:5]

# 복사 항목
converted_data['U'] = data['U']
converted_data['T'] = data['T']

converted_data['ub'] = data['ub']
converted_data['lb'] = data['lb']

# 저장
np.savez("converted_all_data_elbow_up.npz", **converted_data)

ee_lb, ee_ub = estimate_xy_bounds(data['lb'], data['ub'])
print("End-effector space lower bounds:", ee_lb)  # [x_min, y_min, ẋ_min, ẏ_min]
print("End-effector space upper bounds:", ee_ub)  # [x_max, y_max, ẋ_max, ẏ_max]
