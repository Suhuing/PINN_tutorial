import numpy as np
from system import f,f_tensor  # 앞서 정의하신 f(t,x,u)
import torch
import torch.nn as nn
from typing import Union, Tuple
import time


def ik_2dof(x, y, l0=0.2236, l1=0.18, l2=0.5586,
            elbow_up=True, device="cpu", dtype=torch.float64):
    # 1. 입력을 Tensor 로 변환
    x = torch.as_tensor(x, dtype=dtype, device=device)
    y = torch.as_tensor(y, dtype=dtype, device=device)

    # 2. 나머지 계산도 전부 torch 연산
    y_adj = y - l0
    r2 = x**2 + y_adj**2

    eps = 1e-6
    cos_beta = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_beta = torch.clamp(cos_beta, -1.0 + eps, 1.0 - eps)

    raw_beta = torch.acos(cos_beta)
    beta = -raw_beta if elbow_up else raw_beta

    k1 = l1 + l2 * torch.cos(beta)
    k2 = l2 * torch.sin(beta)

    alpha = torch.atan2(-x, y_adj) - torch.atan2(k2, k1)

    # wrap to (-π,π]
    alpha = torch.atan2(torch.sin(alpha), torch.cos(alpha))
    beta  = torch.atan2(torch.sin(beta),  torch.cos(beta))
    return alpha, beta

def J_inv(
        alpha: Union[float, torch.Tensor],
        beta : Union[float, torch.Tensor],
        *,
        l1: float        = 0.18,
        l2: float        = 0.5586,
        eps_det: float   = 1e-3,     # det|J| ≤ eps_det → DLS
        lam: float       = 3e-2,     # DLS damping λ
        device=None,
        dtype: torch.dtype = torch.float64,
        squeeze_singleton: bool = True,  # batch=1이면 (2,2)로 반환
    ) -> torch.Tensor:
    """
    Safe inverse / damped-pseudo-inverse of the planar 2-DOF Jacobian.

    Parameters
    ----------
    alpha, beta : float | Tensor[...,]   (radians)
    returns     : Tensor[..., 2, 2]      (batch×2×2) or (2×2) if batch==1
    """

    # 0) 입력을 Tensor로 통일
    if device is None:
        device = alpha.device if torch.is_tensor(alpha) else "cpu"

    alpha = torch.as_tensor(alpha, dtype=dtype, device=device).flatten()
    beta  = torch.as_tensor(beta , dtype=dtype, device=device).flatten()
    B     = alpha.size(0)

    # 1) Jacobian
    J11 = -l1*torch.cos(alpha) - l2*torch.cos(alpha + beta)
    J12 = -l2*torch.cos(alpha + beta)
    J21 = -l1*torch.sin(alpha) - l2*torch.sin(alpha + beta)
    J22 = -l2*torch.sin(alpha + beta)
    J = torch.stack((
            torch.stack((J11, J12), dim=-1),   # row1
            torch.stack((J21, J22), dim=-1)    # row2
        ), dim=-2)                              # (B,2,2)

    # 2) det 기준으로 정확 역 vs DLS
    det  = J[:, 0, 0]*J[:, 1, 1] - J[:, 0, 1]*J[:, 1, 0]
    safe = det.abs() > eps_det
    Jinv = torch.empty_like(J)

    if safe.any():
        Jinv[safe] = torch.linalg.inv(J[safe])          # 정확 역

    if (~safe).any():
        I  = torch.eye(2, dtype=dtype, device=device).expand(B, -1, -1)
        JT = J.transpose(-1, -2)
        J_dls = JT @ torch.linalg.inv(J @ JT + (lam**2) * I)
        Jinv[~safe] = J_dls[~safe]                      # DLS 의사역

    # 3) 배치가 1개면 (2,2)로 평탄화
    if squeeze_singleton and Jinv.size(0) == 1:
        Jinv = Jinv.squeeze(0)   # (2,2)

    return Jinv

    
def compute_qdot_from_xdot(
        alpha, beta, dxdy,
        *,
        l1: float = 0.18,
        l2: float = 0.5586,
        eps_det: float = 1e-3,
        lam: float = 3e-2,
        device: torch.device = torch.device("cpu"),
        dtype : torch.dtype  = torch.float64
    ):
    """
    alpha, beta : scalar or (B,) Tensor
    dxdy        : (2,) or (B,2) Tensor     – [ẋ, ẏ]
    returns     : dq1, dq2  (scalar or (B,))
    """

    # 0) 입력을 Tensor로 통일
    alpha = torch.as_tensor(alpha, dtype=dtype, device=device).flatten()   # (B,)
    beta  = torch.as_tensor(beta , dtype=dtype, device=device).flatten()   # (B,)
    dxdy  = torch.as_tensor(dxdy , dtype=dtype, device=device)
    if dxdy.dim() == 1:                                                    # (2,) → (1,2)
        dxdy = dxdy.unsqueeze(0)

    # 1) 안전 Jacobian 역/의사역 구하기
    Jinv = J_inv(alpha, beta,
                 l1=l1, l2=l2,
                 eps_det=eps_det, lam=lam,
                 device=device, dtype=dtype,
                 squeeze_singleton=False)          # (B,2,2)

    # 2) q̇ = J^{-1} ẋ   (batch matmul)
    dq = torch.bmm(Jinv, dxdy.unsqueeze(-1)).squeeze(-1)  # (B,2)

    # 3) 배치가 1개면 스칼라처럼 반환
    if dq.size(0) == 1:
        dq = dq.squeeze(0)                          # (2,)

    return dq[..., 0], dq[..., 1]  


def simulate_rk4(x0, us, dt):
    N = us.shape[0]
    X = np.zeros((N+1, x0.shape[0]))
    # alpha, beta = ik_2dof(x0[0],x0[1])
    # alpha_dot, beta_dot = compute_qdot_from_xdot(alpha,beta,x0[2:4])
    # # 모두 NumPy로 변환하고 float로 캐스팅
    # alpha = alpha.item() if alpha.dim() == 0 else alpha.detach().cpu().numpy()
    # beta = beta.item() if beta.dim() == 0 else beta.detach().cpu().numpy()
    # alpha_dot = alpha_dot.item() if alpha_dot.dim() == 0 else alpha_dot.detach().cpu().numpy()
    # beta_dot = beta_dot.item() if beta_dot.dim() == 0 else beta_dot.detach().cpu().numpy()
    # x0_new = np.array([alpha, beta, alpha_dot, beta_dot])
    X[0] = x0 
    t = 0.0
    # print("x0",x0)
    for k in range(N):
        
            
        xk = X[k]
        uk = us[k]
        k1 = f_tensor(t,            xk,               uk)
        k2 = f_tensor(t + dt/2,     xk + dt/2*k1,     uk)
        k3 = f_tensor(t + dt/2,     xk + dt/2*k2,     uk)
        k4 = f_tensor(t + dt,       xk + dt*k3,       uk)
        X[k+1] = xk + dt*(k1 + 2*k2 + 2*k3 + k4)/6
        # if k==0:
        #     print(f"alpha dot, beta dot: ",(k1 + 2*k2 + 2*k3 + k4)/6)
        t += dt
    return X



def subsample_rk4_simulation(x0, us, dt_fine=0.005, dt_sample=0.0625):
    """
    Run RK4 simulation with fine time step and subsample at coarse time steps.
    
    Parameters:
        x0 (np.ndarray): Initial state, shape (4,)
        us (np.ndarray): Control inputs, shape (N, 2)
        dt_fine (float): Fine time step for integration (e.g., 0.005)
        dt_sample (float): Time step for saving data (e.g., 0.0625)
    
    Returns:
        np.ndarray: Subsampled state trajectory, shape (len(time_stamps), 4)
    """
    from scipy.interpolate import interp1d
    
    N = us.shape[0]
    T = N * dt_sample
    steps_per_sample = int(dt_sample / dt_fine)
    total_steps = N * steps_per_sample

    X_fine = np.zeros((total_steps + 1, x0.shape[0]))
    X_fine[0] = x0
    t = 0.0

    def f_tensor(t, x, u):
        # Dummy dynamics for placeholder; replace with actual f_tensor implementation
        return np.array([x[2], x[3], u[0], u[1]])

    for k in range(total_steps):
        idx_u = min(k // steps_per_sample, N - 1)
        uk = us[idx_u]
        xk = X_fine[k]

        k1 = f_tensor(t, xk, uk)
        k2 = f_tensor(t + dt_fine/2, xk + dt_fine/2 * k1, uk)
        k3 = f_tensor(t + dt_fine/2, xk + dt_fine/2 * k2, uk)
        k4 = f_tensor(t + dt_fine,   xk + dt_fine * k3, uk)

        X_fine[k+1] = xk + dt_fine * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt_fine

    # Subsample the result every steps_per_sample
    X_sampled = X_fine[::steps_per_sample]
    return X_sampled




def simulate_euler(x0, us, dt):
    """
    Simulates system dynamics using Euler integration.
    
    :param x0: np.ndarray, shape (n,)
    :param us: np.ndarray, shape (N, m), control inputs
    :param dt: float, time step
    :return: X, np.ndarray, shape (N+1, n)
    """
    N = us.shape[0]
    X = np.zeros((N + 1, x0.shape[0]))
    X[0] = x0
    t = 0.0
    
    for k in range(N):
        xk = X[k]
        uk = us[k]
        dxdt = f_tensor(t, xk, uk)  # f_tensor은 dx/dt 반환
        X[k + 1] = xk + dt * dxdt
        t += dt

    return X



def simulate_rk4_joint(x0_ee, us, dt):
    """
    Simulate joint-space trajectory using RK4, given EE initial state.
    
    :param x0_ee: numpy array (4,) = [x, y, xdot, ydot]
    :param us:    numpy array (N,2) control inputs
    :param dt:    float timestep
    :return:      numpy array (N+1,4) = [q1, q2, dq1, dq2] sequence
    """ 
    N = us.shape[0]
    Xq = np.zeros((N+1, 4))
    
    # 1) EE initial -> joint initial
    x0, y0, xdot0, ydot0 = x0_ee
    q1_0, q2_0    = ik_2dof(x0, y0)
    dq1_0, dq2_0  = compute_qdot_from_xdot(q1_0, q2_0, np.array([xdot0, ydot0]))
    # to Python floats
    q1_0   = float(q1_0)
    q2_0   = float(q2_0)
    dq1_0  = float(dq1_0)
    dq2_0  = float(dq2_0)
    
    Xq[0] = [q1_0, q2_0, dq1_0, dq2_0]
    t = 0.0
    
    for k in range(N):
        qk   = Xq[k]
        uk   = us[k]
        
        # RK4 in joint space
        k1 = f_tensor(t,         qk,               uk)
        k2 = f_tensor(t + dt/2,  qk + dt/2 * k1,   uk)
        k3 = f_tensor(t + dt/2,  qk + dt/2 * k2,   uk)
        k4 = f_tensor(t + dt,    qk +     dt * k3, uk)
        
        Xq[k+1] = qk + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        t += dt
    
    return Xq


def print_det_of_jacobian_inv(X_chunk, name=""):
    print(f"\n🧾 Checking Jacobian Inverse det for chunk: {name}")
    for i, row in enumerate(X_chunk):
        t, x, y, xdot, ydot, _, _ = row  # X_test는 [t, x, y, xdot, ydot, u1, u2]
        alpha, beta = ik_2dof(x, y)
        dxdy = torch.tensor([xdot, ydot], dtype=torch.float64)
        
        Jinv = J_inv(alpha, beta, squeeze_singleton=True)  # (2,2)
        det = torch.linalg.det(Jinv)
        
        print(f"[t = {t:.4f}] det(J_inv) = {det.item():.6f}")



def main():
    # 1) 데이터 로드
    data_ts = np.load('data.npz')
    data_js = np.load('data.npz')
    X_test = data_ts['X_test']   # shape (800, 7)
    Y_test = data_ts['Y_test']   # shape (800, 4)

    X_test_js = data_js['X_test']   # shape (800, 7)
    Y_test_js = data_js['Y_test']   # shape (800, 4)

    dt = 0.005
    chunk_size = 40

    # dt = 0.0625
    # chunk_size = 5
    num_chunks = X_test.shape[0] // chunk_size

    mse_list = []
    all_preds = []  # 예측 상태 저장
    chunk_data = []  # 각 청크별 (X, Y, Y_pred) 저장
    chunk_data_js = []
    start = time.perf_counter()
    for i in range(num_chunks):
        # 2) i번째 청크 분리
        chunk_X = X_test[i*chunk_size:(i+1)*chunk_size]
        chunk_Y = Y_test[i*chunk_size:(i+1)*chunk_size]

        # chunk_X_js = X_test_js[i*chunk_size:(i+1)*chunk_size]
        # chunk_Y_js = Y_test_js[i*chunk_size:(i+1)*chunk_size]

        # 3) 초기 상태 & 제어 입력
        x0 = chunk_X[0, 1:5]        # [x, y, xdot, ydot]
        us = chunk_X[:, 5:7]        # shape (80,2)

        # 4) 시뮬레이션
        # X_pred = simulate_euler(x0, us, dt)  # shape (81,4)
        X_pred = simulate_rk4(x0, us, dt)  # shape (81,4)

        X_pred = X_pred[:-1]                 # (80,4)

        all_preds.append(X_pred)
        chunk_data.append((chunk_X, chunk_Y, X_pred))
        # chunk_data_js.append((chunk_X_js,chunk_Y_js))
        # 5) MSE 계산
        mse = np.mean((X_pred - chunk_Y) ** 2)
        mse_list.append(mse)
        print(f"Chunk #{i:02d}  MSE = {mse:.6f}")
    end = time.perf_counter()
    print(f"time: ",end-start)
    # 전체 평균
    print("\nAverage MSE over all chunks:", np.mean(mse_list))

    # 6) 가장 큰/작은 MSE 인덱스
    max_idx = np.argmax(mse_list)
    min_idx = np.argmin(mse_list)
    
    print(f"\n🔴 Max MSE Chunk #{max_idx:02d}  MSE = {mse_list[max_idx]:.6f}")
    # X_max, Y_max, Y_pred_max = chunk_data[max_idx]
    # X_max_js, Y_max_js = chunk_data_js[max_idx]
    # print("X_test (max error):", X_max)
    # print("Y_pred (max error):", Y_pred_max)
    # print("Y_test (max error):", Y_max)
    # print("X_test_js (max error):", X_max_js)
    # print("Y_test_js (max error):", Y_max_js)

    # print(f"\n🟢 Min MSE Chunk #{min_idx:02d}  MSE = {mse_list[min_idx]:.6f}")
    # X_min, Y_min, Y_pred_min = chunk_data[min_idx]
    # X_min_js, Y_min_js = chunk_data_js[min_idx]
    # print("X_test (min error):", X_min)
    # print("Y_pred (min error):", Y_pred_min)
    # print("Y_test (min error):", Y_min)
    # print("X_test_js (min error):", X_min_js)
    # print("Y_test_js (min error):", Y_min_js)


    # print_det_of_jacobian_inv(X_max, name="Max MSE")
    # print_det_of_jacobian_inv(X_min, name="Min MSE")
if __name__ == '__main__':
    main()
