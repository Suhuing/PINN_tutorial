import torch
from scipy.constants import g


# ---------------------------------------------------------------------
# Single-trajectory dynamics: SciPy solve_ivp에서 쓰는 함수
# ---------------------------------------------------------------------
def f(t, x, u):
    """
    Continuous dynamics:
        M * d2q + k = q + B*u

    x = [q1, q2, dq1_dt, dq2_dt]
    SciPy solve_ivp에서 호출되므로
    x, u 는 numpy array/float 로 들어온다 → torch.Tensor 로 변환해서 계산.
    """

    # x, u 를 CPU double tensor로 변환
    x_t = torch.as_tensor(x, dtype=torch.float64)   # shape (4,)
    u_t = torch.as_tensor(u, dtype=torch.float64)   # shape (2,)

    # 상태 분해
    alpha    = x_t[0]
    beta     = x_t[1]
    dalpha   = x_t[2]
    dbeta    = x_t[3]

    # 동역학 계산 (torch 버전 사용)
    M_tf = M(beta)                                 # (2,2)
    k_tf = k(dalpha, beta, dbeta)                  # (2,)
    q_tf = q(alpha, dalpha, beta, dbeta)           # (2,)
    B_tf = B()                                     # (2,2)

    dx12dt = x_t[2:]                               # (2,) = [dalpha, dbeta]

    rhs = -k_tf + q_tf + torch.mv(B_tf, u_t)       # (2,)
    dx34dt = torch.linalg.solve(M_tf, rhs.unsqueeze(1))[:, 0]  # (2,)

    dxdt = torch.cat((dx12dt, dx34dt), dim=0)      # (4,)

    # solve_ivp 는 numpy.ndarray 를 기대하므로 numpy 로 변환해서 반환
    return dxdt.detach().cpu().numpy()


def M(beta: torch.Tensor, i_PR90: float | torch.Tensor = 161.0) -> torch.Tensor:
    """
    Returns mass matrix of the robot for beta.

    :param beta: scalar or tensor
    :param i_PR90: motor constant
    :return: 2x2 mass matrix
    """
    beta = torch.as_tensor(beta)
    i = torch.as_tensor(i_PR90, dtype=beta.dtype, device=beta.device)

    M_1 = torch.stack(
        [
            0.00005267 * i**2 + 0.6215099724 * torch.cos(beta) + 0.9560375168565,
            0.00005267 * i + 0.3107549862 * torch.cos(beta) + 0.6608899068565,
        ],
        dim=0,
    )

    M_2 = torch.stack(
        [
            0.00005267 * i + 0.3107549862 * torch.cos(beta) + 0.6608899068565,
            0.00005267 * i**2 + 0.6608899068565,
        ],
        dim=0,
    )

    M_tf = torch.stack([M_1, M_2], dim=1)  # (2,2)

    return M_tf


def k(dalpha_dt: torch.Tensor, beta: torch.Tensor, dbeta_dt: torch.Tensor) -> torch.Tensor:
    """
    Returns stiffness vector of the robot for a set of generalized coordinates.

    :return: (2,) stiffness vector
    """

    term = torch.stack(
        [
            0.040968 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.5586)
            - 0.18
            * torch.sin(beta)
            * (
                1.714 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 1.714 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 1.714 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 1.714 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.30852 * dalpha_dt**2 * torch.cos(beta)
                + 1.714 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 1.714 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            )
            - 0.36
            * torch.sin(beta)
            * (
                0.1138 * (0.06415 * dalpha_dt + 0.06415 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.1138 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.1138 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.1138 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.1138 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.020484 * dalpha_dt**2 * torch.cos(beta)
                + 0.1138 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.1138 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.1138 * (0.03 * dalpha_dt + 0.03 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            )
            - 0.18
            * torch.sin(beta)
            * (
                2.751 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.49518 * dalpha_dt**2 * torch.cos(beta)
            )
            - 0.18
            * torch.sin(beta)
            * (
                1.531 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 1.531 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.27558 * dalpha_dt**2 * torch.cos(beta)
                + 1.531 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            )
            - 0.18
            * torch.sin(beta)
            * (
                0.934 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.934 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.934 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt)
                + 0.16812 * dalpha_dt**2 * torch.cos(beta)
                + 0.934 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            )
            + 0.16812 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.335)
            + 0.49518 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.04321)
            + 0.30852 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.46445)
            + 0.27558 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.24262),
            0.3107549862 * dalpha_dt**2 * torch.sin(beta),
        ],
        dim=0,
    )

    return term


def q(alpha: torch.Tensor, dalpha_dt: torch.Tensor, beta: torch.Tensor, dbeta_dt: torch.Tensor) -> torch.Tensor:
    """
    Returns reaction forces vector of the robot for a set of generalized coordinates.

    :return: (2,) reaction forces
    """

    term = torch.stack(
        [
            0.33777 * g * torch.sin(alpha)
            - 3.924 * torch.tanh(5 * dalpha_dt)
            - 10.838 * torch.tanh(10 * dalpha_dt)
            - 2.236 * torch.tanh(20 * dalpha_dt)
            - 76.556 * dalpha_dt
            - 1.288368 * g * torch.cos(alpha + beta) * torch.sin(beta)
            + 0.2276 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.5586)
            + 0.934 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.335)
            + 2.751 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.04321)
            + 1.714 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.46445)
            + 1.531 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.24262),
            1.72641659 * g * torch.sin(alpha + beta)
            - 0.368 * torch.tanh(5 * dbeta_dt)
            - 0.368 * torch.tanh(10 * dbeta_dt)
            - 8.342 * torch.tanh(100 * dbeta_dt)
            - 0.492 * torch.sign(dbeta_dt)
            - 56.231 * dbeta_dt,
        ],
        dim=0,
    )

    return term


def B(i_PR90: float | torch.Tensor = 161.0) -> torch.Tensor:
    """
    Returns input matrix of the robot (2x2).

    :param i_PR90: constant or tensor
    :return: (2,2) input matrix
    """
    i = torch.as_tensor(i_PR90, dtype=torch.float64)
    zero = torch.zeros((), dtype=torch.float64, device=i.device)

    B_1 = torch.stack([i, zero], dim=0)
    B_2 = torch.stack([zero, i], dim=0)

    B_tf = torch.stack([B_1, B_2], dim=1)  # (2,2)

    return B_tf


# ---------------------------------------------------------------------
# Batch 버전: train_pinn.py에서 사용하는 함수들
# ---------------------------------------------------------------------
def M_tensor(beta: torch.Tensor, i_PR90: torch.Tensor | float) -> torch.Tensor:
    """
    Returns mass matrices of the robot for multiple values for beta.

    :param beta: (N,) tensor of beta values
    :param i_PR90: (N,) tensor or scalar
    :return: (N, 2, 2) mass matrices
    """
    beta = torch.as_tensor(beta)
    i = torch.as_tensor(i_PR90, dtype=beta.dtype, device=beta.device)

    M_1 = torch.stack(
        [
            0.00005267 * i**2 + 0.6215099724 * torch.cos(beta) + 0.9560375168565,
            0.00005267 * i + 0.3107549862 * torch.cos(beta) + 0.6608899068565,
        ],
        dim=1,
    )  # (N,2)

    M_2 = torch.stack(
        [
            0.00005267 * i + 0.3107549862 * torch.cos(beta) + 0.6608899068565,
            0.00005267 * i**2 + 0.6608899068565,
        ],
        dim=1,
    )  # (N,2)

    M_tf = torch.stack([M_1, M_2], dim=2)  # (N,2,2)

    return M_tf


def k_tensor(dalpha_dt: torch.Tensor, beta: torch.Tensor, dbeta_dt: torch.Tensor) -> torch.Tensor:
    """
    Returns stiffness vectors of the robot for multiple values of generalized coordinates.

    :param dalpha_dt: (N,) tensor
    :param beta: (N,) tensor
    :param dbeta_dt: (N,) tensor
    :return: (N,2) stiffness vectors
    """

    term1 = (
        0.040968 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.5586)
        - 0.18
        * torch.sin(beta)
        * (
            1.714 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 1.714 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 1.714 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 1.714 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.30852 * dalpha_dt**2 * torch.cos(beta)
            + 1.714 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 1.714 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
        )
        - 0.36
        * torch.sin(beta)
        * (
            0.1138 * (0.06415 * dalpha_dt + 0.06415 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.1138 * (0.07205 * dalpha_dt + 0.07205 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.1138 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.1138 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.1138 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.020484 * dalpha_dt**2 * torch.cos(beta)
            + 0.1138 * (0.0574 * dalpha_dt + 0.0574 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.1138 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.1138 * (0.03 * dalpha_dt + 0.03 * dbeta_dt) * (dalpha_dt + dbeta_dt)
        )
        - 0.18
        * torch.sin(beta)
        * (
            2.751 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.49518 * dalpha_dt**2 * torch.cos(beta)
        )
        - 0.18
        * torch.sin(beta)
        * (
            1.531 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 1.531 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.27558 * dalpha_dt**2 * torch.cos(beta)
            + 1.531 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
        )
        - 0.18
        * torch.sin(beta)
        * (
            0.934 * (0.08262 * dalpha_dt + 0.08262 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.934 * (0.04321 * dalpha_dt + 0.04321 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.934 * (0.09238 * dalpha_dt + 0.09238 * dbeta_dt) * (dalpha_dt + dbeta_dt)
            + 0.16812 * dalpha_dt**2 * torch.cos(beta)
            + 0.934 * (0.11679 * dalpha_dt + 0.11679 * dbeta_dt) * (dalpha_dt + dbeta_dt)
        )
        + 0.16812 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.335)
        + 0.49518 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.04321)
        + 0.30852 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.46445)
        + 0.27558 * dalpha_dt**2 * torch.sin(beta) * (0.18 * torch.cos(beta) + 0.24262)
    )

    term2 = 0.3107549862 * dalpha_dt**2 * torch.sin(beta)

    return torch.stack([term1, term2], dim=1)  # (N,2)


def q_tensor(alpha: torch.Tensor, dalpha_dt: torch.Tensor, beta: torch.Tensor, dbeta_dt: torch.Tensor) -> torch.Tensor:
    """
    Returns reaction forces vectors of the robot for multiple values of generalized coordinates.

    :return: (N,2) reaction forces
    """

    term1 = (
        0.33777 * g * torch.sin(alpha)
        - 3.924 * torch.tanh(5 * dalpha_dt)
        - 10.838 * torch.tanh(10 * dalpha_dt)
        - 2.236 * torch.tanh(20 * dalpha_dt)
        - 76.556 * dalpha_dt
        - 1.288368 * g * torch.cos(alpha + beta) * torch.sin(beta)
        + 0.2276 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.5586)
        + 0.934 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.335)
        + 2.751 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.04321)
        + 1.714 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.46445)
        + 1.531 * g * torch.sin(alpha + beta) * (0.18 * torch.cos(beta) + 0.24262)
    )

    term2 = (
        1.72641659 * g * torch.sin(alpha + beta)
        - 0.368 * torch.tanh(5 * dbeta_dt)
        - 0.368 * torch.tanh(10 * dbeta_dt)
        - 8.342 * torch.tanh(100 * dbeta_dt)
        - 0.492 * torch.sign(dbeta_dt)
        - 56.231 * dbeta_dt
    )

    return torch.stack([term1, term2], dim=1)  # (N,2)


def B_tensor(i_PR90: torch.Tensor | float) -> torch.Tensor:
    """
    Returns input matrices of the robot.

    :param i_PR90: (N,) tensor or scalar
    :return: (N,2,2) input matrices
    """
    i = torch.as_tensor(i_PR90)
    zeros = torch.zeros_like(i)

    B_1 = torch.stack([i, zeros], dim=1)      # (N,2)
    B_2 = torch.stack([zeros, i], dim=1)      # (N,2)

    B_tf = torch.stack([B_1, B_2], dim=2)     # (N,2,2)

    return B_tf
