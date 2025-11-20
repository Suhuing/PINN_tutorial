import logging
import os

import numpy as np
import torch

from controller.mpc import MPC
from train_pinn import ManipulatorInformedNN
from utils.data import load_ref_trajectory, load_data
from utils.plotting import plot_input_sequence, plot_states, plot_absolute_error, animate
from utils.system import f


def to_numpy(x):
    """torch.Tensor면 detach+cpu+numpy, 아니면 그대로 반환."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA available: {}".format(torch.cuda.is_available()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    resources_path = os.path.join("../resources")
    # EE 기반 데이터 사용 (train_pinn.py와 동일)
    data_path = os.path.join(resources_path, "converted_all_data.npz")
    weights_path = os.path.join("../checkpoints")

    # 데이터 로드
    lb, ub, input_dim, output_dim, _, _, _, _ = load_data(data_path)

    # Hyper parameter
    N_l = 4
    N_n = 64
    layers = [input_dim, *N_l * [N_n], output_dim]

    logging.info("MPC parameters:")
    H = 5
    logging.info(f"\tH:\t{H}")
    u_ub = np.array([0.5, 0.5])
    u_lb = -u_ub

    # 참조 궤적 (joint/EE 어디든, utils.load_ref_trajectory 정의에 따름)
    X_ref, T_ref = load_ref_trajectory(resources_path)

    x0 = X_ref[0]
    T_ref = T_ref[:-H, 0]

    tau = T_ref[1] - T_ref[0]
    logging.info(f"\ttau:\t{tau}")

    # PINN 모델 초기화 및 weight 로드
    pinn = ManipulatorInformedNN(layers, lb, ub)
    pinn.load_weights(os.path.join(weights_path, "easy_checkpoint"))

    # MPC 컨트롤러 초기화
    Q = torch.diag(torch.tensor([1, 1, 0, 0], dtype=torch.float64, device=device))
    R = 1e-6 * torch.eye(2, dtype=torch.float64, device=device)

    controller = MPC(
        f,
        pinn.model,   # 순수 NN 모델 (nn.Sequential)
        u_ub=u_ub,
        u_lb=u_lb,
        t_sample=tau,
        H=H,
        Q=Q,
        R=R,
    )

    # ===================== Self-loop prediction 테스트 =====================

    H_sl = 20

    # self loop 입력 시퀀스 생성
    U1_sl = 0.5 * np.sin(np.linspace(0, 2 * np.pi, H_sl))
    U2_sl = -U1_sl
    U_sl = np.hstack((U1_sl[:, np.newaxis], U2_sl[:, np.newaxis]))

    # 초기 상태 (필요시 수정)
    x0_sl = np.zeros(4)
    # 예시로 y 위치만 살짝 올려놓은 것 같아서 유지
    x0_sl[1] = 0.9622

    # 실제 plant 시스템 시뮬레이션
    X_ref_sl = controller.sim_open_loop_plant(
        x0_sl,
        U_sl,
        t_sample=tau,
        H=H_sl,
    )
    X_ref_sl = to_numpy(X_ref_sl)

    # PINN 모델 기반 시스템 시뮬레이션
    X_sl = controller.sim_open_loop(
        x0_sl,
        U_sl,
        t_sample=tau,
        H=H_sl,
    )
    X_sl = to_numpy(X_sl)

    T_sl = np.arange(0.0, H_sl * tau + tau, tau)


    # ===================== Closed-loop MPC 테스트 =====================

    X_mpc, U_mpc, X_pred = controller.sim(x0, X_ref, T_ref)

    X_mpc = to_numpy(X_mpc)
    U_mpc = to_numpy(U_mpc)
    X_pred = to_numpy(X_pred)

    plot_input_sequence(T_ref, U_mpc)
    plot_states(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    plot_absolute_error(T_ref, X_ref[:-H], Z_mpc=X_mpc)
    animate(X_ref[:-H], [X_mpc], ["MPC"], fps=1 / tau)
