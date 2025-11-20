import os
import click
import logging
import time

import numpy as np
import torch

from model.pinn import PINN
from utils.data import (
    generate_data_points,
    generate_collocation_points,
    load_data,
)
from utils.plotting import (
    animate,
    plot_states,
    plot_input_sequence,
    plot_absolute_error,
)
from utils.system import M_tensor, k_tensor, q_tensor, B_tensor


class ManipulatorInformedNN(PINN):
    """
    Manipulator Informed PINN (2-DOF Schunk PowerCube),
    PINN 베이스 클래스를 상속한 구현.
    """

    def __init__(self, layers, lb, ub, X_f=None):
        super().__init__(layers, lb, ub)

        # collocation points 전체는 PINN.x_phys에 저장
        if X_f is not None:
            self.set_collocation_points(X_f)

    # --------------------------------------------------------------
    # collocation points 세팅 (편의 함수)
    # --------------------------------------------------------------
    def set_collocation_points(self, X_f: np.ndarray):
        """
        X_f: [t, x0(4), u(2)] shape (N_f, 7)
        """
        # PINN 베이스 클래스에서 쓰는 멤버
        self.x_phys = X_f

    # --------------------------------------------------------------
    # f_model(x): PDE residual 계산
    # --------------------------------------------------------------
    def f_model(self, X_f: torch.Tensor) -> torch.Tensor:
        """
        Physics Informed residual for the Schunk PowerCube Serial Robot.
        """

        device = self.device
        dtype = self.dtype

        # --------------------------------------------
        # ① 입력 구성 (grad는 t만 켬)
        # --------------------------------------------
        X_f = X_f.to(device=device, dtype=dtype)

        t  = X_f[:, 0:1].clone().detach().requires_grad_(True)
        x0 = X_f[:, 1:5]     # grad 불필요
        u  = X_f[:, 5:7]     # grad 불필요

        # input to network: [t, x0, u]
        input_tensor = torch.cat([t, x0, u], dim=1)   # (N_f, 7)

        # --------------------------------------------
        # ② 네트워크 출력
        # --------------------------------------------
        y_pred = self.model(input_tensor)

        q1      = y_pred[:, 0:1]
        q2      = y_pred[:, 1:2]
        dq1_net = y_pred[:, 2:3]
        dq2_net = y_pred[:, 3:4]

        # --------------------------------------------
        # ③ autograd: dq/dt (1차 미분)
        # --------------------------------------------
        dq1_dt_tf = torch.autograd.grad(
            outputs=q1,
            inputs=t,
            grad_outputs=torch.ones_like(q1),
            create_graph=True,
            retain_graph=True
        )[0]

        dq2_dt_tf = torch.autograd.grad(
            outputs=q2,
            inputs=t,
            grad_outputs=torch.ones_like(q2),
            create_graph=True,
            retain_graph=True
        )[0]

        dq_dt_tf = torch.cat([dq1_dt_tf, dq2_dt_tf], dim=1)   # (N_f,2)
        dq_dt_net = torch.cat([dq1_net, dq2_net], dim=1)

        # --------------------------------------------
        # ④ autograd: d²q/dt² (2차 미분)
        # --------------------------------------------
        d2q1_dt_tf = torch.autograd.grad(
            outputs=dq1_net,
            inputs=t,
            grad_outputs=torch.ones_like(dq1_net),
            create_graph=True,
            retain_graph=True
        )[0]

        d2q2_dt_tf = torch.autograd.grad(
            outputs=dq2_net,
            inputs=t,
            grad_outputs=torch.ones_like(dq2_net),
            create_graph=True,
            retain_graph=True
        )[0]

        d2q_dt_tf = torch.cat([d2q1_dt_tf, d2q2_dt_tf], dim=1)  # (N_f,2)

        # --------------------------------------------
        # ⑤ Dynamics (Schunk PowerCube)
        # --------------------------------------------
        i_PR90 = torch.full((X_f.shape[0],), 161.0,
                            dtype=dtype, device=device)

        beta      = q2.squeeze(1)
        dalpha_dt = dq1_dt_tf.squeeze(1)
        dbeta_dt  = dq2_dt_tf.squeeze(1)

        M_tf = M_tensor(beta, i_PR90)        # (N,2,2)
        k_tf = k_tensor(dalpha_dt, beta, dbeta_dt)   # (N,2)
        q_tf = q_tensor(q1.squeeze(1), dalpha_dt, beta, dbeta_dt)   # (N,2)
        B_tf = B_tensor(i_PR90)              # (N,2,2)

        M_d2q = torch.bmm(M_tf, d2q_dt_tf.unsqueeze(-1)).squeeze(-1)   # (N,2)
        Bu    = torch.bmm(B_tf, u.unsqueeze(-1)).squeeze(-1)           # (N,2)

        # --------------------------------------------
        # ⑥ Residuals
        # --------------------------------------------
        res_dq  = dq_dt_tf - dq_dt_net           # (N,2)
        res_dyn = M_d2q + k_tf - q_tf - Bu       # (N,2)

        # --------------------------------------------
        # ⑦ Concatenate: (N,4)
        # --------------------------------------------
        f_pred = torch.cat([res_dq, res_dyn], dim=1)

        return f_pred

if __name__ == "__main__":
    LOAD_WEIGHTS = False
    TRAIN_NET = True

    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logging.info("PyTorch version: {}".format(torch.__version__))
    logging.info("CUDA available: {}".format(torch.cuda.is_available()))

    # Hyper parameter
    N_train = 2
    epochs = 500000
    N_phys = 20000
    N_data = 100


    logging.info(f'Epochs: {epochs}')
    logging.info(f'N_data: {N_data}')
    logging.info(f'N_phys: {N_phys}')

    # Paths
    data_path = os.path.join('../resources/data.npz')
    weights_path = os.path.join('../checkpoints/easy_checkpoint')

    lb, ub, input_dim, output_dim, X_test, Y_test, X_star, Y_star = load_data(data_path)

    N_layer = 4
    N_neurons = 64
    layers = [input_dim, *N_layer * [N_neurons], output_dim]

    # PINN initialization
    pinn = ManipulatorInformedNN(layers, lb, ub)

    if LOAD_WEIGHTS:
        pinn.load_weights(weights_path)

    # PINN training
    if TRAIN_NET:
        for i in range(N_train):
            # Generate training data via LHS
            X_data, Y_data = generate_data_points(N_data, lb, ub)
            X_phys = generate_collocation_points(N_phys, lb, ub)
            pinn.set_collocation_points(X_phys)

            logging.info(f'\t{i + 1}/{N_train} Start training of the PINN')
            start_time = time.time()
            pinn.fit(
                X_data,
                Y_data,
                epochs,
                X_star,
                Y_star,
                optimizer='lbfgs',
                learning_rate=1.0,
                val_freq=1000,
                log_freq=1000,
            )
            logging.info(f'\t{i + 1}/{N_train} Training time: {time.time() - start_time:.2f} s')

    # --------- numpy / torch 안전 변환 함수 ----------
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x  # already numpy

    # PINN Evaluation
    Y_pred, F_pred = pinn.predict(X_test)

    X_test_np = to_numpy(X_test)
    Y_test_np = to_numpy(Y_test)
    Y_pred_np = to_numpy(Y_pred)

    # 시간 step 추출 (torch 또는 numpy 모두 대응)
    if isinstance(X_test, torch.Tensor):
        t_step = float((X_test[1, 0] - X_test[0, 0]).detach().cpu())
    else:
        t_step = float(X_test[1, 0] - X_test[0, 0])

    tau = 0.2
    T = np.arange(t_step, 20 * tau + t_step, t_step)

    plot_input_sequence(T, X_test_np[:, 5:])
    plot_states(T, Y_test_np, Y_pred_np)
    plot_absolute_error(T, Y_test_np, Y_pred_np)

    animate(Y_test_np[::10], [Y_pred_np[::10]], ['PINN'], fps=1 / (10 * t_step))

    if click.confirm('Do you want to save (overwrite) the models weights?'):
        pinn.save_weights(os.path.join(weights_path, 'easy_checkpoint'))
