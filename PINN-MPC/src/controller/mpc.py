import logging
import time
import numpy as np
import torch

from scipy.integrate import solve_ivp


class MPC:
    """
    PyTorch 버전의 Model Predictive Controller.
    TensorFlow MPC와 구조/로직을 최대한 동일하게 맞춘 구현.
    """

    def __init__(
        self,
        plant,
        model,
        u_ub,
        u_lb,
        t_sample=0.1,
        H=10,
        Q=None,
        R=None,
    ):
        # TF 쪽과 동일하게 double precision 사용
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        self.plant = plant                     # solve_ivp에서 쓰는 연속계 함수
        self.model = model.to(self.device)     # PINN (torch.nn.Module)
        self.H = H
        self.t_sample = t_sample

        # ====== TF와 동일하게: optimizer 인스턴스를 생성자에서 한 번만 생성 ======
        # TF: self.optimizer = tf.keras.optimizers.RMSprop()
        # Keras 기본 lr ≈ 1e-3 이라 여기서도 1e-3로 맞춤
        self.u_ub = np.array(u_ub, dtype=np.float64)
        self.u_lb = np.array(u_lb, dtype=np.float64)
        self.input_dim = len(self.u_ub)

        # control input sequence (H x m)
        self.u = torch.zeros(
            (self.H, self.input_dim),
            dtype=self.dtype,
            device=self.device,
            requires_grad=True,
        )

        self.optimizer = torch.optim.RMSprop([self.u], lr=1e-3)

        # ====== Q, R 기본값 (TF 기본과 동일하게 eye(1)) ======
        if Q is None:
            self.Q = torch.eye(1, dtype=self.dtype, device=self.device)
        else:
            self.Q = torch.as_tensor(Q, dtype=self.dtype, device=self.device)

        if R is None:
            self.R = torch.eye(1, dtype=self.dtype, device=self.device)
        else:
            self.R = torch.as_tensor(R, dtype=self.dtype, device=self.device)

        self.solving_times = {}

    # ------------------------------------------------------------------
    # 비용 함수 (TF costs()와 동일한 형태)
    # ------------------------------------------------------------------
    def costs(self, x_ref: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
        """
        J = Σ (x_ref - x_pred)^T Q (x_ref - x_pred) + Σ u^T R u
        TF 코드:
            J = tf.reduce_sum(tf.square(x_ref - x_pred) @ Q) + ...
        과 같은 구조.
        """
        J_state = torch.sum((x_ref - x_pred) ** 2 @ self.Q)
        J_control = torch.sum(self.u ** 2 @ self.R)
        return J_state + J_control

    # ------------------------------------------------------------------
    # OCP solve (수렴 조건, 루프 구조를 TF와 동일)
    # ------------------------------------------------------------------
    def solve_ocp(self, x0, x_ref, iterations=1000, tol=1e-8):
        """
        TF 버전과 동일한 형태:
        - for epoch in range(iterations):
            J, x_pred = optimization_step(...)
            if |J - J_prev| < tol: 종료
        """
        # x0, x_ref를 torch 텐서로 캐스팅
        x0_t = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
        x_ref_t = torch.as_tensor(x_ref, dtype=self.dtype, device=self.device)

        J_prev = -1.0
        J_val = None
        x_pred = None

        for epoch in range(iterations):
            J_val, x_pred = self.optimization_step(x0_t, x_ref_t)

            if abs(J_val - J_prev) < tol:
                return J_val, x_pred

            J_prev = J_val

        return J_val, x_pred

    # ------------------------------------------------------------------
    # optimization_step: TF의 @tf.function optimization_step와 역할 동일
    # ------------------------------------------------------------------
    def optimization_step(self, x0: torch.Tensor, x_ref: torch.Tensor):
        """
        - sim_open_loop로 x_pred 계산
        - cost 계산
        - self.u에 대한 gradient 계산
        - RMSprop step
        - 제약 적용 (ensure_constraints)
        """
        self.optimizer.zero_grad()

        x_pred = self.sim_open_loop(x0, self.u, t_sample=self.t_sample, H=self.H)
        cost = self.costs(x_ref, x_pred)

        cost.backward()
        self.optimizer.step()
        self.ensure_constraints()

        # TF 쪽은 J를 tf scalar로 반환 → 여기서는 float로 반환
        return cost.item(), x_pred

    # ------------------------------------------------------------------
    # 입력 saturate (TF ensure_constraints와 동일 역할)
    # ------------------------------------------------------------------
    def ensure_constraints(self):
        """
        각 step마다 u를 [u_lb, u_ub] 범위로 projection.
        """
        with torch.no_grad():
            # 벡터화된 clamp 사용
            u_lb_t = torch.as_tensor(self.u_lb, dtype=self.dtype, device=self.device)
            u_ub_t = torch.as_tensor(self.u_ub, dtype=self.dtype, device=self.device)
            self.u[:] = torch.max(torch.min(self.u, u_ub_t), u_lb_t)

    # ------------------------------------------------------------------
    # closed-loop simulation (sim) – TF 구현과 동일 로직
    # ------------------------------------------------------------------
    def sim(self, x0, X_ref, T_ref):
        """
        Closed-loop MPC simulation.
        TF sim()과 구조를 동일하게 유지.
        """
        x0_t = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
        X_ref_t = torch.as_tensor(X_ref, dtype=self.dtype, device=self.device)

        N = len(T_ref)
        state_dim = x0_t.shape[0]

        X_mpc = np.zeros((N, state_dim))
        X_pred = np.zeros((N, state_dim))
        U_mpc = np.zeros((N, self.input_dim))

        # 초기 상태
        X_mpc[0] = x0_t.detach().cpu().numpy()
        X_pred[0] = x0_t.detach().cpu().numpy()
        U_mpc[0] = self.u[0].detach().cpu().numpy()

        for i, t in enumerate(T_ref[:-1]):
            start_time = time.time()

            # OCP 풀기
            J, x_pred = self.solve_ocp(X_mpc[i], X_ref_t[i : i + self.H + 1])
            ocp_solving_time = time.time() - start_time
            self.solving_times[i] = ocp_solving_time

            u_k = self.u[0].detach()

            # plant 실제 시스템 시뮬레이션
            x_true = self.sim_plant_system(X_mpc[i], u_k, self.t_sample)

            X_pred[i + 1] = x_pred[1].detach().cpu().numpy()
            X_mpc[i + 1] = x_true.detach().cpu().numpy()
            U_mpc[i + 1] = u_k.detach().cpu().numpy()

            # 로그 출력 (TF 버전과 유사 포맷)
            log_str = (
                f'\tIter: {str(i + 1).zfill(len(str(N - 1)))}/{N - 1},'
                f'\tJ: {J:.2e},\tt: {t + self.t_sample:.2f} s,'
            )

            for j in range(self.input_dim):
                log_str += f'\tu{j + 1}: {U_mpc[i + 1, j]:.2f},'

            for j in range(state_dim // 2):
                log_str += f'\tx{j + 1}(t, u): {X_mpc[i + 1, j]:.2f},'

            log_str += f'\tOCP-solving-time: {ocp_solving_time:.2e} s'
            logging.info(log_str)

        return X_mpc, U_mpc, X_pred

    # ------------------------------------------------------------------
    # plant simulation (solve_ivp) – TF와 동일
    # ------------------------------------------------------------------
    def sim_plant_system(self, x0, u, tau):
        """
        SciPy solve_ivp로 참 시스템 시뮬레이션.
        """
        if isinstance(x0, torch.Tensor):
            x0_np = x0.detach().cpu().numpy()
        else:
            x0_np = np.asarray(x0, dtype=np.float64)

        if isinstance(u, torch.Tensor):
            u_np = u.detach().cpu().numpy()
        else:
            u_np = np.asarray(u, dtype=np.float64)

        ivp_solution = solve_ivp(self.plant, [0, tau], x0_np, args=(u_np,))
        z_true = ivp_solution.y[:, -1]  # shape: (state_dim,)

        return torch.as_tensor(z_true, dtype=self.dtype, device=self.device)

    # ------------------------------------------------------------------
    # PINN 기반 모델로 open-loop 예측 – TF sim_open_loop와 구조 동일
    # ------------------------------------------------------------------
    def sim_open_loop(self, x0, u_array, t_sample, H):
        """
        PINN model을 이용한 open-loop 예측.
        TF:
            t = const
            x_i = x0
            for i in range(H):
                x = concat(t, x_i, u[i])
                x_pred = model(x)
        """
        t = torch.full((1, 1), t_sample, dtype=self.dtype, device=self.device)

        x_i = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
        if x_i.ndim == 1:
            x_i = x_i.unsqueeze(0)

        X_pred = [x_i]

        for i in range(H):
            u_i = u_array[i : i + 1]
            if isinstance(u_i, np.ndarray):
                u_i = torch.as_tensor(u_i, dtype=self.dtype, device=self.device)
            else:
                u_i = u_i.to(dtype=self.dtype, device=self.device)

            if u_i.ndim == 1:
                u_i = u_i.unsqueeze(0)

            x_input = torch.cat([t, x_i, u_i], dim=1)
            x_pred = self.model(x_input)

            X_pred.append(x_pred)
            x_i = x_pred

        return torch.cat(X_pred, dim=0)  # (H+1, state_dim)

    # ------------------------------------------------------------------
    # plant 기반 open-loop (필요하면, TF sim_open_loop_plant와 동일)
    # ------------------------------------------------------------------
    def sim_open_loop_plant(self, x0, u_array, t_sample, H):
        x_i = torch.as_tensor(x0, dtype=self.dtype, device=self.device)
        X = [x_i.detach().cpu().numpy()]

        for i in range(H):
            x_true = self.sim_plant_system(x_i, u_array[i], t_sample)
            X.append(x_true.detach().cpu().numpy())
            x_i = x_true

        return np.vstack(X)  # (H+1, state_dim)
