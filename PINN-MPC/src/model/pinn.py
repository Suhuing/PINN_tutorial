import abc
import torch

from model.nn import NN


class PINN(NN, metaclass=abc.ABCMeta):
    """
    Physics-Informed Neural Network base class.
    - NN(순수 데이터 기반 네트워크)를 상속
    - loss_object를 PINN 전용 loss로 교체
    """

    def __init__(self, layers, lb, ub):
        super().__init__(layers, lb, ub)

        # collocation points (physics domain)
        # 외부에서 반드시 세팅해줘야 함: self.x_phys = ...
        self.x_phys = None

        # NN에서 쓰던 기본 MSE 대신, PINN용 loss로 교체
        self.loss_object = self.pinn_loss

    # ------------------------------------------------------------------
    # PINN loss: 데이터 + 물리 residual
    # ------------------------------------------------------------------
    def pinn_loss(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        w_data = 1.0
        w_phys = 1.0

        L_data = torch.mean((y_pred - y) ** 2)

        if self.x_phys is not None:
            # 🔥 여기서는 grad 필요 없음
            x_phys_t = self.tensor(self.x_phys, requires_grad=False)
            f_pred = self.f_model(x_phys_t)
            L_phys = torch.mean(f_pred ** 2)
        else:
            L_phys = torch.zeros(1, dtype=self.dtype, device=self.device)

        return w_data * L_data + w_phys * L_phys

    # ------------------------------------------------------------------
    # f_model: 각 문제별로 구현 (추상)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def f_model(self, x: torch.Tensor) -> torch.Tensor:
        """
        PDE residual f(x)를 계산하는 함수.

        Parameters
        ----------
        x : torch.Tensor, shape (N_phys, input_dim)
            collocation points (필요하면 requires_grad=True)

        Returns
        -------
        f : torch.Tensor
            PDE residual 값. 보통 shape (N_phys, 1) 또는 (N_phys,)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # predict: y_pred만 또는 (y_pred, f_pred(x)) 반환
    # ------------------------------------------------------------------
    def predict(self, x):
        """
        Calls the model prediction function and returns:
        (y_pred, f_pred(x))

        :param x: input np.ndarray or torch.Tensor
        :return: (y_pred, f_pred)
        """
        # x는 data/test input (예: [t, x0, u] 형태)
        x_t = self.tensor(x, requires_grad=False)

        # 네트워크 출력 (단순 forward, grad 불필요)
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_t)

        # f_model(x)에 대해서는 autograd 필요하면 no_grad 빼야 하는데,
        # 여기서는 '분석용'이라고 보면 no_grad 안에서 써도 무방.
        f_pred = self.f_model(x_t)

        return y_pred, f_pred
