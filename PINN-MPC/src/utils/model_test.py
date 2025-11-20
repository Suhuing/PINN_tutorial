import torch
import torch.nn as nn
import numpy as np
import time
import abc

CHECKPOINTS_PATH = './'  # 필요에 따라 경로 수정

class NN(object, metaclass=abc.ABCMeta):
    """
    Neural Network 모델 클래스
    """

    def __init__(self, layers: list, lb: np.ndarray, ub: np.ndarray) -> None:
        self.checkpoints_dir = CHECKPOINTS_PATH
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

        self.input_dim = layers[0]
        self.output_dim = layers[-1]

        self.lb = torch.tensor(lb, dtype=self.dtype, device=self.device)
        self.ub = torch.tensor(ub, dtype=self.dtype, device=self.device)

        class Normalize(nn.Module):
            def __init__(self, lb, ub):
                super().__init__()
                self.lb = lb
                self.ub = ub
            def forward(self, x):
                return 2.0 * (x - self.lb) / (self.ub - self.lb) - 1.0

        modules = [Normalize(self.lb, self.ub)]

        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            modules.append(nn.Linear(in_dim, out_dim))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))

        self.model = nn.Sequential(*modules).to(self.device).double()
        self.loss_object = nn.MSELoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            pred = self.model(x_tensor)
        return pred.cpu().numpy()


def simulate_model(nn_model: NN, chunk_X: np.ndarray, dt: float) -> np.ndarray:
    """
    학습된 NN 모델을 이용해 상태 시퀀스를 예측합니다.
    :param nn_model: NN 객체
    :param chunk_X: 입력 데이터 chunk (N, 7)
    :param dt: 시간 간격
    :return: 예측된 상태 시퀀스 (N, 4)
    """
    N = chunk_X.shape[0]
    x = chunk_X[0, 1:5].copy()  # 초기 상태
    X_pred = []

    for k in range(N):
        model_input = chunk_X[k][None, :]  # (1, 7)
        dxdt = nn_model.predict(model_input)[0]  # (4,)
        x = x + dt * dxdt
        X_pred.append(x.copy())

    return np.array(X_pred)



def main():
    # 1) 데이터 로드
    data = np.load('data.npz')
    X_test = data['X_test']   # (800, 7)
    Y_test = data['Y_test']   # (800, 4)

    dt = 0.005
    chunk_size = 40
    num_chunks = X_test.shape[0] // chunk_size

    # 2) 모델 로드
    input_dim = 7
    output_dim = 4
    N_layer = 4
    N_neurons = 64
    layers = [input_dim] + [N_neurons] * N_layer + [output_dim]

    lb = data['lb']
    ub = data['ub']

    model = NN(layers, lb, ub)
    model.model.load_state_dict(torch.load("model_weights_new.pth", map_location=model.device))
    model.model.eval()
    
    state_dict = torch.load("model_weights_new.pth")
    for k, v in state_dict.items():
        print(f"{k}: {v.shape}")

    # 3) 예측 및 평가
    mse_list = []
    chunk_data = []
    all_preds = []

    start = time.perf_counter()
    for i in range(num_chunks):
        chunk_X = X_test[i*chunk_size:(i+1)*chunk_size]
        chunk_Y = Y_test[i*chunk_size:(i+1)*chunk_size]


        X_pred = simulate_model(model, chunk_X, dt)

        all_preds.append(X_pred)
        chunk_data.append((chunk_X, chunk_Y, X_pred))

        mse = np.mean((X_pred - chunk_Y) ** 2)
        mse_list.append(mse)
        print(f"Chunk #{i:02d}  MSE = {mse:.6f}")
    end = time.perf_counter()

    print(f"\n⏱️ Inference Time: {end - start:.4f} sec")
    print("📉 Average MSE over all chunks:", np.mean(mse_list))

    # max_idx = np.argmax(mse_list)
    # min_idx = np.argmin(mse_list)
    # print(f"🔴 Max MSE Chunk #{max_idx:02d}  MSE = {mse_list[max_idx]:.6f}")
    # print(f"🟢 Min MSE Chunk #{min_idx:02d}  MSE = {mse_list[min_idx]:.6f}")

if __name__ == '__main__':
    main()
