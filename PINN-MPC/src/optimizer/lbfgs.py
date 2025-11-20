import numpy
import numpy as np
import torch

from optimizer.custom_lbfgs import lbfgs, Struct


def function_factory(model, loss_fcn, x, y, callback_fcn, epochs,
                     x_test=None, y_test=None,
                     val_freq=1000, log_freq=1000, verbose=1):
    """
    A factory to create a function required by the L-BFGS implementation.

    :param model: an instance of torch.nn.Module
    :param loss_fcn: a function with signature loss_value = loss_fcn(y_pred, y_true)
    :param torch.Tensor x: input tensor of the training dataset
    :param torch.Tensor y: output tensor of the training dataset
    :param callback_fcn: callback function, which is called after each epoch
    :param int epochs: number of epochs
    :param x_test: input tensor of the test dataset, used to evaluate accuracy
    :param y_test: output tensor of the test dataset, used to evaluate accuracy
    :return: object: a function that has the signature of loss_value, gradients = f(model_parameters)
    """

    # parameters list
    params = [p for p in model.parameters() if p.requires_grad]

    # shapes and sizes of all trainable parameters
    shapes = [p.shape for p in params]
    n_tensors = len(shapes)
    n_params = [p.numel() for p in params]
    total_params = sum(n_params)

    device = params[0].device
    dtype = params[0].dtype

    def assign_new_model_parameters(weights: torch.Tensor):
        """
        Updates the model's weights from a flat parameter vector.

        :param torch.Tensor weights: flat vector representing the model's weights
        """
        offset = 0
        with torch.no_grad():
            for p, n in zip(params, n_params):
                p.copy_(weights[offset:offset + n].view_as(p))
                offset += n

    def train_step(weights: torch.Tensor):
        """
        One evaluation of loss and gradient for given flat weights.
        """
        # update model parameters
        assign_new_model_parameters(weights)

        # forward + loss
        model.zero_grad()
        y_pred = model(x)
        # NOTE: loss_fcn(y_pred, y) 를 기준으로 맞춰둠 (NN.train_step과 동일)
        loss_value = loss_fcn(y_pred, y)

        # gradients w.r.t. model parameters
        grads = torch.autograd.grad(loss_value, params, create_graph=False)
        grads_flat = torch.cat([g.reshape(-1) for g in grads])

        return loss_value, grads_flat

    def f(weights: torch.Tensor):
        """
        Function that can be used in the L-BFGS implementation.
        This function is created by function_factory.

        :param torch.Tensor weights: flat vector representing the model's weights
        :return: (loss_value, grads_flat)
        """
        loss_value, grads_flat = train_step(weights)

        # iteration count
        f.iter += 1
        # epoch-style callback (원래 TF 코드와 동일하게 사용)
        callback_fcn(f.iter, loss_value.item(), epochs,
                     x_test, y_test,
                     val_freq=val_freq, log_freq=log_freq, verbose=verbose)

        # store loss history
        f.history.append(loss_value.item())

        return loss_value, grads_flat

    # store meta info as attributes (원본 구조 최대한 유지)
    f.iter = 0
    f.shapes = shapes
    f.n_params = n_params
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f


class LBFGS:
    """
    Class used to represent the L-BFGS optimizer.
    """

    def minimize(self, model, loss_fcn, x, y, callback_fcn, epochs=2000, learning_rate=1.,
                 x_test=None, y_test=None, val_freq=1000, log_freq=1000, verbose=1):
        """
        Performs the Neural Network training with the L-BFGS implementation.

        :param model: an instance of torch.nn.Module
        :param loss_fcn: a function with signature loss_value = loss_fcn(y_pred, y_true)
        :param torch.Tensor x: input tensor of the training dataset
        :param torch.Tensor y: output tensor of the training dataset
        :param callback_fcn: callback function, which is called after each epoch
        :param int epochs: number of epochs
        :param x_test: input tensor of the test dataset, used to evaluate accuracy
        :param y_test: output tensor of the test dataset, used to evaluate accuracy
        """

        # x, y 는 이미 NN 쪽에서 tensor(...)를 거쳐 들어온다고 가정 (device / dtype 맞춰짐)
        x = x
        y = y

        # create opfunc
        func = function_factory(
            model, loss_fcn, x, y,
            callback_fcn, epochs,
            x_test=x_test, y_test=y_test,
            val_freq=val_freq, log_freq=log_freq, verbose=verbose
        )

        # flatten initial model parameters to 1D tensor
        params = [p for p in model.parameters() if p.requires_grad]
        device = params[0].device
        dtype = params[0].dtype

        init_params = torch.cat([p.detach().reshape(-1) for p in params]).to(device=device, dtype=dtype)

        # L-BFGS config
        nt_epochs = epochs
        nt_config = Struct()
        nt_config.learningRate = learning_rate
        nt_config.maxIter = nt_epochs
        nt_config.nCorrection = 50
        nt_config.tolFun = 1.0 * np.finfo(float).eps
        nt_config.maxEval = None
        nt_config.tolX = None
        nt_config.lineSearch = None
        nt_config.lineSearchOptions = None
        nt_config.verbose = verbose

        state = Struct()
        state.funcEval = 0
        state.nIter = 0

        # run L-BFGS
        new_params, f_hist, currentFuncEval = lbfgs(
            func,
            init_params,
            nt_config,
            state,
            True,
            lambda _iter, _loss, _is_iter: None
        )

        # 최종 파라미터를 모델에 다시 반영 (마지막 step에서 이미 assign 되긴 함)
        func.assign_new_model_parameters(new_params)

        return f_hist
