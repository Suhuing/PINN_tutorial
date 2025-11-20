# Adapted from https://github.com/yaroslavvb/stuff/blob/master/eager_lbfgs/eager_lbfgs.py
# PyTorch 포팅 + 그래프/메모리 안전 수정 버전

import time
import types
import torch

# ---------------------------------------------------------------------------
# Time tracking (원본과 동일)
# ---------------------------------------------------------------------------
global_time_list = []
global_last_time = 0


def reset_time():
    global global_time_list, global_last_time
    global_time_list = []
    global_last_time = time.perf_counter()


def record_time():
    global global_last_time, global_time_list
    new_time = time.perf_counter()
    global_time_list.append(new_time - global_last_time)
    global_last_time = time.perf_counter()


def last_time():
    """Returns last interval records in millis."""
    global global_last_time, global_time_list
    if global_time_list:
        return 1000 * global_time_list[-1]
    else:
        return 0.0


def dot(a, b):
    """Dot product function (scalar)."""
    return torch.sum(a * b)


def verbose_func(s):
    print(s)


final_loss = None
times = []


# ---------------------------------------------------------------------------
# L-BFGS 메인 루프
# ---------------------------------------------------------------------------
def lbfgs(opfunc, x, config, state, do_verbose, log_fn):
    """
    Limited-memory BFGS optimizer (Lua/Torch lbfgs.lua 포트).

    Parameters
    ----------
    opfunc : function
        f, g = opfunc(x)를 만족하는 함수.
        - x: 1D torch.Tensor, flat parameters
        - f: scalar torch.Tensor (loss)
        - g: 1D torch.Tensor (gradient)
    x : torch.Tensor
        초기 파라미터 (1D)
    config : Struct
        maxIter, maxEval, tolFun, tolX, nCorrection, learningRate, lineSearch 등
    state : Struct
        funcEval, nIter 등 상태 저장용
    do_verbose : bool
        중간 log_fn 호출 여부
    log_fn : callable
        log_fn(iter, loss_value, True) 형태의 콜백

    Returns
    -------
    x : torch.Tensor
        업데이트된 파라미터
    f_hist : list[float]
        각 스텝의 loss 값 (float)
    currentFuncEval : int
        함수 평가 횟수
    """

    if config.maxIter == 0:
        return x, [], 0

    global final_loss, times

    maxIter = int(config.maxIter)
    # maxEval이 None이면 약간 더 크게 설정
    maxEval = int(config.maxEval) if config.maxEval else int(maxIter * 1.25)
    tolFun = float(config.tolFun) if config.tolFun else 1e-5
    tolX = float(config.tolX) if config.tolX else 1e-19
    nCorrection = int(config.nCorrection) if config.nCorrection else 100
    lineSearch = config.lineSearch
    lineSearchOpts = config.lineSearchOptions
    learningRate = float(config.learningRate) if config.learningRate else 1.0
    isverbose = bool(config.verbose) if config.verbose else False

    # verbose function
    verbose = verbose_func if isverbose else (lambda _x: None)

    # ------------------------------------------------------------
    # evaluate initial f(x) and df/dx
    # ------------------------------------------------------------
    f, g = opfunc(x)          # f: scalar tensor, g: 1D tensor

    # 🔥 f_hist에는 tensor가 아니라 float만 저장해서 그래프 레퍼런스 제거
    f_val = f.item()
    f_hist = [f_val]

    currentFuncEval = 1
    state.funcEval = state.funcEval + 1
    p = g.shape[0]

    # check optimality of initial point
    tmp1 = torch.abs(g)
    if torch.sum(tmp1).item() <= tolFun:
        verbose("optimality condition below tolFun")
        return x, f_hist, currentFuncEval

    # optimize for a max of maxIter iterations
    nIter = 0
    times = []

    # old_dirs/old_stps/Hdiag 등은 while 안에서 정의되지만,
    # 루프를 한 번도 안 돌고 나오는 경우를 위해 기본값 준비
    old_dirs = []
    old_stps = []
    Hdiag = 1.0
    g_old = g.detach().clone()

    while nIter < maxIter:
        start_time = time.time()

        # keep track of nb of iterations
        nIter = nIter + 1
        state.nIter = state.nIter + 1

        ############################################################
        # compute gradient descent direction
        ############################################################
        if state.nIter == 1:
            d = -g
            old_dirs = []
            old_stps = []
            Hdiag = 1.0
        else:
            # do lbfgs update (update memory)
            y = g - g_old
            s = d * t
            ys = dot(y, s)

            if ys.item() > 1e-10:
                # updating memory
                if len(old_dirs) == nCorrection:
                    # shift history by one (limited-memory)
                    del old_dirs[0]
                    del old_stps[0]

                # store new direction/step
                old_dirs.append(s)
                old_stps.append(y)

                # update scale of initial Hessian approximation
                Hdiag = ys / dot(y, y)

            # compute the approximate (L-BFGS) inverse Hessian
            # multiplied by the gradient
            k = len(old_dirs)

            # need to be accessed element-by-element, so don't re-type tensor:
            ro = [0.0] * nCorrection
            for i in range(k):
                ro[i] = 1.0 / dot(old_stps[i], old_dirs[i])

            # iteration in L-BFGS loop collapsed to use just one buffer
            al = [0.0] * nCorrection

            q = -g
            for i in range(k - 1, -1, -1):
                al[i] = dot(old_dirs[i], q) * ro[i]
                q = q - al[i] * old_stps[i]

            # multiply by initial Hessian
            r = q * Hdiag
            for i in range(k):
                be_i = dot(old_stps[i], r) * ro[i]
                r += (al[i] - be_i) * old_dirs[i]

            d = r
            # final direction is in r/d (same object)

        # 그래프 없는 gradient 저장 (다음 iter에서 y = g - g_old 계산용)
        g_old = g.detach().clone()
        f_old = f_val  # float 로만 저장

        ############################################################
        # compute step length
        ############################################################
        # directional derivative
        gtd = dot(g, d)

        # check that progress can be made along that direction
        if gtd.item() > -tolX:
            verbose("Can not make progress along direction.")
            break

        # reset initial guess for step size
        if state.nIter == 1:
            tmp1 = torch.abs(g)
            denom = torch.sum(tmp1).item()
            if denom == 0.0:
                t = 1.0
            else:
                t = min(1.0, 1.0 / denom)
        else:
            t = learningRate

        # optional line search: user function
        lsFuncEval = 0
        if lineSearch and isinstance(lineSearch, types.FunctionType):
            # perform line search, using user function
            f, g, x, t, lsFuncEval = lineSearch(
                opfunc, x, t, d, f, g, gtd, lineSearchOpts
            )
            f_val = f.item()
            f_hist.append(f_val)
        else:
            # no line search, simply move with fixed-step
            # x += t * d
            with torch.no_grad():
                x.add_(t * d)

            if nIter != maxIter:
                # re-evaluate function only if not in last iteration
                f, g = opfunc(x)
                lsFuncEval = 1
                f_val = f.item()
                f_hist.append(f_val)

        # update func eval
        currentFuncEval = currentFuncEval + lsFuncEval
        state.funcEval = state.funcEval + lsFuncEval

        ############################################################
        # check conditions
        ############################################################
        if nIter == maxIter:
            break

        if currentFuncEval >= maxEval:
            # max nb of function evals
            verbose("max nb of function evals")
            break

        tmp1 = torch.abs(g)
        if torch.sum(tmp1).item() <= tolFun:
            # check optimality
            verbose("optimality condition below tolFun")
            break

        tmp1 = torch.abs(d * t)
        if torch.sum(tmp1).item() <= tolX:
            # step size below tolX
            verbose("step size below tolX")
            break

        # function value changing less than tolX
        if abs(f_val - f_old) < tolX:
            verbose("function value changing less than tolX" + str(abs(f_val - f_old)))
            break

        if do_verbose:
            log_fn(nIter, f_val, True)
            record_time()
            times.append(last_time())

        if nIter == maxIter - 1:
            final_loss = f_val

    # save state
    state.old_dirs = old_dirs
    state.old_stps = old_stps
    state.Hdiag = Hdiag
    state.g_old = g_old
    state.f_old = f_old
    state.t = t
    state.d = d

    return x, f_hist, currentFuncEval


# ---------------------------------------------------------------------------
# Lua-style Struct
# ---------------------------------------------------------------------------
class dummy(object):
    pass


class Struct(dummy):
    def __getattribute__(self, key):
        if key == "__dict__":
            return super(dummy, self).__getattribute__("__dict__")
        return self.__dict__.get(key, 0)
