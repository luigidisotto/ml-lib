import numpy as np
import random
import copy
import logging

from .activations import Linear, Sigmoid
from .loss import MeanSquaredError

logger = logging.getLogger(__name__)


class MLP:
    def __init__(
            self,
            learning_rate=0.1,
            momentum=0.0,
            lamda=0.0,
            max_epochs=2000,
            sigma=0.01,
            mu=0.0,
            layers=[],
            activation=Linear,
            activation_out=Sigmoid,
            loss=MeanSquaredError,
            eval_error=MeanSquaredError,
            task='regression',
            threshold=0.5,
            k=100,
            s=10,
            prog=1e-12,
            debug_interval=100,
            verbose='no',
            do_early_stopping=False
    ):

        self._lrate = learning_rate
        self._momentum = momentum
        self._lamda = lamda
        self._k = k
        self._s = s
        self._sigma = sigma
        self._mu = mu
        self._task = task
        self._threshold = threshold
        self._layers = layers
        self._nlayers = len(layers)
        self._max_epochs = max_epochs
        self._verbose = verbose
        self._prog = prog
        self._activation = activation
        self._activation_out = activation_out
        self._loss = MeanSquaredError
        self._eval = MeanSquaredError
        self._debug_interval = debug_interval
        self._do_early_stopping = do_early_stopping

        self.init_free_vars()

    def init_free_vars(self):
        self._ws = [np.random.normal(self._mu, self._sigma, (self._layers[i], self._layers[i+1]))
                    for i in range(self._nlayers-1)]
        self._bs = [np.random.normal(self._mu, self._sigma, (self._layers[i+1]))
                    for i in range(self._nlayers-1)]

    def n_hidden(self):
        return self._ws[0].shape[1]

    def split(self, X, Y, p):
        n = X.shape[0]
        # random dataset samples going to be the training set
        idx_tr = random.sample(range(n), int((1. - p)*n))
        # remaining dataset samples building up the validation set
        idx_va = [i for i in range(n) if i not in idx_tr]
        X_tr, Y_tr = X.take(idx_tr, axis=0), Y.take(idx_tr, axis=0)
        X_va, Y_va = X.take(idx_va, axis=0), Y.take(idx_va, axis=0)

        return X_tr, Y_tr, X_va, Y_va

    def penalty(self):
        # regularization penalty term
        penalty = np.sum(np.linalg.norm(self._ws[l], 2)**2 +
                         np.linalg.norm(self._bs[l], 2)**2 for l in range(self._nlayers-1))
        return penalty

    def fit(self, X, Y, p=.25):
        """
                X, Y: input and ouput training samples.
                p: fraction (default 20%) of X and Y to be used as validation set.

        """

        X_tr, Y_tr, X_va, Y_va = self.split(X, Y, p)

        tr_err_history, va_err_history = [], []
        tr_acc_history, va_acc_history = [], []
        loss_history = []

        grad_w, grad_b = [0.0]*(self._nlayers-1), [0.0]*(self._nlayers-1)

        epoch = 0

        h1, h2 = 1., False
        va_err_min = float('inf')
        ws_snapshot, bs_snapshot = [], []
        e_snap = 0

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        # (h1). early stop heuristic:
        # - Training error progress - (in per thousand) measured after every strip of ``k`` epochs has the following (informal) semantics:
        # 'how much was the average training error during a strip larger than the minimum train. err. during that strip?'.
        # Note that this progress measure is high for unstable phases of training, where the training set error goes up
        # instead of down. This is intended, because many training algorithms sometimes produce such "jitter" by taking
        # inappropiately large steps in weight space.
        # The progress measure is, however, guaranteed to approach zero in the long run unless the training
        # error is globally unstable (e.g. oscillating).

    	# (h2). early stop heuristic:
        # - Validation error progress - stop when validation error increased in
        # ``s`` successive strips of ``k`` epochs. In that way we have the advantage of measuring change locally,
        # allowing validation error to remain higher than previous minima over long training periods.

        while(epoch <= self._max_epochs and h1 > self._prog and h2 == False):

            xs, zs = [X_tr], []

            # forward
            for l in range(self._nlayers-2):
                # linearizations
                zs += [(xs[l].dot(self._ws[l]) + self._bs[l])]
                # introducing non-linearity
                xs += [(self._activation.apply(zs[-1]))]

            zs += [(xs[-1].dot(self._ws[-1]) + self._bs[-1])]
            xs += [(self._activation_out.apply(zs[-1]))]

            penalty = self.penalty() if self._lamda > 0.0 else 0.0
            loss = self._loss.apply(
                Y_tr, xs[-1])/Y_tr.shape[0] + 0.5*self._lamda*penalty
            loss_history += [loss]
            tr_err = self._eval.apply(Y_tr, xs[-1])/Y_tr.shape[0]
            tr_err_history += [tr_err]

            debug = f'[Epoch {epoch+1}] loss = {loss}, tr. mse = {tr_err}'

            # (h1). early stop heuristic:
            if ((epoch+1) % self._k) == 0 and self._do_early_stopping:
                h1 = 1000 * \
                    (sum(tr_err_history[-self._k:]) /
                     (self._k*min(tr_err_history[-self._k:])) - 1)

            Y_pred_va = self.predict(X_va)
            va_err_history += [self._eval.apply(Y_va, Y_pred_va)/Y_va.shape[0]]

            debug += f', va. mse = {va_err_history[-1]}'

            # (h2). early stop checkpoint: we keep track of free vars. values at
            # lowest point in validation error curve
            if ((epoch+1) % self._k) == 0 and self._do_early_stopping:
                if va_err_history[-1] < va_err_min:
                    va_err_min = va_err_history[-1]
                    ws_snapshot = copy.deepcopy(self._ws)
                    bs_snapshot = copy.deepcopy(self._bs)
                    e_snap = epoch

            # (h2). early stop heuristic: ``s`` strips of ``k`` epochs have been run
            if len(va_err_history) != 0 and (len(va_err_history) % (self._s*self._k)) == 0 and self._do_early_stopping:
                # validation error monotonicity history
                ms = [va_err_history[i]
                      for i in range(-1, -(self._s*self._k)-1, -self._k)]
                h2 = all(ms[i] > ms[i+1] for i in range(len(ms)-1))

            if self._task == 'classification':
                va_acc = self.accuracy(
                    Y_va, (Y_pred_va >= self._threshold).astype(int))
                va_acc_history += [va_acc]
                tr_acc = self.accuracy(
                    Y_tr, (xs[-1] >= self._threshold).astype(int))
                tr_acc_history += [tr_acc]
                debug += f', tr. accuracy = {tr_acc}, va. accuracy = {va_acc}'

            # backward
            ds = [self._loss.derivative(
                Y_tr, xs[-1]) * self._activation_out.derivative(zs[-1])/X_tr.shape[0]]  # deltas matrix list
            for l in range(-2, -self._nlayers, -1):
                # pre-pending deltas
                ds = [ds[l+1].dot(self._ws[l+1].T) *
                      self._activation.derivative(zs[l])] + ds

            # adding momentum to past gradients
            grad_w = [self._lrate*xs[l].T.dot(ds[l]) + self._momentum*grad_w[l]
                      for l in range(self._nlayers-1)]

            grad_b = [self._lrate*ds[l].sum(axis=0) + self._momentum*grad_b[l]
                      for l in range(self._nlayers-1)]

            # weights update
            for l in range(self._nlayers-1):
                # self._lrate * xs[l].T.dot(ds[l])
                self._ws[l] -= (grad_w[l] + self._lamda*self._ws[l])
                # self._lrate * ds[l].sum(axis=0)
                self._bs[l] -= (grad_b[l] + self._lamda*self._bs[l])

            if (epoch == 0 or ((epoch+1) % self._debug_interval) == 0) and self._verbose == 'yes':
                logger.info(debug)

            epoch += 1

        if h1 < self._prog and self._do_early_stopping:
            logger.info(
                f'[training error progress]: poor learning, epoch {epoch}')

        # backtrack to the solution we found at the epoch where the validation error curve
        # gave the lowest mse
        if h2 == True:
            self._ws = ws_snapshot
            self._bs = bs_snapshot
            logger.info(
                f"[validation error]: overfitting starting from epoch {epoch}")
            logger.info(f"Backtracked to network state at epoch {e_snap}")

        return {	"tr_err_hist": tr_err_history,
                 "tr_err": tr_err_history[-1],
                 "va_err_hist": va_err_history,
                 "va_err": va_err_history[-1] if va_err_history else [],
                 "tr_acc_hist": tr_acc_history,
                 "va_acc_hist": va_acc_history,
                 "tr_acc": tr_acc_history[-1] if tr_acc_history else [],
                 "va_acc": va_acc_history[-1] if va_acc_history else [],
                 "loss_hist": loss_history,
                 "loss": loss_history[-1] if loss_history else []
                 }

    def predict(self, X):
        output = X
        for l in range(self._nlayers-2):
            output = self._activation.apply(
                output.dot(self._ws[l]) + self._bs[l])
        output = output.dot(self._ws[-1] + self._bs[-1])
        output = self._activation_out.apply(output)
        return output

    def score(self, X, Y):
        Y_pred = self.predict(X)
        return self._eval.apply(Y, Y_pred)/Y.shape[0]

    def accuracy(self, X, Y):
        return np.mean(X == Y)
