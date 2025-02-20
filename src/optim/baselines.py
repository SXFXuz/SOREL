import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights, get_smooth_weights_sorted

class Optimizer:
    def __init__(self):
        pass

    def start_epoch(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_epoch(self):
        raise NotImplementedError

    def get_epoch_len(self):
        raise NotImplementedError


class SubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01):
        super(SubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )

    def start_epoch(self):
        pass

    def step(self):
        g = self.objective.get_batch_subgrad(self.weights)
        self.weights = self.weights - self.lr * g

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return 1


class StochasticSubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01, batch_size=64, seed=25, epoch_len=None, change_weights=False):
        super(StochasticSubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr
        self.batch_size = batch_size

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            if change_weights is False:
                self.weights = torch.zeros(
                    self.objective.d, requires_grad=True, dtype=torch.float64
                )
            else:
                self.weights = torch.ones(
                    self.objective.d, requires_grad=True, dtype=torch.float64
                )
        self.order = None
        self.iter = None
        torch.manual_seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0

    def step(self):
        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        self.weights.requires_grad = True
        g = self.objective.get_batch_subgrad(self.weights, idx=idx)
        self.weights.requires_grad = False
        self.weights = self.weights - self.lr * g
        self.iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


class SmoothedLSVRG(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=1.0,
        smoothing="l2",
        seed=25,
        length_epoch=None,
        change_weights = False,
    ):
        super(SmoothedLSVRG, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        self.lr = lr
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            if change_weights is False:
                self.weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
            else:
                self.weights = torch.ones(d, requires_grad=True, dtype=torch.float64)
        self.spectrum = self.objective.sigmas
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smoothing = smoothing
        if length_epoch:
            self.length_epoch = length_epoch
        else:
            self.length_epoch = int(nb_passes * n)
        self.nb_checkpoints = 0
        self.step_no = 0

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n

        # start epoch
        if self.step_no % n == 0:
            losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
            sorted_losses, self.argsort = torch.sort(losses, stable=True)
            self.sigmas = get_smooth_weights_sorted(
                sorted_losses, self.spectrum, self.smooth_coef, self.smoothing
            )
            with torch.enable_grad():
                self.subgrad_checkpt = self.objective.get_batch_subgrad(self.weights, include_reg=False)

            self.weights_checkpt = torch.clone(self.weights)
            self.nb_checkpoints += 1

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
        else:
            i = torch.tensor([np.random.choice(n, p=self.sigmas)])
        x = self.objective.X[self.argsort[i]]
        y = self.objective.y[self.argsort[i]]

        # Compute gradient at current iterate.
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        g_checkpt = self.objective.get_indiv_grad(self.weights_checkpt, x, y).squeeze()

        if self.uniform:
            direction = n * self.sigmas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction += self.objective.l2_reg * self.weights / n

        self.weights.copy_(self.weights - self.lr * direction)
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch


class SmoothedLSVRGBatch(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=0,
        smoothing="l2",
        seed=25,
        batch_size = 64,
        length_epoch=None,
        change_weights = False,
    ):
        super(SmoothedLSVRGBatch, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        self.lr = lr
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            if change_weights is False:
                self.weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
            else:
                self.weights = torch.full((d,), 0.5, dtype=torch.double, requires_grad=True)
        self.spectrum = self.objective.sigmas
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smoothing = smoothing
        self.nb_checkpoints = 0
        self.step_no = 0
        self.batch_size = 64
        if length_epoch:
            self.length_epoch = min(length_epoch, self.objective.n // self.batch_size)
        else:
            self.length_epoch = self.objective.n // self.batch_size
        losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
        self.sigmas = self.spectrum[torch.argsort(torch.argsort(losses))]
        torch.manual_seed(seed)

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0
        losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
        self.sigmas = get_smooth_weights(
                losses, self.spectrum, self.smooth_coef, self.smoothing
            )
        with torch.enable_grad():
            loss_gradient = self.objective.get_indiv_grad(self.weights)
        self.subgrad_checkpt = torch.matmul(self.sigmas, loss_gradient)
        self.weights_checkpt = torch.clone(self.weights)
        self.nb_checkpoints += 1

    @torch.no_grad()
    def step(self):
        n = self.objective.n

        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        self.weights.requires_grad = True
        g = self.objective.get_indiv_grad_idx(self.weights, idx)
        self.weights.requires_grad = False
        self.iter += 1

        # Compute gradient at current iterate.
        g_checkpt = self.objective.get_indiv_grad_idx(self.weights_checkpt, idx=idx)
        # g_checkpt = self.objective.get_indiv_grad(self.weights_checkpt, x, y).squeeze()

        direction = n * torch.matmul(self.sigmas[idx], (g-g_checkpt)) / self.batch_size + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction += self.objective.l2_reg * self.weights / n 
        self.weights.copy_(self.weights - self.lr * direction)
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch