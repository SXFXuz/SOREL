import torch
import numpy as np
from src.optim.smoothing import get_smooth_weights
from src.optim.baselines import Optimizer
from numba import jit
import warnings

class Prospect(Optimizer):
    def __init__(
        self,
        objective,
        lrp=0.01,
        lrd=None,
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        shift_cost=1.0,
        penalty="l2",
        oracle_reg="prox",
        change_weights = False,
    ):
        super(Prospect, self).__init__()
        self.objective = objective
        self.lrp = lrp
        self.lrd = 1.0 if lrd is None else lrd
        n, d = self.objective.n, self.objective.d
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
            self.grad_table = torch.zeros(n, objective.n_class * d, dtype=torch.float64)
        else:
            if change_weights is False:
                self.weights = torch.zeros(
                    self.objective.d, requires_grad=True, dtype=torch.float64
                )
            else:
                self.weights = torch.ones(
                    self.objective.d, requires_grad=True, dtype=torch.float64
                )
            self.grad_table = torch.zeros(n, d, dtype=torch.float64)
        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)
        self.shift_cost = n * shift_cost
        self.penalty = penalty
        assert oracle_reg in ["prox", "grad"]
        self.oracle_reg = oracle_reg

        # Generate loss and gradient tables.
        self.losses = self.objective.get_indiv_loss(self.weights).detach()
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        self.rho = self.lam.clone()
        real_l2_reg = self.objective.l2_reg / n

        if self.oracle_reg == "grad":
            self.grad_table = self.objective.get_indiv_grad(self.weights) + real_l2_reg * self.weights[None, :]
        else:
            self.grad_table = self.objective.get_indiv_grad(self.weights)
        self.running_subgrad = torch.matmul(self.grad_table.T, self.rho)

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = self.objective.n

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n
        real_l2_reg = self.objective.l2_reg / n

        # Compute gradient at current iterate.
        i = torch.tensor([self.rng_grad.randint(0, n)])
        x = self.objective.X[i]
        y = self.objective.y[i]
        loss = self.objective.loss(self.weights, x, y)
        g = self.objective.get_indiv_grad(self.weights, x, y).squeeze()
        if self.oracle_reg == "grad":
            g += real_l2_reg * self.weights

        # Compute gradient at from table.
        g_old = self.grad_table[i].reshape(-1)

        v = n * self.lam[i] * g - n * self.rho[i] * g_old + self.running_subgrad

        # Update iterate.
        if self.oracle_reg == "prox":
            self.weights.copy_(
                1 / (self.lrp * real_l2_reg + 1) * (self.weights - self.lrp * v)
            )
        else:
            self.weights.copy_(self.weights - self.lrp * v)

        # update dual weights
        # self.losses[i] = self.lrd*loss + (1-self.lrd)*self.losses[i]
        self.losses[i] = loss.detach()
        cur_lam = self.lam[i]
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        rho_old = self.rho[i]
        self.rho[i] = cur_lam

        # update table
        self.grad_table[i] = g[None, :]
        self.running_subgrad += cur_lam * g - rho_old * g_old

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len
    
class ProspectBatch(Optimizer):
    def __init__(
        self,
        objective,
        lrp=0.01,
        lrd=None,
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        shift_cost=1.0,
        penalty="l2",
        oracle_reg="prox",
        batch_size = 64,
        change_weights = False,
    ):
        super(ProspectBatch, self).__init__()
        self.objective = objective
        self.lrp = lrp
        self.lrd = 1.0 if lrd is None else lrd
        n, d = self.objective.n, self.objective.d
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
            self.grad_table = torch.zeros(n, objective.n_class * d, dtype=torch.float64)
        else:
            if change_weights is False:
                self.weights = torch.zeros(
                    self.objective.d, requires_grad=True, dtype=torch.float64
                )
            else:
                self.weights = torch.ones(
                    self.objective.d, requires_grad=True, dtype=torch.float64
                )
            self.grad_table = torch.zeros(n, d, dtype=torch.float64)
        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)
        self.shift_cost = n * shift_cost
        self.penalty = penalty
        assert oracle_reg in ["prox", "grad"]
        self.oracle_reg = oracle_reg

        # Generate loss and gradient tables.
        self.losses = self.objective.get_indiv_loss(self.weights).detach()
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        self.rho = self.lam.clone()
        real_l2_reg = self.objective.l2_reg / n

        if self.oracle_reg == "grad":
            self.grad_table = self.objective.get_indiv_grad(self.weights) + real_l2_reg * self.weights[None, :]
        else:
            self.grad_table = self.objective.get_indiv_grad(self.weights)
        self.running_subgrad = torch.matmul(self.grad_table.T, self.rho)

        self.order = None
        self.batch_size = batch_size

        if epoch_len:
            self.length_epoch = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.length_epoch = self.objective.n // self.batch_size
        self.iter = 0
        torch.manual_seed(seed_grad)

    def start_epoch(self):
        self.order = torch.randperm(self.objective.n)
        self.iter = 0


    @torch.no_grad()
    def step(self):
        n = self.objective.n
        real_l2_reg = self.objective.l2_reg / n

        # Compute gradient at current iterate.
        idx = self.order[
            self.iter
            * self.batch_size : min(self.objective.n, (self.iter + 1) * self.batch_size)
        ]
        loss = self.objective.get_indiv_loss_idx(self.weights, idx)
        self.weights.requires_grad = True
        # g = self.objective.get_batch_subgrad(self.weights, idx=idx)
        g = self.objective.get_indiv_grad_idx(self.weights, idx)
        self.weights.requires_grad = False
        g_old = self.grad_table[idx]
        v = n * (torch.matmul(self.lam[idx], g) - torch.matmul(self.rho[idx], g_old)) / self.batch_size + self.running_subgrad

        # Update iterate.
        if self.oracle_reg == "prox":
            self.weights.copy_(
                1 / (self.lrp * real_l2_reg + 1) * (self.weights - self.lrp * v)
            )
        else:
            self.weights.copy_(self.weights - self.lrp * v)


        self.losses[idx] = loss.detach()
        cur_lam = self.lam[idx]
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )
        rho_old = self.rho[idx]
        self.rho[idx] = cur_lam
        self.grad_table[idx] = g
        self.running_subgrad += torch.matmul(cur_lam, g) - torch.matmul(rho_old, g_old)

        self.iter += 1


    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch


class ProspectMoreau(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        seed_grad=25,
        seed_table=123,
        epoch_len=None,
        shift_cost=1.0,
        penalty="l2",
    ):
        super(ProspectMoreau, self).__init__()
        self.objective = objective
        self.lr = lr
        n, d = self.objective.n, self.objective.d
        if self.objective.autodiff == False:
            warnings.warn("ProspectMoreau does not currently have a non-autodiff implementation")
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
            self.grad_table = torch.zeros(n, objective.n_class * d, dtype=torch.float64)
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
            self.grad_table = torch.zeros(n, d, dtype=torch.float64)
        self.sigmas = self.objective.sigmas
        self.rng_grad = np.random.RandomState(seed_grad)
        self.rng_table = np.random.RandomState(seed_table)
        self.shift_cost = n * shift_cost
        self.penalty = penalty

        # Generate loss and gradient tables.
        self.losses = self.objective.get_indiv_loss(self.weights)
        self.lam = get_smooth_weights(
            self.losses, self.sigmas, self.shift_cost, self.penalty
        )

        for i in range(n):
            loss = self.objective.loss(
                self.weights, self.objective.X[i, :], self.objective.y[i]
            )
            self.grad_table[i] = (
                torch.autograd.grad(outputs=loss, inputs=self.weights)[0]
                + self.objective.l2_reg * self.weights / n
            )

        self.grad_table = self.grad_table

        self.running_subgrad = torch.matmul(
            self.grad_table.T, self.lam
        ).cpu()

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = self.objective.n

    def start_epoch(self):
        pass

    def step(self):
        n = self.objective.n

        p = torch.abs(self.lam) / (torch.sum(abs(self.lam)))
        i = torch.tensor([np.random.choice(n, p=p)])
        g_old_i = self.grad_table[i].reshape(-1).cpu()

        j = torch.tensor([self.rng_grad.randint(0, n)])
        x_j = self.objective.X[j]
        y_j = self.objective.y[j]
        loss_j = self.objective.loss(self.weights, x_j, y_j)
        g_old_j = self.grad_table[j].reshape(-1).cpu()

        v_weights = self.weights + self.lr * (g_old_i - self.running_subgrad)
        v_table = self.weights + self.lr * (g_old_j - self.running_subgrad)

        self.weights = self.objective.get_indiv_prox_loss(v_weights, self.lr, i)

        mor_grad_j = self.objective.get_indiv_mor_grad(v_table, self.lr, j)
        self.grad_table[j] = mor_grad_j.reshape(1, -1)

        self.losses[j] = loss_j
        self.lam = get_smooth_weights(
            self.losses.detach(), self.sigmas, self.shift_cost, self.penalty
        )

        self.running_subgrad = torch.matmul(
            self.lam, self.grad_table
        ).cpu()

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len


@jit(nopython=True)
def bubble_sort(idx, sort, argsort):
    n = len(sort)

    # Bubble left.
    j = idx
    while j > 0 and sort[j] < sort[j - 1] - 1e-10:
        # Swap elements in sorted vector.
        temp = sort[j]
        sort[j] = sort[j - 1]
        sort[j - 1] = temp

        # Swap elements in "argsort" vector.
        temp = argsort[j]
        argsort[j] = argsort[j - 1]
        argsort[j - 1] = temp

        j -= 1

    # Bubble right.
    j = idx
    while j < n - 1 and sort[j] > sort[j + 1] + 1e-10:
        # Swap elements in sorted vector.
        temp = sort[j]
        sort[j] = sort[j + 1]
        sort[j + 1] = temp

        # Swap elements in "argsort" vector.
        temp = argsort[j]
        argsort[j] = argsort[j + 1]
        argsort[j + 1] = temp

        j += 1
