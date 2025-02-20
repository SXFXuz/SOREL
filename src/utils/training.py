import pandas as pd
import time
from tqdm import tqdm
import torch
import os
import datetime

from src.optim.baselines import (
    StochasticSubgradientMethod,
    SmoothedLSVRG,
    SmoothedLSVRGBatch,
)
from src.optim.prospect import Prospect, ProspectMoreau, ProspectBatch
from src.optim.SOREL import Sorel
from src.optim.objectives import (
    Objective,
    get_extremile_weights,
    get_superquantile_weights,
    get_esrm_weights,
    get_erm_weights,
)
from src.utils.data import load_dataset

SUCCESS_CODE = 0
FAIL_CODE = -1

class OptimizationError(RuntimeError):
    pass


def get_optimizer(optim_cfg, objective, seed, device="cpu", change_weights=False):
    name, lr, epoch_len, shift_cost = (
        optim_cfg["optimizer"],
        optim_cfg["lr"],
        optim_cfg["epoch_len"],
        optim_cfg["shift_cost"],
    )

    lrd = 0.5 if "lrd" not in optim_cfg.keys() else optim_cfg["lrd"]
    penalty = "l2"

    if name == "sgd":
        return StochasticSubgradientMethod(
            objective, lr=lr, seed=seed, epoch_len=epoch_len, change_weights=change_weights
        )
    elif name == "lsvrg":
        return SmoothedLSVRG(
            objective,
            lr=lr,
            smooth_coef=shift_cost,
            smoothing=penalty,
            seed=seed,
            length_epoch=epoch_len,
        )
    elif name == "prospect":
        return Prospect(
            objective,
            lrp=lr,
            epoch_len=epoch_len,
            shift_cost=shift_cost,
            penalty=penalty,
            seed_grad=seed,
            seed_table=3 * seed,
        )
    elif name == "soreal":
        return Sorel(
            objective,
            lr=lr,
            smooth_coef=shift_cost,
            smoothing=penalty,
            seed=seed,
            length_epoch=epoch_len,
        )
    elif name == "lsvrg_batch":
        return SmoothedLSVRGBatch(
            objective,
            lr=lr,
            smooth_coef=shift_cost,
            smoothing=penalty,
            seed=seed,
            length_epoch=epoch_len,
            change_weights=change_weights,
        )
    elif name == "prospect_batch":
        return ProspectBatch(
            objective,
            lrp=lr,
            epoch_len=epoch_len,
            shift_cost=shift_cost,
            penalty=penalty,
            seed_grad=seed,
            seed_table=3 * seed,
            change_weights=change_weights
        )
    else:
        raise ValueError("Unreocgnized optimizer!")


def get_objective(model_cfg, X, y, dataset=None, autodiff=False):
    name, l2_reg, loss, n_class, shift_cost, par_value = (
        model_cfg["objective"],
        model_cfg["l2_reg"],
        model_cfg["loss"],
        model_cfg["n_class"],
        model_cfg["shift_cost"],
        model_cfg["para_value"],
    )
    if name == "erm":
        weight_function = lambda n: get_erm_weights(n)
    elif par_value is not None:
        if name == "cvar":
            weight_function = lambda n: get_superquantile_weights(n, par_value)
        elif name == "esrm":
            weight_function = lambda n: get_esrm_weights(n, par_value)
        elif name == "extremile":
            weight_function = lambda n: get_extremile_weights(n, par_value)
    else:
        if name == "extremile":
            weight_function = lambda n: get_extremile_weights(n, 2.5)
        elif name == "cvar":
            weight_function = lambda n: get_superquantile_weights(n, 0.5)
        elif name == "esrm":
            weight_function = lambda n: get_esrm_weights(n, 2.0)

    return Objective(
        X,
        y,
        weight_function,
        l2_reg=l2_reg,
        loss=loss,
        n_class=n_class,
        risk_name=name,
        dataset=dataset,
        shift_cost=shift_cost,
        penalty="l2",
        autodiff=autodiff,
    )
