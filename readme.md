# SOREL: A Stochastic Algorithm for Spectral Risks Minimization

We consider stochastic optimization of the spectral risk combined with a strongly convex regularizer:
$$
\min _{\boldsymbol{w}} \sum_{i=1}^n \sigma_i \ell_{[i]}(\boldsymbol{w})+g(\boldsymbol{w}),
$$
where $\ell_{[1]}(\cdot) \leq \cdots \leq \ell_{[n]}(\cdot)$ denotes the order statistics of the empirical loss distribution, and $0 \leq$ $\sigma_1 \leq \cdots \leq \sigma_n, \sum_{i=1}^n \sigma_i=1$.

## Dependencies

All algorithms are implemented in Python 3.8. Install the dependencies by running the following code in your terminal

```
pip install -r requirements.txt
```

## Reproducing Figures

Run `draw_regression.py`,  `draw_fair.py`,  `draw_dro.py` and  `draw_NN.py` to reproduce figures in the experiments. Experimental results can be found in the  `result` folder.

## Quickstart

`regression.ipynb`, `fair.ipynb`, and  `robust optimization.ipynb `  contain quick start guides for the three experiments in our paper. All the hyperparameters are summarized in `hyperparamters.py`.