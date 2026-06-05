# sophiaupdate — Sophia Optimizer for MATLAB Deep Learning Toolbox

A MATLAB implementation of **Sophia** (*Second-order Clipped Stochastic Optimization*),
a scalable second-order optimizer for training deep learning models — including large
language models — faster than Adam with the same or lower compute budget.

> Liu et al. (2023). *Sophia: A Scalable Stochastic Second-order Optimizer for Language
> Model Pre-training.* [arXiv:2305.14342](https://arxiv.org/abs/2305.14342)

---

## Why Sophia?

Adam applies the same effective step size to all parameter dimensions — it normalises by
the square root of the gradient's second moment, which is only a rough proxy for
curvature. Sophia replaces that proxy with a **cheap diagonal Hessian estimate** and adds
**per-coordinate clipping** so that:

- **Sharp dimensions** (high curvature) receive a shrunk update — preventing overshooting.
- **Flat dimensions** (low curvature) receive the full update — accelerating convergence.
- **Negative / near-zero curvature** automatically falls back to momentum SignSGD.

In practice this equalises loss-decrease across all parameter dimensions and has been
shown to reach the same validation loss as Adam in roughly **half the number of steps**.

---

## Quick Start

```matlab
avg_g    = [];
avg_hess = [];
bs       = batch_size * seq_len;   % effective token count
hess_interval = 10;

k = 0;
for t = 1:max_iters
    do_hess = (mod(t, hess_interval) == 0) || (t == 1);

    [loss, g] = dlfeval(@modelGradients, model, X, Y);

    hess_est = [];
    if do_hess
	    k = k + 1; 
        [~, g_sampled] = dlfeval(@sampledGradients, model, X);
        hess_est = bs .* g_sampled .* g_sampled;   % GNB estimator
    end

    [model, avg_g, avg_hess] = sophiaupdate( ...
        model, g, avg_g, avg_hess, hess_est, do_hess, t, ...
        k, 3e-4, 0.965, 0.99, 0.04, bs, 0.1, 1e-15);
end
