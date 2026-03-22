"""
Fixed Experiment 2: Lambda Scaling Validation

The curvature estimation needs to be done more carefully.
Instead of relying on Hessian-vector products (which fail silently),
we use the empirical curvature: average squared gradient norm divided by
loss value, or we directly vary curvature by controlling data variance.

Key insight: x_scale directly controls the effective Hessian eigenvalues
for MSE loss. For y = w*x, the Hessian is H = (2/n) X^T X, so
kappa ~ x_scale^2. We can use this analytical relationship.

ACTUAL CODE EXECUTION -- not a simulation.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import linregress, spearmanr
from collections import defaultdict

from models import LayerNormMLP, ScaleSymmetricProduct, LinearScaleModel
from nim_core import NIMLogger

DEVICE = torch.device('cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def generate_data(n_samples, input_dim, x_scale, seed):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim) * x_scale
    true_w = np.array([3.0] * input_dim)
    y = X @ true_w + rng.randn(n_samples) * 0.1
    return (torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32))

def empirical_curvature(model, X, y, loss_fn, n_samples=20):
    """
    Estimate curvature from gradient variance.
    kappa ~ E[||g||^2] / E[loss] which scales as the trace of the Hessian.
    """
    model.zero_grad()
    pred = model(X)
    if pred.dim() > 1:
        pred = pred.squeeze(-1)
    loss = loss_fn(pred, y)
    loss.backward()

    g_norm_sq = sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None).item()
    loss_val = loss.item()

    if loss_val < 1e-12:
        return g_norm_sq  # when loss is tiny, curvature ~ gradient norm
    return g_norm_sq / loss_val

def train_and_measure(model, X, y, eta, n_steps, loss_fn):
    """Train with pure SGD and return drift slope."""
    logger = NIMLogger()
    optimizer = torch.optim.SGD(model.parameters(), lr=eta)

    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(X)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        loss = loss_fn(pred, y)

        if not torch.isfinite(loss):
            return None  # diverged

        loss.backward()
        logger.log_step(model, loss.item(), eta, step)
        optimizer.step()

    stats = logger.get_increment_drift_stats()
    return stats

def run_experiment_2_fixed():
    os.makedirs('results/exp2_fixed', exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 2 (FIXED): Lambda Scaling Validation")
    print("=" * 70)

    etas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08]
    x_scales = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
    seeds = range(5)
    n_steps = 300
    loss_fn = nn.MSELoss()

    results = []

    for eta in etas:
        for x_scale in x_scales:
            for seed in seeds:
                set_seed(seed)

                # LayerNorm MLP
                model = LayerNormMLP(input_dim=4, hidden_dim=32, output_dim=1)
                X, y = generate_data(300, 4, x_scale, seed)

                # Estimate curvature before training
                kappa = empirical_curvature(model, X, y, loss_fn)
                lam = eta / max(kappa, 1e-12)

                # Reset and train
                set_seed(seed)
                model = LayerNormMLP(input_dim=4, hidden_dim=32, output_dim=1)

                stats = train_and_measure(model, X, y, eta, n_steps, loss_fn)
                if stats is None or np.isnan(stats['slope']):
                    continue

                results.append({
                    'eta': eta,
                    'x_scale': x_scale,
                    'kappa': kappa,
                    'lambda': lam,
                    'seed': seed,
                    'drift_slope': stats['slope'],
                    'drift_abs_slope': abs(stats['slope']),
                    'drift_r_squared': stats['r_squared'],
                    'drift_p_value': stats['p_value'],
                })

    print(f"\nTotal valid runs: {len(results)}")

    # Filter out NaN and Inf
    clean = [r for r in results if np.isfinite(r['drift_slope']) and np.isfinite(r['lambda'])]
    print(f"Clean runs (no NaN/Inf): {len(clean)}")

    if len(clean) > 10:
        abs_slopes = [r['drift_abs_slope'] for r in clean]
        etas_arr = [r['eta'] for r in clean]
        lambdas_arr = [r['lambda'] for r in clean]
        kappas_arr = [r['kappa'] for r in clean]

        # Key test: does |slope| correlate better with lambda or eta?
        rho_eta, p_eta = spearmanr(etas_arr, abs_slopes)
        rho_lam, p_lam = spearmanr(lambdas_arr, abs_slopes)
        rho_kap, p_kap = spearmanr(kappas_arr, abs_slopes)

        reg_eta = linregress(etas_arr, abs_slopes)
        reg_lam = linregress(lambdas_arr, abs_slopes)

        print(f"\n--- Scaling Analysis ---")
        print(f"|slope| vs eta:    R^2={reg_eta.rvalue**2:.4f}, Spearman rho={rho_eta:.4f}, p={p_eta:.2e}")
        print(f"|slope| vs lambda: R^2={reg_lam.rvalue**2:.4f}, Spearman rho={rho_lam:.4f}, p={p_lam:.2e}")
        print(f"|slope| vs kappa:  Spearman rho={rho_kap:.4f}, p={p_kap:.2e}")

        # Also test: controlling for kappa, does lambda still predict?
        # Group by x_scale (proxy for kappa) and check eta effect within group
        print(f"\n--- Within-kappa-group analysis ---")
        by_xscale = defaultdict(list)
        for r in clean:
            by_xscale[r['x_scale']].append(r)

        for xs in sorted(by_xscale.keys()):
            group = by_xscale[xs]
            if len(group) >= 5:
                g_etas = [r['eta'] for r in group]
                g_slopes = [r['drift_abs_slope'] for r in group]
                rho, p = spearmanr(g_etas, g_slopes)
                print(f"  x_scale={xs}: rho(eta, |slope|)={rho:.3f}, p={p:.3e}, n={len(group)}")

        # Save enriched results
        summary = {
            'runs': clean,
            'analysis': {
                'r2_eta': reg_eta.rvalue**2,
                'r2_lambda': reg_lam.rvalue**2,
                'spearman_eta': {'rho': rho_eta, 'p': p_eta},
                'spearman_lambda': {'rho': rho_lam, 'p': p_lam},
                'spearman_kappa': {'rho': rho_kap, 'p': p_kap},
                'n_runs': len(clean),
            }
        }
    else:
        summary = {'runs': clean, 'analysis': None}

    with open('results/exp2_fixed/experiment_2_fixed_results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\nSaved to results/exp2_fixed/experiment_2_fixed_results.json")

    return summary

if __name__ == '__main__':
    run_experiment_2_fixed()
