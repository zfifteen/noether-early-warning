"""
NIM Experiment Runner: All four experiments in a single reproducible pipeline.

Experiment 1: Temporal Ordering Test
  - Does increment-norm drift become detectable before covariance asymmetry?

Experiment 2: Lambda Scaling Validation
  - Does drift amplitude scale with lambda = eta/kappa (not eta alone)?

Experiment 3: Adam vs SGD Comparison
  - Does momentum/adaptive learning mask or amplify the NIM signal?

Experiment 4: WeightNorm ResNet Extension
  - Does NIM generalize beyond simple models and LayerNorm?

ACTUAL CODE EXECUTION -- not a simulation.
"""

import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import linregress, spearmanr
from collections import defaultdict
import warnings

from models import (
    ToyScalarProduct, LinearScaleModel, LayerNormMLP,
    ScaleSymmetricProduct, WeightNormResNet
)
from nim_core import NIMLogger, estimate_curvature, compute_lambda_ratio, temporal_ordering_summary

# Reproducibility
DEVICE = torch.device('cpu')  # CPU for reproducibility and Hessian computation


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_regression_data(n_samples=500, input_dim=1, noise_std=0.1,
                              x_scale=2.0, true_slope=3.0, seed=42):
    """Generate synthetic regression data: y = true_slope * x + noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim) * x_scale
    y = true_slope * X.sum(axis=1) + rng.randn(n_samples) * noise_std
    return (torch.tensor(X, dtype=torch.float32, device=DEVICE),
            torch.tensor(y, dtype=torch.float32, device=DEVICE))


def generate_multidim_regression_data(n_samples=500, input_dim=4,
                                        noise_std=0.1, x_scale=2.0, seed=42):
    """Generate multi-dimensional regression data."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim) * x_scale
    true_w = rng.randn(input_dim) * 2.0
    y = X @ true_w + rng.randn(n_samples) * noise_std
    return (torch.tensor(X, dtype=torch.float32, device=DEVICE),
            torch.tensor(y, dtype=torch.float32, device=DEVICE))


def train_model(model, X, y, eta, n_steps, optimizer_type='sgd',
                loss_fn=None, logger=None, momentum=0.0, betas=(0.9, 0.999)):
    """
    Train a model and log NIM diagnostics.

    Args:
        model: nn.Module
        X, y: training data
        eta: learning rate
        n_steps: number of gradient steps
        optimizer_type: 'sgd' or 'adam'
        loss_fn: loss function (default MSE)
        logger: NIMLogger instance
        momentum: SGD momentum (default 0, pure SGD)
        betas: Adam betas
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    if logger is None:
        logger = NIMLogger()

    if optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=eta, momentum=momentum)
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=eta, betas=betas)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    for step in range(n_steps):
        optimizer.zero_grad()
        pred = model(X)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        loss = loss_fn(pred, y)
        loss.backward()

        # Log BEFORE step (gradients are fresh)
        logger.log_step(model, loss.item(), eta, step)

        optimizer.step()

    return logger


# =========================================================================
# EXPERIMENT 1: Temporal Ordering Test
# =========================================================================
def run_experiment_1(seeds=range(10), n_steps=600, output_dir='results/exp1'):
    """
    Test: Does NIM drift become detectable before covariance asymmetry?

    For each model type and seed:
      1. Train with finite eta (SGD, no momentum)
      2. Record onset step for:
         a. Increment-norm drift (via sliding-window linear regression)
         b. Covariance asymmetry (via eigenvalue ratio bootstrap test)
         c. Instantaneous symmetry proxy |g.w|
      3. Compare onset ordering across all seeds
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 70)
    print("EXPERIMENT 1: Temporal Ordering Test")
    print("=" * 70)

    results = {}

    # Model configurations
    configs = {
        'toy_scalar': {
            'eta': 0.05, 'n_steps': n_steps,
            'factory': lambda seed: ToyScalarProduct(
                u_init=1.0 + 0.1 * np.random.randn(),
                v_init=1.0 + 0.1 * np.random.randn()
            ),
            'data_fn': lambda seed: generate_regression_data(
                n_samples=200, input_dim=1, x_scale=2.0, seed=seed
            ),
        },
        'linear_scale': {
            'eta': 0.02, 'n_steps': n_steps,
            'factory': lambda seed: LinearScaleModel(
                w_init=1.0 + 0.1 * np.random.randn()
            ),
            'data_fn': lambda seed: generate_regression_data(
                n_samples=200, input_dim=1, x_scale=2.0, seed=seed
            ),
        },
        'layernorm_mlp': {
            'eta': 0.08, 'n_steps': n_steps,
            'factory': lambda seed: LayerNormMLP(input_dim=4, hidden_dim=32, output_dim=1),
            'data_fn': lambda seed: generate_multidim_regression_data(
                n_samples=300, input_dim=4, x_scale=2.0, seed=seed
            ),
        },
        'scale_symmetric': {
            'eta': 0.03, 'n_steps': n_steps,
            'factory': lambda seed: ScaleSymmetricProduct(input_dim=4, hidden_dim=8, output_dim=1),
            'data_fn': lambda seed: generate_multidim_regression_data(
                n_samples=300, input_dim=4, x_scale=2.0, seed=seed
            ),
        },
    }

    for model_name, cfg in configs.items():
        print(f"\n--- {model_name} ---")
        model_results = []

        for seed in seeds:
            set_seed(seed)
            model = cfg['factory'](seed)
            X, y = cfg['data_fn'](seed)
            logger = NIMLogger(window_size=50)

            train_model(model, X, y, eta=cfg['eta'], n_steps=cfg['n_steps'],
                       optimizer_type='sgd', logger=logger)

            # Detect onsets
            nim_onset = logger.detect_drift_onset(window=50, p_threshold=1e-6)
            proxy_onset = logger.detect_symmetry_proxy_onset(window=50, p_threshold=1e-6)

            # Covariance test needs larger window
            cov_onset = logger.detect_covariance_asymmetry_onset(
                window=min(100, cfg['n_steps'] // 3), p_threshold=1e-6
            )

            ordering = temporal_ordering_summary(
                nim_onset, cov_onset, proxy_onset, cfg['n_steps']
            )

            # Also record drift stats
            drift_stats = logger.get_increment_drift_stats()
            weight_drift = logger.get_weight_norm_drift()

            run_result = {
                'seed': seed,
                'ordering': ordering,
                'drift_stats': drift_stats,
                'weight_norm_drift': weight_drift,
                'increment_norms': logger.increment_norms,
                'weight_norms': logger.weight_norms,
                'symmetry_proxies': logger.symmetry_proxies,
                'losses': logger.losses,
            }
            if hasattr(model, 'noether_charge'):
                run_result['noether_charges'] = logger.noether_charges

            model_results.append(run_result)

            status = "NIM FIRST" if ordering['nim_before_covariance'] else "COV FIRST or TIE"
            print(f"  seed={seed}: NIM@{nim_onset}, Cov@{cov_onset}, "
                  f"Proxy@{proxy_onset} -> {status}")

        results[model_name] = model_results

    # Save results
    _save_results(results, output_dir, 'experiment_1_results.json')
    return results


# =========================================================================
# EXPERIMENT 2: Lambda Scaling Validation
# =========================================================================
def run_experiment_2(seeds=range(5), n_steps=500, output_dir='results/exp2'):
    """
    Test: Does drift amplitude scale with lambda = eta/kappa, not eta alone?

    Procedure:
      1. For each (eta, kappa) pair, train the model and measure drift slope
      2. Compute lambda = eta / kappa for each run
      3. Fit: slope vs eta (should be weak) and slope vs lambda (should be strong)
      4. Compare R^2 values
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Lambda Scaling Validation")
    print("=" * 70)

    # Vary eta and x_scale (which controls effective kappa)
    etas = [0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    x_scales = [0.5, 1.0, 2.0, 4.0]  # Controls effective curvature

    results = []

    for eta in etas:
        for x_scale in x_scales:
            for seed in seeds:
                set_seed(seed)

                # Use LayerNorm MLP as primary testbed
                model = LayerNormMLP(input_dim=4, hidden_dim=32, output_dim=1)
                X, y = generate_multidim_regression_data(
                    n_samples=300, input_dim=4, x_scale=x_scale, seed=seed
                )
                loss_fn = nn.MSELoss()

                # Estimate curvature before training
                try:
                    kappa = estimate_curvature(model, loss_fn, X, y, n_samples=20)
                except Exception:
                    kappa = 1.0  # fallback

                lam = compute_lambda_ratio(eta, kappa)

                # Fresh model (curvature estimation may have changed state)
                set_seed(seed)
                model = LayerNormMLP(input_dim=4, hidden_dim=32, output_dim=1)
                logger = NIMLogger()

                try:
                    train_model(model, X, y, eta=eta, n_steps=n_steps,
                               optimizer_type='sgd', logger=logger)
                except RuntimeError:
                    # Training may diverge for high eta
                    print(f"  DIVERGED: eta={eta}, x_scale={x_scale}, seed={seed}")
                    continue

                drift_stats = logger.get_increment_drift_stats()
                if drift_stats is None:
                    continue

                results.append({
                    'eta': eta,
                    'x_scale': x_scale,
                    'kappa': kappa,
                    'lambda': lam,
                    'seed': seed,
                    'drift_slope': drift_stats['slope'],
                    'drift_r_squared': drift_stats['r_squared'],
                    'drift_p_value': drift_stats['p_value'],
                    'weight_norm_drift': logger.get_weight_norm_drift(),
                })

    # Analysis: slope vs eta and slope vs lambda
    if len(results) > 5:
        slopes = [r['drift_slope'] for r in results]
        etas_arr = [r['eta'] for r in results]
        lambdas_arr = [r['lambda'] for r in results if r['lambda'] < 100]
        slopes_for_lambda = [r['drift_slope'] for r in results if r['lambda'] < 100]

        # Linear regression: slope vs eta
        if len(etas_arr) > 2:
            reg_eta = linregress(etas_arr, slopes)
            print(f"\n  slope vs eta:    R^2={reg_eta.rvalue**2:.4f}, p={reg_eta.pvalue:.2e}")

        # Linear regression: slope vs lambda
        if len(lambdas_arr) > 2:
            reg_lambda = linregress(lambdas_arr, slopes_for_lambda)
            print(f"  slope vs lambda: R^2={reg_lambda.rvalue**2:.4f}, p={reg_lambda.pvalue:.2e}")

        # Spearman rank correlation (more robust)
        if len(lambdas_arr) > 2:
            rho_eta, p_eta = spearmanr(etas_arr, slopes)
            rho_lam, p_lam = spearmanr(lambdas_arr, slopes_for_lambda)
            print(f"  Spearman(slope, eta):    rho={rho_eta:.4f}, p={p_eta:.2e}")
            print(f"  Spearman(slope, lambda): rho={rho_lam:.4f}, p={p_lam:.2e}")

    _save_results(results, output_dir, 'experiment_2_results.json')
    return results


# =========================================================================
# EXPERIMENT 3: Adam vs SGD Comparison
# =========================================================================
def run_experiment_3(seeds=range(10), n_steps=600, output_dir='results/exp3'):
    """
    Test: Does momentum/adaptive learning mask or amplify the NIM signal?

    Compare SGD (no momentum), SGD (momentum=0.9), and Adam on the same
    models and data, measuring:
      - Drift slope magnitude
      - Detection onset step
      - Signal-to-noise ratio of drift
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Adam vs SGD Comparison")
    print("=" * 70)

    optimizer_configs = {
        'sgd_pure': {'type': 'sgd', 'momentum': 0.0},
        'sgd_momentum': {'type': 'sgd', 'momentum': 0.9},
        'adam': {'type': 'adam'},
    }

    eta = 0.01  # moderate learning rate
    results = defaultdict(list)

    for opt_name, opt_cfg in optimizer_configs.items():
        print(f"\n--- {opt_name} ---")

        for seed in seeds:
            set_seed(seed)
            model = LayerNormMLP(input_dim=4, hidden_dim=32, output_dim=1)
            X, y = generate_multidim_regression_data(
                n_samples=300, input_dim=4, x_scale=2.0, seed=seed
            )
            logger = NIMLogger()

            try:
                train_model(
                    model, X, y, eta=eta, n_steps=n_steps,
                    optimizer_type=opt_cfg['type'],
                    logger=logger,
                    momentum=opt_cfg.get('momentum', 0.0)
                )
            except RuntimeError:
                print(f"  DIVERGED: {opt_name}, seed={seed}")
                continue

            drift_stats = logger.get_increment_drift_stats()
            nim_onset = logger.detect_drift_onset(window=50, p_threshold=1e-6)

            run_result = {
                'optimizer': opt_name,
                'seed': seed,
                'drift_stats': drift_stats,
                'nim_onset': nim_onset,
                'increment_norms': logger.increment_norms,
                'weight_norms': logger.weight_norms,
                'losses': logger.losses,
            }
            results[opt_name].append(run_result)

            slope_str = f"{drift_stats['slope']:.2e}" if drift_stats else "N/A"
            print(f"  seed={seed}: onset={nim_onset}, slope={slope_str}")

    _save_results(dict(results), output_dir, 'experiment_3_results.json')
    return dict(results)


# =========================================================================
# EXPERIMENT 4: WeightNorm ResNet Extension
# =========================================================================
def run_experiment_4(seeds=range(10), n_steps=600, output_dir='results/exp4'):
    """
    Test: Does NIM generalize to deeper architectures with WeightNorm?

    Uses a simple ResNet with WeightNorm blocks. Tests:
      - Whether increment-norm drift is detectable
      - Temporal ordering (NIM vs covariance)
      - Per-block drift patterns
    """
    os.makedirs(output_dir, exist_ok=True)
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: WeightNorm ResNet Extension")
    print("=" * 70)

    eta = 0.01
    results = []

    for seed in seeds:
        set_seed(seed)
        model = WeightNormResNet(input_dim=8, hidden_dim=16, output_dim=1, n_blocks=2)
        X, y = generate_multidim_regression_data(
            n_samples=400, input_dim=8, x_scale=2.0, seed=seed
        )
        logger = NIMLogger()

        try:
            train_model(model, X, y, eta=eta, n_steps=n_steps,
                       optimizer_type='sgd', logger=logger)
        except RuntimeError:
            print(f"  DIVERGED: seed={seed}")
            continue

        nim_onset = logger.detect_drift_onset(window=50, p_threshold=1e-6)
        drift_stats = logger.get_increment_drift_stats()
        proxy_onset = logger.detect_symmetry_proxy_onset(window=50, p_threshold=1e-6)

        # Covariance test
        cov_onset = logger.detect_covariance_asymmetry_onset(
            window=100, p_threshold=1e-6
        )

        ordering = temporal_ordering_summary(
            nim_onset, cov_onset, proxy_onset, n_steps
        )

        run_result = {
            'seed': seed,
            'ordering': ordering,
            'drift_stats': drift_stats,
            'nim_onset': nim_onset,
            'increment_norms': logger.increment_norms,
            'weight_norms': logger.weight_norms,
            'losses': logger.losses,
        }
        results.append(run_result)

        status = "NIM FIRST" if ordering.get('nim_before_covariance') else "COV FIRST or TIE"
        print(f"  seed={seed}: NIM@{nim_onset}, Cov@{cov_onset}, "
              f"Proxy@{proxy_onset} -> {status}")

    _save_results(results, output_dir, 'experiment_4_results.json')
    return results


# =========================================================================
# Helper: save results (convert numpy/tensor types for JSON)
# =========================================================================
def _save_results(results, output_dir, filename):
    """Save results dict to JSON, handling numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        if isinstance(obj, bool):
            return bool(obj)
        return obj

    filepath = os.path.join(output_dir, filename)
    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\n  Results saved to {filepath}")


# =========================================================================
# MAIN: Run all experiments
# =========================================================================
def run_all():
    """Execute the full NIM experiment pipeline."""
    print("=" * 70)
    print("  NOETHER INCREMENT MONITOR (NIM) -- FULL EXPERIMENT PIPELINE")
    print("  Validating/Falsifying: L2 increment drift as early KSB detector")
    print("=" * 70)
    print()

    start_time = time.time()

    exp1_results = run_experiment_1(seeds=range(10), n_steps=400)
    exp2_results = run_experiment_2(seeds=range(3), n_steps=300)
    exp3_results = run_experiment_3(seeds=range(10), n_steps=400)
    exp4_results = run_experiment_4(seeds=range(10), n_steps=400)

    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"  ALL EXPERIMENTS COMPLETE in {elapsed:.1f}s")
    print(f"{'=' * 70}")

    return {
        'exp1': exp1_results,
        'exp2': exp2_results,
        'exp3': exp3_results,
        'exp4': exp4_results,
    }


if __name__ == '__main__':
    run_all()
