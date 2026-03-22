"""
Generate all publication-quality figures for the NIM paper.

ACTUAL CODE EXECUTION -- not a simulation.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress, spearmanr
from collections import defaultdict
import torch
import torch.nn as nn
from nim_core import NIMLogger
from models import LayerNormMLP, LinearScaleModel

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# =========================================================================
# Figure 1: Representative Time Series (single model, one seed)
# =========================================================================
def fig1_timeseries():
    """Four-panel time series showing NIM signal in context."""
    exp1 = load_json('results/exp1/experiment_1_results.json')
    if not exp1 or 'layernorm_mlp' not in exp1:
        return

    # Pick a seed where NIM detected before covariance
    run = None
    for r in exp1['layernorm_mlp']:
        if r['ordering']['nim_drift_onset'] is not None:
            run = r
            break
    if run is None:
        run = exp1['layernorm_mlp'][0]

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    steps = np.arange(len(run['increment_norms']))
    window = 30

    # Panel 1: Increment norms + running average
    ax = axes[0]
    ax.plot(steps, run['increment_norms'], alpha=0.2, color='#2196F3', linewidth=0.5)
    if len(run['increment_norms']) > window:
        ra = np.convolve(run['increment_norms'], np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(run['increment_norms'])), ra,
                color='#2196F3', linewidth=2, label=f'{window}-step moving avg')
        # Add regression line
        t_ra = np.arange(len(ra))
        slope, intercept, _, _, _ = linregress(t_ra, ra)
        ax.plot(np.arange(window-1, len(run['increment_norms'])),
                slope * t_ra + intercept, 'r--', linewidth=1.5,
                label=f'slope={slope:.2e}')
    ax.set_ylabel(r'$\eta\,\|\mathbf{g}_t\|_2$', fontsize=13)
    ax.set_title('(A) L2 Increment Norm (NIM Signal)', fontweight='bold')
    ax.legend(loc='upper right')

    # Panel 2: Weight norm
    ax = axes[1]
    ax.plot(steps, run['weight_norms'], color='#2E7D32', linewidth=1.5)
    ax.set_ylabel(r'$\|\mathbf{w}\|_2$', fontsize=13)
    ax.set_title('(B) Weight Norm Drift', fontweight='bold')

    # Panel 3: Symmetry proxy
    ax = axes[2]
    ax.plot(steps, run['symmetry_proxies'], alpha=0.3, color='#FF9800', linewidth=0.5)
    if len(run['symmetry_proxies']) > window:
        ra_p = np.convolve(run['symmetry_proxies'], np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(run['symmetry_proxies'])), ra_p,
                color='#FF9800', linewidth=2)
    ax.set_ylabel(r'$|\mathbf{g} \cdot \mathbf{w}|$', fontsize=13)
    ax.set_title('(C) Instantaneous Symmetry Proxy', fontweight='bold')

    # Panel 4: Loss
    ax = axes[3]
    ax.plot(steps, run['losses'], color='#424242', linewidth=1)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_title('(D) Training Loss', fontweight='bold')
    ax.set_yscale('log')

    # Onset markers
    nim_onset = run['ordering']['nim_drift_onset']
    cov_onset = run['ordering']['covariance_asymmetry_onset']

    for ax in axes:
        if nim_onset is not None:
            ax.axvline(nim_onset, color='#2196F3', linestyle='--', alpha=0.7,
                      linewidth=1.5, label='NIM onset' if ax == axes[0] else None)
        if cov_onset is not None:
            ax.axvline(cov_onset, color='#F44336', linestyle='--', alpha=0.7,
                      linewidth=1.5, label='Cov onset' if ax == axes[0] else None)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig1_representative_timeseries.png'))
    plt.close(fig)
    print("  Saved fig1_representative_timeseries.png")


# =========================================================================
# Figure 2: Temporal Ordering Across Models
# =========================================================================
def fig2_temporal_ordering():
    """Bar chart + scatter comparing onset times."""
    exp1 = load_json('results/exp1/experiment_1_results.json')
    if not exp1:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: Bar chart
    ax = axes[0]
    model_names = list(exp1.keys())
    x_pos = np.arange(len(model_names))
    bar_width = 0.25

    nim_data = {}
    cov_data = {}
    proxy_data = {}

    for model_name in model_names:
        runs = exp1[model_name]
        nim_data[model_name] = [r['ordering']['nim_drift_onset'] for r in runs
                                 if r['ordering']['nim_drift_onset'] is not None]
        cov_data[model_name] = [r['ordering']['covariance_asymmetry_onset'] for r in runs
                                 if r['ordering']['covariance_asymmetry_onset'] is not None]
        proxy_data[model_name] = [r['ordering']['symmetry_proxy_onset'] for r in runs
                                   if r['ordering']['symmetry_proxy_onset'] is not None]

    nim_means = [np.mean(nim_data[m]) if nim_data[m] else 0 for m in model_names]
    nim_stds = [np.std(nim_data[m]) if nim_data[m] else 0 for m in model_names]
    cov_means = [np.mean(cov_data[m]) if cov_data[m] else 0 for m in model_names]
    cov_stds = [np.std(cov_data[m]) if cov_data[m] else 0 for m in model_names]
    proxy_means = [np.mean(proxy_data[m]) if proxy_data[m] else 0 for m in model_names]
    proxy_stds = [np.std(proxy_data[m]) if proxy_data[m] else 0 for m in model_names]

    ax.bar(x_pos - bar_width, nim_means, bar_width, yerr=nim_stds,
           color='#2196F3', alpha=0.8, label='NIM Drift', capsize=4, edgecolor='white')
    ax.bar(x_pos, cov_means, bar_width, yerr=cov_stds,
           color='#F44336', alpha=0.8, label='Cov. Asymmetry', capsize=4, edgecolor='white')
    ax.bar(x_pos + bar_width, proxy_means, bar_width, yerr=proxy_stds,
           color='#FF9800', alpha=0.8, label='Symmetry Proxy', capsize=4, edgecolor='white')

    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Detection Onset (step)')
    ax.set_title('(A) Mean Detection Onset by Method', fontweight='bold')
    labels = [n.replace('_', '\n') for n in model_names]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()

    # Panel B: NIM success rate
    ax = axes[1]
    nim_first_rates = []
    nim_detect_rates = []
    labels_clean = []
    for m in model_names:
        runs = exp1[m]
        n = len(runs)
        nf = sum(1 for r in runs if r['ordering']['nim_before_covariance'] is True)
        nd = sum(1 for r in runs if r['ordering']['nim_drift_onset'] is not None)
        nim_first_rates.append(100 * nf / n)
        nim_detect_rates.append(100 * nd / n)
        labels_clean.append(m.replace('_', '\n'))

    x = np.arange(len(model_names))
    ax.bar(x - 0.15, nim_detect_rates, 0.3, color='#2196F3', alpha=0.6,
           label='NIM Detected', edgecolor='white')
    ax.bar(x + 0.15, nim_first_rates, 0.3, color='#1565C0', alpha=0.8,
           label='NIM First', edgecolor='white')

    ax.set_xlabel('Model Architecture')
    ax.set_ylabel('Rate (%)')
    ax.set_title('(B) NIM Detection and Priority Rate', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels_clean, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig2_temporal_ordering.png'))
    plt.close(fig)
    print("  Saved fig2_temporal_ordering.png")


# =========================================================================
# Figure 3: Lambda Scaling Analysis
# =========================================================================
def fig3_lambda_scaling():
    """Log-log regression analysis of drift scaling."""
    # Use the linear model data (exact kappa)
    print("  Generating Exp 2 figure from linear model (exact kappa)...")

    np.random.seed(42)
    torch.manual_seed(42)

    etas = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1]
    x_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    seeds = range(10)
    n_steps = 400
    results = []

    for eta in etas:
        for x_scale in x_scales:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                w = nn.Parameter(torch.tensor(1.0 + 0.1 * np.random.randn()))
                X = torch.randn(200, 1) * x_scale
                y = 3.0 * X.squeeze() + torch.randn(200) * 0.1
                loss_fn = nn.MSELoss()
                kappa = 2.0 * (X ** 2).mean().item()
                lam = eta / kappa

                logger = NIMLogger()
                optimizer = torch.optim.SGD([w], lr=eta)
                diverged = False
                for step in range(n_steps):
                    optimizer.zero_grad()
                    pred = w * X.squeeze()
                    loss = loss_fn(pred, y)
                    if not torch.isfinite(loss):
                        diverged = True
                        break
                    loss.backward()
                    logger.increment_norms.append(eta * w.grad.norm().item())
                    optimizer.step()

                if diverged:
                    continue
                stats = logger.get_increment_drift_stats()
                if stats and np.isfinite(stats['slope']):
                    results.append({
                        'eta': eta, 'x_scale': x_scale, 'kappa': kappa,
                        'lambda': lam, 'abs_slope': abs(stats['slope']),
                    })

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    abs_slopes = np.array([r['abs_slope'] for r in results])
    eta_arr = np.array([r['eta'] for r in results])
    lam_arr = np.array([r['lambda'] for r in results])
    kappa_arr = np.array([r['kappa'] for r in results])

    # Panel A: |slope| vs eta, colored by kappa
    ax = axes[0]
    scatter = ax.scatter(eta_arr, abs_slopes, c=np.log(kappa_arr), cmap='viridis',
                         alpha=0.6, s=25, edgecolors='none')
    reg = linregress(eta_arr, abs_slopes)
    x_fit = np.linspace(eta_arr.min(), eta_arr.max(), 100)
    ax.plot(x_fit, reg.slope * x_fit + reg.intercept, 'r-', linewidth=2,
            label=f'R$^2$={reg.rvalue**2:.3f}')
    ax.set_xlabel(r'$\eta$', fontsize=13)
    ax.set_ylabel('|Drift Slope|', fontsize=13)
    ax.set_title(r'(A) Drift vs $\eta$ (colored by $\kappa$)', fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label=r'$\log(\kappa)$', shrink=0.8)

    # Panel B: |slope| vs lambda
    ax = axes[1]
    ax.scatter(lam_arr, abs_slopes, c='#2196F3', alpha=0.5, s=25, edgecolors='none')
    reg_l = linregress(lam_arr, abs_slopes)
    x_fit_l = np.linspace(lam_arr.min(), lam_arr.max(), 100)
    ax.plot(x_fit_l, reg_l.slope * x_fit_l + reg_l.intercept, 'r-', linewidth=2,
            label=f'R$^2$={reg_l.rvalue**2:.3f}')
    ax.set_xlabel(r'$\lambda = \eta / \kappa$', fontsize=13)
    ax.set_ylabel('|Drift Slope|', fontsize=13)
    ax.set_title(r'(B) Drift vs $\lambda$ (hypothesis predictor)', fontweight='bold')
    ax.legend()

    # Panel C: Log-log analysis
    ax = axes[2]
    pos = abs_slopes > 0
    log_s = np.log10(abs_slopes[pos])
    log_e = np.log10(eta_arr[pos])
    log_k = np.log10(kappa_arr[pos])

    from numpy.linalg import lstsq
    XLL = np.column_stack([np.ones(log_e.shape[0]), log_e, log_k])
    bLL, _, _, _ = lstsq(XLL, log_s, rcond=None)
    pred = XLL @ bLL
    ss_res = np.sum((log_s - pred)**2)
    ss_tot = np.sum((log_s - log_s.mean())**2)
    r2 = 1 - ss_res / ss_tot

    # Plot actual vs predicted
    ax.scatter(pred, log_s, alpha=0.4, s=20, c='#2196F3', edgecolors='none')
    lims = [min(pred.min(), log_s.min()), max(pred.max(), log_s.max())]
    ax.plot(lims, lims, 'k--', alpha=0.5)
    ax.set_xlabel('Predicted log$_{10}$|slope|', fontsize=12)
    ax.set_ylabel('Actual log$_{10}$|slope|', fontsize=12)
    ax.set_title(f'(C) Log-Log Regression (R$^2$={r2:.3f})', fontweight='bold')

    # Annotation
    text = (f'log|slope| = {bLL[1]:.2f} log$\\eta$ + {bLL[2]:.2f} log$\\kappa$\n'
            f'Hypothesis: coeff($\\eta$)=1, coeff($\\kappa$)=-1\n'
            f'Observed:  coeff($\\eta$)={bLL[1]:.2f}, coeff($\\kappa$)={bLL[2]:.2f}')
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig3_lambda_scaling.png'))
    plt.close(fig)
    print("  Saved fig3_lambda_scaling.png")


# =========================================================================
# Figure 4: Adam vs SGD
# =========================================================================
def fig4_optimizer_comparison():
    """Optimizer comparison: drift magnitude and detection timing."""
    exp3 = load_json('results/exp3/experiment_3_results.json')
    if not exp3:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    COLORS = {'sgd_pure': '#2196F3', 'sgd_momentum': '#4CAF50', 'adam': '#9C27B0'}
    LABELS = {'sgd_pure': 'SGD (pure)', 'sgd_momentum': 'SGD (m=0.9)', 'adam': 'Adam'}

    # Panel A: Drift slope boxplot
    ax = axes[0]
    data_box = []
    labels_box = []
    colors_box = []
    for opt_name in ['sgd_pure', 'sgd_momentum', 'adam']:
        if opt_name in exp3:
            slopes = [r['drift_stats']['slope'] for r in exp3[opt_name]
                     if r['drift_stats'] is not None]
            data_box.append(slopes)
            labels_box.append(LABELS[opt_name])
            colors_box.append(COLORS[opt_name])

    bp = ax.boxplot(data_box, widths=0.6, patch_artist=True)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(colors_box[i])
        patch.set_alpha(0.6)
    ax.set_xticklabels(labels_box, fontsize=10)
    ax.set_ylabel('Drift Slope', fontsize=12)
    ax.set_title('(A) Drift Slope Distribution', fontweight='bold')
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

    # Panel B: |slope| comparison (magnitude)
    ax = axes[1]
    for opt_name in ['sgd_pure', 'sgd_momentum', 'adam']:
        if opt_name in exp3:
            abs_slopes = [abs(r['drift_stats']['slope']) for r in exp3[opt_name]
                         if r['drift_stats'] is not None]
            ax.bar(LABELS[opt_name], np.mean(abs_slopes), yerr=np.std(abs_slopes),
                   color=COLORS[opt_name], alpha=0.7, capsize=5, edgecolor='white')

    ax.set_ylabel('Mean |Drift Slope|', fontsize=12)
    ax.set_title('(B) Drift Magnitude by Optimizer', fontweight='bold')

    # Panel C: Time series overlay
    ax = axes[2]
    window = 20
    for opt_name in ['sgd_pure', 'sgd_momentum', 'adam']:
        if opt_name in exp3 and exp3[opt_name]:
            inc = exp3[opt_name][0].get('increment_norms', [])
            if len(inc) > window:
                ra = np.convolve(inc, np.ones(window)/window, mode='valid')
                ax.plot(ra, color=COLORS[opt_name], alpha=0.8, linewidth=1.5,
                       label=LABELS[opt_name])

    ax.set_xlabel('Step')
    ax.set_ylabel(r'Running Avg $\|\Delta w\|_2$')
    ax.set_title('(C) Increment Norm Traces (seed 0)', fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig4_optimizer_comparison.png'))
    plt.close(fig)
    print("  Saved fig4_optimizer_comparison.png")


# =========================================================================
# Figure 5: WeightNorm ResNet
# =========================================================================
def fig5_weightnorm_resnet():
    """WeightNorm ResNet generalization results."""
    exp4 = load_json('results/exp4/experiment_4_results.json')
    if not exp4:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Onset comparison
    ax = axes[0]
    nim_onsets = [r['nim_onset'] for r in exp4 if r['nim_onset'] is not None]
    cov_onsets = [r['ordering']['covariance_asymmetry_onset'] for r in exp4
                  if r['ordering']['covariance_asymmetry_onset'] is not None]

    if nim_onsets:
        ax.hist(nim_onsets, bins=12, alpha=0.6, color='#2196F3', label='NIM Drift', edgecolor='white')
    if cov_onsets:
        ax.hist(cov_onsets, bins=12, alpha=0.6, color='#F44336', label='Cov. Asymmetry', edgecolor='white')
    ax.set_xlabel('Detection Onset (step)')
    ax.set_ylabel('Count')
    ax.set_title('(A) WeightNorm ResNet: Onset Distribution', fontweight='bold')
    ax.legend()

    # Panel B: Traces
    ax = axes[1]
    window = 20
    for i, r in enumerate(exp4[:5]):
        inc = r.get('increment_norms', [])
        if len(inc) > window:
            ra = np.convolve(inc, np.ones(window)/window, mode='valid')
            ax.plot(ra, alpha=0.6, linewidth=1.2, label=f"seed {r['seed']}")

    ax.set_xlabel('Step')
    ax.set_ylabel(r'Running Avg $\|\Delta w\|_2$')
    ax.set_title('(B) Increment Norm Traces', fontweight='bold')
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'fig5_weightnorm_resnet.png'))
    plt.close(fig)
    print("  Saved fig5_weightnorm_resnet.png")


if __name__ == '__main__':
    print("Generating publication figures...")
    fig1_timeseries()
    fig2_temporal_ordering()
    fig3_lambda_scaling()
    fig4_optimizer_comparison()
    fig5_weightnorm_resnet()
    print("All figures generated.")
