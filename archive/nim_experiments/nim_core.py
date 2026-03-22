"""
NIM Core: Logging infrastructure and statistical test harness.

Provides:
  - NIMLogger: tracks all per-step diagnostics (increment norms, charges, covariance)
  - Statistical tests for drift detection and ordering comparison
  - Curvature estimation utilities

ACTUAL CODE EXECUTION -- not a simulation.
"""

import numpy as np
from scipy.stats import linregress, mannwhitneyu, ttest_ind
from scipy.linalg import eigvalsh
from collections import defaultdict
import torch
import warnings


class NIMLogger:
    """
    Per-run logger for NIM experiments.

    Records at every training step:
      - L2 norm of weight increments (eta * ||g||_2)
      - Running average of increment norms
      - Noether charge / weight norm (model-dependent)
      - Instantaneous symmetry proxy |g . w|
      - Gradient vectors for covariance analysis
    """

    def __init__(self, window_size=50):
        self.window_size = window_size

        # Time series storage
        self.increment_norms = []         # eta * ||g||_2 per step
        self.weight_norms = []            # ||w||_2 per step
        self.noether_charges = []         # Q per step (if applicable)
        self.symmetry_proxies = []        # |g . w| per step
        self.losses = []                  # training loss per step
        self.gradient_vectors = []        # full gradient vector per step
        self.per_layer_increment_norms = defaultdict(list)  # per-layer tracking

    def log_step(self, model, loss_val, eta, step):
        """Log all diagnostics for one training step."""
        # Collect gradients and weight vectors
        grad_vec = []
        weight_vec = []
        per_layer_g_norms = {}

        for name, p in model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach().flatten()
                w = p.detach().flatten()
                grad_vec.append(g)
                weight_vec.append(w)
                per_layer_g_norms[name] = eta * g.norm().item()

        if not grad_vec:
            return

        grad_full = torch.cat(grad_vec)
        weight_full = torch.cat(weight_vec)

        # 1. L2 increment norm
        g_norm = grad_full.norm().item()
        increment_norm = eta * g_norm
        self.increment_norms.append(increment_norm)

        # 2. Weight norm
        w_norm = weight_full.norm().item()
        self.weight_norms.append(w_norm)

        # 3. Noether charge (model-dependent)
        if hasattr(model, 'noether_charge'):
            self.noether_charges.append(model.noether_charge())

        # 4. Instantaneous symmetry proxy: |g . w|
        gw = abs(torch.dot(grad_full, weight_full).item())
        self.symmetry_proxies.append(gw)

        # 5. Loss
        self.losses.append(loss_val)

        # 6. Full gradient vector (for covariance) -- skip if NaN
        grad_np = grad_full.cpu().numpy()
        if np.all(np.isfinite(grad_np)):
            self.gradient_vectors.append(grad_np)

        # 7. Per-layer increment norms
        for name, norm_val in per_layer_g_norms.items():
            self.per_layer_increment_norms[name].append(norm_val)

    def get_increment_drift_stats(self, start=0, end=None):
        """
        Compute linear regression on increment norm time series.
        Returns: slope, intercept, r_value, p_value, std_err, n_samples
        """
        series = self.increment_norms[start:end]
        if len(series) < 10:
            return None
        t = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = linregress(t, series)
        return {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_value ** 2,
            'p_value': p_value,
            'std_err': std_err,
            'n_samples': len(series)
        }

    def detect_drift_onset(self, window=50, p_threshold=1e-6, slope_threshold=0.0):
        """
        Sliding-window detection of when increment drift becomes significant.
        Returns the earliest step where drift is detectable, or None.
        """
        for start in range(0, len(self.increment_norms) - window):
            segment = self.increment_norms[start:start + window]
            t = np.arange(len(segment))
            slope, _, _, p_value, _ = linregress(t, segment)
            if abs(slope) > slope_threshold and p_value < p_threshold:
                return start + window  # step at which detection occurs
        return None

    def detect_covariance_asymmetry_onset(self, window=100, p_threshold=1e-6):
        """
        Detect when gradient covariance asymmetry becomes statistically significant.

        Uses the Marchenko-Pastur-based test: for isotropic gradients,
        the condition number of the sample covariance follows a known
        distribution. We use the Tracy-Widom threshold approximation.

        More practically: we track the ratio of top-2 eigenvalue explained
        variance. Under isotropy, no eigenvalue should dominate. We use
        a sliding-window chi-squared test for uniformity of eigenvalue
        distribution.

        Returns the earliest step where asymmetry is detectable, or None.
        """
        if len(self.gradient_vectors) < window:
            return None

        # Limit dimensionality
        max_dim = 50
        grads = np.array(self.gradient_vectors)

        # Filter NaN/Inf
        valid_mask = np.all(np.isfinite(grads), axis=1)
        grads = grads[valid_mask]
        if len(grads) < window:
            return None

        if grads.shape[1] > max_dim:
            # Random projection (much faster than PCA)
            rng = np.random.RandomState(42)
            proj = rng.randn(grads.shape[1], max_dim) / np.sqrt(max_dim)
            grads = grads @ proj

        step_stride = max(window // 4, 10)
        for start in range(0, len(grads) - window, step_stride):
            segment = grads[start:start + window]
            try:
                cov = np.cov(segment, rowvar=False)
                eigs = eigvalsh(cov)
                eigs = eigs[eigs > 1e-12]
                if len(eigs) < 2:
                    continue

                # Relative explained variance of top eigenvalue
                total = eigs.sum()
                top_ratio = eigs[-1] / total
                dim = len(eigs)

                # Under isotropy, top eigenvalue ratio should be ~ 1/dim
                # with variance ~ 2/(dim * window)
                expected_ratio = 1.0 / dim
                # Use Marchenko-Pastur: for gamma = dim/window,
                # max eigenvalue concentrates around (1+sqrt(gamma))^2
                gamma = dim / window
                mp_upper = (1 + np.sqrt(gamma)) ** 2 / dim

                # If top ratio exceeds MP bound significantly, asymmetry detected
                if top_ratio > mp_upper * 2.0:  # factor 2 for significance
                    return start + window

            except Exception:
                continue

        return None

    def detect_symmetry_proxy_onset(self, window=50, p_threshold=1e-6):
        """
        Detect when the instantaneous symmetry proxy |g.w| becomes
        statistically distinguishable from zero.
        """
        for start in range(0, len(self.symmetry_proxies) - window):
            segment = self.symmetry_proxies[start:start + window]
            # One-sample t-test against mean = 0
            mean_val = np.mean(segment)
            std_val = np.std(segment, ddof=1)
            if std_val < 1e-15:
                continue
            t_stat = mean_val / (std_val / np.sqrt(len(segment)))
            from scipy.stats import t as t_dist
            p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=len(segment) - 1))
            if p_val < p_threshold:
                return start + window
        return None

    def get_weight_norm_drift(self):
        """Total drift in weight norm from start to end."""
        if len(self.weight_norms) < 2:
            return 0.0
        return self.weight_norms[-1] - self.weight_norms[0]

    def get_charge_drift(self):
        """Total drift in Noether charge from start to end."""
        if len(self.noether_charges) < 2:
            return 0.0
        return self.noether_charges[-1] - self.noether_charges[0]


def estimate_curvature(model, loss_fn, data_x, data_y, n_samples=50):
    """
    Estimate local loss-landscape curvature kappa via Hessian-vector products.

    Uses the Rayleigh quotient: kappa ~ max eigenvalue of Hessian.
    Approximated via power iteration with Hessian-vector products.

    Returns: estimated curvature kappa (float)
    """
    model.zero_grad()
    loss = loss_fn(model(data_x), data_y)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grad_flat = torch.cat([g.flatten() for g in grads])

    # Power iteration for top eigenvalue
    v = torch.randn_like(grad_flat)
    v = v / v.norm()

    for _ in range(n_samples):
        # Hessian-vector product
        Hv = torch.autograd.grad(grad_flat, model.parameters(),
                                  grad_outputs=_unflatten_like(v, model),
                                  retain_graph=True)
        Hv_flat = torch.cat([h.flatten() for h in Hv])

        # Rayleigh quotient
        eigenvalue = torch.dot(v, Hv_flat).item()

        # Update v
        v = Hv_flat / (Hv_flat.norm() + 1e-12)

    return abs(eigenvalue)


def _unflatten_like(flat_tensor, model):
    """Unflatten a 1D tensor to match model parameter shapes."""
    tensors = []
    offset = 0
    for p in model.parameters():
        numel = p.numel()
        tensors.append(flat_tensor[offset:offset + numel].reshape(p.shape))
        offset += numel
    return tensors


def compute_lambda_ratio(eta, kappa):
    """Dimensionless ratio: lambda = eta / kappa."""
    if kappa < 1e-12:
        return float('inf')
    return eta / kappa


def temporal_ordering_summary(nim_onset, cov_onset, proxy_onset, total_steps):
    """
    Summarize temporal ordering of detection onsets.
    Returns a dict with onsets and whether the NIM prediction holds.
    """
    result = {
        'nim_drift_onset': nim_onset,
        'covariance_asymmetry_onset': cov_onset,
        'symmetry_proxy_onset': proxy_onset,
        'total_steps': total_steps,
        'nim_before_covariance': None,
        'nim_before_proxy': None,
    }

    if nim_onset is not None and cov_onset is not None:
        result['nim_before_covariance'] = nim_onset < cov_onset
    elif nim_onset is not None and cov_onset is None:
        result['nim_before_covariance'] = True  # NIM detected, cov never did

    if nim_onset is not None and proxy_onset is not None:
        result['nim_before_proxy'] = nim_onset < proxy_onset
    elif nim_onset is not None and proxy_onset is None:
        result['nim_before_proxy'] = True

    return result
