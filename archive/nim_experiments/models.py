"""
Model definitions for NIM (Noether Increment Monitor) experiments.

Each model embodies a known continuous-time symmetry that finite-eta SGD breaks.
Models are kept minimal to isolate the KSB mechanism cleanly.

ACTUAL CODE EXECUTION -- not a simulation.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Model 1: Toy Scalar Product  (exact continuous-time charge Q = u^2 - v^2)
# ---------------------------------------------------------------------------
class ToyScalarProduct(nn.Module):
    """
    f(x) = u * v * x
    Loss = (1/2)(f(x) - y)^2

    Under gradient flow the scale symmetry (u,v) -> (alpha*u, v/alpha)
    yields the exact conserved charge Q = u^2 - v^2.
    Finite eta breaks this conservation via KSB.
    """
    def __init__(self, u_init=1.0, v_init=1.0):
        super().__init__()
        self.u = nn.Parameter(torch.tensor(u_init))
        self.v = nn.Parameter(torch.tensor(v_init))

    def forward(self, x):
        return self.u * self.v * x

    def noether_charge(self):
        """Q = u^2 - v^2 (should be conserved under gradient flow)."""
        return (self.u.item() ** 2) - (self.v.item() ** 2)

    def weight_norm(self):
        return math.sqrt(self.u.item() ** 2 + self.v.item() ** 2)


# ---------------------------------------------------------------------------
# Model 2: Linear Regression (scale symmetry when loss is homogeneous)
# ---------------------------------------------------------------------------
class LinearScaleModel(nn.Module):
    """
    f(x) = w * x
    Scale symmetry: w -> alpha * w leaves the loss gradient direction invariant
    (loss landscape is homogeneous degree 2 in w for MSE).
    """
    def __init__(self, w_init=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w_init))

    def forward(self, x):
        return self.w * x

    def weight_norm(self):
        return abs(self.w.item())


# ---------------------------------------------------------------------------
# Model 3: Two-Layer MLP with LayerNorm (standard KSB testbed)
# ---------------------------------------------------------------------------
class LayerNormMLP(nn.Module):
    """
    Two-layer MLP with LayerNorm between layers.
    LayerNorm induces scale invariance in the pre-norm weights,
    making KSB the dominant dynamics driver.

    Architecture: Linear(in, hidden) -> LayerNorm -> ReLU -> Linear(hidden, out)
    """
    def __init__(self, input_dim=4, hidden_dim=32, output_dim=1):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.ln = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.layer1(x)
        h = self.ln(h)
        h = self.relu(h)
        return self.layer2(h)

    def pre_norm_weight_norm(self):
        """L2 norm of layer1 weights (scale-invariant due to LayerNorm)."""
        return self.layer1.weight.norm().item()


# ---------------------------------------------------------------------------
# Model 4: Scale-Symmetric Product Network
# ---------------------------------------------------------------------------
class ScaleSymmetricProduct(nn.Module):
    """
    Two-layer product: f(x) = W2 @ diag(W1 @ x)
    This has explicit rescale symmetry:
        (W1, W2) -> (alpha * W1, W2 / alpha)
    Conserved charge under gradient flow: ||W1||^2 - ||W2||^2
    """
    def __init__(self, input_dim=4, hidden_dim=8, output_dim=1):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.5)
        self.W2 = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.5)

    def forward(self, x):
        h = self.W1 @ x.unsqueeze(-1)  # (batch, hidden, 1)
        h = h.squeeze(-1)               # (batch, hidden)
        out = self.W2 @ h.unsqueeze(-1)  # (batch, out, 1)
        return out.squeeze(-1)

    def noether_charge(self):
        """Q = ||W1||^2 - ||W2||^2"""
        return (self.W1.norm() ** 2 - self.W2.norm() ** 2).item()

    def weight_norm(self):
        return math.sqrt(self.W1.norm().item() ** 2 + self.W2.norm().item() ** 2)


# ---------------------------------------------------------------------------
# Model 5: ResNet Block with WeightNorm (extension for generalizability)
# ---------------------------------------------------------------------------
class WeightNormResBlock(nn.Module):
    """
    Residual block with WeightNorm.
    WeightNorm decomposes w = g * (v / ||v||), inducing a different
    form of scale symmetry than LayerNorm.
    """
    def __init__(self, dim=16):
        super().__init__()
        self.fc1 = nn.utils.weight_norm(nn.Linear(dim, dim))
        self.relu = nn.ReLU()
        self.fc2 = nn.utils.weight_norm(nn.Linear(dim, dim))

    def forward(self, x):
        residual = x
        h = self.fc1(x)
        h = self.relu(h)
        h = self.fc2(h)
        return h + residual


class WeightNormResNet(nn.Module):
    """
    Simple ResNet with WeightNorm blocks for NIM testing.
    """
    def __init__(self, input_dim=8, hidden_dim=16, output_dim=1, n_blocks=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([WeightNormResBlock(hidden_dim) for _ in range(n_blocks)])
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)

    def weight_norm_stats(self):
        """Return weight norms per block for monitoring."""
        norms = {}
        for i, block in enumerate(self.blocks):
            norms[f"block_{i}_fc1_g"] = block.fc1.weight_g.norm().item()
            norms[f"block_{i}_fc2_g"] = block.fc2.weight_g.norm().item()
        return norms
