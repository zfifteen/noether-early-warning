from __future__ import annotations

import math

import torch


class PairedMLP(torch.nn.Module):
    """Two-layer MLP with explicit pairwise permutation symmetry."""

    def __init__(self, input_dim: int, pair_count: int, seed: int, init_noise: float) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pair_count = pair_count
        self.hidden_dim = pair_count * 2
        self.fc1 = torch.nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.fc2 = torch.nn.Linear(self.hidden_dim, 1, bias=False)

        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for pair_index, (left, right) in enumerate(self.pairs):
                base_in = torch.randn(input_dim, generator=gen, dtype=torch.float64) / math.sqrt(input_dim)
                asym_in = torch.randn(input_dim, generator=gen, dtype=torch.float64)
                asym_in = asym_in / (torch.linalg.vector_norm(asym_in) + 1e-12)

                base_out = torch.randn((), generator=gen, dtype=torch.float64) / math.sqrt(self.hidden_dim)
                asym_out = torch.randn((), generator=gen, dtype=torch.float64)

                self.fc1.weight[left].copy_(base_in + init_noise * asym_in)
                self.fc1.weight[right].copy_(base_in - init_noise * asym_in)
                self.fc2.weight[0, left] = base_out + init_noise * asym_out
                self.fc2.weight[0, right] = base_out - init_noise * asym_out

        self.to(dtype=torch.float64)

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return [(2 * idx, 2 * idx + 1) for idx in range(self.pair_count)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.fc1(x))
        return self.fc2(hidden).squeeze(-1)


class ToyScaleProduct(torch.nn.Module):
    """Low-dimensional scale-product model for sanity checking."""

    def __init__(self, u_init: float, v_init: float) -> None:
        super().__init__()
        self.u = torch.nn.Parameter(torch.tensor(float(u_init), dtype=torch.float64))
        self.v = torch.nn.Parameter(torch.tensor(float(v_init), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.u * self.v) * x.squeeze(-1)
