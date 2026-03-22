from __future__ import annotations

import math

import torch


EPS = 1e-12


def _paired_row_init(weight: torch.Tensor, gen: torch.Generator, init_noise: float, fan_in: int) -> None:
    hidden_dim = weight.shape[0]
    if hidden_dim % 2 != 0:
        raise ValueError("paired initialization expects an even hidden dimension")
    for left in range(0, hidden_dim, 2):
        right = left + 1
        base = torch.randn(weight.shape[1], generator=gen, dtype=weight.dtype) / math.sqrt(fan_in)
        asym = torch.randn(weight.shape[1], generator=gen, dtype=weight.dtype)
        asym = asym / (torch.linalg.vector_norm(asym) + EPS)
        weight[left].copy_(base + init_noise * asym)
        weight[right].copy_(base - init_noise * asym)


class PairedMLP(torch.nn.Module):
    """Paired-hidden-unit MLP with explicit pairwise permutation symmetry."""

    def __init__(
        self,
        input_dim: int,
        pair_count: int,
        seed: int,
        init_noise: float,
        hidden_layers: int = 1,
        use_layer_norm: bool = False,
    ) -> None:
        super().__init__()
        if hidden_layers not in (1, 2):
            raise ValueError("hidden_layers must be 1 or 2")

        self.input_dim = input_dim
        self.pair_count = pair_count
        self.hidden_layers = hidden_layers
        self.use_layer_norm = use_layer_norm
        self.hidden_dim = pair_count * 2

        self.input_layer = torch.nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.norm = (
            torch.nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
            if use_layer_norm
            else torch.nn.Identity()
        )
        self.hidden_layer = (
            torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False) if hidden_layers == 2 else None
        )
        self.output_layer = torch.nn.Linear(self.hidden_dim, 1, bias=False)

        self.fc1 = self.input_layer
        self.fc_hidden = self.hidden_layer
        self.fc2 = self.output_layer

        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            _paired_row_init(self.input_layer.weight, gen, init_noise, input_dim)
            if self.hidden_layer is not None:
                _paired_row_init(self.hidden_layer.weight, gen, init_noise, self.hidden_dim)
            for left, right in self.pairs:
                base = torch.randn((), generator=gen, dtype=torch.float64) / math.sqrt(self.hidden_dim)
                asym = torch.randn((), generator=gen, dtype=torch.float64)
                self.output_layer.weight[0, left] = base + init_noise * asym
                self.output_layer.weight[0, right] = base - init_noise * asym

        self.to(dtype=torch.float64)

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return [(2 * idx, 2 * idx + 1) for idx in range(self.pair_count)]

    def hidden_representation(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.input_layer(x))
        hidden = self.norm(hidden)
        if self.hidden_layer is not None:
            hidden = torch.tanh(self.hidden_layer(hidden))
        return hidden

    def monitored_weight_grad(self) -> torch.Tensor:
        layer = self.hidden_layer if self.hidden_layer is not None else self.input_layer
        return layer.weight.grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.hidden_representation(x)).squeeze(-1)


class ToyScaleProduct(torch.nn.Module):
    """Low-dimensional scale-product model for sanity checking."""

    def __init__(self, u_init: float, v_init: float) -> None:
        super().__init__()
        self.u = torch.nn.Parameter(torch.tensor(float(u_init), dtype=torch.float64))
        self.v = torch.nn.Parameter(torch.tensor(float(v_init), dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.u * self.v) * x.squeeze(-1)
