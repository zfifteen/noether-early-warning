from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Literal


@dataclass(frozen=True)
class DetectorConfig:
    drift_window: int = 50
    drift_running_mean_window: int = 10
    drift_effect_floor: float = 0.05
    drift_p_threshold: float = 1e-6
    drift_consecutive: int = 2
    symmetry_baseline_probes: int = 3
    symmetry_z_threshold: float = 2.5
    symmetry_floor: float = 0.02
    symmetry_consecutive: int = 2


@dataclass(frozen=True)
class ProbeConfig:
    every_steps: int = 15
    microbatches: int = 12
    microbatch_size: int = 32


@dataclass(frozen=True)
class SweepConfig:
    seeds: tuple[int, ...]
    learning_rates: tuple[float, ...]
    input_scales: tuple[float, ...]


@dataclass(frozen=True)
class TrainingConfig:
    total_steps: int
    train_size: int
    probe_size: int


@dataclass(frozen=True)
class PairedMLPConfig:
    input_dim: int = 8
    pair_count: int = 2
    init_noise: float = 2e-3
    teacher_hidden: int = 3


@dataclass(frozen=True)
class ToyScaleProductConfig:
    input_dim: int = 1
    init_u: float = 1.0
    init_v: float = 1.0
    init_noise: float = 1e-2
    teacher_slope: float = 3.0


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    description: str
    kind: Literal["paired_mlp", "toy_scale_product"]
    training: TrainingConfig
    sweep: SweepConfig
    probe: ProbeConfig
    detector: DetectorConfig
    model: PairedMLPConfig | ToyScaleProductConfig


def _default_detector() -> DetectorConfig:
    return DetectorConfig()


def _default_probe() -> ProbeConfig:
    return ProbeConfig()


def build_suite_registry() -> dict[str, SuiteConfig]:
    detector = _default_detector()
    probe = _default_probe()
    training = TrainingConfig(total_steps=300, train_size=256, probe_size=256)

    main_model = PairedMLPConfig(input_dim=8, pair_count=2, init_noise=2e-3, teacher_hidden=3)
    control_model = PairedMLPConfig(input_dim=8, pair_count=2, init_noise=1.5e-1, teacher_hidden=3)
    toy_model = ToyScaleProductConfig(input_dim=1, init_u=1.0, init_v=1.0, init_noise=1e-2, teacher_slope=3.0)

    return {
        "main_paired_mlp": SuiteConfig(
            name="main_paired_mlp",
            description="Claim-bearing paired-hidden-unit MLP with near-symmetric initialization.",
            kind="paired_mlp",
            training=training,
            sweep=SweepConfig(
                seeds=(0, 1, 2),
                learning_rates=(0.02, 0.04, 0.08),
                input_scales=(0.75, 1.25, 1.75),
            ),
            probe=probe,
            detector=detector,
            model=main_model,
        ),
        "instant_break_control": SuiteConfig(
            name="instant_break_control",
            description="Matched paired-hidden-unit MLP with intentionally broken symmetry at initialization.",
            kind="paired_mlp",
            training=training,
            sweep=SweepConfig(
                seeds=(0, 1, 2),
                learning_rates=(0.02, 0.04, 0.08),
                input_scales=(0.75, 1.25, 1.75),
            ),
            probe=probe,
            detector=detector,
            model=control_model,
        ),
        "toy_sanity": SuiteConfig(
            name="toy_sanity",
            description="Low-dimensional scale-product sanity check. Included as a sidecar, not primary evidence.",
            kind="toy_scale_product",
            training=TrainingConfig(total_steps=240, train_size=192, probe_size=192),
            sweep=SweepConfig(
                seeds=(0, 1, 2),
                learning_rates=(0.01, 0.02, 0.04),
                input_scales=(0.75, 1.25),
            ),
            probe=ProbeConfig(every_steps=12, microbatches=1, microbatch_size=192),
            detector=replace(detector, symmetry_floor=0.01, drift_effect_floor=0.04),
            model=toy_model,
        ),
    }


def get_suite(name: str, smoke: bool = False) -> SuiteConfig:
    registry = build_suite_registry()
    if name not in registry:
        raise KeyError(f"Unknown suite: {name}")
    suite = registry[name]
    return build_smoke_suite(suite) if smoke else suite


def list_suite_names() -> tuple[str, ...]:
    return tuple(build_suite_registry().keys())


def build_smoke_suite(suite: SuiteConfig) -> SuiteConfig:
    if suite.kind == "paired_mlp":
        return replace(
            suite,
            training=replace(suite.training, total_steps=180, train_size=128, probe_size=128),
            sweep=SweepConfig(
                seeds=(0, 1),
                learning_rates=(0.02, 0.04),
                input_scales=(0.75, 1.25),
            ),
            probe=replace(suite.probe, every_steps=12, microbatches=8, microbatch_size=16),
            detector=replace(
                suite.detector,
                drift_window=30,
                drift_running_mean_window=8,
                symmetry_baseline_probes=2,
            ),
        )
    return replace(
        suite,
        training=replace(suite.training, total_steps=140, train_size=128, probe_size=128),
        sweep=SweepConfig(
            seeds=(0, 1),
            learning_rates=(0.01, 0.02),
            input_scales=(0.75, 1.25),
        ),
        probe=replace(suite.probe, every_steps=10),
        detector=replace(
            suite.detector,
            drift_window=24,
            drift_running_mean_window=6,
            symmetry_baseline_probes=2,
        ),
    )


def suite_to_dict(suite: SuiteConfig) -> dict[str, object]:
    return asdict(suite)
