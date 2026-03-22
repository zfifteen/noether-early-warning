from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from early_warning_research.benchmark import build_canonical_benchmark, build_fixed_point_negative_pair, run_canonical_benchmark


@pytest.fixture(scope="module")
def smoke_benchmark(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output_root = tmp_path_factory.mktemp("canonical_benchmark")
    return run_canonical_benchmark(output_root=output_root, smoke=True, quiet=True)


def test_fixed_point_negative_generator_is_exact_and_quiet() -> None:
    suite = build_canonical_benchmark(smoke=True)["stochastic_fixed_point_negative"].suite
    student, data, initial_loss = build_fixed_point_negative_pair(
        suite,
        seed=0,
        learning_rate=0.01,
        input_scale=0.75,
    )

    for left, right in student.pairs:
        assert torch.allclose(student.input_layer.weight[left], student.input_layer.weight[right])
        if student.hidden_layer is not None:
            assert torch.allclose(student.hidden_layer.weight[left], student.hidden_layer.weight[right])
        assert torch.allclose(student.output_layer.weight[0, left], student.output_layer.weight[0, right])

    assert initial_loss < 1e-20

    student.zero_grad(set_to_none=True)
    loss = torch.mean((student(data["x_train"]) - data["y_train"]) ** 2)
    loss.backward()
    grad_norm = torch.linalg.vector_norm(
        torch.cat([param.grad.detach().flatten() for param in student.parameters() if param.grad is not None])
    ).item()
    assert grad_norm < 1e-16


def test_smoke_benchmark_writes_expected_artifacts(smoke_benchmark: dict[str, object]) -> None:
    artifact_files = smoke_benchmark["artifact_files"]
    for key in (
        "benchmark_summary_json",
        "suite_scorecard_csv",
        "detector_scorecard_csv",
        "short_window_runs_csv",
        "resolved_runs_csv",
        "resolution_curve_csv",
        "lead_distributions_png",
        "censor_resolution_png",
    ):
        assert Path(artifact_files[key]).exists()


def test_smoke_benchmark_runs_all_six_suites(smoke_benchmark: dict[str, object]) -> None:
    suite_names = {row["suite_name"] for row in smoke_benchmark["suite_scorecards"]}
    assert suite_names == {
        "full_batch_positive",
        "full_batch_instant_break",
        "full_batch_fixed_point_negative",
        "stochastic_positive",
        "stochastic_instant_break",
        "stochastic_fixed_point_negative",
    }


def test_smoke_benchmark_resolves_positive_followups(smoke_benchmark: dict[str, object]) -> None:
    resolved_path = Path(smoke_benchmark["artifact_files"]["resolved_runs_csv"])
    rows = list(csv.DictReader(resolved_path.open()))
    positive_rows = [row for row in rows if row["suite_role"] == "positive"]
    assert any(row["reran_to_long_cap"] == "True" for row in positive_rows)
    assert any(row["resolved_status"] == "supportive" for row in positive_rows)


def test_smoke_benchmark_fixed_point_negatives_stay_clean(smoke_benchmark: dict[str, object]) -> None:
    detector_rows = smoke_benchmark["detector_scorecards"]
    negative_rows = [row for row in detector_rows if row["suite_role"] == "fixed_point_negative"]
    assert negative_rows
    for row in negative_rows:
        assert row["drift_false_positive_rate"] <= 0.1
        assert row["symmetry_false_positive_rate"] == 0.0


def test_smoke_benchmark_names_a_stochastic_detector(smoke_benchmark: dict[str, object]) -> None:
    recommended = smoke_benchmark["recommended_stochastic_direct_detector"]
    assert recommended in {"covariance_mismatch", "mean_gradient_mismatch", "activation_stat_mismatch"}
