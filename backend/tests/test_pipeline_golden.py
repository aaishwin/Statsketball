"""Golden-output equivalence tests for the clustering pipelines.

Both pipelines are deterministic given fixed ``random_state`` (PCA, UMAP with
seed, KMeans, agglomerative, HDBSCAN are all seeded/deterministic), so a
refactor that preserves behavior must reproduce byte-identical cluster
assignments and evaluation metrics.

Workflow:
- First run (pre-refactor): golden snapshots are captured into ``tests/golden/``.
- Later runs (post-refactor): outputs are compared against the snapshots.

These are slow (full UMAP + clustering); run explicitly:

    pytest tests/test_pipeline_golden.py -m golden -v
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.conftest import DATA_DIR, GOLDEN_DIR

pytestmark = pytest.mark.golden

_FLOAT_TOL = 1e-9


def _snapshot_from_outputs(output_dir: Path, export_csv: str, label_col: str, id_cols: list[str]) -> dict[str, Any]:
    """Distill a pipeline output directory into a comparable snapshot dict."""
    df = pd.read_csv(output_dir / export_csv)
    assignments = {
        "|".join(str(row[c]) for c in id_cols): int(row[label_col])
        for _, row in df.iterrows()
    }
    with (output_dir / "evaluation.json").open() as f:
        evaluation: dict[str, Any] = json.load(f)
    with (output_dir / "cluster_profiles.json").open() as f:
        profiles: dict[str, Any] = json.load(f)
    cluster_summary = {
        cid: {"name": p["name"], "size": p["size"]} for cid, p in profiles.items()
    }
    return {
        "n_rows": len(df),
        "assignments": assignments,
        "evaluation": evaluation,
        "cluster_summary": cluster_summary,
    }


def _assert_snapshots_equal(actual: dict[str, Any], golden: dict[str, Any]) -> None:
    assert actual["n_rows"] == golden["n_rows"], "Row count changed"
    assert actual["assignments"] == golden["assignments"], "Cluster assignments diverged"
    assert actual["cluster_summary"] == golden["cluster_summary"], "Cluster names/sizes diverged"
    _assert_json_close(actual["evaluation"], golden["evaluation"], path="evaluation")


def _assert_json_close(a: object, b: object, path: str) -> None:
    """Recursively compare JSON values; floats within tolerance."""
    if isinstance(a, float) or isinstance(b, float):
        af, bf = float(a), float(b)  # type: ignore[arg-type]
        if math.isnan(af) and math.isnan(bf):
            return
        assert math.isclose(af, bf, rel_tol=_FLOAT_TOL, abs_tol=_FLOAT_TOL), (
            f"{path}: {a!r} != {b!r}"
        )
    elif isinstance(a, dict) and isinstance(b, dict):
        assert set(a.keys()) == set(b.keys()), f"{path}: keys differ {set(a) ^ set(b)}"
        for k in a:
            _assert_json_close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"{path}: list length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_json_close(x, y, f"{path}[{i}]")
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def _run_and_check(
    *,
    kind: str,
    output_dir: Path,
    export_csv: str,
    label_col: str,
    id_cols: list[str],
) -> None:
    golden_path = GOLDEN_DIR / f"{kind}_pipeline_snapshot.json"
    actual = _snapshot_from_outputs(output_dir, export_csv, label_col, id_cols)

    if not golden_path.exists():
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        with golden_path.open("w") as f:
            json.dump(actual, f, indent=1, sort_keys=True)
        pytest.skip(f"Golden snapshot captured at {golden_path} — re-run to compare.")

    with golden_path.open() as f:
        golden: dict[str, Any] = json.load(f)
    _assert_snapshots_equal(actual, golden)


@pytest.fixture(scope="module")
def data_dir() -> str:
    assert DATA_DIR.exists(), f"Data directory missing: {DATA_DIR}"
    return str(DATA_DIR)


def test_player_pipeline_golden(data_dir: str, tmp_path_factory: pytest.TempPathFactory) -> None:
    from app.clustering.player_pipeline import (
        run_player_pipeline,
        get_similar_players,
        get_player_profile,
        compare_players,
    )

    output_dir = tmp_path_factory.mktemp("player_output")
    result = run_player_pipeline(
        data_dir=data_dir,
        output_dir=str(output_dir),
        enable_cache=False,
        n_clusters=12,
        show_plots=False,
    )

    # Query-function smoke checks (frozen behavior)
    similar = get_similar_players(result, "LeBron James", top_k=5)
    assert len(similar) == 5
    profile = get_player_profile(result, "LeBron James")
    assert profile["player"] == "LeBron James"
    assert len(profile["top_features"]) == 5
    comparison = compare_players(result, "Stephen Curry", "Ray Allen")
    assert "cosine_similarity" in comparison
    assert "same_archetype" in comparison

    assert result.labels is not None
    _run_and_check(
        kind="player",
        output_dir=output_dir,
        export_csv="players_with_archetypes.csv",
        label_col="archetype_label",
        id_cols=["player", "debut_season"],
    )
