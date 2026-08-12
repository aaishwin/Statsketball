"""
Archetype Data Store
====================
Read-only store for the player clustering pipeline outputs:

- ``players_with_archetypes.csv`` — one row per player with UMAP
  coordinates and HDBSCAN cluster label.
- ``cluster_profiles.json`` — per-cluster feature z-score profiles
  produced by the labeling step.

The store is loaded lazily on first access and cached for the process
lifetime (the files only change when the clustering pipeline re-runs,
which requires an API restart anyway). Paths resolve relative to the
workspace root and can be overridden with the ``PLAYER_OUTPUT_DIR``
environment variable.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

logger = logging.getLogger(__name__)

# backend/app/api/archetype_store.py -> workspace root is 3 parents up from app/
_WORKSPACE_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

_CSV_NAME: Final[str] = "players_with_archetypes.csv"
_PROFILES_NAME: Final[str] = "cluster_profiles.json"

#: Curated display names per cluster id, derived from the dominant
#: feature z-scores in cluster_profiles.json (see labeling step).
#: Falls back to the raw profile name for any id not listed here.
_CLUSTER_DISPLAY_NAMES: Final[dict[int, str]] = {
    -1: "Unclassified",
    0: "Offensive Hubs",
    1: "Floor Spacers",
    2: "Low-Usage Role Players",
    3: "Volume Scorers",
    4: "Wing Finishers",
    5: "Defensive Anchors",
    6: "Floor Generals",
    7: "Rim Runners",
    8: "Traditional Centers",
    9: "Glass Cleaners",
    10: "Low-Efficiency Scorers",
    11: "Franchise Anchors",
}

_NOISE_DESCRIPTION: Final[str] = (
    "Players HDBSCAN could not confidently assign to a dense cluster — "
    "stylistic outliers and hybrid profiles."
)


@dataclass(slots=True, frozen=True)
class PlayerPoint:
    """A single player's position in archetype space."""

    entity_id: str
    entity_name: str
    cluster_id: int
    umap_x: float
    umap_y: float
    position: str
    hof: bool
    debut_season: int
    final_season: int
    total_seasons: int


@dataclass(slots=True, frozen=True)
class ClusterProfile:
    """Summary of one HDBSCAN cluster."""

    cluster_id: int
    name: str
    size: int
    description: str
    key_traits: tuple[str, ...]
    example_players: tuple[str, ...]


@dataclass(slots=True)
class ArchetypeStore:
    """Loaded archetype dataset: players + cluster profiles."""

    players: dict[str, PlayerPoint] = field(default_factory=dict)
    clusters: dict[int, ClusterProfile] = field(default_factory=dict)

    @property
    def loaded(self) -> bool:
        return bool(self.players)


_store: ArchetypeStore | None = None
_lock: threading.Lock = threading.Lock()


def _output_dir() -> Path:
    override: str | None = os.environ.get("PLAYER_OUTPUT_DIR")
    if override:
        return Path(override)
    return _WORKSPACE_ROOT / "output_players"


def _parse_bool(raw: str) -> bool:
    return raw.strip().lower() in {"true", "1", "yes"}


def _load_players(csv_path: Path) -> dict[str, PlayerPoint]:
    players: dict[str, PlayerPoint] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                point = PlayerPoint(
                    entity_id=row["player_id"],
                    entity_name=row["player"],
                    cluster_id=int(row["archetype_label"]),
                    umap_x=float(row["umap_x"]),
                    umap_y=float(row["umap_y"]),
                    position=row.get("primary_pos") or "—",
                    hof=_parse_bool(row.get("hof", "")),
                    debut_season=int(float(row["debut_season"])),
                    final_season=int(float(row["final_season"])),
                    total_seasons=int(float(row["total_seasons"])),
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed archetype row %r: %s", row.get("player_id"), exc)
                continue
            players[point.entity_id] = point
    return players


def _traits_from_profile(profile: dict[str, object]) -> tuple[str, ...]:
    """Extract human-readable trait labels from a cluster profile entry."""
    traits: list[str] = []
    top_features = profile.get("top_features")
    if isinstance(top_features, list):
        for entry in top_features:
            if isinstance(entry, dict):
                label = entry.get("label")
                if isinstance(label, str) and label and "_" not in label:
                    traits.append(label)
    # De-duplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for t in traits:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return tuple(unique[:4])


def _example_players(players: dict[str, PlayerPoint], cluster_id: int) -> tuple[str, ...]:
    """Pick representative players: Hall of Famers first, then longest careers."""
    members: list[PlayerPoint] = [p for p in players.values() if p.cluster_id == cluster_id]
    members.sort(key=lambda p: (not p.hof, -p.total_seasons, p.entity_name))
    return tuple(p.entity_name for p in members[:3])


def _load_clusters(
    profiles_path: Path, players: dict[str, PlayerPoint]
) -> dict[int, ClusterProfile]:
    raw_text: str = profiles_path.read_text(encoding="utf-8")
    raw: object = json.loads(raw_text)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected dict in {profiles_path}, got {type(raw).__name__}")

    clusters: dict[int, ClusterProfile] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        try:
            cluster_id = int(key)
        except ValueError:
            continue
        raw_name = value.get("name")
        fallback_name: str = raw_name if isinstance(raw_name, str) else f"Cluster {cluster_id}"
        raw_size = value.get("size")
        size: int = raw_size if isinstance(raw_size, int) else 0
        traits = _traits_from_profile(value)
        clusters[cluster_id] = ClusterProfile(
            cluster_id=cluster_id,
            name=_CLUSTER_DISPLAY_NAMES.get(cluster_id, fallback_name),
            size=size,
            description=" · ".join(traits) if traits else fallback_name,
            key_traits=traits,
            example_players=_example_players(players, cluster_id),
        )

    # Synthesize the HDBSCAN noise cluster (-1) — present in the CSV but
    # absent from cluster_profiles.json by design.
    noise_count: int = sum(1 for p in players.values() if p.cluster_id == -1)
    if noise_count and -1 not in clusters:
        clusters[-1] = ClusterProfile(
            cluster_id=-1,
            name=_CLUSTER_DISPLAY_NAMES[-1],
            size=noise_count,
            description=_NOISE_DESCRIPTION,
            key_traits=("Outlier", "Hybrid Profile"),
            example_players=_example_players(players, -1),
        )
    return clusters


def get_store() -> ArchetypeStore:
    """
    Return the process-wide archetype store, loading it on first call.

    Raises:
        FileNotFoundError: if the clustering output files do not exist.
        ValueError: if the files are structurally invalid.
    """
    global _store
    if _store is not None:
        return _store
    with _lock:
        if _store is not None:
            return _store
        out_dir: Path = _output_dir()
        csv_path: Path = out_dir / _CSV_NAME
        profiles_path: Path = out_dir / _PROFILES_NAME
        if not csv_path.exists() or not profiles_path.exists():
            raise FileNotFoundError(
                f"Archetype outputs not found in {out_dir}. "
                "Run run_player_clustering.py or set PLAYER_OUTPUT_DIR."
            )
        players: dict[str, PlayerPoint] = _load_players(csv_path)
        clusters: dict[int, ClusterProfile] = _load_clusters(profiles_path, players)
        logger.info(
            "Archetype store loaded: %d players, %d clusters from %s",
            len(players),
            len(clusters),
            out_dir,
        )
        _store = ArchetypeStore(players=players, clusters=clusters)
        return _store
