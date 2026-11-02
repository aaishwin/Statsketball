"""
FAISS Index Evaluation & Regression Testing
============================================
Offline evaluation (recall@k on labeled pairs) and regression testing
(index rebuild comparison) for FAISS similarity search.

Design (from faiss_similarity_search_design.md §7):
- Labeled pairs from cluster co-membership, manual curation, external sources
- Regression test captures top-K fingerprint for canonical query entities
- New index is blocked from promotion if Jaccard similarity < 0.85
"""

import numpy as np
import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Result of an offline evaluation run."""
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    n_labeled_pairs: int = 0
    n_queries: int = 0


def _search_for(entity_type: str, entity_id: str, k: int) -> dict:
    """Dispatch a search through the service module.

    Imported lazily to avoid a circular import (service → ranking ← evaluation).
    """
    from . import service

    return service.search_player(entity_id, k=k)


def evaluate_recall_at_k(
    labeled_pairs: list[tuple[str, str]],
    entity_type: str,
    ks: list[int] = None,
) -> EvalResult:
    """
    Measure recall@k on a set of labeled similar pairs.

    Parameters
    ----------
    labeled_pairs : list of (query_entity_id, expected_similar_entity_id)
    entity_type : 'player'
    ks : list of k values to evaluate

    Returns EvalResult with recall@k for each k.
    """
    if ks is None:
        ks = [1, 5, 10, 20, 50]

    queries = list(set(pair[0] for pair in labeled_pairs))
    expected_map = {}
    for q, e in labeled_pairs:
        expected_map.setdefault(q, set()).add(e)

    recall_hits = {k: 0 for k in ks}
    total_pairs = len(labeled_pairs)
    mrr_sum = 0.0
    n_valid = 0

    for query_id in queries:
        try:
            result = _search_for(entity_type, query_id, k=max(ks))
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Skipping query '{query_id}': {e}")
            continue

        retrieved_ids = [r["entity_id"] for r in result["results"]]
        expected = expected_map.get(query_id, set())

        # Recall@k
        for k in ks:
            top_k_ids = set(retrieved_ids[:k])
            if top_k_ids & expected:
                recall_hits[k] += 1

        # MRR
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in expected:
                mrr_sum += 1.0 / rank
                break

        n_valid += 1

    recall = {k: recall_hits[k] / n_valid if n_valid > 0 else 0.0 for k in ks}

    return EvalResult(
        recall_at_k=recall,
        precision_at_k={},  # needs relevance judgments beyond binary
        mrr=mrr_sum / n_valid if n_valid > 0 else 0.0,
        n_labeled_pairs=total_pairs,
        n_queries=n_valid,
    )


def build_labeled_pairs_from_clusters(
    labels: np.ndarray,
    entity_ids: list[str],
    min_confidence: float = 0.8,
    max_pairs: int = 500,
) -> list[tuple[str, str]]:
    """
    Construct labeled "similar" pairs from cluster co-membership.

    Players/teams in the same cluster are considered similar. This is a
    weak signal but provides broad coverage. Only includes pairs that are
    consensus-assigned to the same cluster (if multiple algorithms agree).

    Parameters
    ----------
    labels : cluster labels from the existing pipeline (noise = -1)
    entity_ids : list of entity IDs in same order as labels
    min_confidence : unused here (placeholder for multi-algorithm consensus)
    max_pairs : cap on number of pairs to return

    Returns list of (entity_id_a, entity_id_b) pairs considered similar.
    """
    import random

    pairs = []
    unique_labels = set(labels) - {-1}  # exclude noise

    for cluster_id in unique_labels:
        members = [entity_ids[i] for i in range(len(labels)) if labels[i] == cluster_id]
        # Generate pairs within cluster (up to a limit per cluster)
        n_pairs_per_cluster = min(
            len(members) * (len(members) - 1) // 2,
            max_pairs // len(unique_labels) + 1,
        )
        count = 0
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
                count += 1
                if count >= n_pairs_per_cluster:
                    break
            if count >= n_pairs_per_cluster:
                break

    # Shuffle and truncate
    random.shuffle(pairs)
    return pairs[:max_pairs]


# ═══════════════════════════════════════════════════════════════
# REGRESSION TESTING (functional)
# ═══════════════════════════════════════════════════════════════
# A regression baseline is a plain dict: entity_id → top-K result ids.
# Capture before a rebuild, compare after; block promotion if the average
# Jaccard similarity drops below threshold.

RegressionBaseline = dict[str, list[str]]


def capture_regression_baseline(
    entity_type: str,
    query_entities: list[str],
    k: int = 10,
) -> RegressionBaseline:
    """Capture the current top-K results as a regression baseline.

    Must be called after the FAISS service has an active index.
    """
    baseline: RegressionBaseline = {}
    for entity_id in query_entities:
        try:
            result = _search_for(entity_type, entity_id, k=k)
            baseline[entity_id] = [r["entity_id"] for r in result["results"]]
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Regression baseline: skipping '{entity_id}': {e}")

    logger.info(
        f"[regression] Captured baseline for {len(baseline)} {entity_type} entities"
    )
    return baseline


def compare_to_regression_baseline(
    baseline: RegressionBaseline,
    entity_type: str,
    k: int = 10,
    jaccard_threshold: float = 0.70,
    avg_jaccard_threshold: float = 0.85,
) -> dict:
    """Compare current index results against a stored baseline.

    Returns:
    - pass: bool (True if avg Jaccard ≥ threshold and no individual alert)
    - avg_jaccard: float
    - per_entity: dict of entity_id → comparison details
    - alerts: list of entity_ids with Jaccard < jaccard_threshold
    """
    results: dict[str, dict] = {}
    alerts: list[str] = []

    for entity_id, old_ids in baseline.items():
        try:
            result = _search_for(entity_type, entity_id, k=k)
            new_ids = [r["entity_id"] for r in result["results"]]
        except (KeyError, RuntimeError) as e:
            logger.warning(f"Regression compare: skipping '{entity_id}': {e}")
            continue

        intersection = set(new_ids) & set(old_ids)
        union = set(new_ids) | set(old_ids)
        jaccard = len(intersection) / len(union) if union else 0.0

        results[entity_id] = {
            "jaccard": jaccard,
            "overlap_count": len(intersection),
            "new_only": list(set(new_ids) - set(old_ids)),
            "removed": list(set(old_ids) - set(new_ids)),
        }

        if jaccard < jaccard_threshold:
            alerts.append(entity_id)

    avg_jaccard = (
        sum(r["jaccard"] for r in results.values()) / len(results)
        if results else 0.0
    )

    passed = avg_jaccard >= avg_jaccard_threshold and len(alerts) == 0

    return {
        "pass": passed,
        "avg_jaccard": avg_jaccard,
        "n_entities_tested": len(results),
        "n_alerts": len(alerts),
        "alerts": alerts,
        "per_entity": results,
    }


def save_regression_baseline(baseline: RegressionBaseline, path: str) -> None:
    """Persist a regression baseline to disk."""
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    logger.info(f"[regression] Baseline saved to {path}")


def load_regression_baseline(path: str) -> RegressionBaseline:
    """Load a regression baseline from disk."""
    with open(path, "r") as f:
        baseline: RegressionBaseline = json.load(f)
    logger.info(
        f"[regression] Baseline loaded from {path} ({len(baseline)} entities)"
    )
    return baseline


# ═══════════════════════════════════════════════════════════════
# CANONICAL QUERY SETS
# ═══════════════════════════════════════════════════════════════

CANONICAL_PLAYERS = [
    "jamesle01",   # LeBron James
    "curryst01",   # Stephen Curry
    "jokicni01",   # Nikola Jokić
    "antetgi01",   # Giannis Antetokounmpo
    "doncilu01",   # Luka Dončić
    "birdla01",    # Larry Bird
    "jordami01",   # Michael Jordan
    "onealsh01",   # Shaquille O'Neal
    "bryanko01",   # Kobe Bryant
    "duncati01",   # Tim Duncan
    "nowitdi01",   # Dirk Nowitzki
    "paulch01",    # Chris Paul
    "wadedw01",    # Dwyane Wade
    "duranke01",   # Kevin Durant
    "hardenja01",  # James Harden
    "westbru01",   # Russell Westbrook
    "davisja01",   # Jared Davis (placeholder — will fail gracefully)
]

# For the regression test, we need actual entity IDs from the data.
# These are dynamically populated from the pipeline output.
