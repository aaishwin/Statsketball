"""API contract tests — freeze the HTTP interface before/after the functional refactor.

These tests exercise the FastAPI app against the pre-built FAISS artifacts in
``backend/faiss_output/`` (loaded by the app lifespan). They assert response
shapes and key invariants, not exact floating-point scores, so they stay stable
across index rebuilds while still catching any contract regression.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FAISS_OUTPUT_DIR

SEARCH_RESULT_KEYS: frozenset[str] = frozenset(
    {
        "entity_id",
        "entity_name",
        "score",
        "cosine_similarity",
        "rank",
        "metadata",
        "top_contributing_features",
    }
)

SEARCH_RESPONSE_KEYS: frozenset[str] = frozenset(
    {
        "query_id",
        "query_entity",
        "results",
        "total_candidates_searched",
        "filters_applied",
        "timing_ms",
        "index_version",
        "cache_hit",
    }
)


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """App with lifespan run (loads FAISS indices from disk)."""
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def sample_player_id() -> str:
    """A real player entity_id from the built index metadata."""
    meta_path = FAISS_OUTPUT_DIR / "faiss_player_metadata.json"
    with meta_path.open() as f:
        pkg: dict[str, Any] = json.load(f)
    entity_id: str = pkg["metadata"][0]["entity_id"]
    return entity_id


@pytest.fixture(scope="module")
def player_dimension() -> int:
    meta_path = FAISS_OUTPUT_DIR / "faiss_player_metadata.json"
    with meta_path.open() as f:
        pkg: dict[str, Any] = json.load(f)
    dim: int = pkg["dimension"]
    return dim


def _assert_search_response_shape(body: dict[str, Any]) -> None:
    assert SEARCH_RESPONSE_KEYS <= set(body.keys())
    for result in body["results"]:
        assert SEARCH_RESULT_KEYS <= set(result.keys())
        assert isinstance(result["rank"], int)
        assert isinstance(result["score"], float)
        assert isinstance(result["metadata"], dict)
        assert isinstance(result["top_contributing_features"], list)


class TestHealth:
    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["indices"] == {"player": True}
        assert "cache_size" in body


class TestSuggest:
    def test_suggest_lebron(self, client: TestClient) -> None:
        resp = client.get("/api/v1/search/players", params={"q": "lebron"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "lebron"
        assert body["total"] == len(body["suggestions"])
        assert body["total"] >= 1
        first = body["suggestions"][0]
        assert {"entity_id", "entity_name", "metadata"} <= set(first.keys())
        assert "lebron" in first["entity_name"].lower()

    def test_suggest_no_match_is_empty(self, client: TestClient) -> None:
        resp = client.get("/api/v1/search/players", params={"q": "zzzzqqqq"})
        assert resp.status_code == 200
        assert resp.json()["suggestions"] == []

    def test_suggest_validation(self, client: TestClient) -> None:
        resp = client.get("/api/v1/search/players", params={"q": ""})
        assert resp.status_code == 422


class TestPlayerSearch:
    def test_search_by_id(self, client: TestClient, sample_player_id: str) -> None:
        resp = client.get(f"/api/v1/search/player/{sample_player_id}", params={"k": 5})
        assert resp.status_code == 200
        body = resp.json()
        _assert_search_response_shape(body)
        assert body["query_entity"]["entity_id"] == sample_player_id
        assert len(body["results"]) == 5
        # Results are rank-ordered with descending scores
        scores = [r["score"] for r in body["results"]]
        assert scores == sorted(scores, reverse=True)
        assert [r["rank"] for r in body["results"]] == [1, 2, 3, 4, 5]
        # Self never appears in results
        assert all(r["entity_id"] != sample_player_id for r in body["results"])

    def test_search_by_name_resolution(self, client: TestClient) -> None:
        resp = client.get("/api/v1/search/player/LeBron James", params={"k": 3})
        assert resp.status_code == 200
        body = resp.json()
        assert "lebron" in body["query_entity"]["entity_name"].lower()

    def test_search_cache_hit_flag(self, client: TestClient, sample_player_id: str) -> None:
        params = {"k": 7}
        first = client.get(f"/api/v1/search/player/{sample_player_id}", params=params).json()
        second = client.get(f"/api/v1/search/player/{sample_player_id}", params=params).json()
        assert second["cache_hit"] is True
        assert [r["entity_id"] for r in first["results"]] == [
            r["entity_id"] for r in second["results"]
        ]

    def test_search_position_filter(self, client: TestClient, sample_player_id: str) -> None:
        resp = client.get(
            f"/api/v1/search/player/{sample_player_id}",
            params={"k": 5, "position": "C"},
        )
        assert resp.status_code == 200
        for r in resp.json()["results"]:
            assert r["metadata"]["primary_position"] == "C"

    def test_search_unknown_player_404(self, client: TestClient) -> None:
        resp = client.get("/api/v1/search/player/zzzznotaplayer99")
        assert resp.status_code == 404


class TestRawVectorSearch:
    def test_raw_vector_query(self, client: TestClient, player_dimension: int) -> None:
        vector = [0.1] * player_dimension
        resp = client.post(
            "/api/v1/search/query",
            json={"entity_type": "player", "vector": vector, "k": 5, "normalize": True},
        )
        assert resp.status_code == 200
        body = resp.json()
        _assert_search_response_shape(body)
        assert body["query_entity"]["entity_id"] == "raw_vector"
        assert len(body["results"]) == 5

    def test_raw_vector_wrong_dimension_400(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/search/query",
            json={"entity_type": "player", "vector": [0.1, 0.2], "k": 5},
        )
        assert resp.status_code == 400


class TestIndexInfo:
    @pytest.mark.parametrize("entity_type", ["player"])
    def test_index_info(self, client: TestClient, entity_type: str) -> None:
        resp = client.get(f"/api/v1/index/info/{entity_type}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["entity_type"] == entity_type
        assert body["vector_count"] > 0
        assert body["dimension"] > 0
        assert body["memory_bytes"] > 0
        assert body["index_type"] == "IndexHNSWFlat"


class TestRebuildAuth:
    def test_rebuild_requires_admin_key(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/index/rebuild",
            json={"entity_type": "player", "force": True},
        )
        # Fail-closed: 503 when ADMIN_API_KEY unset, 401/403 when key mismatch.
        assert resp.status_code in {401, 403, 503}


class TestArchetypes:
    def test_archetype_data(self, client: TestClient) -> None:
        resp = client.get("/api/v1/archetypes")
        assert resp.status_code == 200
        body = resp.json()
        assert {"clusters", "players", "total_players"} <= set(body.keys())
        assert body["total_players"] == len(body["players"])
        assert body["total_players"] > 0
        cluster = body["clusters"][0]
        assert {
            "cluster_id",
            "cluster_name",
            "size",
            "description",
            "key_traits",
            "example_players",
        } <= set(cluster.keys())
        player = body["players"][0]
        assert {
            "entity_id",
            "entity_name",
            "cluster_id",
            "umap_x",
            "umap_y",
            "position",
            "hof",
        } <= set(player.keys())


class TestPlayerGraph:
    def test_player_graph(self, client: TestClient) -> None:
        resp = client.get("/api/v1/graph/player/LeBron James", params={"k": 6})
        assert resp.status_code == 200
        body = resp.json()
        assert {"center_id", "nodes", "edges"} <= set(body.keys())
        assert body["nodes"][0]["is_center"] is True
        assert body["center_id"] == body["nodes"][0]["entity_id"]
        assert len(body["edges"]) == 6
        for edge in body["edges"]:
            assert edge["source"] == body["center_id"]

    def test_player_graph_unknown_404(self, client: TestClient) -> None:
        resp = client.get("/api/v1/graph/player/zzzznotaplayer99")
        assert resp.status_code == 404


class TestHeadshots:
    def test_headshot_map(self, client: TestClient) -> None:
        resp = client.get("/api/v1/headshots")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == len(body["headshots"])
        assert body["total"] > 0
        # Every URL must be on the NBA CDN allowlist
        for url in list(body["headshots"].values())[:25]:
            assert url.startswith("https://")
