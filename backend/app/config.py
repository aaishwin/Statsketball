"""
Centralized Configuration for NBA Clustering Pipelines
======================================================
Dataclass-based config with sensible defaults for all tunable parameters.
Replaces scattered hardcoded values across feature_engineering.py,
player_feature_engineering.py, clustering.py, dimensionality.py,
and both run scripts.

Supports hashing for cache invalidation.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import hashlib
import json


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""

    # Era adjustment
    min_seasons: int = 5  # player pipeline only
    block_weights: Optional[dict[str, float]] = None  # e.g. {"offense": 2.0}

    # RobustScaler
    scaler_quantile_range: tuple[float, float] = (5.0, 95.0)


@dataclass
class DimensionalityConfig:
    """Dimensionality reduction parameters."""

    pca_variance: float = 0.90
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"
    random_state: int = 42


@dataclass
class ClusteringConfig:
    """Clustering algorithm parameters."""

    n_clusters: int = 8  # player default = 12
    use_hdbscan: bool = True
    hdbscan_min_cluster_size: int = 25
    hdbscan_min_samples: int = 3
    ensemble_min_consensus: float = 0.5
    random_state: int = 42
    # HDBSCAN is excluded from ensemble if <2 clusters or >50% noise
    hdbscan_min_clusters_for_ensemble: int = 2
    hdbscan_max_noise_for_ensemble: float = 0.50


@dataclass
class LabelingConfig:
    """Cluster/archetype labeling parameters."""

    top_n_features: int = 4


@dataclass
class EvaluationConfig:
    """Evaluation parameters."""

    bootstrap_samples: int = 10
    random_state: int = 42


@dataclass
class SimilarityConfig:
    """Similarity computation parameters."""

    # If n_samples > this, switch from full cosine_similarity to sparse KNN
    large_n_threshold: int = 10_000
    sparse_n_neighbors: int = 50


@dataclass
class OutputConfig:
    """Output and caching parameters."""

    output_dir: str = "./output"
    enable_cache: bool = True
    cache_dir_name: str = ".cache"


@dataclass
class DataQualityConfig:
    """Data quality monitoring thresholds."""

    # Warn if any column exceeds this missing rate (before imputation)
    missing_rate_warn: float = 0.30
    # Flag outliers beyond this Z-score (after scaling)
    outlier_z_threshold: float = 4.0


@dataclass
class PipelineConfig:
    """
    Master configuration for a clustering pipeline run.

    Usage:
        config = PipelineConfig()  # all defaults
        config = PipelineConfig(n_clusters=10, pca_variance=0.95)
        cache_key = config.cache_key()
    """

    # ── Sub-configs ──
    features: FeatureConfig = field(default_factory=FeatureConfig)
    dimensionality: DimensionalityConfig = field(default_factory=DimensionalityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    labeling: LabelingConfig = field(default_factory=LabelingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)

    # ── Convenience shortcuts (synced from sub-configs) ──
    @property
    def n_clusters(self) -> int:
        return self.clustering.n_clusters

    @n_clusters.setter
    def n_clusters(self, value: int):
        self.clustering.n_clusters = value

    @property
    def pca_variance(self) -> float:
        return self.dimensionality.pca_variance

    @pca_variance.setter
    def pca_variance(self, value: float):
        self.dimensionality.pca_variance = value

    @property
    def random_state(self) -> int:
        return self.clustering.random_state

    @random_state.setter
    def random_state(self, value: int):
        self.clustering.random_state = value
        self.dimensionality.random_state = value
        self.evaluation.random_state = value

    @property
    def output_dir(self) -> str:
        return self.output.output_dir

    @output_dir.setter
    def output_dir(self, value: str):
        self.output.output_dir = value

    @property
    def min_seasons(self) -> int:
        return self.features.min_seasons

    @min_seasons.setter
    def min_seasons(self, value: int):
        self.features.min_seasons = value

    @property
    def min_cluster_size(self) -> int:
        return self.clustering.hdbscan_min_cluster_size

    @min_cluster_size.setter
    def min_cluster_size(self, value: int):
        self.clustering.hdbscan_min_cluster_size = value

    # ── Hashing for cache invalidation ──

    def to_dict(self) -> dict:
        """Convert to a deterministic, JSON-serializable dict."""
        d = {}
        for field_name in self.__dataclass_fields__:
            val = getattr(self, field_name)
            if hasattr(val, "__dataclass_fields__"):
                d[field_name] = asdict(val)
            else:
                d[field_name] = val
        return d

    def cache_key(self) -> str:
        """Generate a hash key for caching based on config values."""
        d = self.to_dict()
        # Sort keys for determinism
        serialized = json.dumps(d, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode(), usedforsecurity=False).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════
# Pre-built configurations
# ═══════════════════════════════════════════════════════════════

def get_player_default_config() -> PipelineConfig:
    """Default configuration for player archetype clustering."""
    config = PipelineConfig()
    config.clustering.n_clusters = 12
    config.features.min_seasons = 5
    return config
