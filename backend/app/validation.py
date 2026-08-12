"""
Validation Layer for NBA Clustering Pipelines
=============================================
Pydantic v2 models for validating inputs, parameters, data integrity,
and output paths BEFORE expensive computation begins.

Design:
- Schema/param validation: BLOCKING (raises ValidationError, exits early)
- Data quality checks: NON-BLOCKING (warns, saves report, continues)
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, Any, TYPE_CHECKING
import os

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


# ═══════════════════════════════════════════════════════════════
# PRE-FLIGHT EXCEPTIONS
# ═══════════════════════════════════════════════════════════════

class CSVSchemaError(Exception):
    """Raised when one or more CSV files fail schema validation."""
    pass


# ═══════════════════════════════════════════════════════════════
# PARAMETER VALIDATION
# ═══════════════════════════════════════════════════════════════

class PipelineParams(BaseModel):
    """
    Validates all pipeline parameters before execution.

    Usage:
        params = PipelineParams(n_clusters=8, pca_variance=0.90)
        # or catch validation errors:
        try:
            params = PipelineParams(n_clusters=0)  # raises ValidationError
        except ValidationError as e:
            print(e)
    """

    n_clusters: int = Field(default=8, ge=2, le=50,
                            description="Target number of clusters (2–50)")
    pca_variance: float = Field(default=0.90, gt=0.50, le=0.99,
                                description="PCA variance to retain (0.50–0.99)")
    min_seasons: int = Field(default=5, ge=1, le=30,
                             description="Minimum seasons for player filtering (1–30)")
    min_cluster_size: int = Field(default=25, ge=5,
                                  description="HDBSCAN minimum cluster size (≥5)")
    use_hdbscan: bool = Field(default=True)
    random_state: int = Field(default=42, ge=0)

    @field_validator("pca_variance")
    @classmethod
    def pca_must_be_reasonable(cls, v: float) -> float:
        if v < 0.50:
            raise ValueError("PCA variance must be ≥ 0.50 to retain meaningful structure")
        if v > 0.99:
            raise ValueError("PCA variance > 0.99 retains near-all dimensions; use 0.90–0.95")
        return v

    @field_validator("n_clusters")
    @classmethod
    def clusters_must_be_reasonable(cls, v: int) -> int:
        if v < 2:
            raise ValueError("At least 2 clusters required")
        if v > 50:
            raise ValueError("More than 50 clusters is likely overfitting")
        return v


# ═══════════════════════════════════════════════════════════════
# CSV SCHEMA VALIDATION
# ═══════════════════════════════════════════════════════════════

class ColumnSpec(BaseModel):
    """Specification for a single CSV column."""
    name: str
    dtype: str = "any"  # "int", "float", "str", "bool", "any"
    nullable: bool = True


class CSVSchema(BaseModel):
    """Schema definition for a single CSV file."""
    file_name: str
    required_columns: list[str] = Field(default_factory=list)
    min_rows: int = 0
    max_rows: Optional[int] = None
    description: str = ""


class CSVValidationResult(BaseModel):
    """Result of validating a single CSV against its schema."""
    file_name: str
    exists: bool = True
    missing_columns: list[str] = Field(default_factory=list)
    extra_columns: list[str] = Field(default_factory=list)
    row_count: int = 0
    row_count_ok: bool = True
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.exists and len(self.missing_columns) == 0 and self.row_count_ok


class CSVValidationReport(BaseModel):
    """Aggregate report for all CSV validations."""
    results: list[CSVValidationResult] = Field(default_factory=list)
    all_valid: bool = True

    def add_result(self, result: CSVValidationResult):
        self.results.append(result)
        if not result.is_valid:
            self.all_valid = False


# ═══════════════════════════════════════════════════════════════
# MERGE INTEGRITY VALIDATION
# ═══════════════════════════════════════════════════════════════

class MergeIntegrityResult(BaseModel):
    """Result of a merge integrity check."""
    step_name: str
    row_count: int = 0
    column_count: int = 0
    duplicate_rows: int = 0
    has_suffix_columns: bool = False  # _x / _y columns from overlapping merges
    warnings: list[str] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.row_count > 0 and not self.has_suffix_columns and self.duplicate_rows == 0


# ═══════════════════════════════════════════════════════════════
# DATA QUALITY VALIDATION
# ═══════════════════════════════════════════════════════════════

class ColumnQuality(BaseModel):
    """Per-column data quality metrics."""
    column: str
    missing_count: int = 0
    missing_rate: float = 0.0
    outlier_count: int = 0  # Z > threshold after scaling
    flagged: bool = False  # exceeds warning threshold


class DataQualityReport(BaseModel):
    """Full data quality report."""
    dataset: str = ""  # "player"
    n_samples: int = 0
    n_features: int = 0
    total_missing_before_impute: int = 0
    columns_flagged: list[ColumnQuality] = Field(default_factory=list)
    overall_ok: bool = True

    def flag_column(self, cq: ColumnQuality):
        if cq.flagged:
            self.columns_flagged.append(cq)
            self.overall_ok = False


# ═══════════════════════════════════════════════════════════════
# OUTPUT PATH VALIDATION
# ═══════════════════════════════════════════════════════════════

class OutputPathResult(BaseModel):
    """Result of output directory validation."""
    path: str
    exists: bool = False
    is_writable: bool = False
    was_created: bool = False
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.is_writable


# ═══════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def validate_csv_exists(file_path: str, schema: CSVSchema) -> CSVValidationResult:
    """
    Validate a single CSV file against its schema.

    Args:
        file_path: Full path to the CSV file.
        schema: Expected schema for this CSV.

    Returns:
        CSVValidationResult with details.
    """
    result = CSVValidationResult(file_name=schema.file_name)

    if not os.path.exists(file_path):
        result.exists = False
        result.warnings.append(f"File not found: {file_path}")
        return result

    import pandas as pd
    try:
        df = pd.read_csv(file_path, nrows=0)  # headers only for speed
    except Exception as e:
        result.warnings.append(f"Failed to read CSV: {e}")
        result.exists = False
        return result

    actual_columns = set(df.columns)
    required = set(schema.required_columns)
    result.missing_columns = sorted(required - actual_columns)
    result.extra_columns = sorted(actual_columns - required)

    # Quick row count (re-read with just column 0)
    try:
        result.row_count = len(pd.read_csv(file_path, usecols=[0]))
    except Exception:
        result.row_count = -1

    if schema.min_rows > 0 and result.row_count < schema.min_rows:
        result.row_count_ok = False
        result.warnings.append(
            f"Row count {result.row_count} < min expected {schema.min_rows}"
        )

    return result


def validate_merge_result(
    df: "pd.DataFrame",
    step_name: str,
    expected_min_rows: int = 1,
) -> MergeIntegrityResult:
    """
    Validate a DataFrame after a merge step.

    Args:
        df: The merged DataFrame to check.
        step_name: Human-readable step description.
        expected_min_rows: Minimum expected row count.

    Returns:
        MergeIntegrityResult with details.
    """
    import pandas as pd

    result = MergeIntegrityResult(step_name=step_name)
    result.row_count = len(df)
    result.column_count = len(df.columns)

    # Check for _x / _y suffix columns (merge overlap artifacts)
    suffix_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if suffix_cols:
        result.has_suffix_columns = True
        result.warnings.append(
            f"Found {len(suffix_cols)} suffix columns from overlapping merges: "
            f"{suffix_cols[:10]}{'...' if len(suffix_cols) > 10 else ''}"
        )

    # Check for duplicates
    result.duplicate_rows = int(df.duplicated().sum())
    if result.duplicate_rows > 0:
        result.warnings.append(f"Found {result.duplicate_rows} duplicate rows")

    # Check row count
    if result.row_count == 0:
        result.warnings.append(f"Merge produced 0 rows — check join keys")
    elif result.row_count < expected_min_rows:
        result.warnings.append(
            f"Row count {result.row_count} below expected minimum {expected_min_rows}"
        )

    if not result.warnings:
        print(f"[validate] ✓ {step_name}: {result.row_count} rows, "
              f"{result.column_count} cols OK")
    else:
        for w in result.warnings:
            print(f"[validate] ⚠ {step_name}: {w}")

    return result


def validate_output_dir(output_dir: str) -> OutputPathResult:
    """
    Validate that the output directory exists and is writable.
    Creates it if missing.

    Args:
        output_dir: Path to the output directory.

    Returns:
        OutputPathResult with details.
    """
    result = OutputPathResult(path=output_dir)

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            result.was_created = True
            result.exists = True
        except PermissionError as e:
            result.error = f"Permission denied creating directory: {e}"
            return result
        except Exception as e:
            result.error = f"Failed to create directory: {e}"
            return result
    else:
        result.exists = True

    # Check writability
    test_file = os.path.join(output_dir, ".write_test")
    try:
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        result.is_writable = True
    except Exception as e:
        result.is_writable = False
        result.error = f"Directory not writable: {e}"

    if result.was_created:
        print(f"[validate] Created output directory: {output_dir}")
    elif result.is_valid:
        pass  # all good, no need to log
    else:
        print(f"[validate] ❌ Output directory issue: {result.error}")

    return result


def generate_data_quality_report(
    X_raw: "np.ndarray",
    feature_names: list[str],
    dataset_label: str = "",
    missing_rate_warn: float = 0.30,
    outlier_z_threshold: float = 4.0,
) -> DataQualityReport:
    """
    Generate a data quality report from a raw feature matrix.

    Args:
        X_raw: Raw feature matrix (n_samples, n_features) BEFORE imputation.
        feature_names: Column names for the features.
        dataset_label: "player".
        missing_rate_warn: Missing rate threshold for flagging.
        outlier_z_threshold: Z-score threshold for outlier flagging.

    Returns:
        DataQualityReport with per-column metrics.
    """
    import numpy as np

    report = DataQualityReport(
        dataset=dataset_label,
        n_samples=X_raw.shape[0],
        n_features=X_raw.shape[1],
    )

    # Missing value analysis
    for j, col_name in enumerate(feature_names):
        col = X_raw[:, j]
        n_missing = int(np.isnan(col).sum())
        missing_rate = n_missing / len(col) if len(col) > 0 else 0.0
        report.total_missing_before_impute += n_missing

        cq = ColumnQuality(
            column=col_name,
            missing_count=n_missing,
            missing_rate=round(missing_rate, 4),
        )

        if missing_rate > missing_rate_warn:
            cq.flagged = True

        # Outlier detection (on non-NaN values)
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            mean, std = valid.mean(), valid.std()
            if std > 0:
                z_scores = np.abs((valid - mean) / std)
                cq.outlier_count = int(np.sum(z_scores > outlier_z_threshold))

        report.flag_column(cq)

    if report.columns_flagged:
        print(f"[quality] ⚠ {len(report.columns_flagged)} columns flagged:")
        for cq in report.columns_flagged[:5]:
            print(f"  - {cq.column}: {cq.missing_rate:.1%} missing, "
                  f"{cq.outlier_count} outliers")
        if len(report.columns_flagged) > 5:
            print(f"  ... and {len(report.columns_flagged) - 5} more")
    else:
        print(f"[quality] ✓ All {report.n_features} columns within thresholds")

    return report


# ═══════════════════════════════════════════════════════════════
# PRE-FLIGHT ORCHESTRATION
# ═══════════════════════════════════════════════════════════════

def resolve_and_validate(
    data_dir: Optional[str],
    default_data_dir: str,
    params: "PipelineParams",
    schemas: list["CSVSchema"],
) -> str:
    """
    Resolve the data directory, confirm parameters are valid, and validate
    all CSV schemas against the resolved data directory.

    Consolidates the pre-flight sequence shared by the clustering run scripts
    (data-dir resolution, parameter validation, CSV schema validation) into a
    single call that raises typed exceptions instead of calling sys.exit().

    Args:
        data_dir: Explicit data directory from CLI args, or None to use default.
        default_data_dir: Fallback data directory when data_dir is None.
        params: Already-constructed PipelineParams (validated by Pydantic).
        schemas: List of CSVSchema definitions to validate against.

    Returns:
        The resolved data directory path.

    Raises:
        FileNotFoundError: If the resolved data directory does not exist.
        ValidationError: If params fail Pydantic validation (re-raised).
        CSVSchemaError: If one or more CSVs fail schema validation.
    """
    # ── Resolve data directory ──
    resolved = data_dir if data_dir is not None else default_data_dir
    if not os.path.isdir(resolved):
        raise FileNotFoundError(f"Data directory not found: {resolved}")

    # ── Validate CSV schemas ──
    # Imported here to avoid a circular import (schema_definitions imports
    # validation types at module load).
    from .schema_definitions import validate_all_csvs

    print("\n🔍 Validating CSV schemas...")
    csv_report = validate_all_csvs(resolved, schemas)
    if not csv_report.all_valid:
        raise CSVSchemaError("CSV schema validation failed — see details above.")

    return resolved
