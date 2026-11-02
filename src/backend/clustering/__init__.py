"""
NBA Player Playing-Style Clustering Pipeline
===========================================
Groups NBA players by playing style using historical player data.
Produces interpretable clusters and a visual embedding space for exploration.

Exports:
    - constants: shared era buckets, feature blocks, label mappings, colors
    - config: PipelineConfig and sub-config dataclasses
    - validation: Pydantic validation models and functions
    - schema_definitions: CSV schema definitions for pre-flight checks
"""

__version__ = "1.1.0"
