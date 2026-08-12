#!/usr/bin/env python3
"""
NBA Similarity Search API — Run Script
=======================================
Starts the FastAPI server with the FAISS similarity search endpoints.

Usage:
    python run_api.py
    python run_api.py --host 0.0.0.0 --port 8000
    python run_api.py --rebuild  # Force full rebuild on startup

Environment variables:
    NBA_DATA_DIR      — Path to CSV data directory
    FAISS_OUTPUT_DIR  — Where to save/load FAISS indices

Security:
    The default bind address is 127.0.0.1 (localhost only). Use
    --host 0.0.0.0 only when you intend to expose the API on the network,
    and ensure ADMIN_API_KEY and CORS_ALLOWED_ORIGINS are configured.
"""

import argparse
import os
import sys

# Resolve OpenMP conflict between faiss-cpu and sklearn/hdbscan
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="NBA Similarity Search API Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_api.py
  python run_api.py --port 8080
  python run_api.py --rebuild
  python run_api.py --data-dir ./custom_data --host 0.0.0.0
        """,
    )
    # Default to localhost only — 0.0.0.0 must be an explicit opt-in so an
    # auth-less API is not accidentally exposed to the network.
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Bind address (default: 127.0.0.1; use 0.0.0.0 to expose on all interfaces)",
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev only)")
    parser.add_argument("--rebuild", action="store_true", help="Force full index rebuild on startup")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to CSV data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Path to FAISS output directory")

    args = parser.parse_args()

    # Set environment variables from args
    if args.data_dir:
        os.environ["NBA_DATA_DIR"] = args.data_dir
    if args.output_dir:
        os.environ["FAISS_OUTPUT_DIR"] = args.output_dir

    import uvicorn

    print("=" * 60)
    print("  NBA SIMILARITY SEARCH API")
    print("=" * 60)
    print(f"  Host:     {args.host}:{args.port}")
    print(f"  Docs:     http://{args.host}:{args.port}/docs")
    print(f"  Health:   http://{args.host}:{args.port}/api/v1/health")
    print(f"  Rebuild:  {'Yes — force rebuild' if args.rebuild else 'No — use cached indices'}")
    print("=" * 60)

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
