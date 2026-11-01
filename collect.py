"""Independent data collection entrypoint (no model or GPU dependencies)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from pl_mmb_collection.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
