"""Local live browser and hosted snapshot export, independent of the scraper writer."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pl_mmb_collection.browser.server import main

if __name__ == "__main__":
    raise SystemExit(main())
