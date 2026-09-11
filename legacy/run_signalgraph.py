# run_signalgraph.py
import os, sys
from pathlib import Path

# Make "src" importable when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dagster import materialize, fs_io_manager
from dags.signalgraph_assets import sg_market_daily

if __name__ == "__main__":
    os.environ.setdefault("SF_DATA_ROOT", "./_data")

    # Runs the asset immediately with a local filesystem IO manager
    result = materialize(
        [sg_market_daily],
        resources={"io_manager": fs_io_manager},
        run_config={},  # nothing to configure yet
    )
    print("success:", result.success)
