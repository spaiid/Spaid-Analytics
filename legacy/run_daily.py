# run_daily.py
import sys
from pathlib import Path
SRC_DIR = (Path(__file__).resolve().parent / "src"); sys.path.insert(0, str(SRC_DIR))

from dags.daily_assets import defs
job = defs.get_job_def("daily_job")
job.execute_in_process(run_config={})
