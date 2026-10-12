"""One job of the Modal ADS-B stage run (modal/final_adsb_stage_app.py), importable without modal, and its local dry run.

    python final/_adsb_job.py --frame FRAME --inp S0SHIP [--stream 103] [--serve-days 2026-01-01]
"""
import os
import sys


def _stage21():
    """final/21_adsb_stage.py as a module (its flags() builds the shipped command line)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("stage21", os.path.join(os.environ.get("REPO", "/repo"), "final/21_adsb_stage.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def build_cmd(stream, job, frame, inp, out, days, calib_days=None):
    """The command of one job. calib_days (comma list) replaces the 264-day calibration list: dry runs only."""
    m = _stage21()
    fl = m.flags(stream, days)
    if calib_days:
        i = fl.index("--calib-days")
        fl[i + 1] = calib_days
    return [sys.executable, "-u", "-W", "ignore", "final/_run_as_ranking.py", "--ranking", frame, m.APPLY, "--in", inp, "--out", out, *fl]


def env():
    return dict(os.environ, PYTHONPATH=os.environ.get("REPO", "/repo"), PYTHONIOENCODING="utf-8",
                POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage", OMP_NUM_THREADS="4", POLARS_MAX_THREADS="4")


def dryrun(argv):
    """Run the container entry point's command locally on a sample: the same build_cmd, a short calibration list."""
    import argparse
    import subprocess
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True)
    ap.add_argument("--inp", required=True)
    ap.add_argument("--stream", default="103")
    ap.add_argument("--serve-days", default="2026-01-01")
    ap.add_argument("--calib-days", default="2025-01-06,2025-01-07,2025-01-08,2025-01-09,2025-07-01,2025-07-02")
    ap.add_argument("--out", default="data/final/_t2x/dryrun_adsb.parquet")
    a = ap.parse_args(argv)
    os.environ["REPO"] = os.getcwd()
    cmd = build_cmd(a.stream, "1", a.frame, a.inp, a.out, a.serve_days, a.calib_days)
    print(" ".join(cmd)[:300], "...")
    sys.exit(subprocess.call(cmd, env=env()))




if __name__ == "__main__":
    dryrun(sys.argv[1:])
