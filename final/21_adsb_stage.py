"""Stage 21: the ADS-B off-block stage for the chosen months, on top of the base model.

Follows archive/merged_chain.sh as run by archive/build_v103_np.sh, archive/build_v105_taxinm.sh and, for the larger stage,
archive/agent_stagecap_stageserve.sh. For the departures of the chosen months:

  1. Rome pre-rules on the base: day-early rule (archive/apply_rome_dayearly.py), plan gate (archive/apply_rome_plan_gate.py)
     -> <tag>_base.parquet, <tag>_s0ship.parquet; Rome path: the plan gate bagged over 20 seeds at gap cap 10800, weight 0.60
     -> <tag>_gatebag_s0.parquet.
  2. Per stream, N realisations of the ADS-B stage (calibration on the 264 listed days of 2025, serving on the chosen days),
     rounded mean -> <tag>_v<stream>_s1.parquet; and one run on the Rome-path input -> <tag>_v<stream>_gatebag_s1.parquet.
     Streams: 103 (airborne-callsign and no-plan matching, anchors), 105 (103 plus taxi-phase matching at nine airports),
     105sc (105 with the larger floor-stack and stack settings).

Every shipped script runs unchanged through final/_run_as_ranking.py, which points its ranking file at the month frame
(final/_mkrank.py). The 103 stream uses the later copy of the serving script with taxi-phase matching off (the shipped
self-test shows it reproduces the original at those settings). The stage is a randomised fit, so a re-run reproduces a shipped
realisation to its realisation noise, not bit for bit.

    python final/21_adsb_stage.py --months 2,6 [--sample] [--tag J5] [--streams 103,105,105sc] [--realisations 10] [--workers 3]

Memory: about 3 GB per serving process (11 GB for three at a time in the shipped runs); about 1 h 40 min for a stream of ten
realisations. Needs data/adsb/points day files of the chosen days and the base model file (default data/final/J2_v94model.parquet).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, OUT, REPO, TARGET_COL, parse_months  # noqa: E402

PY = sys.executable
RUN = ["final/_run_as_ranking.py"]
APPLY = "archive/agent_stagecap_apply_taxinm.py"
NINE = "EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH"
CALIB_PREDS = ("results/preds_n_feat_1_7.parquet,results/preds_n_feat_2_8.parquet,results/preds_n3_bulk_3_9.parquet,"
               "data/interim/calib2026/preds_oct_f4.parquet,data/interim/calib2026/preds_nov_c.parquet")
SC_FLAGS = ("--stack-trees 600 --stack-leaves 31 --stack-minchild 50 --stack-lr 0.05 --floor-stack-trees 500 "
            "--floor-stack-leaves 31 --floor-stack-minchild 100 --floor-stack-lr 0.05")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def run(cmd, logfile=None):
    t0 = time.time()
    if logfile:
        with open(logfile, "w", encoding="utf-8") as f:
            r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, cwd=REPO)
    if r.returncode:
        raise SystemExit(f"failed ({r.returncode}) after {time.time() - t0:.0f}s: {' '.join(map(str, cmd))[:300]}")


def average(paths, out):
    """Rounded mean of several outputs on the same ids (the avg() of archive/merged_chain.sh)."""
    fr = [pl.read_parquet(p).sort(ID_COL) for p in paths]
    for f in fr[1:]:
        assert (f[ID_COL] == fr[0][ID_COL]).all()
    m = np.vstack([f[TARGET_COL].to_numpy().astype(float) for f in fr])
    fr[0].with_columns(pl.Series(TARGET_COL, np.rint(m.mean(0))).cast(fr[0][TARGET_COL].dtype)).write_parquet(out)


def flags(stream, days):
    cd = Path(REPO / "data/interim/calib_days_264_octnov.txt").read_text().strip()
    f = (f"--osm-fill on --floor-stack-feats quality --stack-feats quality_anch --floor-q zero --spill-tracks on --calib-days {cd} "
         f"--floor-stack-lower 1800 --floor-stack-lower-skip LIRF,LFPG --calib-preds {CALIB_PREDS} "
         f"--floor-stack-anchors on --floor-stack-anch-skip LIRF --air-cs-all on --air-cs-skip LIRF --noplan-callsign on "
         f"--serve-days {days}")
    if stream in ("105", "105sc"):
        f += f" --floor-stack-taxinm-apts {NINE}"
    if stream == "105sc":
        f += " " + SC_FLAGS
    return f.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(REPO / "data/raw/final_ranking.parquet"))
    ap.add_argument("--months", default="2,6")
    ap.add_argument("--tag", default="J5")
    ap.add_argument("--base", default=str(OUT / "J2_v94model.parquet"))
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--streams", default="103,105,105sc")
    ap.add_argument("--realisations", type=int, default=10)
    ap.add_argument("--gate-seeds", type=int, default=20)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--serve-days", default=None, help="comma list; default = every day of the months that has a points file")
    ap.add_argument("--sample", action="store_true", help="2 serving days, 1 realisation, 2 gate seeds")
    ap.add_argument("--prerules-only", action="store_true", help="stop after the Rome pre-rules (step 1)")
    ap.add_argument("--month-frame", default=None, help="an existing month frame (skips final/_mkrank.py)")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    def P(s):
        return str(out / f"{args.tag}_{s}.parquet")

    frame = args.month_frame or P("frame")
    if not args.month_frame and not Path(frame).exists():
        run([PY, "final/_mkrank.py", "--ranking", args.ranking, "--months", args.months, "--out", frame])
    dep = (pl.scan_parquet(frame).filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col(ID_COL), pl.col("ADEP_mvt").alias("apt"),
                   pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"),
                   pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).collect())
    days = args.serve_days.split(",") if args.serve_days else sorted(dep["day"].unique().to_list())
    have = [d for d in days if (REPO / f"data/adsb/points/{d}.parquet").exists()]
    if len(have) < len(days):
        log(f"serving days without a points file: {sorted(set(days) - set(have))}")
    if args.sample:
        have = have[:2]
    dayarg = ",".join(have)
    nreal = 1 if args.sample else args.realisations
    nseed = 2 if args.sample else args.gate_seeds
    log(f"{dep.height:,} departures, serving days {len(have)}: {have[0]}..{have[-1]}; realisations {nreal}")

    base_in = P("base_in")
    b = pl.read_parquet(args.base).join(dep.select(ID_COL), on=ID_COL, how="semi")
    assert b.height == dep.height, f"base has {b.height} of {dep.height} departures"
    b.write_parquet(base_in)

    def rr(script, *a):
        return [PY, *RUN, "--ranking", frame, script, *map(str, a)]

    # 1. pre-rules
    if not Path(P("base")).exists():
        run(rr("archive/apply_rome_dayearly.py", "--in", base_in, "--out", P("base")))
    if not Path(P("s0ship")).exists():
        run(rr("archive/apply_rome_plan_gate.py", "--in", P("base"), "--out", P("s0ship")))
    if not Path(P("gatebag_s0")).exists():
        gs = []
        for sd in range(1, nseed + 1):
            o = P(f"gate_{sd}")
            if not Path(o).exists():
                run(rr("archive/apply_rome_plan_gate.py", "--gap-cap", "10800", "--blend-weight", "0.60", "--seed", sd,
                       "--in", P("base"), "--out", o))
            gs.append(o)
        average(gs, P("gatebag_s0"))
    log("pre-rules done")
    if args.prerules_only:
        return

    # 2. ADS-B stage per stream
    for stream in args.streams.split(","):
        fl = flags(stream, dayarg)
        jobs = [(k, P("s0ship"), P(f"v{stream}_adsb_{k}")) for k in range(1, nreal + 1)]
        jobs = [j for j in jobs if not Path(j[2]).exists()]
        for i in range(0, len(jobs), args.workers):
            procs = []
            for k, inp, o in jobs[i:i + args.workers]:
                lf = open(o.replace(".parquet", ".log"), "w", encoding="utf-8")
                procs.append((subprocess.Popen([PY, *RUN, "--ranking", frame, APPLY, "--in", inp, "--out", o, *fl], cwd=REPO,
                                               stdout=lf, stderr=subprocess.STDOUT), lf, o))
            for p, lf, o in procs:
                rc = p.wait()
                lf.close()
                if rc:
                    raise SystemExit(f"stream {stream} realisation {o} failed ({rc}); see {o.replace('.parquet', '.log')}")
            log(f"stream {stream}: {min(i + args.workers, len(jobs))}/{len(jobs)} realisations")
        average([P(f"v{stream}_adsb_{k}") for k in range(1, nreal + 1)], P(f"v{stream}_s1"))
        if not Path(P(f"v{stream}_gatebag_s1")).exists():
            run([PY, *RUN, "--ranking", frame, APPLY, "--in", P("gatebag_s0"), "--out", P(f"v{stream}_gatebag_s1"), *fl],
                P(f"v{stream}_gatebag_s1").replace(".parquet", ".log"))
        log(f"stream {stream} done")

    # rows per airport-month and rows moved by the stage
    s0 = pl.read_parquet(P("s0ship")).rename({TARGET_COL: "s0"})
    for stream in args.streams.split(","):
        s1 = pl.read_parquet(P(f"v{stream}_s1")).rename({TARGET_COL: "s1"})
        t = (dep.join(s0, on=ID_COL).join(s1, on=ID_COL)
             .group_by("apt", "month").agg(pl.len().alias("rows"), (pl.col("s1") != pl.col("s0")).sum().alias("moved"),
                                           (pl.col("s1") - pl.col("s0")).mean().round(2).alias("mean_shift")).sort("apt", "month"))
        print(f"stream {stream}: rows, rows moved by the stage, mean shift (s) per airport-month")
        with pl.Config(tbl_rows=100):
            print(t)


if __name__ == "__main__":
    main()
