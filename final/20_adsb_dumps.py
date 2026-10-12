"""Stage 20: per-departure ADS-B dumps for the chosen months (the inputs of the ADS-B stage and the re-stack).

For the departures of the chosen months (UTC month of take-off) of a ranking file, writes under data/final/ (default tag J6):
  <tag>_rows.parquet  off-block estimates, anchors and track-shape columns   (final/_dump_rows.py)
  <tag>_ff.parquet    first ground fix geometry                               (final/_dump_ff.py)
  <tag>_evt.parquet   push-back events of the matched airframe                (final/_dump_evt.py)
  <tag>_seq.parquet   take-off-sequence features                              (final/_dump_seq.py)
  <tag>_bx.parquet    back-extrapolated off-block estimates                   (final/_dump_bx.py)
  <tag>_trk.parquet   id + anchor and track-shape columns (a view of the rows file)
  <tag>_osm.parquet   stand-map distances to OpenStreetMap stands             (final/_dump_osm_occ.py osm)
  <tag>_occ.parquet   apron occupancy around the off-block estimate           (final/_dump_osm_occ.py occ)
  <tag>_census.parquet  why a departure has no matched ground track           (final/_dump_census.py)
Each part is a parameterised copy of the script that built the January/July 2026 files (named in its header). Days without a
points file are skipped and listed. Nothing here reads a 2026 departure label. Prints rows per airport-month and the share
of departures with a surface track.

    python final/20_adsb_dumps.py --months 2,6 [--sample 2] [--tag J6] [--ranking data/raw/final_ranking.parquet]

Memory: under 3 GB per part for ~60 days (shipped runs: 2 GB). Run under the slot and the job lock of final/README.md.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import OUT, REPO, parse_months  # noqa: E402

APTS = "EDDF,EDDM,EGLL,EHAM,LEBL,LEMD,LFPG,LIRF,LSZH,LTFM"
NINE = APTS.replace(",LTFM", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(REPO / "data/raw/final_ranking.parquet"))
    ap.add_argument("--months", default="2,6")
    ap.add_argument("--tag", default="J6")
    ap.add_argument("--days", default=None, help="comma list of YYYY-MM-DD; overrides --months")
    ap.add_argument("--sample", type=int, default=0, help="only the first N days that have a points file")
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--only", default="rows,ff,evt,seq,bx,trk,osm,occ,census")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    months = parse_months(args.months)
    dep = (pl.scan_parquet(args.ranking).filter(pl.col("PHASE_mvt") == "DEP")
           .select(pl.col("MVT_ID_mvt").alias("id"), pl.col("ADEP_mvt").alias("apt"), pl.col("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"),
                   pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).collect())
    if args.days:
        days = args.days.split(",")
    else:
        days = sorted(dep.filter(pl.col("month").is_in(months))["day"].unique().to_list())
    have = [d for d in days if (REPO / f"data/adsb/points/{d}.parquet").exists()]
    miss = [d for d in days if d not in have]
    if miss:
        print(f"days without a points file ({len(miss)}): {miss[:5]}{'...' if len(miss) > 5 else ''}", flush=True)
    if args.sample:
        have = have[:args.sample]
    assert have, "no day with a points file"
    dayarg = ",".join(have)
    print(f"{len(have)} days: {have[0]} .. {have[-1]}", flush=True)
    py = sys.executable
    f = lambda s: str(out / f"{args.tag}_{s}.parquet")  # noqa: E731
    common = ["--ranking", args.ranking, "--days", dayarg]
    steps = {
        "rows": [py, "-u", "-W", "ignore", "final/_dump_rows.py", *common, "--apts", NINE, "--out", f("rows")],
        "ff": [py, "-u", "-W", "ignore", "final/_dump_ff.py", *common, "--apts", NINE, "--out", f("ff")],
        "evt": [py, "-u", "-W", "ignore", "final/_dump_evt.py", *common, "--apts", APTS, "--out", f("evt")],
        "seq": [py, "-u", "-W", "ignore", "final/_dump_seq.py", "--ranking", args.ranking, "--rows", f("rows"), "--out", f("seq")],
        "bx": [py, "-u", "-W", "ignore", "final/_dump_bx.py", "--rows", f("rows"), "--ff", f("ff"), "--out", f("bx")],
    }
    steps["osm"] = [py, "-u", "-W", "ignore", "final/_dump_osm_occ.py", "osm", "--rows", f("rows"), "--ff", f("ff"), "--out", f("osm")]
    steps["occ"] = [py, "-u", "-W", "ignore", "final/_dump_osm_occ.py", "occ", "--rows", f("rows"), "--ff", f("ff"), "--out", f("occ")]
    steps["census"] = [py, "-u", "-W", "ignore", "final/_dump_census.py", "--ranking", args.ranking, "--months", ",".join(sorted({str(int(d[5:7])) for d in have})),
                       "--days", dayarg, "--out", f("census")]
    steps["trk"] = None
    for name in args.only.split(","):
        t0 = time.time()
        if name == "trk":
            sys.path.insert(0, str(REPO))
            from archive.agent_restack3_restack import ANCH, TRK
            pl.read_parquet(f("rows"), columns=["MVT_ID_mvt"] + ANCH + TRK).write_parquet(f("trk"))
            print("part trk done", flush=True)
            continue
        r = subprocess.run(steps[name], cwd=REPO)
        if r.returncode:
            raise SystemExit(f"part {name} failed with code {r.returncode}")
        print(f"part {name} done in {time.time() - t0:.0f}s", flush=True)
    # rows per airport-month and share of departures with a surface track (first ground fix of the departure segment)
    if "rows" in args.only and "ff" in args.only:
        rows = pl.read_parquet(f("rows"), columns=["MVT_ID_mvt", "e_start", "src", "prod_t"]).with_columns(pl.col("MVT_ID_mvt").cast(pl.Float64))
        d = dep.filter(pl.col("day").is_in(have)).with_columns(pl.col("id").cast(pl.Float64))
        j = d.join(rows, left_on="id", right_on="MVT_ID_mvt", how="left")
        tab = (j.group_by("apt", "month").agg(pl.len().alias("departures"), pl.col("src").is_not_null().sum().alias("dump_rows"),
                                              pl.col("e_start").is_not_null().sum().alias("with_track"),
                                              pl.col("prod_t").is_not_null().sum().alias("offblock_measured"))
               .with_columns((pl.col("with_track") / pl.col("departures")).round(3).alias("track_share")).sort("apt", "month"))
        with pl.Config(tbl_rows=200):
            print(tab)


if __name__ == "__main__":
    main()
