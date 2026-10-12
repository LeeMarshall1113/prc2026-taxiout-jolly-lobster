"""Stage 22: the pooled re-stack on the ADS-B stage output, the post-chain, the splices and the monster rule, for chosen months.

Follows the four piece builds of archive/agent_stagecap_builds.sh (mode repro) and archive/agent_stagecap_build9s.sh, i.e.
archive/agent_stagecap_bxbuild.sh and archive/agent_stagecap_build.sh:

  piece  serve script / recipe                                      re-stack training      stage streams
  a      agent_restack5_bxserve.py   cap,evt,trk,bx,big             TX_S6                  103 + 105
  b      agent_restack5_bxserve.py   cap,evt,trk,seq,bx,big         TX_S6                  103 + 105
  c      agent_restack5_serve.py     cap,evt,trk,seq,big            TX_S6                  103 + 105
  d      agent_restack5_bxserve.py   cap,evt,trk,bx,big             CAND_E9f               103 + 105sc (splice also EDDM,EHAM,LEBL)

Per piece and stream: serve the re-stack on the stage output (s1 and the Rome-path gatebag_s1), run the post-chain (gap clamp,
plan-clock window with hard-binding blend 0.5, floor, v35 rules, upper bound, hidden reading) on both, take the Rome path at
LIRF, splice the 105 stream over the 103 stream at EDDF,EGLL,LEMD,LSZH,LFPG,LIRF, then apply the monster rule. The
January-only Munich shift (archive/apply_eddm_january.py) is left out: it matches no row in February or June and was removed
from the chain at v144.

The airport-wise composite is written as <tag>_restack.parquet: EGLL from piece a, LEMD from b, LFPG and LIRF from c,
EDDF/EDDM/EHAM/LEBL/LSZH from d (the map of archive/agent_sb2_compose.py), every other row from piece a. The later additive
steps of the chain (neighbour blend, Istanbul refit, the E9S delta on the blend base) are not part of this stage.

Every shipped script runs unchanged through final/_run_as_ranking.py (ranking file -> the month frame of final/_mkrank.py).

    python final/22_restack.py --months 2,6 [--tag J7] [--stage-tag J5] [--dump-tag J6] [--pieces a,b,c,d]

Memory: one re-stack fit on 1.11 M training rows is about 3.2 GB (shipped peak, 6 GB cap), the rest under 1 GB; about 5 minutes
per piece.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, OUT, REPO, TARGET_COL  # noqa: E402

PY = sys.executable
RUN = ["final/_run_as_ranking.py"]
I = "data/interim/"
PIECES = {
    "a": dict(serve="archive/agent_restack5_bxserve.py", recipe="cap,evt,trk,bx,big", train=I + "modal/TX_S6.npz",
              p2="105", spl="EDDF,EGLL,LEMD,LSZH,LFPG,LIRF"),
    "b": dict(serve="archive/agent_restack5_bxserve.py", recipe="cap,evt,trk,seq,bx,big", train=I + "modal/TX_S6.npz",
              p2="105", spl="EDDF,EGLL,LEMD,LSZH,LFPG,LIRF"),
    "c": dict(serve="archive/agent_restack5_serve.py", recipe="cap,evt,trk,seq,big", train=I + "modal/TX_S6.npz",
              p2="105", spl="EDDF,EGLL,LEMD,LSZH,LFPG,LIRF"),
    "d": dict(serve="archive/agent_restack5_bxserve.py", recipe="cap,evt,trk,bx,big", train=I + "agent_stagecap_CAND_E9f.npz",
              p2="105sc", spl="EDDF,EGLL,LEMD,LSZH,LFPG,LIRF,EDDM,EHAM,LEBL"),
}
COMPOSITE = {"EGLL": "a", "LEMD": "b", "LFPG": "c", "LIRF": "c", "EDDF": "d", "EDDM": "d", "EHAM": "d", "LEBL": "d", "LSZH": "d"}


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=str(OUT / "J5_frame.parquet"), help="month frame (final/_mkrank.py output)")
    ap.add_argument("--tag", default="J7")
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--stage-dir", default=str(OUT), help="directory of the stage-2 outputs")
    ap.add_argument("--stage-tag", default="J5_", help="file prefix: <dir>/<prefix>v<stream>_<s1|gatebag_s1>.parquet")
    ap.add_argument("--stage-s0", default=None, help="prefix of the stage inputs for the nbr token (not used by these recipes)")
    ap.add_argument("--rows", default=str(OUT / "J6_rows.parquet"))
    ap.add_argument("--evt", default=str(OUT / "J6_evt.parquet"))
    ap.add_argument("--trk", default=None, help="default: the rows file (it carries the anchor and track columns)")
    ap.add_argument("--seq", default=str(OUT / "J6_seq.parquet"))
    ap.add_argument("--bx", default=str(OUT / "J6_bx.parquet"))
    ap.add_argument("--pieces", default="a,b,c,d")
    ap.add_argument("--skip-monster", action="store_true")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    trk = args.trk or args.rows

    def P(s):
        return str(out / f"{args.tag}_{s}.parquet")

    def stage(stream, s):
        return str(Path(args.stage_dir) / f"{args.stage_tag}v{stream}_{s}.parquet")

    def rr(script, *a):
        return [PY, *RUN, "--ranking", args.frame, script, *map(str, a)]

    def chain(src, dst):
        """post-chain of archive/merged_chain.sh without the January-only Munich shift"""
        steps = [("archive/apply_gap_clamp.py", []), ("archive/apply_lobt_box.py", ["--hard-binding-blend", "0.5"]),
                 ("archive/apply_min_floor.py", []), ("archive/apply_v35_rules.py", []),
                 ("archive/apply_upper_bound.py", []), ("archive/apply_hidden_reading.py", [])]
        cur = src
        for i, (sc, extra) in enumerate(steps):
            nxt = f"{dst}_{i}.parquet" if i < len(steps) - 1 else f"{dst}_final.parquet"
            if not Path(nxt).exists():
                run(rr(sc, *extra, "--in", cur, "--out", nxt))
            cur = nxt
        return cur

    done = {}
    for pc in args.pieces.split(","):
        cfg = PIECES[pc]
        final = P(f"piece_{pc}")
        if Path(final).exists():
            done[pc] = final
            continue
        raw = {}
        for stream in ("103", cfg["p2"]):
            tag = f"{args.tag}_{pc}_v{stream}"
            fins = {}
            for s in ("s1", "gatebag_s1"):
                o = str(out / f"{tag}_{s}.parquet")
                if not Path(o).exists():
                    cmd = [PY, *RUN, "--ranking", args.frame, cfg["serve"], "--train-npz", cfg["train"],
                           "--train-rows", I + "agent_offblock_rows_c238.parquet", "--rows-2026", args.rows,
                           "--feats", "est_v2", "--recipe", cfg["recipe"], "--evt-2026", args.evt, "--trk-2026", trk,
                           "--in", stage(stream, s), "--out", o]
                    if "seq" in cfg["recipe"]:
                        cmd += ["--seq-2026", args.seq]
                    if "bx" in cfg["recipe"]:
                        cmd += ["--bx-2026", args.bx]
                    run(cmd, o.replace(".parquet", ".log"))
                fins[s] = o
            nl = chain(fins["s1"], str(out / f"{tag}_nl"))
            li = chain(fins["gatebag_s1"], str(out / f"{tag}_li"))
            r = str(out / f"{tag}_raw.parquet")
            if not Path(r).exists():
                run(rr("archive/splice_apt_parquet.py", nl, li, "LIRF", r))
            raw[stream] = r
        sp = P(f"piece_{pc}_spliced")
        if not Path(sp).exists():
            run(rr("archive/splice_apt_parquet.py", raw["103"], raw[cfg["p2"]], cfg["spl"], sp))
        if args.skip_monster:
            Path(sp).replace(final)
        else:
            run(rr("archive/agent_monster_apply.py", "--in", sp, "--out", final))
        done[pc] = final
        # the "ref" variant file the corrector inputs read (final/40_serve_inputs.py VARIANTS['ref'] = sb2_t_<piece>): the
        # no-token piece build, i.e. this piece
        import shutil
        shutil.copyfile(final, P(f"sb2_t_{pc}"))
        log(f"piece {pc} done")

    # airport-wise composite
    if all(p in done for p in "abcd"):
        cur = done["a"]
        comp = P("restack")
        by_piece = {}
        for apt, pc in COMPOSITE.items():
            by_piece.setdefault(pc, []).append(apt)
        later = sorted(p for p in by_piece if p != "a")      # piece a is the frame; its airport (EGLL) needs no splice
        for k, pc in enumerate(later):
            nxt = comp if k == len(later) - 1 else P(f"restack_{k}")
            run(rr("archive/splice_apt_parquet.py", cur, done[pc], ",".join(by_piece[pc]), nxt))
            cur = nxt
        dep = (pl.scan_parquet(args.frame).filter(pl.col("PHASE_mvt") == "DEP")
               .select(pl.col(ID_COL), pl.col("ADEP_mvt").alias("apt"), pl.col("MVT_TIME_UTC_mvt").dt.month().alias("month")).collect())
        fin = pl.read_parquet(comp).join(dep, on=ID_COL)
        print("rows per airport-month in the composite:")
        with pl.Config(tbl_rows=100):
            print(fin.group_by("apt", "month").agg(pl.len().alias("rows"), pl.col(TARGET_COL).mean().round(1).alias("mean_taxi"),
                                                   (pl.col(TARGET_COL) < 0).sum().alias("negatives")).sort("apt", "month"))
        log(f"composite -> {comp}")


if __name__ == "__main__":
    main()
