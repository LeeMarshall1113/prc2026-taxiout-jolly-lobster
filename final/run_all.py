"""Driver of the whole final-phase pipeline: stages in order, inputs and outputs, skip a stage whose outputs are newer than its inputs.

    python final/run_all.py --dry-run            # the plan, and which outputs exist right now
    python final/run_all.py --run                # run every stage that is out of date, in order, stop at the first failure
    python final/run_all.py --run --only 50      # one stage
    python final/run_all.py --run --from 41      # from a stage on

Every heavy stage runs through its experiment's launcher (scheduler slot, cross-experiment job lock, hard commit cap); the sizes are the ones in the
experiment documentation (final/README.md) and in final/2x_chain.sh. Stages marked MANUAL run on rented or free cloud machines or need a
decision of the operator; the driver prints the command and stops there (use --skip-manual to go on to the next stage).
A stage is out of date when one of its outputs is missing or older than its newest input. Inputs outside data/final are the raw files.

Months: --months (default 2,6) are the months built from the stage outputs; January and July (--carried, default 1,7) come from the stored
lineage and need none of the heavy stages.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
F = REPO / "data" / "final"
RAW = REPO / "data" / "raw"
PTS = REPO / "data" / "adsb" / "points"
BASH = "bash"


def mt(paths: list[Path]) -> float:
    ts = [p.stat().st_mtime for p in paths if p.exists()]
    return max(ts) if ts else 0.0


def adsb_days(months: list[int]) -> list[Path]:
    return [p for m in months for p in sorted(PTS.glob(f"2026-{m:02d}-*.parquet"))]


def stages(months: str, carried: str) -> list[dict]:
    mm = [int(x) for x in months.split(",")]
    L = lambda gb, mins, eta: f"LEASE_GB={gb} LEASE_MIN={mins} LEASE_ETA={eta} "  # noqa: E731
    fin = RAW / "final_ranking.parquet"
    return [
        dict(id="05", name="ADS-B day files, 58 days", manual=True, cmd="bash final/05_adsb_download.sh   # slot prc-taxiout-adsb-final, already launched by the team",
             inputs=[], outputs=lambda: adsb_days(mm) or [PTS / "(none yet)"], count=("days", lambda: len(adsb_days(mm)), 58)),
        dict(id="10", name="base model refit (J2)", cmd="bash final/10_base.sh   # slot prc-taxiout-final-base, 9-10 GB, about 13 h", manual=True,
             inputs=[fin], outputs=lambda: [F / "J2_v94model.parquet"]),
        dict(id="20", name="ADS-B dumps (rows, ff, evt, seq, bx, trk, osm, occ, census)",
             cmd=L(3, 120, 60) + f'bash final/2x_run.sh "dumps" final/20_adsb_dumps.py --months {months}',
             inputs=lambda: adsb_days(mm) + [fin], outputs=lambda: [F / f"J6_{n}.parquet" for n in ("rows", "ff", "evt", "seq", "bx", "osm", "occ", "census")]),
        dict(id="21", name="Rome pre-rules + ADS-B stage (streams 103, 105, 105sc)",
             cmd=L(11, 420, 240) + f'bash final/2x_run.sh "ADS-B stage" final/21_adsb_stage.py --months {months} --workers 3',
             inputs=lambda: [F / "J2_v94model.parquet"] + adsb_days(mm), outputs=lambda: [F / "J5_v105_s1.parquet", F / "J5_v103_s1.parquet", F / "J5_v105sc_s1.parquet"]),
        dict(id="22", name="re-stack, post-chain, splices (J7)", cmd=L(3, 180, 90) + f'bash final/2x_run.sh "re-stack" final/22_restack.py --months {months}',
             inputs=lambda: [F / "J5_v105_s1.parquet", F / "J6_rows.parquet", F / "J6_evt.parquet"], outputs=lambda: [F / "J7_restack.parquet"]),
        dict(id="43a", name="Istanbul bulk and snow refits, rows (J3)", manual=True,
             cmd="see final/README.md 'Istanbul path': final/43_kaggle_prepare.py + modal/final_istanbul_app.py (control, snow k0, snow k12), then "
                 "python final/43_ist_build.py build --base data/final/J7_restack.parquet --ctrl X --k0 A --k12 B --months " + months + " --out data/final/J3_ist.parquet",
             inputs=lambda: [F / "J7_restack.parquet"], outputs=lambda: [F / "J3_ist.parquet"]),
        dict(id="50a", name="chain value (input of the small stages)", cmd=L(3, 40, 10) + f'bash final/5x_run.sh "chain value" final/50_assemble.py --months 1,2,6,7 --carried {carried} --stop-after-chain',
             inputs=lambda: [F / "J7_restack.parquet", F / "J3_ist.parquet"], outputs=lambda: [F / "J11_chain.parquet"]),
        dict(id="42", name="small stages M1-M4, tail (J8)", cmd=L(3, 120, 60) + f'bash final/4x_run.sh "small stages" final/42_small_stages.py --base data/final/J11_chain.parquet --months {months}',
             inputs=lambda: [F / "J11_chain.parquet", F / "J6_rows.parquet"], outputs=lambda: [F / "J8_small.parquet", F / "J8_pv.parquet", F / "J8_deltas.npz"]),
        dict(id="40", name="serve-side inputs of the correctors (J9)",
             cmd=L(3, 120, 60) + f'bash final/4x_run.sh "serve inputs" final/40_serve_inputs.py --months {months} --pv data/final/J8_pv.parquet --profile final',
             inputs=lambda: [F / "J8_pv.parquet", F / "J7_restack.parquet"], outputs=lambda: [F / "J9_serve"]),
        dict(id="41s", name="corrector stage directory (J10)", cmd=L(3, 40, 10) + 'bash final/4x_run.sh "stage" final/41_corrector_serve.py stage --serve-dir data/final/J9_serve --tag J10',
             inputs=lambda: [F / "J9_serve"], outputs=lambda: [F / "J10_stage"]),
        dict(id="41a", name="arithmetic corrector, 12 seeds", cmd=L(10, 300, 240) + 'bash final/4x_run.sh "arith serve fits" final/41_corrector_serve.py arith --stage data/final/J10_stage --seeds 0-11',
             inputs=lambda: [F / "J10_stage"], outputs=lambda: [F / "J10_stage" / "out" / "arith" / f"KYQSR-pw-s0-r0_serve_s{s}.npz" for s in range(12)]),
        dict(id="41p", name="plan-clock corrector, 12 seeds", cmd=L(10, 300, 240) + 'bash final/4x_run.sh "plan-clock serve fits" final/41_corrector_serve.py pcd --stage data/final/J10_stage --seeds 0-11',
             inputs=lambda: [F / "J10_stage"], outputs=lambda: [F / "J10_stage" / "out" / "pcd" / f"C4ff7V5_s{s}.npz" for s in range(12)]),
        dict(id="43b", name="Istanbul corrector term", manual=True, cmd="see final/README.md 'Istanbul path' (43_gen_lv, 43_ist_serve_inputs, ist_corrector_serve.py); writes data/final/*ist*_term.npz",
             inputs=lambda: [F / "J3_ist.parquet"], outputs=lambda: sorted(F.glob("*ist*_term.npz")) or [F / "(ist term .npz)"]),
        dict(id="50", name="assembly: variants A, B, C", cmd=L(3, 60, 15) + f'bash final/5x_run.sh "assembly" final/50_assemble.py --months 1,2,6,7 --carried {carried}',
             inputs=lambda: [F / "J7_restack.parquet", F / "J8_small.parquet", F / "J3_ist.parquet"] + sorted((F / "J10_stage" / "out").glob("*/*.npz")),
             outputs=lambda: [F / "final_A.parquet", F / "final_B.parquet", F / "final_C.parquet"]),
    ]


def resolve(x):
    return x() if callable(x) else x


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--from", dest="start", default="")
    ap.add_argument("--months", default="2,6")
    ap.add_argument("--carried", default="1,7")
    ap.add_argument("--skip-manual", action="store_true")
    a = ap.parse_args()
    if not (a.dry_run or a.run):
        ap.error("give --dry-run or --run")
    sts = stages(a.months, a.carried)
    go = not a.start
    print(f"final-phase pipeline, months built {a.months}, carried {a.carried}, {time.strftime('%Y-%m-%d %H:%M')}\n")
    for s in sts:
        ins, outs = resolve(s.get("inputs", [])), resolve(s["outputs"])
        missing_out = [o for o in outs if not o.exists()]
        missing_in = [i for i in ins if not i.exists()]
        t_in, t_out = mt(ins), mt(outs) if not missing_out else 0.0
        stale = bool(missing_out) or t_out < t_in
        state = "MISSING" if missing_out else ("OUT OF DATE" if stale else "up to date")
        if "count" in s:
            label, fn, want = s["count"]
            state += f" ({fn()} of {want} {label})"
        print(f"[{s['id']:>3}] {s['name']:<62} {state}{'  MANUAL' if s.get('manual') else ''}")
        if missing_out:
            print("        missing: " + ", ".join(o.name for o in missing_out[:6]) + (f" (+{len(missing_out) - 6})" if len(missing_out) > 6 else ""))
        if missing_in and stale:
            print("        waits for: " + ", ".join(i.name for i in missing_in[:6]))
        print(f"        cmd: {s['cmd']}")
        if a.start and s["id"] == a.start:
            go = True
        if not a.run or not go or (a.only and s["id"] != a.only) or not stale:
            continue
        if s.get("manual"):
            print("        MANUAL stage: not run by the driver")
            if a.skip_manual:
                continue
            sys.exit(f"stopped at manual stage {s['id']}")
        if missing_in:
            sys.exit(f"stage {s['id']} cannot run: inputs missing ({', '.join(i.name for i in missing_in[:4])})")
        r = subprocess.run(["bash", "-c", s["cmd"]], cwd=str(REPO))
        if r.returncode:
            sys.exit(f"stage {s['id']} failed with exit {r.returncode}")


if __name__ == "__main__":
    main()
