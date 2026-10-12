"""Prepare (NOT push, NOT start) the Kaggle free-CPU route for the Istanbul arms of the final phase.

Builds, under kaggle-stage/ :
  prc-ltfm-final/            flat private dataset dir (hard links, no copy): current prc/*.py, the scripts the three arms import, the
                             generated wrappers final/ist_*.py and final/10_base.py, 2025 training files, weather, the training-feature
                             cache + hash, final_ranking.parquet and final_submitting.parquet
  kernels/final-ist-<arm>/   run.py + kernel-metadata.json for the arms ctrl (4.2 h), snow_k0 (2.25 h), snow_k12 (3.1 h) at the shipped
                             Kaggle sizes; the treatment arm of the shipped chain is not needed (final/43_ist_build.py).
Pushing is a manual step:
    python -m kaggle datasets create -p kaggle-stage/prc-ltfm-final            (private, about 1.6 GB)
    python -m kaggle kernels push -p kaggle-stage/kernels/final-ist-ctrl        (then snow_k0, snow_k12)
    python -m kaggle kernels output <kaggle-user>/prc-final-ist-ctrl -p <dir>                     (preds_<arm>.parquet)
Kaggle runs the arms one or two at a time: about 5.5 h wall with two in parallel (ctrl beside k0 then k12), 9.6 h one after the other.
    python final/43_kaggle_prepare.py
"""
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = Path("kaggle-stage")
USER, DS = "<kaggle-user>", "prc-ltfm-final"
TEMPLATE = REPO / "archive" / "agent_kg_ltfm_kernel_template.py"
BASEDROP = ("airport_plan,arr_slow_180min,arr_slow_60min,arr_sub_day,arr_sub_stand,arr_taxi_180min,arr_taxi_trend,cfg_new,"
            "cfg_shift,clk_aobt_minus_sched,clk_arvt3_minus_aobt,clk_arvt3_minus_arvt1,clk_arvt3_minus_mvt,"
            "clk_eobt_minus_lobt,clk_iobt_minus_lobt,clk_lobt_minus_sched,clk_sec_aobt,clk_sec_mvt,clk_sec_sched,doy,"
            "lobt_minus_aobt,month,mvt_sec_00,prev_rwy_gap,prev_stand_gap,prev_stand_headway,ref_level,ref_taxi_s,"
            "turn_in_delay,turn_rwy_match,turn_sched_slack,wake_heavy_60min,wake_prev_heavy,wake_sep_req")
COMMON = ("--iterations 2500 --depth 8 --lr 0.08 --l2 3.0 --threads 4 --baseline blend_apt --noplan-iterations 300 --noplan-seeds 5 "
          "--regime-correction 1.0 --residual --bulk-plan-only --noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF").split()
PINS = ["catboost==1.2.10", "polars==1.44.1"]
# arm: (wrapper, seq, wrapper flags before --out, base flags after the common ones, expected feature count)
ARMS = {"ctrl": ("ist_kg_ltfm_final.py", "off", [], ["--seeds", "3"], 80),
        "snow_k0": ("ist_ltreg_final.py", "on", [], ["--seeds", "1"], 104),
        "snow_k12": ("ist_ltreg_final.py", "on", ["--seed-offset", "1"], ["--seeds", "2"], 104)}


def link(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage():
    d = STAGE / DS
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    n = 0
    for p in sorted((REPO / "prc").glob("*.py")):
        link(p, d / f"prc__{p.name}"); n += 1
    for s in ("agent_ltfm_seqfeat.py", "agent_ltreg_feat.py"):
        link(REPO / "archive" / s, d / f"scripts__{s}"); n += 1
    for s in ("ist_kg_ltfm_final.py", "ist_ltreg_final.py"):
        link(REPO / "final" / s, d / f"scripts__{s}"); n += 1
    link(REPO / "final" / "10_base.py", d / "scripts__final_10_base.py"); n += 1
    raw = REPO / "data" / "raw"
    for p in sorted(raw.glob("training_2025-*.parquet")) + [raw / "final_ranking.parquet", raw / "final_submitting.parquet"]:
        link(p, d / f"raw__{p.name}"); n += 1
    for p in sorted((raw / "weather").glob("*.csv")):
        link(p, d / f"raw__weather__{p.name}"); n += 1
    for name in ("train_features.parquet", "train_features.featurehash"):
        link(REPO / "data" / "interim" / name, d / f"interim__{name}"); n += 1
    (d / "dataset-metadata.json").write_text(json.dumps({"title": DS, "id": f"{USER}/{DS}", "licenses": [{"name": "other"}]}, indent=1))
    print(f"staged {n} files, {sum(p.stat().st_size for p in d.iterdir()) / 1e9:.2f} GB (hard links) -> {d}")


def kernels():
    tpl = TEMPLATE.read_text(encoding="utf-8").replace("\r\n", "\n")
    branch = ('elif KIND == "ifinal":\n'
              '    cmd = [sys.executable, "-u", "archive/" + WRAPPER, "--ltfm-seq", SEQ, *WFLAGS, "--out", f"{W}/preds_{TAG}.parquet",\n'
              '           "--basedrop", BASEDROP, *COMMON, *EXTRA, "--ranking", "/tmp/data/raw/final_ranking.parquet",\n'
              '           "--template", "/tmp/data/raw/final_submitting.parquet", "--months", "2,6", "--out-dir", W, "--tag", TAG]\n'
              'else:\n    raise SystemExit(KIND)')
    assert "else:\n    raise SystemExit(KIND)" in tpl
    tpl = tpl.replace("else:\n    raise SystemExit(KIND)", branch, 1)
    tpl = tpl.replace('"OMP_NUM_THREADS": "4",', '"OMP_NUM_THREADS": "4", "FINAL_RANKING": "/tmp/data/raw/final_ranking.parquet", "FINAL_MONTHS": "2,6",', 1)
    tpl = tpl.replace('EXPECT = @EXPECT@', 'EXPECT = @EXPECT@\nWRAPPER = "@WRAPPER@"\nWFLAGS = @WFLAGS@', 1)
    for arm, (wrapper, seq, wflags, extra, expect) in ARMS.items():
        tag = f"final_ist_{arm}"
        code = (tpl.replace("@TAG@", tag).replace("@KIND@", "ifinal").replace("@SEQ@", seq).replace("@FOLD@", "").replace("@EXTRA@", repr(extra))
                .replace("@EXPECT@", str(expect)).replace("@PINS@", repr(PINS)).replace("@BASEDROP@", BASEDROP).replace("@COMMON@", repr(COMMON))
                .replace("@WRAPPER@", wrapper).replace("@WFLAGS@", repr(wflags)))
        compile(code, "run.py", "exec")
        kd = STAGE / "kernels" / f"final-ist-{arm}"
        if kd.exists():
            shutil.rmtree(kd)
        kd.mkdir(parents=True)
        (kd / "run.py").write_text(code, encoding="utf-8")
        slug = f"prc-final-ist-{arm.replace('_', '-')}"
        (kd / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{slug}", "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": False, "enable_tpu": False, "enable_internet": True, "dataset_sources": [f"{USER}/{DS}"], "competition_sources": [],
            "kernel_sources": []}, indent=1))
        print("kernel dir", kd, "expect", expect, "features")


if __name__ == "__main__":
    stage()
    kernels()
    print("prepared only: nothing was pushed or started")
