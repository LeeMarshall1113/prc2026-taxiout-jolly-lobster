"""Stage / push / poll / pull the LTFM bulk-refit arms on Kaggle FREE CPU (tag "kg-ltfm", 2026-09-25).

    python archive/agent_kg_ltfm_kaggle.py stage                 # build the flat PRIVATE dataset dir (no upload)
    python archive/agent_kg_ltfm_kaggle.py kernels TAG [TAG..]   # write kernel dirs for the named arms
    python archive/agent_kg_ltfm_kaggle.py status
    python archive/agent_kg_ltfm_kaggle.py pull TAG [TAG..]      # download outputs into results/ (+ logs/kg/)

Kaggle CLI = SYSTEM python (`python -m kaggle`), never a project venv. Everything is private: dataset created without
--public, kernels "is_private": true, "enable_gpu": false. Internet is ON only so the kernel can pip-pin catboost 1.2.10
and polars 1.44.1 (the local versions); nothing is downloaded locally except the kernels' own outputs.
Flags = agent_ltfm_app.py COMMON + BASEDROP (the late2500_treat configuration).
"""
import json
import pathlib
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
STAGE = pathlib.Path("kaggle-stage")
USER = __import__("os").environ["KAGGLE_USERNAME"]
DS = "prc-ltfm-kg"
BASEDROP = ("airport_plan,arr_slow_180min,arr_slow_60min,arr_sub_day,arr_sub_stand,arr_taxi_180min,arr_taxi_trend,cfg_new,"
            "cfg_shift,clk_aobt_minus_sched,clk_arvt3_minus_aobt,clk_arvt3_minus_arvt1,clk_arvt3_minus_mvt,"
            "clk_eobt_minus_lobt,clk_iobt_minus_lobt,clk_lobt_minus_sched,clk_sec_aobt,clk_sec_mvt,clk_sec_sched,doy,"
            "lobt_minus_aobt,month,mvt_sec_00,prev_rwy_gap,prev_stand_gap,prev_stand_headway,ref_level,ref_taxi_s,"
            "turn_in_delay,turn_rwy_match,turn_sched_slack,wake_heavy_60min,wake_prev_heavy,wake_sep_req")
COMMON = ("--seeds 1 --iterations 2500 --depth 8 --lr 0.08 --l2 3.0 --threads 4 --baseline blend_apt "
          "--noplan-iterations 300 --noplan-seeds 5 --regime-correction 1.0 --save-preds --residual --bulk-plan-only "
          "--noplan-model --noplan-routes LIRF --noplan-gate-routes LIRF").split()
# prc.final twin: same model flags, no --save-preds (crossval-only); --seeds set per arm
FINAL_COMMON = [a for a in COMMON if a != "--save-preds"]
i = FINAL_COMMON.index("--seeds")
del FINAL_COMMON[i:i + 2]
PINS = ["catboost==1.2.10", "polars==1.44.1"]
N_CTRL, N_SEQ = 80, 19

# arm table: tag -> (kind, seq, fold, extra)
ARMS = {}
for f in ("1", "2", "3", "4", "6"):
    ARMS[f"ltfmseq_ctrl_f{f}"] = ("fold", "off", f, [])
    ARMS[f"ltfmseq_treat_f{f}"] = ("fold", "on", f, [])
ARMS["ltfmseq_smoke_f1"] = ("fold", "on", "1", ["--iterations", "30", "--noplan-iterations", "20", "--noplan-seeds", "1"])
ARMS["ltfmfin_smoke_treat"] = ("final", "on", "", ["--seeds", "1", "--iterations", "30", "--noplan-iterations", "20", "--noplan-seeds", "1"])
for f in ("1", "2", "3"):
    ARMS[f"bulkseq_treat_f{f}"] = ("bfold", "on", f, [])
ARMS["bulkseq_smoke_f1"] = ("bfold", "on", "1", ["--iterations", "30", "--noplan-iterations", "20", "--noplan-seeds", "1"])
ARMS["bulkfin_smoke_treat"] = ("bfinal", "on", "", ["--seeds", "1", "--iterations", "30", "--noplan-iterations", "20", "--noplan-seeds", "1"])
ARMS["bulkfin_treat_s3"] = ("bfinal", "on", "", ["--seeds", "3"])
ARMS["bulkfin_treat_s2"] = ("bfinal", "on", "", ["--seeds", "2"])
for s in ("2", "3"):
    ARMS[f"ltfmfin_ctrl_s{s}"] = ("final", "off", "", ["--seeds", s])
    ARMS[f"ltfmfin_treat_s{s}"] = ("final", "on", "", ["--seeds", s])


def slug(tag):
    return "prc-ltfm-kg-" + tag.replace("_", "-")


def stage():
    d = STAGE / DS
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    n = 0
    for p in sorted((REPO / "prc").glob("*.py")):
        shutil.copy(p, d / f"prc__{p.name}"); n += 1
    for s in ("agent_ltfm_crossval.py", "agent_ltfm_seqfeat.py", "agent_kg_ltfm_final.py", "agent_seqpost_feat.py",
              "agent_kg_bulkseq_crossval.py", "agent_kg_bulkseq_final.py"):
        shutil.copy(REPO / "archive" / s, d / f"scripts__{s}"); n += 1
    raw = REPO / "data" / "raw"
    for p in sorted(raw.glob("training_2025-*.parquet")) + [raw / "ranking.parquet", raw / "submitting.parquet"]:
        shutil.copy(p, d / f"raw__{p.name}"); n += 1
    for p in sorted((raw / "weather").glob("*.csv")):
        shutil.copy(p, d / f"raw__weather__{p.name}"); n += 1
    for name in ("train_features.parquet", "train_features.featurehash"):
        shutil.copy(REPO / "data" / "interim" / name, d / f"interim__{name}"); n += 1
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": DS, "id": f"{USER}/{DS}", "licenses": [{"name": "other"}]}, indent=1))
    size = sum(p.stat().st_size for p in d.iterdir())
    print(f"staged {n} files, {size / 1e6:.0f} MB -> {d}")


def kernels(tags):
    tpl = (REPO / "archive" / "agent_kg_ltfm_kernel_template.py").read_text(encoding="utf-8")
    for tag in tags:
        kind, seq, fold, extra = ARMS[tag]
        expect = N_CTRL + (0 if seq == "off" else (21 if kind.startswith("b") else N_SEQ))
        common = COMMON if kind in ("fold", "bfold") else FINAL_COMMON
        code = (tpl.replace("@TAG@", tag).replace("@KIND@", kind).replace("@SEQ@", seq).replace("@FOLD@", fold)
                .replace("@EXTRA@", repr(extra)).replace("@EXPECT@", str(expect)).replace("@PINS@", repr(PINS))
                .replace("@BASEDROP@", BASEDROP).replace("@COMMON@", repr(common)))
        assert "@" not in code.split("# ---------------------------------------------")[0].split("CONFIG")[1]
        kd = STAGE / "kernels" / tag
        if kd.exists():
            shutil.rmtree(kd)
        kd.mkdir(parents=True)
        (kd / "run.py").write_text(code, encoding="utf-8")
        (kd / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{slug(tag)}", "title": slug(tag), "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [f"{USER}/{DS}"], "competition_sources": [],
            "kernel_sources": []}, indent=1))
        print(f"kernel dir {kd} ({kind} {seq} fold={fold} extra={extra} expect={expect})")


def kag(*a):
    r = subprocess.run(["python", "-m", "kaggle", *a], capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def status(tags):
    for tag in tags:
        print(f"{tag:<24} {kag('kernels', 'status', f'{USER}/{slug(tag)}')[-160:]}")


def pull(tags):
    for tag in tags:
        out = STAGE / "out" / tag
        out.mkdir(parents=True, exist_ok=True)
        print(tag, kag("kernels", "output", f"{USER}/{slug(tag)}", "-p", str(out), "-o")[-300:])
        (REPO / "logs" / "kg").mkdir(parents=True, exist_ok=True)
        for p in out.iterdir():
            if p.name.startswith("preds_") and p.suffix == ".parquet":
                shutil.copy(p, REPO / "results" / p.name)
            elif p.suffix in (".log", ".json", ".jsonl"):
                shutil.copy(p, REPO / "logs" / "kg" / (p.name if p.suffix != ".jsonl" else f"{tag}_crossval.jsonl"))


if __name__ == "__main__":
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "stage":
        stage()
    elif cmd == "kernels":
        kernels(rest)
    elif cmd == "status":
        status(rest or [t for t in ARMS if "smoke" not in t])
    elif cmd == "pull":
        pull(rest)
    else:
        raise SystemExit(cmd)
