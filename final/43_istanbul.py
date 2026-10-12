"""Stage 43: the Istanbul path for the served months (runbook jobs J3 and J11). PLAN AND PREFLIGHT ONLY: no fit is started.

What the Istanbul path is (docs/PIPELINE.md stages 6 and 9, S4 and S13):
  1  bulk refit with take-off-sequence features      archive/agent_kg_ltfm_final.py (--ltfm-seq on, "treat"; off, "ctrl")
                                                      + archive/agent_kg_ltfm_delta.py                         (Kaggle CPU, 8.5 h / 4.2 h)
  2  snow-aware refit, two member runs (k0, k12)     archive/agent_ltreg_final.py + archive/agent_ltreg_build.py (Kaggle CPU, 2.25 h + 3.1 h)
  3  views members at Istanbul only (7 arms)         modal/agent_lv_app.py -> archive/agent_lv_members.py       (Modal, about 2 USD)
  4  views corrector + stand handover                archive/agent_lv_04_serve.py, agent_lv_07_handover.py      (local, minutes)
  5  Istanbul corrector with the nine-airport lists  archive/agent_istnext_04_make_serve.py -> agent_istnext_04_serve.py --cols D1,
                                                      agent_istnext_05_build.py                                  (local, 2.6 GB, minutes)

Why nothing is ported here yet. Steps 1 and 2 are wrappers that monkey-patch prc.final (its ranking path, its feature lists and
its output file are module globals) and then call prc.final.main(); the final-phase base is final/10_base.py, which has a
different main(). A port means re-pointing each wrapper at final/10_base.py (its build_by_month, load_training_features and
resolve_transform are module globals and can be patched the same way), serving all four months in one call (the wrappers build
sequence features for the ranking frame in ONE frame, so Feb and Jun must be served together with Jan and Jul or as their own
frame with the same demeaning), and rebuilding the asserts of agent_kg_ltfm_delta.py:41,73 and agent_ltreg_build.py:61,81 (the full-file row
count). Steps 3 to 5 read results of steps 1 and 2 and the Istanbul rows of the J7 base. Estimated engineering: about a day.
That is why the runbook's cut order puts the Istanbul refits first ("fall back to the stage-1 Istanbul rows").

    python final/43_istanbul.py plan        # the table below plus the slot sizes
    python final/43_istanbul.py preflight   # which inputs of each step exist for the chosen months
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# step, shipped script, where it ran, hours, peak RAM GB (logs/kg/*_meta.json, logs/modal, notes), inputs that must exist
STEPS = [
    ("1a bulk refit with sequence features (treat)", "archive/agent_kg_ltfm_final.py --ltfm-seq on", "Kaggle CPU (shared account)", 8.5, 11.0,
     ["data/final/J2_base.parquet"], "logs/kg/ltfmfin_treat_s3_meta.json"),
    ("1b bulk refit without them (ctrl)", "archive/agent_kg_ltfm_final.py --ltfm-seq off", "Kaggle CPU", 4.2, 9.6, [], "logs/kg/ltfmfin_ctrl_s3_meta.json"),
    ("2a snow refit k0", "archive/agent_ltreg_final.py", "Kaggle CPU", 2.25, 11.2, [], "logs/kg/lrfin_snow_k0_meta.json"),
    ("2b snow refit k12", "archive/agent_ltreg_final.py", "Kaggle CPU", 3.1, 11.2, [], "logs/kg/lrfin_snow_k12_meta.json"),
    ("3 views members, 7 arms (ctrl and treat serve + quantiles)", "modal/agent_lv_app.py::lv", "Modal 8 CPU / 28 GB", 3.0, 12.0,
     [], "modal/agent_lv_app.py docstring: about 2 USD in all, 3 h limit"),
    ("4 views corrector (3 seeds) + stand handover", "archive/agent_lv_04_serve.py, agent_lv_07_handover.py", "local", 0.2, 3.0,
     ["data/final/rankfeat_02.parquet", "data/final/rankfeat_06.parquet"], ""),
    ("5 Istanbul corrector D1 (57 columns, 3 seeds)", "archive/agent_istnext_04_serve.py --cols D1; agent_istnext_05_build.py", "local", 0.3, 2.6,
     ["data/final/J7_base.parquet"], "docs/LAB-NOTES.md"),
]


def plan(_a):
    print("step | shipped script | where | hours | peak GB | note")
    for s in STEPS:
        print(f"  {s[0]} | {s[1]} | {s[2]} | {s[3]} | {s[4]} | {s[6]}")
    kag = sum(s[3] for s in STEPS[:4])
    print(f"""
Totals. Kaggle CPU arms (steps 1 and 2): {kag:.1f} compute hours; two arms can run in parallel (treat 8.5 h is the critical one).
Local alternative: peak 11 to 12 GB per arm exceeds the 10 GB that final-phase jobs may ask for; the machine cannot run them
beside the base refit (J2, 9 to 10 GB for about 13 h). Not recommended.
Modal alternative for steps 1 and 2: 8 CPU / 28 GB containers, the same wall hours (the Kaggle runs used 4 cores; expect
5 to 7 h for the treat arm); price not measured, the earlier 7-arm views run cost about 2 USD for 3 h, so 12 to 16 container hours is
roughly 8 to 12 USD (paid compute, not launched by any script). Needs approval. Steps 3 and 5: about 2 USD
and no cost. Steps 4 and 5 local: 3 GB slot, minutes, once steps 1 to 3 have written their files.
Order if run: 1a and 2 in parallel -> 3 (needs the J7 Istanbul rows as the served reference) -> 4 -> 5.
Value (docs/PIPELINE.md): the Istanbul pieces are worth about 0.4 s on the board together; step 5 alone 0.05 s (v156).""")


def preflight(a):
    months = [int(m) for m in a.months.split(",")]
    print("preflight for months", months)
    for s in STEPS:
        for f in s[5]:
            f = f.replace("_02", f"_{months[0]:02d}") if months else f
            print(f"  step {s[0][:2]} needs {f}: {'present' if (REPO / f).exists() else 'MISSING'}")
    for f in ("archive/agent_ltreg_feat.py", "archive/agent_ltfm_seqfeat.py", "data/interim/agent_istpcd_cols.npz", "data/interim/agent_lv_handover.npz",
              "data/interim/agent_istnext_pool/limits.npz", "data/interim/agent_combo_v129_noM4_final.parquet"):
        print(f"  stored input {f}: {'present' if (REPO / f).exists() else 'MISSING'}")
    print("""
Row-count and ranking-path lines that must be re-pointed (from the runbook's audit): agent_kg_ltfm_final.py:74,132;
agent_ltreg_final.py:93,154; agent_kg_ltfm_kaggle.py:72; agent_kg_ltfm_delta.py:41,73; agent_ltreg_build.py:61,81;
agent_lv_members.py:151,171; agent_lv_07_handover.py:81; agent_istnext_04_serve.py (ranking.parquet read for the time column,
agent_bndl_rank_ltfm.parquet and v137 as fixed names).""")


def route(_a):
    print("""ISTANBUL ROUTE (decided 2026-10-08)
Arms: the shipped wrappers (archive/agent_kg_ltfm_final.py, agent_ltreg_final.py) keep their patching of prc.final, now pointed at
final/10_base.py (per-month served frame): final/ist_kg_ltfm_final.py and final/ist_ltreg_final.py, generated by final/43_gen_wrappers.py.
Dry runs (sampled, under 3 GB) pass: 80 features for the control arm, 104 for the snow arms, training frame and served frame build.
Treatment arm NOT needed: the shipped steps telescope (base + (treat - ctrl) + (snow - treat) = base + (snow - ctrl)): final/43_ist_build.py.
Reuse of data/final/J2_members: NO. The members are depth-10 / blend / 5-seed models of the whole base (fingerprint.json argv); the
Istanbul arms are depth-8, 80/104-feature, 1-3 seed CatBoost fits on a different feature list. Nothing is shared.
Arms and shipped sizes: ctrl (seeds 3) 4.2 h, 9.6 GB; snow k0 (seed 1) 2.25 h, 11.2 GB; snow k12 (seeds 2) 3.1 h, 11.2 GB.
  Kaggle free CPU (prepared, not pushed): python final/43_kaggle_prepare.py -> 3 kernels; about 5.5 h wall with two in parallel, 9.6 h in series; no cost.
  Modal (modal/final_istanbul_app.py): three containers of 5 cores / 16 GB in parallel, same 4 threads as shipped (paired fits must share the
  platform and thread count): about 4.2 h wall, 9.6 container-hours, about 3.5 USD (estimate). Views members (launch_lv): 2 containers of 8 cores /
  24 GB, about 1 USD.
Wiring (all pieces written; env FINAL_RANKING=data/raw/final_ranking.parquet FINAL_MONTHS=2,6):
  1 bulk arms (launcher of the operator's choice)            -> preds_ctrl / preds_snow_k0 / preds_snow_k12 (data/final/ist/)
  2 python final/43_ist_build.py validate / build         -> data/final/J3_ist.parquet (base = J7 value of the Istanbul rows)
  3 views members (modal launch_lv, or ist_lv_members.py)  -> data/final/ist/lv/ltfmm_{ctrl,treat}_srv_99_98_*.npz
  4 python final/43_ist_serve_inputs.py --base data/final/J3_ist.parquet --flags data/final/J9_serve/serve_flags.parquet
                                                           -> data/final/ist_serve/ (needs final/40's rankfeat_<mm>.parquet)
  5 FINAL_IST_SERVE=data/final/ist_serve FINAL_IST_BASE=data/final/J3_ist.parquet FINAL_LV_DIR=data/final/ist/lv python final/ist_corrector_serve.py
      --arm AQCEHSBR --keys agent_lv_keys8.npz --olddelta agent_lv3a_ltfm_final_delta.npz --base data/interim/agent_lv3a_ltfm_final.parquet
      --cfg S2 --k 1.0 --anom A --seeds 0,1,2 --cols D1 --out J11_D1      -> data/final/ist_serve/J11_D1_term.npz (about 5 GB slot, 15 min)
  Istanbul value = J3_ist + term (snow rows 0); the shipped build is agent_istnext_05_build.py.
Choices made on January/July: the 20 key cells, views arm AQCEHSBR, fold months 1-4 and 7-10, the snow definition.""")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "preflight", "route"])
    ap.add_argument("--months", default="2,6")
    a = ap.parse_args()
    {"plan": plan, "preflight": preflight, "route": route}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
