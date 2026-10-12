"""experiment sb2 compose (docs/LAB-NOTES.md): candidate = max(v122 + (S_treat - S_ctrl) + c2, 0) at
the 9 airports, LTFM and every other row = v122 exactly; serving guard (2026 / fold sum dp^2 per row <= 2.0).

    python archive/agent_sb2_compose.py --name bundle --tokens mw,osm --c2 1
S_* per airport: EGLL <- _a (bx,big on TX), LEMD <- _b (seq,bx,big on TX), LFPG/LIRF <- _c (seq,big on TX),
M = EDDF,EDDM,EHAM,LEBL,LSZH <- _d (bx,big on CAND_E9f, v105sc stage). Writes data/interim/agent_sb2_cand.parquet
and data/interim/agent_sb2_guard.json; prints GUARD=PASS|FAIL. v129 is written by the driver only after the guard + lens.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")
import numpy as np
import polars as pl

from archive.agent_sb2_calib import apply_table, fit_table

C = pl.col
I = "data/interim/"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
M = ["EDDF", "EDDM", "EHAM", "LEBL", "LSZH"]
PIECE = {"EGLL": "a", "LEMD": "b", "LFPG": "c", "LIRF": "c", **{a: "d" for a in M}}
PROD = {"a": I + "agent_restack5_x_cap_evt_trk_bx_big_final.parquet", "b": I + "agent_restack5_x_cap_evt_trk_seq_bx_big_final.parquet",
        "c": I + "agent_restack5_b_cap_evt_trk_seq_big_final.parquet", "d": I + "agent_stagecap_9s_bx_big_final.parquet"}
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--tokens", default="")
ap.add_argument("--c2", type=int, default=0)
ap.add_argument("--limit", type=float, default=2.0)
ap.add_argument("--ctrl-prod", type=int, default=0)  # 1: S_ctrl = the production files (C6 serving, s3)
a = ap.parse_args()
res = {"name": a.name, "tokens": a.tokens, "c2": a.c2}

v122 = pl.read_parquet(I + "v122_stage9s_final.parquet").with_columns(C(ID).cast(pl.Float64)).sort(ID)
ids = v122[ID].to_numpy()
base = v122[T].cast(pl.Float64).to_numpy()
hd = pl.DataFrame({ID: ids}).join(pl.read_parquet(I + "agent_sb2_hdow_2026.parquet"), on=ID, how="left", maintain_order="left")
apt = np.asarray(hd["apt"].fill_null("?").to_list())
hour = hd["hour"].fill_null(-1).to_numpy().astype(int)
dow = hd["dow"].fill_null(-1).to_numpy().astype(int)
assert len(ids) == 344_841 and (apt != "?").all(), "2026 airport lookup incomplete"


def rd(p):
    f = pl.read_parquet(p).with_columns(C(ID).cast(pl.Float64)).sort(ID)
    assert (f[ID].to_numpy() == ids).all(), p
    return f[T].cast(pl.Float64).to_numpy()


delta = np.zeros(len(ids))
if a.tokens:
    res["ctrl_repro_rows_differing"] = {}
    for pc in "abcd":
        sc = rd(PROD[pc]) if a.ctrl_prod else rd(I + f"agent_sb2_c_{pc}_final.parquet")
        st = rd(I + f"agent_sb2_t_{pc}_final.parquet")
        res["ctrl_repro_rows_differing"][pc] = int((sc != rd(PROD[pc])).sum())
        for x, p in PIECE.items():
            if p == pc:
                k = apt == x
                delta[k] = st[k] - sc[k]
    print("S_ctrl (sb2 no-token builds) vs production files, rows differing: " + str(res["ctrl_repro_rows_differing"]))
c2 = np.zeros(len(ids))
if a.c2:
    z = np.load(I + f"agent_sb2_{a.name}_preC2_s0.npz", allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    fid, fy, fp, fap = z["ids"][o].astype(float), z["y"][o].astype(float), z["both_guard"][o].astype(float), z["apts"][o].astype(str)
    fh = (pl.DataFrame({ID: fid}).join(pl.read_parquet(I + "agent_sb2_hdow_2025.parquet"), on=ID, how="left", maintain_order="left"))
    fhour, fdow = fh["hour"].fill_null(-1).to_numpy().astype(int), fh["dow"].fill_null(-1).to_numpy().astype(int)
    m = np.isin(fap, AIRPORTS) & (fhour >= 0)
    tab, ah, kap = fit_table(fy[m] - fp[m], fap[m], fhour[m], fdow[m])
    k9 = np.isin(apt, AIRPORTS) & (hour >= 0)
    c2[k9] = apply_table(tab, ah, apt[k9], hour[k9], dow[k9])
    res["c2_kappas"], res["c2_rms"] = kap, float(np.sqrt((c2 ** 2).mean()))
    print(f"C2 serving: kappas {kap}; rms correction {res['c2_rms']:.2f} s on 2026 rows")
at9 = np.isin(apt, AIRPORTS)
cand = np.where(at9, np.maximum(base + delta + c2, 0.0), base)
cand = np.rint(cand)
assert (cand[~at9] == base[~at9]).all()
out = v122.with_columns(pl.Series(T, cand).cast(v122[T].dtype))
out.write_parquet(I + "agent_sb2_cand.parquet")
dp = cand - base
vj = json.load(open(I + "agent_sb2_verdict.json"))
fold = vj[a.name]
ratio = float((dp ** 2).sum() / len(dp)) / fold["fold_dp2_per_row"]
res["dp2_2026_per_row"], res["dp2_fold_per_row"], res["ratio"] = float((dp ** 2).sum() / len(dp)), fold["fold_dp2_per_row"], ratio
res["per_airport"] = {}
for x in AIRPORTS:
    k = apt == x
    f26 = float((dp[k] ** 2).sum() / max(k.sum(), 1))
    ff = fold["fold_dp2_by_apt"][x] / max(fold["fold_rows_by_apt"][x], 1)
    res["per_airport"][x] = {"moved": int((dp[k] != 0).sum()), "mean_dp": float(dp[k].mean()), "dp2_26": f26, "dp2_fold": ff,
                             "ratio": f26 / ff if ff > 0 else float("nan")}
print(f"moved {int((dp != 0).sum()):,} rows; mean shift {dp.mean():+.3f} s; floor binds on {int(((base + delta + c2) < 0).sum())} rows; "
      f"LTFM/other rows untouched: {bool((dp[~at9] == 0).all())}")
print(f"GUARD like-for-like: 2026 sum dp^2/row {res['dp2_2026_per_row']:.2f} vs fold {res['dp2_fold_per_row']:.2f} -> ratio {ratio:.3f} "
      f"(limit {a.limit})")
print("per airport ratio: " + "  ".join(f"{x} {v['ratio']:.2f}" for x, v in res["per_airport"].items()))
res["guard_pass"] = ratio <= a.limit
json.dump(res, open(I + "agent_sb2_guard.json", "w"), indent=1, default=float)
print(f"GUARD={'PASS' if res['guard_pass'] else 'FAIL'}")
