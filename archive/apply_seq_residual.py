"""Sequence residual post-step (docs/LAB-NOTES.md). Moves ONLY rows the ADS-B experiment never touches.

    python archive/apply_seq_residual.py --in submissions/jolly-lobster_v108.parquet --out data/interim/v109_seq_final.parquet
    python archive/apply_seq_residual.py --selftest            # 2025: reproduce the fold OOF for months 3 and 9

Train: in-scope untouched rows of the six current-lineage months (3,4,6,9,10,12) of results/agent_hedge_folds.parquet,
target r = y - s8 winsorised at +-3,600 s; features = archive/agent_seqpost_feat.py SEQ + airport + UTC hour, each training
row built with T fitted on 2025 months other than its own. LightGBM params and seeds exactly as the fold test.
Serve: 2026 departures (data/raw/ranking.parquet), T from all of 2025, level columns demeaned by (airport, month) over the
2026 frame. Moved rows = untouched (not in data/interim/agent_offblock_rows_2026.parquet, the dump every ADS-B stage and
re-stack acts on) AND not rule-placed in the v103 build (day-early, Rome plan gate, gap clamp, v35, v36 upper bound, hidden
reading, EDDM January, monster step). new = in + clip(f, +-600), LOBT box re-applied on plan rows, floor 0, rounded.
--airports A,B restricts the moved rows to those airports (the pre-registered splice realisation).
"""
from __future__ import annotations

import argparse
import glob
import sys

sys.path.insert(0, ".")

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

import archive.agent_seqpost_feat as F  # noqa: E402

C = pl.col
ID, T_ = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
I = "data/interim/"
MONTHS = [3, 4, 6, 9, 10, 12]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH", "LTFM"]
APT_CODE = {a: i for i, a in enumerate(APTS)}
P = dict(objective="regression", learning_rate=0.03, n_estimators=400, num_leaves=15, min_child_samples=300,
         subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0, verbose=-1, n_jobs=4)
FCOLS = F.SEQ + ["apt_code", "hour"]
W_BOX = 3620.0


def train_scope():
    hf = (pl.read_parquet("results/agent_hedge_folds.parquet", columns=[ID, "month", "apt", "y"] + [f"s{i}" for i in range(9)])
          .filter(C("month").is_in(MONTHS)))
    z = np.load(I + "modal/TX_S6.npz", allow_pickle=True)
    dump = pl.read_parquet(I + "agent_offblock_rows_c238.parquet", columns=[ID])[ID]
    hf = hf.with_columns(
        ((C("apt") == "LTFM") | (C(ID).is_in(pl.Series(z["ids"]).implode()) & ~C(ID).is_in(dump.implode()))).alias("untouched"),
        (((C("s1") - C("s0")).abs() > 0.5) | ((C("s2") - C("s1")).abs() > 0.5) | ((C("s5") - C("s4")).abs() > 0.5) | ((C("s6") - C("s5")).abs() > 0.5) | ((C("s7") - C("s6")).abs() > 0.5)
         | ((C("s8") - C("s7")).abs() > 0.5)).alias("ruled"))
    return hf.filter(C("untouched") & ~C("ruled")).select(ID, "month", "apt", "y", "s8")


def design(rows: pl.DataFrame, feats: pl.DataFrame, hour: pl.DataFrame) -> np.ndarray:
    d = (rows.select(ID, "apt").join(feats, on=ID, how="left", maintain_order="left")
         .join(hour, on=ID, how="left", maintain_order="left")
         .with_columns(C("apt").replace_strict(APT_CODE, return_dtype=pl.Int32).alias("apt_code")))
    assert d.height == rows.height
    return d.select(FCOLS).to_numpy().astype(np.float32)


def fit_predict(Xtr, rtr, Xte):
    r = np.clip(rtr, -3600, 3600)
    ps = []
    for seed in (0, 1):
        m = lgb.LGBMRegressor(**P, random_state=seed)
        m.fit(Xtr, r, categorical_feature=[FCOLS.index("apt_code")])
        ps.append(m.predict(Xte))
    return np.clip(np.mean(ps, axis=0), -600, 600)


def tt_excluding(dep25, dep, excl_fn):
    """Per-row planned taxi for frame `dep`: row in month j uses T fitted on 2025 months not in excl_fn(j)."""
    mon = dep["month"].to_numpy()
    tt = np.empty(dep.height)
    cache = {}
    for j in np.unique(mon):
        ex = tuple(sorted(excl_fn(int(j))))
        if ex not in cache:
            cache[ex] = F.fit_T(dep25.filter(~C("month").is_in(list(ex))))
        k = mon == j
        tt[k] = F.apply_T(dep.filter(pl.Series(k)), cache[ex])
    return tt


def selftest():
    dep, arr = F.load_movements(sorted(glob.glob("data/raw/training_2025-*.parquet")))
    sc = train_scope()
    oof = pl.read_parquet("results/agent_seqpost_oof.parquet", columns=[ID, "f", "scope"]).filter(C("scope"))
    hour = dep.select(ID, "hour")
    worst = 0.0
    for m in (3, 9):
        feats = F.build(dep, arr, tt_excluding(dep, dep, lambda j: {m, j}), demean=True)
        tr, te = sc.filter(C("month") != m), sc.filter(C("month") == m)
        f = fit_predict(design(tr, feats, hour), (tr["y"] - tr["s8"]).to_numpy(), design(te, feats, hour))
        ref = te.select(ID).join(oof, on=ID, how="left", maintain_order="left")["f"].to_numpy()
        diff = float(np.nanmax(np.abs(f - ref)))
        worst = max(worst, diff)
        print(f"selftest month {m}: {te.height:,} rows, max |f - fold OOF f| = {diff:.3g} s; mean f {f.mean():+.2f}")
    ok = worst < 1e-6
    print("SELFTEST", "PASS (serve code reproduces the judged fold predictions)" if ok else "FAIL")
    return 0 if ok else 1


def serve(inp, out, airports):
    dep25, arr25 = F.load_movements(sorted(glob.glob("data/raw/training_2025-*.parquet")))
    sc = train_scope()
    feats25 = F.build(dep25, arr25, tt_excluding(dep25, dep25, lambda j: {j}), demean=True)
    X = design(sc, feats25, dep25.select(ID, "hour"))
    r = (sc["y"] - sc["s8"]).to_numpy()
    del feats25
    dep26, arr26 = F.load_movements(["data/raw/ranking.parquet"])
    T_all = F.fit_T(dep25)
    feats26 = F.build(dep26, arr26, F.apply_T(dep26, T_all), demean=True)
    print(f"train rows {sc.height:,}; 2026 departures {dep26.height:,}, arrivals {arr26.height:,}")

    sub = pl.read_parquet(inp)
    assert sub[ID].n_unique() == sub.height == 344_841
    rk = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(C(ID), C("ADEP_mvt").alias("apt"), C("AOBT_3_flt").is_not_null().alias("plan"),
                  (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl")).collect())
    dump = pl.read_parquet(I + "agent_offblock_rows_2026.parquet", columns=[ID])[ID]

    def col(p, name):
        return pl.read_parquet(p).select(ID, C(T_).cast(pl.Float64).alias(name))

    v = rk
    for p, nm in [(I + "v94_model.parquet", "model"), (I + "v103_base.parquet", "base"), (I + "v103_gatebag_s0.parquet", "gb0"),
                  (I + "v103_raw.parquet", "raw"), (I + "v103_p2.parquet", "p2")]:
        v = v.join(col(p, nm), on=ID, how="left", maintain_order="left")
    for ch in ("nl", "li"):
        for stg in ("s2", "s35", "s4", "s5", "s6", "final"):
            v = v.join(col(f"{I}v103_{ch}_{stg}.parquet", f"{ch}_{stg}"), on=ID, how="left", maintain_order="left")
        v = v.join(col(f"{I}v103_s1.parquet" if ch == "nl" else f"{I}v103_gatebag_s1.parquet", f"{ch}_s1"), on=ID,
                   how="left", maintain_order="left")
    ne = lambda a, b: (C(a) - C(b)).abs() > 0.5  # noqa: E731

    def chain_ruled(ch):
        return ne(f"{ch}_s2", f"{ch}_s1") | ne(f"{ch}_s4", f"{ch}_s35") | ne(f"{ch}_s5", f"{ch}_s4") | ne(f"{ch}_s6", f"{ch}_s5") \
            | ne(f"{ch}_final", f"{ch}_s6")

    v = v.with_columns(
        (~C(ID).is_in(dump.implode())).alias("untouched"),
        (ne("base", "model") | ne("p2", "raw")
         | pl.when(C("apt") == "LIRF").then(ne("gb0", "base") | chain_ruled("li")).otherwise(chain_ruled("nl"))).alias("ruled"))
    v = v.with_columns((C("untouched") & ~C("ruled").fill_null(True)).alias("scope"))
    if airports:
        v = v.with_columns(C("scope") & C("apt").is_in(airports))
    print(v.group_by("apt").agg(n=pl.len(), untouched=C("untouched").sum(), ruled=(C("untouched") & C("ruled")).sum(),
                                scope=C("scope").sum()).sort("apt"))
    scope = v.filter(C("scope")).select(ID, "apt")
    f = fit_predict(X, r, design(scope, feats26, dep26.select(ID, "hour")))
    j = (sub.join(v.select(ID, "apt", "plan", "gl", "scope"), on=ID, how="left", maintain_order="left")
         .join(scope.select(ID).with_columns(pl.Series("f", f)), on=ID, how="left", maintain_order="left"))
    p = j[T_].cast(pl.Float64).to_numpy()
    ff = j["f"].fill_null(0.0).to_numpy()
    gl, plan, sc_ = j["gl"].to_numpy(), j["plan"].fill_null(False).to_numpy(), j["scope"].fill_null(False).to_numpy()
    new = p + ff
    box = plan & np.isfinite(gl)
    new = np.where(box, np.clip(new, np.ceil(gl - W_BOX), np.floor(gl + W_BOX)), new)
    new = np.where(sc_, np.clip(np.rint(new), 0, None), p)
    res = sub.with_columns(pl.Series(T_, new).cast(sub[T_].dtype))
    res.write_parquet(out)
    moved = np.abs(new - p) > 0.5
    un = j.select(ID).with_columns(C(ID).is_in(dump.implode()).alias("d"))["d"].to_numpy()
    assert not (moved & un).any(), "moved a row the ADS-B experiment touches"
    apt = j["apt"].to_numpy()
    print(f"wrote {out}: moved {int(moved.sum()):,} rows (all untouched); mean shift {float((new - p)[moved].mean()):+.2f} s, "
          f"rms shift {float(np.sqrt(((new - p)[moved] ** 2).mean())):.1f} s")
    for a in APTS:
        k = moved & (apt == a)
        if k.any():
            print(f"  {a}: moved {int(k.sum()):>6,}  mean dp {float((new - p)[k].mean()):+7.2f}  rms dp {float(np.sqrt(((new - p)[k] ** 2).mean())):6.1f}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp")
    ap.add_argument("--out")
    ap.add_argument("--airports", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    assert a.inp and a.out, "--in and --out are required"
    sys.exit(serve(a.inp, a.out, [x for x in a.airports.split(",") if x]))
