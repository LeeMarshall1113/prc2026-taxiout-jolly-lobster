"""Stage 43c: serve-side inputs of the Istanbul views corrector and the Istanbul corrector (nine-airport column lists, D1) for the served months.

Rebuilds, for the Istanbul departures of the chosen months, the serve halves of four stored January/July files; the 2025 halves are
copied unchanged. Each part is a parameterised copy of its shipped script:
  agent_lv_keys8.npz        <- archive/agent_lks_01_build.py prep()      (20 hashed cells; serve rows appended to the stored fold rows)
  agent_istpcd_cols.npz     <- archive/agent_istpcd_01_cols.py           (nine plan-clock distance columns, current prediction = the base)
  agent_lv_handover.npz     <- archive/agent_lv_07_handover.py feats()   (stand-handover columns; arrivals of the ranking file)
  rank_ltfm.parquet         <- agent_bndl_rank_ltfm.parquet              (rank features of the Istanbul departures, one month at a time)
Output directory data/final/ist_serve/ (name of each file as the shipped serve script reads it).

  --base   parquet (MVT_ID_mvt, TAXITIME_SEC_mvt): the Istanbul chain value before any corrector and before M4
           (= data/final/J3_ist.parquet of final/43_ist_build.py; the shipped agent_combo_v129_noM4_final)
  --flags  parquet (id, flag) AOBT_3 flags of the served rows (final/40 writes <tag>_serve/serve_flags.parquet)
    python final/43_ist_serve_inputs.py --months 2,6 --base data/final/J3_ist.parquet --flags data/final/J9_serve/serve_flags.parquet
    python final/43_ist_serve_inputs.py --ranking data/raw/ranking.parquet --months 1,7 --base data/interim/agent_combo_v129_noM4_final.parquet \
        --rank-cache data/interim/agent_bndl_rank_ltfm.parquet --flags data/final/_chk/stored_flags26.parquet --outdir data/final/_chk/ist_serve --check
Prints serve rows per month. --check compares every array with the stored January/July files by id (reproduction test).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "archive"))
import agent_lv_07_handover as H  # noqa: E402

C = pl.col
ID = "MVT_ID_mvt"
I = REPO / "data" / "interim"
NAMES9 = ["d_SCHED", "d_LOBT", "d_IOBT", "d_EOBT", "d_AOBT3", "clamp", "wmin", "d_AOBT3_ind", "d_AOBT3_der"]


def prep(df):
    """Copy of the prep() of archive/agent_lks_01_build.py (same names, same hash)."""
    s = lambda c: C(c).cast(pl.Utf8).fill_null("NA")  # noqa: E731
    cs = s("CALLSIGN_flt")
    st = s("STAND_mvt")
    hb = (C("MVT_TIME_UTC_mvt").dt.hour() // 3).cast(pl.Utf8)
    pre = cs.str.extract(r"^([A-Z]{3})", 1).fill_null("NA")
    num = cs.str.extract(r"^[A-Z]{3}(\d+)$", 1)
    csfn = pl.when(num.is_not_null()).then(pre + pl.lit("#") + (num.cast(pl.Int64, strict=False) // 100).cast(pl.Utf8)).otherwise(pre + pl.lit("@") + cs.str.len_chars().cast(pl.Utf8))
    area = st.str.extract(r"^([A-Z]+)", 1).fill_null(st.str.slice(0, 1) + pl.lit("n") + st.str.len_chars().cast(pl.Utf8))
    pier = st.str.extract(r"^([A-Z]?\d)", 1).fill_null(st.str.slice(0, 2))
    op, ty, rw, ad = s("AIRCRAFT_OPERATOR_flt"), s("AIRCRAFT_TYPE_mvt"), s("RUNWAY_mvt"), s("ADES_mvt")
    ex = {"op": [op], "csp": [pre], "csfn": [csfn], "ades": [ad], "cty": [ad.str.slice(0, 2)], "reg": [ad.str.slice(0, 1)], "stand": [st], "area": [area], "pier": [pier],
          "type": [ty], "wk": [s("WK_TBL_CAT_flt")], "mkt": [s("MARKET_SEGMENT_flt")], "rwy_stand": [rw, st], "rwy_area": [rw, area], "op_hb": [op, hb], "ty_stand": [ty, st],
          "op_stand": [op, st], "op_rwy": [op, rw], "rwy_hb": [rw, hb], "op_ty": [op, ty]}
    return df.select([C(ID)] + [pl.concat_str(v, separator="|").hash(seed=7).floordiv(2).cast(pl.Int64).alias(k) for k, v in ex.items()]), list(ex)


def pcd_build(pv, g, flag):
    """Copy of build() of archive/agent_istpcd_01_cols.py."""
    d = {k: pv - g[k] for k in ("sched", "lobt", "iobt", "eobt", "aobt")}
    gs = g["sched"]
    clamp = np.where((gs >= 5000) & (gs < 50000), gs - pv, np.nan)
    wmin = 3606.0 - np.abs(d["lobt"])
    ind = np.where(flag == 1, d["aobt"], np.nan)
    der = np.where(flag == 2, d["aobt"], np.nan)
    return np.column_stack([d["sched"], d["lobt"], d["iobt"], d["eobt"], d["aobt"], clamp, wmin, ind, der]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(REPO / "data/raw/final_ranking.parquet"))
    ap.add_argument("--months", default="2,6")
    ap.add_argument("--base", required=True)
    ap.add_argument("--flags", required=True)
    ap.add_argument("--rank-cache", default=None, help="parquet of rank features of the Istanbul rows (default data/final/rankfeat_<mm>.parquet of final/40)")
    ap.add_argument("--outdir", default=str(REPO / "data/final/ist_serve"))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    months = [int(m) for m in a.months.split(",")]
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    base = pl.read_parquet(a.base).with_columns(C(ID).cast(pl.Float64)).sort(ID)
    if a.rank_cache:
        rl = pl.read_parquet(a.rank_cache)
    else:
        rl = pl.concat([pl.read_parquet(REPO / f"data/final/rankfeat_{m:02d}.parquet") for m in months])
    rl = rl.filter(C("ADEP_mvt") == "LTFM").with_columns(C(ID).cast(pl.Float64)).unique(ID).sort(ID)
    rl = rl.filter(C("month").is_in(months)) if "month" in rl.columns and not a.rank_cache else rl
    rl = rl.join(base.select(ID, C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("noM4")), on=ID, how="inner")
    sid = rl[ID].to_numpy()
    assert np.isfinite(rl["noM4"].to_numpy()).all()
    rl.drop("noM4").write_parquet(out / "rank_ltfm.parquet")
    mon_s = rl["month"].to_numpy()
    print("Istanbul serve rows per month:", {int(m): int((mon_s == m).sum()) for m in sorted(set(mon_s.tolist()))})

    # keys
    cols = [ID, "CALLSIGN_flt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "PHASE_mvt"]
    rk = pl.read_parquet(a.ranking, columns=cols).with_columns(C(ID).cast(pl.Float64)).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM"))
    ks, names = prep(rk)
    z = dict(np.load(I / "agent_lv_keys8.npz", allow_pickle=True))
    kn = [str(x) for x in z["kn"]]
    Ks = pl.DataFrame({ID: sid}).join(ks.unique(ID), on=ID, how="left", maintain_order="left").select(kn).to_numpy().astype(np.int64)
    assert not np.isnan(Ks.astype(float)).any()
    nf = len(z["fid"])
    old_sid, old_K = z["sid"], z["Kall"]
    z["Kall"] = np.vstack([old_K[:nf], Ks])
    z["sid"] = sid
    np.savez(out / "agent_lv_keys8.npz", **z)

    # plan-clock columns
    flg = pl.read_parquet(a.flags)
    flg = flg.rename({"id": ID}) if "id" in flg.columns else flg
    flag_s = pl.DataFrame({ID: sid}).join(flg.with_columns(C(ID).cast(pl.Float64)).unique(ID), on=ID, how="left", maintain_order="left")["flag"].cast(pl.Float64).to_numpy()
    g = lambda n: rl[n].cast(pl.Float64).to_numpy()  # noqa: E731
    gaps = dict(sched=g("gap_sched"), lobt=g("gap_lobt"), iobt=g("gap_lobt") - g("clk_iobt_minus_lobt"), eobt=g("gap_eobt"), aobt=g("gap_aobt"))
    Ds = pcd_build(rl["noM4"].to_numpy(), gaps, flag_s)
    zc = np.load(I / "agent_istpcd_cols.npz", allow_pickle=True)
    np.savez(out / "agent_istpcd_cols.npz", D=zc["D"], Ds=Ds, names=zc["names"], fid=zc["fid"], sid=sid, mon_s=mon_s)

    # stand-handover columns
    rkh = pl.read_parquet(a.ranking, columns=H.cols).with_columns(C(ID).cast(pl.Float64))
    rkh = rkh.filter((C("ADEP_mvt") == "LTFM") | (C("ADES_mvt") == "LTFM"))
    arr = H.arrivals(rkh)
    sd = pl.DataFrame({ID: sid, "ref": rl["noM4"].to_numpy()}).join(
        rkh.filter(C("PHASE_mvt") == "DEP").select(ID, H.ep("MVT_TIME_UTC_mvt").alias("mvt"), H.ep("SCHED_TIME_UTC_mvt").alias("sched"), C("STAND_mvt").alias("stand")).unique(ID),
        on=ID, how="left", maintain_order="left").with_columns(H.area("stand").alias("area"))
    miss = sd["mvt"].is_null().to_numpy() | sd["stand"].is_null().to_numpy() | sd["sched"].is_null().to_numpy()
    sd = sd.with_columns(C("mvt").fill_null(0), C("sched").fill_null(0))
    XS, hn = H.feats(sd, arr, sd["ref"].to_numpy())
    XS[miss] = np.nan
    ho = np.load(I / "agent_lv_handover.npz", allow_pickle=True)
    np.savez(out / "agent_lv_handover.npz", fold=ho["fold"], serve=XS.astype(np.float32), names=np.array(hn), fold_ids=ho["fold_ids"], serve_ids=sid)
    print("wrote", out, "| serve NaN share plan-clock", np.round(np.isnan(Ds).mean(0), 3).tolist(), "| handover coverage", np.round(np.isfinite(XS).mean(0), 3).tolist())

    if a.check:
        pos = {v: i for i, v in enumerate(old_sid.tolist())}
        i = np.array([pos[v] for v in sid.tolist()])
        def eq(x, y):
            x, y = np.asarray(x, float), np.asarray(y, float)
            bad = ~((x == y) | (np.isnan(x) & np.isnan(y)))
            both = np.isfinite(x) & np.isfinite(y)
            return (float(np.abs(x[both] - y[both]).max()) if both.any() else 0.0), int(bad.reshape(len(x), -1).any(1).sum())
        print("\nREPRODUCTION vs stored January/July files (by id, %d rows): array | max abs difference | differing rows" % len(sid))
        print("  keys Kall (20 cells)      ", eq(Ks, old_K[nf:][i]))
        zs = np.load(I / "agent_istpcd_cols.npz", allow_pickle=True)
        pos2 = {v: k for k, v in enumerate(zs["sid"].tolist())}
        j = np.array([pos2[v] for v in sid.tolist()])
        print("  plan-clock Ds (9)         ", eq(Ds, zs["Ds"][j]))
        hs = np.load(I / "agent_lv_handover.npz", allow_pickle=True)
        pos3 = {v: k for k, v in enumerate(hs["serve_ids"].tolist())}
        k3 = np.array([pos3[v] for v in sid.tolist()])
        print("  handover serve (6)        ", eq(XS.astype(np.float32), hs["serve"][k3]))


if __name__ == "__main__":
    main()
