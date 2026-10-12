"""Stage 40: serve-side inputs of the residual correctors for the departures of chosen months of a ranking file.

Derived from the scripts that built these inputs for January and July 2026 (the shipped script of each part is named
next to it); nothing here is re-tuned, every list, map and constant is read from the stored 2025-side files:

  serve_in   archive/agent_bigc_v2_36_serve_prep.py + agent_refit2_04_serve.py (base columns, extra columns, drift)
  extra      archive/agent_m12_bigc_inputs.py extra_frame()   (52 columns of push-back, track, map, occupancy, sequence)
  d_NEW/d_ET archive/agent_bigc_v2_10_extra2.py               (differences of two re-stack variants, per airport group)
  keys       archive/agent_bigc_v2_15_keys.py, agent_bigc_v2_20_keys2.py   (hashed cells)
  drift      archive/agent_bigc_drift.py drift_serve()
  flags      archive/agent_aobtflag_01_flag.py apply()        (modes of 2025, stored)
  pcd        archive/agent_pcd_01_cols.py                     (nine plan-clock distance columns)
  time       archive/agent_arith3_05_servecols.py make_time()

Output directory <outdir>/<tag>_serve/ holds the SERVE-SIDE arrays only, under the file names the shipped fit code reads
(final/41_corrector_serve.py links the 2025-side files next to them):

  agent_bigc13_serve_in.npz   Xs_pre (n, 194), cols_pre, Dsv (n, 8), mon_s, pv26, apt, is9, ids
  serve_keys.npz              Ks (n, 11) [keys.npz names], Ks2 (n, 16), Ks_d (n, 4), apt_s
  serve_views.npz             Xs (n, 52)  variant views (see --views)
  agent_pcd_cols_app.npz      Ds (n, 9), ids_s
  serve_flags.parquet         id, yr=2026, flag, K
  agent_arith3_time_serve.npz t, ids
Rows are sorted by MVT_ID_mvt. A departure's month is the UTC month of its take-off time.

Inputs that come from other stages (profile 'final' = files written by final/20 to final/22 and final/3x):
  --pv        parquet (MVT_ID_mvt, pv): the value the corrector is served on, i.e. the chain before the corrector, the
              neighbour blend and the tail step (the B26 of archive/agent_refit2_04_serve.py).
  --profile   where the ADS-B tables and the variant files are. A missing table gives NaN columns and a printed warning.
  --views     npz with ids and Xs (52 variant-view columns); without it the 52 columns are NaN and a warning is printed.

    python final/40_serve_inputs.py --months 2,6 --pv data/final/J8_pv.parquet            (real run, profile final)
    python final/40_serve_inputs.py --ranking data/raw/ranking.parquet --months 1,7 --profile stored --pv <parquet of the
        stored pv26> --views data/final/_chk/stored_views.npz --check     (reproduction of the stored January/July files)

--sample N keeps the first N departures of each airport (sorted by id) so the whole path runs in seconds.
Memory: the rank-feature build is the largest step (about 2 GB per calendar month); run under the slot and the job lock of
final/README.md, one month per call (the per-month feature file is cached in <outdir>).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fserve as F  # noqa: E402
from fserve import AIR9, C, ID, INTERIM, OUT, RAW, REPO, parse_months  # noqa: E402

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "archive"))
import agent_bigc_drift as DR  # noqa: E402

I = "data/interim/"
T0 = time.time()
LOW = ["ADEP_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "airport_plan", "RUNWAY_mvt"]
HIGH = ["STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "callsign_op"]
OBC = ["has_c", "P_span", "P_n", "P_gap", "P_dstand", "P_long", "seg_n", "seg_rate", "seg_start_gap", "push", "first_gs", "first_dstand",
       "revany_gap", "rev_dist", "rev_gs", "rev_dstand"]
ECS = ["e_stand", "e_appear", "e_ci20", "e_dend", "e_dnext", "e_mv5", "e_mv10", "e_mv20", "e_rev", "e_start", "e_revany", "e_pushend"]
LATE = ["arr_late_share60", "arr_late_share180", "arr_late_mean60", "arr_late_mean180"]
# airport groups of the two variant columns (agent_bigc_v2_10_extra2.py); airports in no group get 0
PARTS = {("EGLL",): "a", ("LEMD",): "b", ("EDDF", "EDDM", "EHAM", "LEBL", "LSZH"): "d", ("LFPG", "LIRF"): "c"}

PROFILES = {
    # the January/July 2026 tables as stored (reproduction test)
    "stored": dict(rank=I + "agent_trans_rank_features.parquet", census=I + "agent_coverage_census_2026.parquet",
                   ob=I + "agent_offblock_rows_2026.parquet", bx=I + "agent_backx_bx_2026.parquet",
                   evt=I + "agent_restack3_evt_2026.parquet", trk=I + "agent_restack3_trk_2026.parquet",
                   osm=I + "agent_sb2_osm_2026.parquet", occ=I + "agent_sb2_occ_2026.parquet", seq=I + "agent_smallbundle_seq_2026.parquet",
                   var="data/interim/agent_{name}_final.parquet"),
    # files of the final-phase stages (J6 dumps of final/20; census, track, map and occupancy tables have no writer yet)
    "final": dict(rank=None, census="data/final/J6_census.parquet", ob="data/final/J6_rows.parquet", bx="data/final/J6_bx.parquet",
                  evt="data/final/J6_evt.parquet", trk="data/final/J6_trk.parquet", osm="data/final/J6_osm.parquet",
                  occ="data/final/J6_occ.parquet", seq="data/final/J6_seq.parquet", var="data/final/J7_{name}.parquet"),
}
# the four variant files per part; {name} of the 'var' template is filled with these and {pc} with the part letter
VARIANTS = dict(bulkr="bulkr_t_{pc}", ref="sb2_t_{pc}", ctrl="etrs_t_{pc}", avg="etrs_t_{pc}_avg")


def P(*a):
    print("[%5.0fs]" % (time.time() - T0), *a, flush=True)


def table(path, label):
    """A parquet keyed by MVT_ID_mvt (a column called id is accepted) as Float64 ids, one row per id; None when missing."""
    if path is None or not Path(path).exists():
        P(f"WARNING: table '{label}' not found ({path}): its columns will be NaN")
        return None
    d = pl.read_parquet(path)
    if ID not in d.columns and "id" in d.columns:
        d = d.rename({"id": ID})
    return d.with_columns(C(ID).cast(pl.Float64)).unique(ID, keep="first", maintain_order=True)


def conv(k, dt):
    """Key of a stored label map (always text) as the dtype of the column it is applied to."""
    if dt == pl.Boolean:
        return k in ("true", "True", "1")
    if dt.is_integer():
        return int(k)
    if dt.is_float():
        return float(k)
    return k


# ---------------------------------------------------------------- rank features of the chosen months
def rank_features(ranking: Path, months: list[int], ids: np.ndarray, outdir: Path, cache: str | None) -> pl.DataFrame:
    """Model features of the departures, built one calendar month at a time like the training features (final/10_base.py)."""
    if cache:
        P("rank features from", cache)
        return pl.read_parquet(cache).with_columns(C(ID).cast(pl.Float64)).join(pl.DataFrame({ID: ids}), on=ID, how="semi").unique(ID)
    from prc.features import build
    parts = []
    for m in sorted(months):
        f = outdir / f"rankfeat_{m:02d}.parquet"
        if f.exists():
            P("rank features month", m, "cached", f.name)
            parts.append(pl.read_parquet(f))
            continue
        P("building rank features month", m)
        raw = pl.scan_parquet(ranking).filter(pl.col("MVT_TIME_UTC_mvt").dt.month() == m).collect()   # departures and arrivals of the month
        part = build(raw, with_target=False)
        del raw
        part.write_parquet(f)
        parts.append(part)
    return pl.concat(parts).with_columns(C(ID).cast(pl.Float64)).unique(ID)


# ---------------------------------------------------------------- arrival lateness (agent_bigc_v2_36_serve_prep.py)
def arrival_lateness(ranking: Path, dep: pl.DataFrame) -> pl.DataFrame:
    rk = pl.read_parquet(ranking, columns=[ID, "PHASE_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"])
    arr = rk.filter((C("PHASE_mvt") == "ARR") & C("ADES_mvt").is_in(AIR9)).select(
        "ADES_mvt", C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t"), C("SCHED_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("ts"))
    del rk
    Sx = dep.select(ID, C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Int64).alias("t"))
    out = []
    for ap in AIR9:
        A = arr.filter(C("ADES_mvt") == ap).sort("t")
        at = A["t"].to_numpy()
        lt = np.clip((A["t"] - A["ts"]).to_numpy().astype(float), -3600, 10800)
        cs_l = np.concatenate([[0], np.cumsum(lt)])
        cs_s = np.concatenate([[0], np.cumsum((lt > 900).astype(float))])
        Dd = Sx.filter(C("apt") == ap)
        Tt = Dd["t"].to_numpy()
        f = {}
        for wn, wl in (("60", 3600), ("180", 10800)):
            hi = np.searchsorted(at, Tt, side="right")
            lo = np.searchsorted(at, Tt - wl, side="right")
            nn = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                f["s" + wn] = np.where(nn >= 3, (cs_s[hi] - cs_s[lo]) / np.maximum(nn, 1), np.nan)
                f["m" + wn] = np.where(nn >= 3, (cs_l[hi] - cs_l[lo]) / np.maximum(nn, 1), np.nan)
        out.append(pl.DataFrame({ID: Dd[ID].to_numpy(), "arr_late_share60": f["s60"], "arr_late_share180": f["s180"],
                                 "arr_late_mean60": f["m60"], "arr_late_mean180": f["m180"]}))
    return pl.concat(out)


# ---------------------------------------------------------------- the 192 base columns
def base_matrix(dep, pv, rank, tabs, cols, excols, maps):
    """Xs of the 140 base columns + 52 extra columns, in the stored column order. dep sorted by id."""
    ids = dep[ID].cast(pl.Float64).to_numpy()
    n = len(ids)
    S = pl.DataFrame({ID: ids}).join(rank.unique(ID), on=ID, how="left", maintain_order="left")
    assert S.height == n and S["ADEP_mvt"].null_count() == 0, "rank features missing for some departures"
    for c in LOW + HIGH:
        mp = {conv(k, S.schema[c]): v for k, v in maps[c].items()}
        S = S.with_columns(C(c).replace_strict(mp, default=-1, return_dtype=pl.Int32).fill_null(-1).alias(c))
    cen = tabs["census"]
    if cen is not None:
        cen = cen.select([ID, "category"]).with_columns(C("category").replace_strict(maps["category"], default=-1, return_dtype=pl.Int32).alias("cov_cat")).select([ID, "cov_cat"])
    ob, bx = tabs["ob"], tabs["bx"]
    if ob is not None:
        for c in ("src", "prod_kind"):
            ob = ob.with_columns(C(c).cast(pl.Utf8).replace_strict(maps["ob_" + c], default=-1, return_dtype=pl.Int32).fill_null(-1).alias("ob_" + c))
        ob = ob.select([ID, "ob_src", "ob_prod_kind"] + OBC + [(C("mvt_t") - C(e)).alias("m_" + e) for e in ECS])
    for o in (cen, ob, bx, tabs["wx"]):
        if o is not None:
            S = S.join(o, on=ID, how="left", maintain_order="left")
    assert S.height == n
    Xs = np.full((n, len(cols) + len(excols)), np.nan, np.float32)
    miss = []
    for j, c in enumerate(cols):
        if c == "pv":
            Xs[:, j] = pv
        elif c in S.columns:
            Xs[:, j] = S[c].cast(pl.Float32).to_numpy()
        else:
            miss.append(c)
    if miss:
        P("WARNING base columns left NaN (their table is missing):", miss)
    # extra columns (agent_m12_bigc_inputs.py extra_frame, serve branch)
    parts = []
    srcmap = {v: i for i, v in enumerate(sorted(set(pl.read_parquet(INTERIM / "agent_restack3_evt_2025.parquet", columns=["src"])["src"].cast(pl.Utf8).drop_nulls().to_list())))}
    d = pl.DataFrame({ID: ids})
    for key in ("evt", "trk", "osm", "occ", "seq"):
        p = tabs[key]
        if p is None:
            continue
        if key == "evt":
            p = p.drop(["day", "apt", "t_first", "t_last", "t_gfirst", "t_air_prev"], strict=False)
        if "src" in p.columns:
            p = p.with_columns(C("src").cast(pl.Utf8).replace_strict(srcmap, default=-1, return_dtype=pl.Int32).alias("evt_src")).drop("src")
        d = d.join(p, on=ID, how="left", maintain_order="left")
    have = [c for c in excols if c in d.columns]
    if len(have) < len(excols):
        P("WARNING extra columns left NaN:", [c for c in excols if c not in d.columns])
    for c in have:
        Xs[:, len(cols) + excols.index(c)] = d[c].cast(pl.Float32).to_numpy()
    return Xs


def variant_columns(ids, apt, prof, tabs_dir_name):
    """d_NEW, d_ET (agent_bigc_v2_10_extra2.py): per airport group, bulk-refit minus reference stream and averaged minus single control."""
    dN, dE = np.zeros(len(ids)), np.zeros(len(ids))
    for aa, pc in PARTS.items():
        m = np.isin(apt, aa)
        if not m.any():
            continue
        fr = {}
        for k, tpl in VARIANTS.items():
            f = Path(prof["var"].format(name=tpl.format(pc=pc)))
            if not f.exists():
                P(f"WARNING variant file missing: {f}: d_NEW/d_ET of {aa} left NaN")
                dN[m], dE[m] = np.nan, np.nan
                fr = None
                break
            q = pl.read_parquet(f, columns=[ID, "TAXITIME_SEC_mvt"]).with_columns(C(ID).cast(pl.Float64)).unique(ID)
            fr[k] = pl.DataFrame({ID: ids}).join(q, on=ID, how="left", maintain_order="left")["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()
        if fr is None:
            continue
        dN[m] = (fr["bulkr"] - fr["ref"])[m]
        dE[m] = (fr["avg"] - fr["ctrl"])[m]
    return np.column_stack([dN, dE]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking", default=str(RAW / "final_ranking.parquet"))
    ap.add_argument("--months", default="2,6")
    ap.add_argument("--tag", default="J9")
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--pv", required=True, help="parquet MVT_ID_mvt, pv")
    ap.add_argument("--profile", default="final", choices=sorted(PROFILES))
    ap.add_argument("--views", default=None, help="npz with ids and Xs (52 variant-view columns); default NaN")
    ap.add_argument("--rank-cache", default=None, help="parquet of precomputed rank features (default: build, or the profile's file)")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--check", action="store_true", help="compare every output with the stored January/July files by id (profile stored)")
    ap.add_argument("--flags-only", action="store_true")
    ap.add_argument("--in-tag", default="", help="suffix of the J6/J7 input tags, e.g. jj reads J6jj_* and J7jj_* (January/July rebuilt from the final file)")
    ap.add_argument("--build-rank", action="store_true", help="build the rank features from the ranking file even when the profile names a cache")
    ap.add_argument("--fold-prefix", default="agent_bigc13", choices=["agent_bigc13", "agent_bigc12"],
                    help="2025 fold the drift columns are measured against: agent_bigc13 = the corrector refit base (use with the B26 pv), "
                         "agent_bigc12 = the earlier 12-month corrector whose serve preds feed the tail component (use with the old pv)")
    a = ap.parse_args()
    months = parse_months(a.months)
    prof = dict(PROFILES[a.profile])
    if a.in_tag:
        prof = {k: (v.replace("/J6_", f"/J6{a.in_tag}_").replace("/J7_", f"/J7{a.in_tag}_") if isinstance(v, str) else v) for k, v in prof.items()}
    outdir = Path(a.outdir)
    sd = outdir / f"{a.tag}_serve"
    sd.mkdir(parents=True, exist_ok=True)

    cols_dep = [ID, "ADEP_mvt", "ADES_mvt", "CALLSIGN_flt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "AOBT_3_flt"]
    dep = F.departures(Path(a.ranking), months, cols_dep)
    if a.sample:
        dep = dep.group_by("ADEP_mvt", maintain_order=True).head(a.sample).sort(ID)
    ids = dep[ID].cast(pl.Float64).to_numpy()
    apt = dep["ADEP_mvt"].cast(pl.Utf8).to_numpy().astype(str)
    mon_s = dep["MVT_TIME_UTC_mvt"].dt.month().to_numpy().astype(np.int64)
    n = len(ids)
    P(f"{n:,} departures; per airport-month:")
    cnt = dep.group_by("ADEP_mvt", dep["MVT_TIME_UTC_mvt"].dt.month().alias("month")).len().sort("ADEP_mvt", "month")
    print(cnt.to_pandas().pivot(index="ADEP_mvt", columns="month", values="len").fillna(0).astype(int).to_string())

    pv_df = pl.read_parquet(a.pv)
    if ID not in pv_df.columns:
        pv_df = pv_df.rename({"id": ID})
    pv_df = pv_df.with_columns(C(ID).cast(pl.Float64))
    pv = pl.DataFrame({ID: ids}).join(pv_df.unique(ID), on=ID, how="left", maintain_order="left")["pv"].cast(pl.Float64).to_numpy()
    assert np.isfinite(pv).all(), f"pv missing for {int((~np.isfinite(pv)).sum())} departures"

    # ---- time and flags (cheap, no table needed)
    t = dep["MVT_TIME_UTC_mvt"].dt.epoch("s").to_numpy().astype(np.int64)
    np.savez(sd / "agent_arith3_time_serve.npz", t=t, ids=ids)
    modes = pl.read_parquet(INTERIM / "agent_aobtflag_modes25.parquet")
    fl = F.aobt_flags(F.flag_inputs(Path(a.ranking), months).join(pl.DataFrame({"id": ids}), on="id", how="semi"), modes)
    fl = pl.DataFrame({"id": ids}).join(fl.select("id", "flag", "K"), on="id", how="left", maintain_order="left").with_columns(pl.lit(2026).cast(pl.Int32).alias("yr"))
    fl.write_parquet(sd / "serve_flags.parquet")
    P("flags: share with AOBT_3 default (flag 2) %.4f, no AOBT_3 (flag 0) %.4f" % ((fl["flag"] == 2).mean(), (fl["flag"] == 0).mean()))
    if a.flags_only:
        return

    # ---- keys
    kk = np.load(INTERIM / "agent_bigc13_keys2.npz", allow_pickle=True)
    k1 = np.load(INTERIM / "agent_bigc12_keys.npz", allow_pickle=True)
    kt = F.key_table(dep.select(F.KEY_INPUT_COLS), with_fn=True)
    n1, n2 = [str(x) for x in k1["names"]], [str(x) for x in kk["names"]]
    Ks = kt.select(n1).to_numpy().astype(np.int64)
    Ks2 = kt.select(n2).to_numpy().astype(np.int64)
    Ks_d = kt.select(["op_apt", "op_stand", "st_rwy", "st_hb"]).to_numpy().astype(np.int64)

    # ---- base columns
    z = np.load(INTERIM / "agent_bigc13_serve_in.npz", allow_pickle=True)
    cols_pre = [str(c) for c in z["cols_pre"]]
    excols = [str(c) for c in np.load(INTERIM / "agent_bigc12_extra.npz", allow_pickle=True)["cols"]]
    cols = cols_pre[:len(cols_pre) - 2 - len(excols)]
    assert cols_pre == cols + excols + ["d_NEW", "d_ET"] and len(cols) == 140
    maps = json.load(open(INTERIM / "agent_bigc12_maps.json"))
    tabs = {k: table(prof[k], k) for k in ("census", "ob", "bx", "evt", "trk", "osm", "occ", "seq")}
    tabs["wx"] = arrival_lateness(Path(a.ranking), dep)
    rank = rank_features(Path(a.ranking), months, ids, outdir, None if a.build_rank else (a.rank_cache or prof["rank"]))
    Xb = base_matrix(dep, pv, rank, tabs, cols, excols, maps)
    del rank
    nv = variant_columns(ids, apt, prof, None)
    Xs_pre = np.hstack([Xb, nv]).astype(np.float32)
    assert Xs_pre.shape == (n, len(cols_pre))

    # ---- drift (agent_refit2_04_serve.py): own-month mean of pv and of the ADS-B span against the 2025 fold
    cache = outdir / f"{a.fold_prefix}_drift_cache.npz"
    if not cache.exists():
        f = np.load(INTERIM / f"{a.fold_prefix}_fold.npz", allow_pickle=True)
        fc = [str(c) for c in f["cols"]]
        np.savez(cache, span_f=f["X"][:, fc.index("m_e_start")].astype(np.float64), pv_f=f["pv"].astype(np.float64))
        del f
    fo = np.load(cache)
    Dsv = DR.drift_serve(np.load(INTERIM / f"{a.fold_prefix}_keys2.npz", allow_pickle=True)["Kf_d"], fo["pv_f"], fo["span_f"], Ks_d, pv, Xs_pre[:, cols_pre.index("m_e_start")].astype(float), mon_s)

    # ---- plan-clock distance columns
    gaps, pvv = F.pcd_gaps(Xs_pre, cols_pre)
    flag = fl["flag"].cast(pl.Float64).to_numpy()
    Ds = F.pcd_build(pvv, gaps, flag)

    # ---- variant views
    if a.views:
        vz = np.load(a.views, allow_pickle=True)
        vi = F.id_index(vz["ids"].astype(float), ids)
        Xv = np.where((vi >= 0)[:, None], vz["Xs"][np.maximum(vi, 0)], np.nan).astype(np.float32)
    else:
        P("WARNING --views not given: the 52 variant-view columns are NaN (blocked on the re-stack variant serves)")
        Xv = np.full((n, 52), np.nan, np.float32)
    np.savez(sd / f"{a.fold_prefix}_serve_in.npz", Xs_pre=Xs_pre, cols_pre=np.array(cols_pre), Dsv=Dsv, mon_s=mon_s, pv26=pv, apt=apt,
             is9=np.isin(apt, AIR9), ids=ids)
    np.savez(sd / "serve_keys.npz", Ks=Ks, Ks2=Ks2, Ks_d=Ks_d, apt_s=apt, names1=np.array(n1), names2=np.array(n2))
    np.savez(sd / "serve_views.npz", Xs=Xv, ids=ids)
    np.savez(sd / "agent_pcd_cols_app.npz", Ds=Ds, ids_s=ids)
    P("wrote", sd)
    if a.check:
        check_against_stored(sd, ids, Xs_pre, cols_pre, Dsv, Ks, Ks2, Ks_d, Ds, fl, Xv)


def _cmp(name, new, old):
    """Max abs difference over finite pairs and a per-row 'differs' mask (NaN equals NaN), column by column to save memory."""
    new, old = np.asarray(new), np.asarray(old)
    if new.ndim == 1:
        new, old = new[:, None], old[:, None]
    bad = np.zeros(new.shape, bool)
    d = 0.0
    for j in range(new.shape[1]):
        x, y = new[:, j].astype(np.float64), old[:, j].astype(np.float64)
        bad[:, j] = ~((x == y) | (np.isnan(x) & np.isnan(y)))
        both = np.isfinite(x) & np.isfinite(y)
        if both.any():
            d = max(d, float(np.abs(x[both] - y[both]).max()))
    return d, bad


def check_against_stored(sd, ids, Xs_pre, cols_pre, Dsv, Ks, Ks2, Ks_d, Ds, fl, Xv):
    """Max abs difference and number of differing rows against the stored January/July serve files, by id."""
    z = np.load(INTERIM / "agent_bigc13_serve_in.npz", allow_pickle=True)
    i = F.id_index(z["ids"].astype(float), ids)
    assert (i >= 0).all(), "ids not in the stored serve file"
    rep = []
    d, bad = _cmp("Xs_pre", Xs_pre, z["Xs_pre"][i])
    rep.append(("Xs_pre (all 194)", d, int(bad.any(1).sum())))
    for j, c in enumerate(cols_pre):
        dj, bj = _cmp(c, Xs_pre[:, j], z["Xs_pre"][i, j])
        if bj.any():
            rep.append((" column " + c, dj, int(bj.sum())))
    for nm, new, old in (("Dsv", Dsv, z["Dsv"][i]), ("Ks", Ks, np.load(INTERIM / "agent_bigc12_keys.npz", allow_pickle=True)["Ks"][i]),
                         ("Ks2", Ks2, np.load(INTERIM / "agent_bigc13_keys2.npz", allow_pickle=True)["Ks"][i]),
                         ("Ks_d", Ks_d, np.load(INTERIM / "agent_bigc13_keys2.npz", allow_pickle=True)["Ks_d"][i]),
                         ("pcd Ds", Ds, np.load(INTERIM / "agent_pcd_cols_app.npz", allow_pickle=True)["Ds"][i])):
        d, bad = _cmp(nm, new, old)
        rep.append((nm, d, int(bad.any(1).sum())))
    st = pl.read_parquet(INTERIM / "agent_aobtflag_flags.parquet").filter(C("yr") == 2026).select("id", C("flag").alias("flag0"), C("K").alias("K0"))
    j = fl.join(st, on="id", how="left")
    rep.append(("flags flag", float((j["flag"].cast(pl.Float64) - j["flag0"].cast(pl.Float64)).abs().max() or 0.0), int((j["flag"] != j["flag0"]).sum())))
    rep.append(("flags K", float((j["K"] - j["K0"]).abs().max() or 0.0), int(((j["K"] != j["K0"]) & ~(j["K"].is_null() & j["K0"].is_null())).sum())))
    print("\nREPRODUCTION vs stored January/July files (%d rows): output | max abs difference | differing rows" % len(ids))
    for r in rep:
        print("  %-28s %-12.6g %d" % r)


if __name__ == "__main__":
    main()
