"""experiment m12 bigc inputs (docs/LAB-NOTES.md): the 12-month versions of the bigc fold inputs, new-month rows APPENDED after the 8-month rows
(row order of the old 1,174,088 rows is unchanged, so old LOMO outputs align by prefix). Every file is written under the prefix agent_bigc12_ (the bigc
experiment points modal/agent_bigc_app.py FILES/IN at it); agent_bigc_* files are only READ.

  agent_bigc12_fold.npz     X (N12, 140; same cols/order, last col pv = p + M1+M2+M3), cols, y, pv, p, apt, mon (1..12), ids, cat
  agent_bigc12_extra.npz    Xf (N12, 52) [evt/trk/osm/occ/seq columns, same keep list], Xs (unchanged 2026), cols, nanf
  agent_bigc12_extra2.npz   Xf (N12, 2) d_NEW, d_ET: NaN on the new months
  agent_bigc12_keys.npz     Kf (N12, 11), Ks, names, apt_f (N12), apt_s
  agent_bigc12_keys2.npz    Kf (N12, 16), Kf_d (N12, 4), Df (N12, 8) RECOMPUTED over the 12 months (own month vs the other 11), Ks, Kl, Ks_d, Kl_d, names
  agent_bigc12_vfac_clean{,2a,2b,3}.npz   Xf (N12, k): NaN on the new months, Xs unchanged
  agent_bigc12_maps.json    = agent_bigc_maps.json (label codes stay the 8-month frequency ranks: unseen -> -1, as at serve)
The X/Y/W target statistics (agent_bigc_te.py te_fold / Te2) and the leave-own-month-out logic are computed INSIDE modal/agent_bigc_app.py from (K, mon): with 12
months they become 12-fold LOMO with no change to that code; the 2026 serve side (Ks, Xs, Ds drift) must be recomputed against 12 months by the bigc serve scripts.
--check: rebuild X / extra / keys for a 150k-row sample of OLD rows through the new-row code path and require equality with the existing files (the identity gate).
"""
import glob
import json
import shutil
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "archive")
import agent_bigc_drift as DR

C = pl.col
I = "data/interim/"
ID = "MVT_ID_mvt"
AIR = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
LOW = ["ADEP_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "FLIGHT_TYPE_flt", "FLIGHT_RULE_mvt", "airport_plan", "RUNWAY_mvt"]
HIGH = ["STAND_mvt", "ADES_mvt", "AIRCRAFT_TYPE_mvt", "AIRCRAFT_OPERATOR_flt", "callsign_op"]
check = "--check" in sys.argv
F = np.load(I + "agent_bigc_fold.npz", allow_pickle=True)
fcols = [str(c) for c in F["cols"]]
assert fcols[-1] == "pv"
maps = json.load(open(I + "agent_bigc_maps.json"))
tfl = pl.scan_parquet(I + "train_features.parquet")
tfs = tfl.collect_schema()


def conv(k, dt):
    if dt == pl.Boolean:
        return k in ("true", "True", "1")
    if dt.is_integer():
        return int(k)
    if dt.is_float():
        return float(k)
    return k


def build_X(ids, tabs):
    """tabs: census (ID, category), ob (rows dump), bx, wx (id -> ID). Mirrors archive/agent_bigc_01_build.py with the 8-month maps."""
    N = len(ids)
    base = pl.DataFrame({ID: ids, "_r": np.arange(N)})
    need = [c for c in fcols if c in tfs.names()]
    num = (pl.Float64, pl.Int64, pl.Int32, pl.Int16, pl.Int8)
    tf = (tfl.select([ID] + [C(c).cast(pl.Float32) if tfs[c] in num else C(c) for c in need]).join(base.lazy().select(ID), on=ID, how="semi").collect().unique(ID))
    d = base.join(tf, on=ID, how="left").sort("_r")
    assert d.height == N
    for c in LOW + HIGH:
        if c in d.columns:
            mp = {conv(k, d.schema[c]): v for k, v in maps[c].items()}
            d = d.with_columns(C(c).replace_strict(mp, default=-1, return_dtype=pl.Int32).fill_null(-1).alias(c))
    cen = tabs["census"].select([ID, "category"]).unique(ID)
    cen = cen.with_columns(C("category").replace_strict(maps["category"], default=-1, return_dtype=pl.Int32).alias("cov_cat")).select([ID, "cov_cat"])
    ecs = ["e_stand", "e_appear", "e_ci20", "e_dend", "e_dnext", "e_mv5", "e_mv10", "e_mv20", "e_rev", "e_start", "e_revany", "e_pushend"]
    obc = ["has_c", "P_span", "P_n", "P_gap", "P_dstand", "P_long", "seg_n", "seg_rate", "seg_start_gap", "push", "first_gs", "first_dstand", "revany_gap", "rev_dist", "rev_gs", "rev_dstand"]
    ob = tabs["ob"].select([ID, "src", "prod_kind", "mvt_t"] + obc + ecs).unique(ID)
    for c in ("src", "prod_kind"):
        ob = ob.with_columns(C(c).cast(pl.Utf8).replace_strict(maps["ob_" + c], default=-1, return_dtype=pl.Int32).fill_null(-1).alias("ob_" + c))
    ob = ob.select([ID, "ob_src", "ob_prod_kind"] + obc + [(C("mvt_t") - C(e)).alias("m_" + e) for e in ecs])
    for o in (cen, ob, tabs["bx"].unique(ID), tabs["wx"]):
        d = d.join(o, on=ID, how="left")
    d = d.sort("_r")
    assert d.height == N
    X = np.empty((N, len(fcols)), np.float32)
    for j, c in enumerate(fcols[:-1]):
        X[:, j] = d[c].cast(pl.Float32).to_numpy()
    return X


def tables(old):
    if old:
        return dict(census=pl.read_parquet(I + "agent_coverage_census_2025.parquet"), ob=pl.read_parquet(I + "agent_offblock_rows_c238.parquet"),
                    bx=pl.read_parquet(I + "agent_backx_bx_2025.parquet"),
                    wx=pl.read_parquet(I + "agent_wx2_diag.parquet").select(["id", "arr_late_share60", "arr_late_share180", "arr_late_mean60", "arr_late_mean180"]).rename({"id": ID}).unique(ID))
    w = pl.read_parquet(I + "agent_m12_wx2_new.parquet").select(["id", "arr_late_share60", "arr_late_share180", "arr_late_mean60", "arr_late_mean180"]).rename({"id": ID}).unique(ID)
    return dict(census=pl.read_parquet(I + "agent_m12_census_new.parquet"), ob=pl.read_parquet(I + "agent_md_rows_new.parquet"), bx=pl.read_parquet(I + "agent_md_bx_new.parquet"), wx=w)


EX = np.load(I + "agent_bigc_extra.npz", allow_pickle=True)
excols = [str(c) for c in EX["cols"]]


def extra_frame(ids, tag):
    if tag == "2025":
        parts = [pl.read_parquet(I + f"agent_restack3_evt_2025.parquet").drop(["day", "apt", "t_first", "t_last", "t_gfirst", "t_air_prev"], strict=False).unique(ID),
                 pl.read_parquet(I + "agent_restack3_trk_2025.parquet").unique(ID), pl.read_parquet(I + "agent_sb2_osm_2025.parquet").unique(ID),
                 pl.read_parquet(I + "agent_sb2_occ_2025.parquet").unique(ID), pl.read_parquet(I + "agent_smallbundle_seq_2025.parquet").unique(ID)]
    else:
        parts = [pl.read_parquet(I + "agent_md_evt_new.parquet").drop(["day", "apt", "t_first", "t_last", "t_gfirst", "t_air_prev"], strict=False).unique(ID),
                 pl.read_parquet(I + "agent_md_trk_new.parquet").unique(ID), pl.read_parquet(I + "agent_md_osm_new.parquet").unique(ID),
                 pl.read_parquet(I + "agent_m12_occ_new.parquet").unique(ID), pl.read_parquet(I + "agent_md_seq_new.parquet").unique(ID)]
    old_evt = pl.read_parquet(I + "agent_restack3_evt_2025.parquet", columns=["src"])
    srcmap = {v: i for i, v in enumerate(sorted(set(old_evt["src"].cast(pl.Utf8).drop_nulls().to_list())))}
    d = pl.DataFrame({ID: ids})
    for p in parts:
        if "src" in p.columns:
            p = p.with_columns(C("src").cast(pl.Utf8).replace_strict(srcmap, default=-1, return_dtype=pl.Int32).alias("evt_src")).drop("src")
        d = d.join(p.with_columns(C(ID).cast(pl.Float64)), on=ID, how="left", maintain_order="left")
    return d.select(excols).cast(pl.Float32).to_numpy()


def keys_full(ids):
    cols = [ID, "CALLSIGN_flt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt", "PHASE_mvt"]

    def prep(df):
        s = lambda c: C(c).cast(pl.Utf8).fill_null("NA")  # noqa: E731
        hb = (C("MVT_TIME_UTC_mvt").dt.hour() // 3).cast(pl.Utf8)
        A, op, st, rw, ty, ad = s("ADEP_mvt"), s("AIRCRAFT_OPERATOR_flt"), s("STAND_mvt"), s("RUNWAY_mvt"), s("AIRCRAFT_TYPE_mvt"), s("ADES_mvt")
        ex = {"fn": [A, s("CALLSIGN_flt")], "ades": [A, ad], "cty": [A, ad.str.slice(0, 2)], "op_apt": [A, op], "op_stand": [A, op, st], "op_hb": [A, op, hb], "op_rwy": [A, op, rw],
              "st_rwy": [A, st, rw], "st_hb": [A, st, hb], "ty_st": [A, ty, st], "mkt": [A, s("MARKET_SEGMENT_flt")], "wk": [A, s("WK_TBL_CAT_flt")],
              "op_ades": [A, op, ad], "op_ty": [A, op, ty], "ty_rwy": [A, ty, rw], "ades_hb": [A, ad, hb], "st_op_rwy": [A, st, op, rw]}
        return df.select([C(ID)] + [pl.concat_str(v, separator="|").hash(seed=7).floordiv(2).cast(pl.Int64).alias(k) for k, v in ex.items()]), list(ex)
    idf = pl.DataFrame({ID: ids})
    fr = []
    for f in sorted(glob.glob("data/raw/training_2025-*.parquet")):
        fr.append(prep(pl.read_parquet(f, columns=cols).filter(C("PHASE_mvt") == "DEP").join(idf, on=ID, how="semi"))[0])
    names = prep(pl.read_parquet("data/raw/ranking.parquet", columns=cols, n_rows=5))[1]
    j = idf.with_row_index("_r").join(pl.concat(fr).unique(ID), on=ID, how="left").sort("_r")
    return j.select(names).to_numpy().astype(np.int64), names


K1, K2 = np.load(I + "agent_bigc_keys.npz", allow_pickle=True), np.load(I + "agent_bigc_keys2.npz", allow_pickle=True)

if check:
    rng = np.random.default_rng(0)
    idx = np.sort(rng.choice(len(F["ids"]), 150000, replace=False))
    ids = F["ids"][idx]
    X = build_X(ids, tables(True))
    old = F["X"][idx][:, :-1]
    bad = []
    for j, c in enumerate(fcols[:-1]):
        a, b = X[:, j], old[:, j]
        if not np.array_equal(a, b, equal_nan=True):
            bad.append((c, float(np.nanmax(np.abs(a - b))) if np.isfinite(a - b).any() else "nan-mismatch", int((~((a == b) | (np.isnan(a) & np.isnan(b)))).sum())))
    print("IDENTITY X (150k old rows through the new-row path):", "PASS" if not bad else f"FAIL {bad[:8]}")
    ea = extra_frame(ids, "2025")
    eb = EX["Xf"][idx]
    print("IDENTITY extra:", "PASS" if np.array_equal(ea, eb, equal_nan=True) else f"FAIL maxdiff {np.nanmax(np.abs(ea - eb))}")
    kf, names = keys_full(ids)
    k1 = kf[:, [names.index(n) for n in K1["names"]]]
    k2 = kf[:, [names.index(n) for n in K2["names"]]]
    print("IDENTITY keys:", "PASS" if np.array_equal(k1, K1["Kf"][idx]) and np.array_equal(k2, K2["Kf"][idx]) else "FAIL")
    sys.exit(0)

# ------------------------------------------------------------------ main: append the new months
M = np.load(I + "agent_m12_members_new.npz", allow_pickle=True)
ids_n = M["ids"].astype(float)
Nn, No = len(ids_n), len(F["ids"])
pv_n = M["p"] + M["M1"] + M["M2"] + M["M3"]
Xn = build_X(ids_n, tables(False))
Xn[:, -1] = pv_n
cat = lambda a, b: np.concatenate([a, b])  # noqa: E731
np.savez(I + "agent_bigc12_fold.npz", X=np.vstack([F["X"], Xn]), cols=F["cols"], y=cat(F["y"], M["y"]), pv=cat(F["pv"], pv_n), p=cat(F["p"], M["p"]),
         apt=cat(F["apt"].astype(str), M["apt"].astype(str)), mon=cat(F["mon"], M["mon"].astype(F["mon"].dtype)), ids=cat(F["ids"], ids_n), cat=F["cat"])
mon12 = cat(F["mon"], M["mon"].astype(F["mon"].dtype))
print("fold", No, "+", Nn, "months", sorted(set(mon12.tolist())), flush=True)
ea = extra_frame(ids_n, "new")
Xf = np.vstack([EX["Xf"], ea])
np.savez(I + "agent_bigc12_extra.npz", Xf=Xf, Xs=EX["Xs"], cols=EX["cols"], nanf=np.isnan(Xf).mean(0))
E2 = np.load(I + "agent_bigc_extra2.npz", allow_pickle=True)
np.savez(I + "agent_bigc12_extra2.npz", Xf=np.vstack([E2["Xf"], np.full((Nn, E2["Xf"].shape[1]), np.nan, np.float32)]), Xs=E2["Xs"], cols=E2["cols"])
kf, names = keys_full(ids_n)
k1n, k2n = kf[:, [names.index(n) for n in K1["names"]]], kf[:, [names.index(n) for n in K2["names"]]]
dkn = kf[:, [names.index(n) for n in ["op_apt", "op_stand", "st_rwy", "st_hb"]]]
apt12 = cat(K1["apt_f"].astype(str), M["apt"].astype(str))
np.savez(I + "agent_bigc12_keys.npz", Kf=np.vstack([K1["Kf"], k1n]), Ks=K1["Ks"], names=K1["names"], apt_f=apt12, apt_s=K1["apt_s"])
Kd12 = np.vstack([K2["Kf_d"], dkn])
span12 = np.concatenate([F["X"][:, fcols.index("m_e_start")], Xn[:, fcols.index("m_e_start")]]).astype(float)
pv12 = cat(F["pv"], pv_n).astype(float)
Df = DR.drift_fold(Kd12, pv12, span12, mon12)
np.savez(I + "agent_bigc12_keys2.npz", Kf=np.vstack([K2["Kf"], k2n]), Ks=K2["Ks"], Kl=K2["Kl"], names=K2["names"], Kf_d=Kd12, Ks_d=K2["Ks_d"], Kl_d=K2["Kl_d"], Df=Df)
for sfx in ("", "2a", "2b", "3", "4", "5"):
    V = np.load(I + f"agent_bigc_vfac_clean{sfx}.npz", allow_pickle=True)
    np.savez(I + f"agent_bigc12_vfac_clean{sfx}.npz", Xf=np.vstack([V["Xf"], np.full((Nn, V["Xf"].shape[1]), np.nan, np.float32)]), Xs=V["Xs"], cols=V["cols"])
shutil.copy(I + "agent_bigc_maps.json", I + "agent_bigc12_maps.json")
print("wrote agent_bigc12_*: fold X", (No + Nn, len(fcols)), "extra", Xf.shape, "keys", (No + Nn, K1["Kf"].shape[1]), "keys2", (No + Nn, K2["Kf"].shape[1]), "Df", Df.shape, flush=True)
