"""Compare the four-month file built entirely by the final/ stages (final_B_rebuilt.parquet) with the submitted candidate (final_B.parquet,
January/July carried from the stored lineage) on the January and July rows. Inputs and our own outputs only; no labels.

1. Per airport-month: rows differing, mean and rms of (rebuilt - carried), share of the squared difference carried by the 20 largest rows.
2. Where the difference enters. On the value before the day layer (J11_pre_variants of both builds) the difference is split into
     restack      J7jj_restack minus the stored composite of the four production pieces (ADS-B stage realisation noise and re-drawn jitter)
     small        new small-stage delta (M1..M4) minus the stored one
     corrector    dose x (new serve-fit mean - stored serve-fit mean), 52 view columns and 2 variant columns empty in the new fits
     tail         0.5 x (new - stored tail component)
     ist_term     Istanbul corrector term the carried file has and the rebuilt file does not (LTFM rows)
     other        everything else (Istanbul bulk/snow rows from stored old-frame arms, blend not ported, rounding, clips, rules)
   plus the part added by the layers after that point (day layer and window react to the level). Shares are
   s_i = sum(d_i * d_final) / sum(d_final^2), which add to 1.
   python final/54_rebuilt_compare.py
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL as ID, OUT, RAW, REPO, TARGET_COL as T  # noqa: E402

C = pl.col
I = REPO / "data" / "interim"
DOSE = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25, "EHAM": 1.25, "LFPG": 0.75}
ARITH = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LFPG", "LEMD"]
PROD = {"a": "agent_restack5_x_cap_evt_trk_bx_big_final", "b": "agent_restack5_x_cap_evt_trk_seq_bx_big_final",
        "c": "agent_restack5_b_cap_evt_trk_seq_big_final", "d": "agent_stagecap_9s_bx_big_final"}
COMP = {"EGLL": "a", "LEMD": "b", "LFPG": "c", "LIRF": "c", "EDDF": "d", "EDDM": "d", "EHAM": "d", "LEBL": "d", "LSZH": "d"}
KEYS = ("restack", "small", "corrector", "tail", "ist_term", "other", "layers_after")
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)


def rd(path, name, col=T):
    d = pl.read_parquet(path)
    return d.select(C(ID).cast(pl.Float64), C(col).cast(pl.Float64).alias(name))


def mean_term(pattern):
    acc, ids, n = None, None, 0
    for f in sorted(glob.glob(pattern)):
        z = np.load(f, allow_pickle=True)
        if ids is None and "ids" in z.files:
            ids = z["ids"].astype(float)
        p = z["pred"].astype(float)
        acc = p if acc is None else acc + p
        n += 1
    return ids, acc / n


def main() -> None:
    dep = (pl.scan_parquet(RAW / "final_ranking.parquet").filter((C("PHASE_mvt") == "DEP") & C("MVT_TIME_UTC_mvt").dt.month().is_in([1, 7]))
           .select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("m")).collect())
    j = dep.join(rd(OUT / "final_B.parquet", "B"), on=ID).join(rd(OUT / "final_B_rebuilt.parquet", "R"), on=ID)
    v162 = rd(REPO / "submissions" / "jolly-lobster_v162.parquet", "v162")
    j = j.join(v162, on=ID).with_columns((C("R") - C("B")).alias("d"))

    # 1. per airport-month
    rows = []
    for (apt, m), g in j.group_by(["apt", "m"], maintain_order=True):
        d = g["d"].to_numpy()
        sq = d ** 2
        top = np.sort(sq)[::-1][:20].sum() / max(sq.sum(), 1e-9)
        rows.append((apt, m, len(d), int((d != 0).sum()), float(d.mean()), float(np.sqrt(sq.mean())), float(top),
                     float(np.sqrt(((g["R"] - g["v162"]) ** 2).mean())), float(np.sqrt(((g["B"] - g["v162"]) ** 2).mean()))))
    t1 = pl.DataFrame(rows, schema=["apt", "m", "rows", "differ", "mean", "rms", "top20_share", "rms_R_v162", "rms_B_v162"], orient="row").sort("apt", "m")
    print("1. B_rebuilt - B on January/July rows (s)")
    print(t1.with_columns(pl.col(pl.Float64).round(3)))
    d = j["d"].to_numpy()
    sq = d ** 2
    print(f"all: rows {len(d):,}, differ {(d != 0).sum():,}, mean {d.mean():+.3f}, rms {np.sqrt(sq.mean()):.2f}, top-20 share {np.sort(sq)[::-1][:20].sum() / sq.sum():.4f}, "
          f"top-1000 share {np.sort(sq)[::-1][:1000].sum() / sq.sum():.3f}")
    mmj = j["m"].to_numpy()
    for m in (1, 7):
        mk = mmj == m
        print(f"month {m}: rows {mk.sum():,}, differ {(d[mk] != 0).sum():,}, mean {d[mk].mean():+.3f}, rms {np.sqrt(sq[mk].mean()):.2f}")
    print(f"vs v162 (all Jan/Jul rows): B rms {np.sqrt(((j['B'] - j['v162']) ** 2).mean()):.2f}, B_rebuilt rms {np.sqrt(((j['R'] - j['v162']) ** 2).mean()):.2f}; "
          f"mean B {float((j['B'] - j['v162']).mean()):+.3f}, B_rebuilt {float((j['R'] - j['v162']).mean()):+.3f}")
    print("20 largest |R-B| rows:")
    print(j.sort(C("d").abs(), descending=True).head(20).select("apt", "m", ID, "B", "R", "v162", "d"))

    # 2. stage decomposition on the pre-variant value
    ids = j[ID].to_numpy()
    x = (j.select(ID, "apt", "m")
         .join(rd(OUT / "J11_pre_variants.parquet", "pre_o"), on=ID).join(rd(OUT / "_rebuilt_out" / "J11_pre_variants.parquet", "pre_n"), on=ID))
    new_rs = rd(OUT / "J7jj_restack.parquet", "rs_n")
    old_rs = pl.concat([rd(I / f"{PROD[pc]}.parquet", "rs_o").join(dep.filter(C("apt").is_in([a for a, q in COMP.items() if q == pc])).select(ID), on=ID) for pc in PROD])
    x = x.join(new_rs, on=ID, how="left").join(old_rs, on=ID, how="left")
    zn = np.load(OUT / "J8jj_deltas.npz", allow_pickle=True)
    zo = np.load(I / "agent_bndl_serve_deltas.npz", allow_pickle=True)
    sm_n = pl.DataFrame({ID: zn["ids"].astype(float), "sm_n": sum(zn[q].astype(float) for q in ("M1", "M2", "M3", "M4"))})
    sm_o = pl.DataFrame({ID: zo["ids"].astype(float), "sm_o": sum(zo[q].astype(float) for q in ("M1", "M2", "M3", "M4"))})
    x = x.join(sm_n, on=ID, how="left").join(sm_o, on=ID, how="left")
    zt = np.load(I / "agent_tailc_serve_ehat.npz", allow_pickle=True)
    x = x.join(pl.DataFrame({ID: zn["ids"].astype(float), "tl_n": zn["ehat"].astype(float)}), on=ID, how="left")
    x = x.join(pl.DataFrame({ID: zt["ids"].astype(float), "tl_o": zt["ehat"].astype(float)}), on=ID, how="left")
    si = np.load(I / "agent_bigc13_serve_in.npz", allow_pickle=True)
    ids_old = si["ids"].astype(float)
    cols = []
    for pn, po, apts in (("J10jj_stage/out/arith/KYQSR-pw-s0-r0_serve_s*.npz", "agent_arith3_serve/KYQSR-pw-s0-r0_serve_s*.npz", ARITH),
                         ("J10jj_stage/out/pcd/C4ff7V5_s*.npz", "agent_pcd_modal/serve/C4ff7V5_s*.npz", ["LIRF", "LSZH"])):
        i1, p1 = mean_term(str(OUT / pn))
        i0, p0 = mean_term(str(I / po))
        i0 = ids_old if i0 is None else i0
        keep = dep.filter(C("apt").is_in(apts))[ID]
        cols.append(pl.DataFrame({ID: i1, "cn": p1}).join(pl.DataFrame({ID: i0, "co": p0}), on=ID, how="inner").filter(C(ID).is_in(keep)))
    x = x.join(pl.concat(cols), on=ID, how="left")
    x = x.join(rd(I / "agent_bundle_v150_final.parquet", "v150"), on=ID, how="left").join(rd(I / "agent_istnext_final.parquet", "ist"), on=ID, how="left")
    assert np.array_equal(x[ID].to_numpy(), ids)
    k = x["apt"].replace_strict(DOSE, default=0.0, return_dtype=pl.Float64).to_numpy()

    def f0(c):
        return np.nan_to_num(x[c].to_numpy().astype(float))

    apt, mm = x["apt"].to_numpy(), x["m"].to_numpy()
    is9 = np.isin(apt, list(DOSE))
    is_l = apt == "LTFM"
    comp = {
        "restack": np.where(is9, f0("rs_n") - f0("rs_o"), 0.0),
        "small": f0("sm_n") - f0("sm_o"),
        "corrector": k * (f0("cn") - f0("co")),
        "tail": 0.5 * np.where(is9, f0("tl_n") - f0("tl_o"), 0.0),
        "ist_term": np.where(is_l, -(f0("ist") - f0("v150")), 0.0),
    }
    tot = (x["pre_n"] - x["pre_o"]).to_numpy().astype(float)
    comp["other"] = tot - sum(comp.values())
    final = d
    comp["layers_after"] = final - tot
    out = []
    for a in sorted(set(apt)):
        for m in (1, 7):
            s = (apt == a) & (mm == m)
            row = [a, m, float(np.sqrt((final[s] ** 2).mean())), float(np.sqrt((tot[s] ** 2).mean()))]
            row += [float(np.sqrt((comp[q][s] ** 2).mean())) for q in KEYS]
            out.append(row)
    t2 = pl.DataFrame(out, schema=["apt", "m", "rms_final", "rms_pre"] + list(KEYS), orient="row")
    print("\n2a. rms (s) of each component of (rebuilt - carried); pre = value before the day layer, final = after the window; restack..other sum to pre, layers_after = final - pre")
    print(t2.with_columns(pl.col(pl.Float64).round(2)))

    def shares(s):
        den = (final[s] ** 2).sum()
        return {q: float((comp[q][s] * final[s]).sum() / den) for q in KEYS}

    print("\n2b. share of the final squared difference (sum d_i * d_final / sum d_final^2); adds to 1")
    rowsb = [("all", shares(np.ones(len(final), bool))), ("Jan", shares(mm == 1)), ("Jul", shares(mm == 7)), ("LTFM", shares(is_l)), ("nine airports", shares(is9))]
    for a in sorted(set(apt)):
        rowsb.append((a, shares(apt == a)))
    print(pl.DataFrame([dict(group=g, **{q: round(v, 3) for q, v in s.items()}) for g, s in rowsb]))
    print("\nrms of the pre-layer difference over all rows:", round(float(np.sqrt((tot ** 2).mean())), 2), " final:", round(float(np.sqrt((final ** 2).mean())), 2))


if __name__ == "__main__":
    main()
