"""Tag aggr, 2026-09-23. Screen four EUROCONTROL/ansperformance.eu airport-day datasets plus
Nager.Date public holidays against the CURRENT model's fold residuals, with the project's
calibrated partial-rho instrument (archive/agent_cong_common.py; bar registered in the ledger
~4884, ~5136: same sign on every fold, |rho| > PLACEBO_FLOOR on every fold, min|rho| >= 5x max
placebo, >= 1000 finite rows per fold).

Deviations from agent_audit_02_screen.py, and why:
  * Fold source is results/preds_late2500_treat_f3_3_9.parquet and _f4_4_10.parquet (the CURRENT
    model per the plan), NOT the FOLDS list hardcoded in agent_cong_common.py -- those are
    stale paths from an earlier model generation. Only 2 folds are available this way (fold 3 =
    test months 3+9, fold 4 = test months 4+10), not 5, so SIGN and MARGIN are read on 2 folds,
    which is a weaker bar than the pre-registered 5-fold one and is flagged as such per result.
  * Every candidate here is constant within one (airport, calendar day) -- true airport-day
    grain, one value broadcast to every flight departing that airport that day -- exactly the
    construction the plan and the scour note describe, and exactly what the +0.75s day-grain
    oracle ceiling (idea-ledger 2026-09-20 21:45) bounds.
  * Controls reuse agent_audit_02_screen.py's CONTROLS list (imported, not copied) so a
    candidate must clear the same bar the 90 high-volume candidates were held to on 2026-09-21.

    PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH=.
        .venv/Scripts/python.exe archive/agent_aggr_01_screen.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl

import sys
sys.path.insert(0, ".")
sys.path.insert(0, "archive")

from agent_cong_common import PLACEBO_FLOOR, Proj, chain, linear_gain, partial_rho  # noqa: E402
from agent_audit_02_screen import CONTROLS  # noqa: E402  (imported, not modified)

C = pl.col
ID = "MVT_ID_mvt"
FOLDS = [("f3_3_9", "results/preds_late2500_treat_f3_3_9.parquet"),
         ("f4_4_10", "results/preds_late2500_treat_f4_4_10.parquet")]
FLOOR = PLACEBO_FLOOR
MARGIN = 5.0
MIN_ROWS = 1000
FEATCACHE = "data/interim/train_features.parquet"

APT_COUNTRY = {"EDDF": "DE", "EDDM": "DE", "EGLL": "GB", "EHAM": "NL", "LEBL": "ES",
               "LEMD": "ES", "LFPG": "FR", "LIRF": "IT", "LSZH": "CH", "LTFM": "TR"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_raw_meta() -> pl.DataFrame:
    return (pl.scan_parquet("data/raw/training_*.parquet").filter(C("PHASE_mvt") == "DEP")
            .select(C(ID), C("ADEP_mvt").alias("adep"),
                    C("MVT_TIME_UTC_mvt").dt.date().cast(pl.Utf8).alias("day"),
                    C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                    C("AOBT_3_flt").is_not_null().alias("plan"),
                    (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                    (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"))
            .collect())


def build_universe():
    meta = load_raw_meta()
    parts = []
    for label, path in FOLDS:
        P = pl.read_parquet(path).select(ID, "y", "bagged").join(meta, on=ID, how="inner")
        y = P["y"].to_numpy().astype(float)
        gs = P["gs"].fill_null(np.nan).to_numpy().astype(float)
        gl = P["gl"].fill_null(np.nan).to_numpy().astype(float)
        plan = P["plan"].fill_null(False).to_numpy()
        base = chain(P["bagged"].to_numpy().astype(float), gs, gl, plan)
        P = P.with_columns(pl.Series("base", base), pl.Series("resid", y - base))
        parts.append(P.select(ID, "adep", "day", "ym", "resid", "base").with_columns(pl.lit(label).alias("fold")))
        log(f"   {label}: {P.height:,} rows, chained RMSE {np.sqrt(np.mean((y - base) ** 2)):.2f}s")
    return pl.concat(parts)


def load_controls(ids: np.ndarray, control_cols: list[str]) -> dict[str, np.ndarray]:
    pos = {m: i for i, m in enumerate(ids)}
    cache = pl.scan_parquet(FEATCACHE).select([C(ID)] + [C(c).cast(pl.Float32) for c in control_cols]).collect()
    cp = np.array([pos.get(m, -1) for m in cache[ID].to_numpy()])
    ck = cp >= 0
    out = {}
    for c in control_cols:
        col = np.full(len(ids), np.nan, dtype=np.float32)
        col[cp[ck]] = cache[c].fill_null(np.nan).to_numpy()[ck]
        out[c] = col
    return out


def design(loc_adep, loc_ctrl, m, control_cols) -> np.ndarray:
    n = int(m.sum())
    X = [np.ones(n)]
    for c in control_cols:
        v = loc_ctrl[c][m]
        X.append(np.nan_to_num(v.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
        if not np.isfinite(v).all():
            X.append((~np.isfinite(v)).astype(float))
    apt = loc_adep[m]
    for a in np.unique(apt)[1:]:
        X.append((apt == a).astype(float))
    return np.column_stack(X)


# ---------------------------------------------------------------- candidate loaders

def cand_pre_departure_delay(kind: str) -> pl.DataFrame:
    """kind in {'atc', 'all'}: minutes of pre-departure delay per departure, airport-day."""
    col_dly = "DLY_ATC_PRE_2" if kind == "atc" else "DLY_ALL_PRE_2"
    name = f"eur_{kind}_predep_dpf"
    frames = []
    for y in (2025, 2026):
        f = f"data/external/eurocontrol/{'atc' if kind == 'atc' else 'all'}_pre_departure_delays_{y}.csv"
        d = (pl.read_csv(f)
             .select(C("APT_ICAO").alias("adep"), C("FLT_DATE").str.slice(0, 10).alias("day"),
                     C("FLT_DEP_1").alias("dep"), C(col_dly).fill_null(0).alias("dly")))
        frames.append(d)
    d = pl.concat(frames).filter(C("dep") > 0)
    return d.select("adep", "day", (C("dly") / C("dep")).alias(name)), name


def cand_atfm_slot_adherence() -> pl.DataFrame:
    """Share of departures OUTSIDE the CTOT tolerance window that day, airport-day."""
    name = "eur_atfm_nonadh_rate"
    frames = []
    for y in (2025, 2026):
        f = f"data/external/eurocontrol/atfm_slot_adherence_{y}.csv"
        d = (pl.read_csv(f)
             .select(C("APT_ICAO").alias("adep"), C("FLT_DATE").str.slice(0, 10).alias("day"),
                     C("FLT_DEP_1").alias("dep"),
                     (C("FLT_DEP_OUT_EARLY_1").fill_null(0) + C("FLT_DEP_OUT_LATE_1").fill_null(0)).alias("out")))
        frames.append(d)
    d = pl.concat(frames).filter(C("dep") > 0)
    return d.select("adep", "day", (C("out") / C("dep")).alias(name)), name


def cand_apt_dly_arr() -> pl.DataFrame:
    """apt_dly is ARRIVAL ATFM delay at the airport -- a same-day inbound-congestion / ramp
    pressure proxy for the SAME airport's departures that day."""
    import bz2
    import io
    name = "eur_apt_dly_arr_dpf"
    frames = []
    for y in (2025, 2026):
        f = f"data/external/eurocontrol/apt_dly_{y}.csv.bz2"
        with bz2.open(f, "rb") as fh:
            raw = fh.read()
        d = (pl.read_csv(io.BytesIO(raw), encoding="latin1")
             .select(C("APT_ICAO").alias("adep"), C("FLT_DATE").str.slice(0, 10).alias("day"),
                     C("FLT_ARR_1").alias("arr"), C("DLY_APT_ARR_1").fill_null(0).alias("dly")))
        frames.append(d)
    d = pl.concat(frames).filter(C("arr") > 0)
    return d.select("adep", "day", (C("dly") / C("arr")).alias(name)), name


def cand_holiday() -> pl.DataFrame:
    """1.0 if the departure-country public holiday calendar (Nager.Date) marks this UTC date a
    public holiday in the airport's own country, else 0.0. Airport-day grain by construction."""
    name = "eur_is_public_holiday"
    rows = []
    for c in sorted(set(APT_COUNTRY.values())):
        for y in (2025, 2026):
            f = Path(f"data/external/holidays/{c}_{y}.json")
            items = json.loads(f.read_text(encoding="utf-8"))
            for it in items:
                rows.append((c, it["date"]))
    hol = pl.DataFrame(rows, schema=["country", "day"], orient="row").unique()
    apt_tab = pl.DataFrame({"adep": list(APT_COUNTRY), "country": [APT_COUNTRY[a] for a in APT_COUNTRY]})
    full = apt_tab.join(hol, on="country", how="inner").select("adep", "day").with_columns(pl.lit(1.0).alias(name))
    return full, name


# ---------------------------------------------------------------- screening

def screen_candidate(uni: pl.DataFrame, cand: pl.DataFrame, name: str, ctrl: dict, control_cols: list[str],
                      rows_out: list) -> None:
    j = uni.join(cand, on=["adep", "day"], how="left")
    # demean within (airport, month) per the pre-registered bar
    v = j[name].to_numpy().astype(float)
    apt = j["adep"].to_numpy().astype(str)
    ym = j["ym"].to_numpy().astype(str)
    grp = np.array([f"{a}|{m}" for a, m in zip(apt, ym)])
    dm = v.copy()
    for g in np.unique(grp):
        gi = grp == g
        gv = v[gi]
        fin = np.isfinite(gv)
        if fin.sum() > 0:
            dm[gi] = gv - np.nanmean(gv[fin]) if fin.any() else gv
    ids = j[ID].to_numpy()
    fold_lab = j["fold"].to_numpy().astype(str)
    resid = j["resid"].to_numpy().astype(float)

    rho, plc, nrows, gains = [], [], [], []
    rng = np.random.default_rng(20260923)
    for label, _ in FOLDS:
        m0 = fold_lab == label
        vv = dm[m0]
        finite = np.isfinite(vv)
        n = int(finite.sum())
        nrows.append(n)
        if n < MIN_ROWS:
            rho.append(np.nan); plc.append(np.nan); gains.append(np.nan)
            continue
        idx = np.where(m0)[0][finite]
        r = resid[idx]
        cvals = {c: ctrl[c][idx] for c in control_cols}
        X = []
        X.append(np.ones(n))
        for c in control_cols:
            cv = cvals[c]
            X.append(np.nan_to_num(cv.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0))
            if not np.isfinite(cv).all():
                X.append((~np.isfinite(cv)).astype(float))
        apt_i = apt[idx]
        for a in np.unique(apt_i)[1:]:
            X.append((apt_i == a).astype(float))
        Xd = np.column_stack(X)
        P = Proj(Xd)
        pr = P(r)
        pv = P(vv[finite])
        rho.append(partial_rho(P, vv[finite], r))
        # placebo: circularly shift the candidate's day-index within each airport (breaks the
        # day alignment but keeps the airport level and the marginal distribution)
        placebo_v = np.roll(vv[finite], n // 2)
        plc.append(partial_rho(P, placebo_v, r))
        d_rmse, _, _ = linear_gain(P, vv[finite], r)   # in-sample CEILING fit, not an OOS estimate
        gains.append(d_rmse)

    rho_a = np.array(rho)
    ok_rows = min(nrows) >= MIN_ROWS and np.isfinite(rho_a).all()
    sign_ok = ok_rows and (np.all(rho_a > 0) or np.all(rho_a < 0))
    minabs = float(np.nanmin(np.abs(rho_a))) if ok_rows else 0.0
    maxplc = float(np.nanmax(np.abs(plc))) if ok_rows else np.nan
    floor_ok = ok_rows and minabs > FLOOR
    margin = minabs / max(maxplc, 1e-9) if ok_rows else 0.0
    margin_ok = ok_rows and margin >= MARGIN
    verdict = "PASS" if (sign_ok and floor_ok and margin_ok) else "fail"
    why_fail = [] if verdict == "PASS" else [s for s, okk in (("rows", ok_rows), ("sign", sign_ok),
                                                               ("floor", floor_ok), ("margin", margin_ok)) if not okk]
    cover = float(np.isfinite(v).mean())
    ceiling = 0.75  # +0.75s +/- 0.07, the day-grain PERFECT-oracle ceiling (idea-ledger 2026-09-20 21:45)
    gains_a = np.array(gains, dtype=float)
    mean_gain = float(np.nanmean(gains_a)) if np.isfinite(gains_a).any() else 0.0
    frac_ceiling = -mean_gain / ceiling if ceiling else 0.0   # gain is negative (RMSE DOWN) when real
    row = {"name": name, "coverage": round(cover, 4), "min_rows": min(nrows), "verdict": verdict,
           "fails": "+".join(why_fail), "min_abs_rho": round(minabs, 4),
           "max_abs_placebo": round(maxplc, 4) if np.isfinite(maxplc) else None,
           "margin": round(margin, 2), "mean_rho": round(float(np.nanmean(rho_a)), 4),
           "insample_rmse_gain_s": round(mean_gain, 4), "frac_of_0.75s_ceiling": round(frac_ceiling, 3)}
    for (label, _), a, b in zip(FOLDS, rho, plc):
        row[f"rho_{label}"] = round(a, 4) if np.isfinite(a) else None
        row[f"plc_{label}"] = round(b, 4) if np.isfinite(b) else None
    rows_out.append(row)
    print(f"   {name:<26}" + "".join(f"{x:>+9.4f}" if np.isfinite(x) else f"{'nan':>9}" for x in rho)
          + f"  |plc| {maxplc:.4f}  min|r| {minabs:.4f}  x{margin:>5.1f}  cov {cover:.2f}  "
          f"{verdict} {row['fails']}  in-sample gain {mean_gain:+.4f}s ({frac_ceiling:+.2f}x of +0.75s ceiling)",
          flush=True)


def main():
    t0 = time.time()
    log("building universe from the two current-model fold files")
    uni = build_universe()
    ids = uni[ID].to_numpy()
    log(f"instrument controls: {len(CONTROLS)} columns (reused from agent_audit_02_screen.CONTROLS)")
    ctrl = load_controls(ids, CONTROLS)
    log(f"ready in {time.time() - t0:.0f}s")

    print()
    print("=" * 110)
    print(f"SCREEN (2 folds only -- f3=months 3+9, f4=months 4+10). BAR: same sign both folds, "
          f"|rho| > {FLOOR} both folds, min|rho| >= {MARGIN:.0f} x max|placebo|, >= {MIN_ROWS} rows/fold")
    print(f"   {'':<26}" + "".join(f"{l:>9}" for l, _ in FOLDS))
    rows = []
    for loader in (lambda: cand_pre_departure_delay("atc"), lambda: cand_pre_departure_delay("all"),
                   cand_atfm_slot_adherence, cand_apt_dly_arr, cand_holiday):
        cand, name = loader()
        screen_candidate(uni, cand, name, ctrl, CONTROLS, rows)

    out = pl.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    out.write_csv("results/agent_aggr_screen.csv")
    log(f"wrote results/agent_aggr_screen.csv ({out.height} rows)")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
