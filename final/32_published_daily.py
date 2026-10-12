"""Published daily delay layer: day-level calibration of delayed flights and the Rome record-type posterior.

Source of the figures: the EUROCONTROL performance portal's daily airport file (all_pre_departure_delays_<year>.csv, columns
FLT_DEP_1 and DLY_ALL_PRE_2), which covers every day of the final months. The organisers confirmed on 2026-10-08 that published
EUROCONTROL aggregates of the scored months may be used.

What the step does. The published total delay of an airport-day equals the sum over that day's departures of
max(off-block minus schedule, 0), each departure filed under the date of its off-block stamp (take-off minus label). So a
prediction file implies a daily total, and the difference to the published figure is one equation per airport-day:

  level   (use "level"): the delayed flights of the day (predicted stamp after schedule) are shifted together by the amount that
          makes the implied total equal the published total less the airport's typical bias (bisection on the exact positive-part
          sum), times a shrink factor lam. Only on days whose published departure count is within 15 of ours. Airports and constants
          as shipped: EDDF EDDM EGLL LEBL LEMD LIRF (v153), LTFM with the snow stop-day exclusion (v159), LFPG (v161), LSZH
          on IFR flights (v160, see --note on LSZH).
  types   (use "types"): a Rome no-plan departure with a gap of at least 5,400 s is either a schedule copy (delay 0) or a real late
          departure (delay = gap minus an ordinary taxi). The served value is the posterior mean of the two, the prior copy
          probability being the model's own hedge; the day's published total, with the day's other uncertain rows as noise, updates
          it by enumerating the hypotheses of up to 20 rows a day.
  cells   (--cell-split, default off): the v162 split of the level shift into slopes per coverage class and lateness bin.

Derived from: archive/agent_dayeq_04_level.py (level), agent_dayeq_05_rome_types.py (types), agent_dayeq2_04_ltfm_level.py (Istanbul),
agent_nextpools_05_build.py (Paris), agent_bundle_dayeq.py (combination), agent_bundle_v159/160.py (window re-applied after a piece).
Month-edge guard (new, --no-edge-guard switches it off): a day is used only if the frame also holds departures from the start of
the next UTC day, because late stamps of the last day belong to take-offs just after midnight.

--count-equation (default off, failed its 2025 gate, see docs/LAB-NOTES.md) adds the published IFR count to the Rome record-type step.

Date guard (--no-date-guard switches it off): a shifted row keeps the UTC date of its off-block stamp from before the layer.

--on / --off switch the whole step (default --on). With --off the output equals the input.

  python final/32_published_daily.py --pred P.parquet --ranking data/raw/final_ranking.parquet --out Q.parquet [--months 2,6]
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "archive"))
from fcommon import ID_COL, OUT, RAW, TARGET_COL  # noqa: E402

C = pl.col
EURO = REPO / "data" / "external" / "eurocontrol"
INTERIM = REPO / "data" / "interim"
AMB_GAP, AMB_PRED, DN_MAX, MAXK = 5400.0, 1800.0, 15, 20
FLOOR_MIN, ORD_FLOOR = 33.0, 121.0        # minutes, as shipped (Rome types)
SIG_F, GATE_Z, MIN_N = 5.0, 3.0, 100      # Istanbul stop-day gate, as shipped
FRAME_MONTHS = [1, 2, 3, 7, 8, 9, 10, 11]

# per airport: (lam, bias in seconds) exactly as shipped. Six airports of v153 from data/interim/agent_dayeq_level_cfg.json,
# Istanbul from the 2025 run of v159, Paris from the 2025 run of v161.
SHIPPED = {
    "EDDF": dict(lam=0.9, bias=-1.7302967877185438), "EDDM": dict(lam=1.0, bias=-1.5605391701683402),
    "EGLL": dict(lam=0.9, bias=-9.804013121902244), "LEBL": dict(lam=1.0, bias=-0.6308384821750224),
    "LEMD": dict(lam=1.0, bias=1.4123306660621893), "LIRF": dict(lam=0.8, bias=-1.5997403948858846),
    "LFPG": dict(lam=0.8, bias=-11.229803192691179),
    "LTFM": dict(lam=0.8, bias=-14.511),
}
LSZH_LAM = 0.96   # mean nested lam of the Zurich IFR run (logs/agent_dayeq2_01_LSZH_hold_2025.log)


def load_rules_module():
    spec = importlib.util.spec_from_file_location("mod_30", HERE / "30_general_rules.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod_30"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------------------------------
def load_published(apts: list[str]) -> pl.DataFrame:
    """Published departures (FLT_DEP_1) and all-cause pre-departure delay (DLY_ALL_PRE_2, minutes) per airport and day, all years on disk."""
    parts = []
    for f in sorted(EURO.glob("all_pre_departure_delays_20*.csv")):
        d = pl.read_csv(f, infer_schema_length=0).filter(C("APT_ICAO").is_in(apts))
        parts.append(d.select(C("APT_ICAO").alias("apt"), C("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"),
                              C("DLY_ALL_PRE_2").cast(pl.Float64, strict=False).alias("pub"),
                              C("FLT_DEP_1").cast(pl.Float64, strict=False).alias("n_pub")))
    return pl.concat(parts)


def load_frame(ranking: Path, pred: pl.DataFrame, months: list[int] | None, sample: float | None) -> pl.DataFrame:
    """Departures with the prediction, in id order. Input columns only."""
    d = (pl.scan_parquet(ranking).filter(C("PHASE_mvt") == "DEP")
         .select(C(ID_COL).alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                 C("FLIGHT_ID_mvt").is_not_null().alias("hasplan"), C("FLIGHT_RULE_mvt").alias("rule"),
                 C("FLIGHT_mvt").is_null().alias("fnull"),
                 (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().cast(pl.Float64).alias("gl"),
                 C("MVT_TIME_UTC_mvt").dt.month().alias("mon"))
         .collect().sort("id"))
    if months:
        d = d.filter(C("mon").is_in(months))
    p = pred.select(C(ID_COL).alias("id"), C(TARGET_COL).cast(pl.Float64).alias("pred"))
    d = d.join(p, on="id", how="inner")
    if sample:
        d = d.sample(fraction=sample, seed=0).sort("id")
    d = d.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"))
    taxi = d.filter(C("gs").abs() < 3600).group_by("apt", "mon").agg(taxi=C("pred").median())
    d = d.join(taxi, on=["apt", "mon"], how="left")
    d = d.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"),
                       (C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day"))
    d = d.with_columns(((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"))
    return d


def complete_days(all_t: pl.Series, guard: bool) -> set:
    """UTC dates whose following date (before 03:00) has departures in the frame, i.e. whose late stamps are all present."""
    if not guard:
        return set()
    t = all_t.dt.replace_time_zone(None) if all_t.dtype.time_zone else all_t
    early = t.filter(t.dt.hour() < 3).dt.date().unique().to_list()
    return {d - datetime.timedelta(days=1) for d in early}


def solve_delta(gs: np.ndarray, yh: np.ndarray, target_min: float) -> float:
    """delta (s) with sum max(gs - yh - delta, 0) / 60 = target_min, bisection on [-7200, 7200] (as shipped)."""
    def S(d):
        return np.clip(gs - yh - d, 0, None).sum() / 60.0
    lo, hi = -7200.0, 7200.0
    if S(lo) < target_min:
        return lo
    if S(hi) > target_min:
        return hi
    for _ in range(40):
        m = 0.5 * (lo + hi)
        if S(m) > target_min:
            lo = m
        else:
            hi = m
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------------------------------------
# level
# ---------------------------------------------------------------------------------------------------------------------
def day_table(f: pl.DataFrame, pub: pl.DataFrame, apt: str, ifr_only: bool, shift_only_solve: bool, usable: set, guard: bool) -> pl.DataFrame:
    """Per day of one airport: our count, the published figures, and the delay shift delta (s, lam 1, no bias) that meets the published total."""
    g = f.filter(C("apt") == apt)
    if ifr_only:
        g = g.filter(C("rule") == "I")
    g = g.with_columns(pl.when(C("amb")).then(((1 - C("p_copy")) * (C("gs") - C("taxi")) / 60.0).clip(lower_bound=0))
                       .otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm"))
    full = f.filter(C("apt") == apt)
    nall = full.group_by("day").agg(n=pl.len())
    parts = g.partition_by("day", as_dict=True, maintain_order=True)
    rows = []
    pubd = {r["day"]: r for r in pub.filter(C("apt") == apt).iter_rows(named=True)}
    cnt = dict(nall.iter_rows())
    for (d,), grp in parts.items():
        r = pubd.get(d)
        if r is None:
            continue
        n = cnt.get(d, 0)
        dn = r["n_pub"] - n if r["n_pub"] is not None else np.nan
        ok = (r["pub"] is not None and np.isfinite(r["pub"]) and np.isfinite(dn) and abs(dn) <= DN_MAX and (not guard or d in usable))
        delta = np.nan
        O = float(grp["dm"].sum())
        if ok:
            amb = grp["amb"].to_numpy()
            gs, yh = grp["gs"].to_numpy(), grp["pred"].to_numpy()
            if shift_only_solve:
                sh = ~amb & (gs > yh)
                fixed = float(grp["dm"].to_numpy()[~sh].sum())
                delta = solve_delta(gs[sh], yh[sh], r["pub"] - fixed) if sh.any() else np.nan
            else:
                fixed = float(grp["dm"].to_numpy()[amb].sum())
                delta = solve_delta(gs[~amb], yh[~amb], r["pub"] - fixed)
        rows.append(dict(day=d, n=n, O=O, D=(r["pub"] - O) if r["pub"] is not None else np.nan, delta=delta, ok=ok))
    return pl.DataFrame(rows, schema=dict(day=pl.Date, n=pl.Int64, O=pl.Float64, D=pl.Float64, delta=pl.Float64, ok=pl.Boolean))


def gate_from_2025(apt: str) -> dict:
    """Stop-day gate parameters of one airport fitted on the stored 2025 held-out chain values (as archive/agent_dayeq2_04_ltfm_level.py
    for Istanbul, archive/agent_dayeq2_06_excl.py for the other airports): median and robust spread of D = published - implied delay,
    the spread of the unshifted long-gap component and the median implied total. Cached in data/final/_gen."""
    cache = OUT / "_gen" / (f"gate_2025_{apt}.json" if apt != "LTFM" else "ltfm_gate_2025.json")
    if cache.exists():
        return json.loads(cache.read_text())
    files = sorted(RAW.glob("training_*.parquet"))
    raw = (pl.scan_parquet([str(x) for x in files]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == apt))
           .select(C("MVT_ID_mvt").alias("id"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S")).collect())
    eb = pl.scan_parquet(INTERIM / "agent_eb136_tbl.parquet").filter((C("apt") == apt) & C("mon").is_in(FRAME_MONTHS)).select("id", "mon", "pred").collect()
    f = raw.join(eb, on="id", how="inner").with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"))
    taxi = f.filter(C("gs").abs() < 3600).group_by("mon").agg(taxi=C("pred").median())
    f = f.join(taxi, on="mon", how="left").with_columns((C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day"))
    f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"), ((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"))
    f = f.with_columns(pl.when(C("amb")).then(((1 - C("p_copy")) * (C("gs") - C("taxi")) / 60.0).clip(lower_bound=0)).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm"),
                       pl.when(C("amb")).then(C("p_copy") * (1 - C("p_copy")) * ((C("gs") - C("taxi")) / 60.0) ** 2).otherwise(0.0).alias("var_amb"))
    pub = load_published([apt])
    days = (f.group_by("day").agg(n=pl.len(), O=C("dm").sum(), Vamb=C("var_amb").sum()).join(pub.select("day", "pub", "n_pub"), on="day", how="inner")
            .with_columns((C("n_pub") - C("n")).alias("dn")))
    days = days.filter((C("dn").abs() <= DN_MAX) & (C("n") >= MIN_N) & (C("n_pub") >= MIN_N)).with_columns((C("pub") - C("O")).alias("D"))
    Dv = days["D"].drop_nulls()
    bias = float(Dv.median())
    rsd = 1.4826 * float((Dv - bias).abs().median())
    sig_f = SIG_F_APT.get(apt, SIG_F)
    sigL2 = max(rsd ** 2 - sig_f ** 2 - float(days["Vamb"].median()), 100.0)
    out = dict(bias=bias, sigL2=sigL2, O_med=float(days["O"].median()))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def ltfm_gate_from_2025() -> dict:
    return gate_from_2025("LTFM")


def day_snow_set(f: pl.DataFrame, apt: str) -> set:
    """Dates (by predicted stamp) with at least one departure of the airport under a METAR snow / freezing report in the preceding 3 hours."""
    g = f.filter(C("apt") == apt)
    try:
        from agent_eusnow_feat import snow_features_apts
        sn = snow_features_apts(g.select(C("id").alias("MVT_ID_mvt"), pl.lit(apt).alias("ADEP_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), RAW, [apt])
    except (IndexError, FileNotFoundError, pl.exceptions.PolarsError):
        return set()          # the airport's METAR file carries no snow or freezing observation
    flag = (sn["lt_snow_3h"] > 0).to_numpy()
    j = g.select(C("id").alias("MVT_ID_mvt"), "day").join(sn.select("MVT_ID_mvt").with_columns(pl.Series("fl", flag)), on="MVT_ID_mvt", how="inner")
    return set(j.filter(C("fl"))["day"].unique().to_list())


STOP_REPORT: list[dict] = []     # filled by level_layer: one record per airport-day (see --stop-report)
SIG_F_APT = {"LIRF": 33.0}       # spread of the unshifted long-gap component in minutes: 33 at Rome, 5 elsewhere (SIG_F)


SNOW_AIRPORTS = ["EDDF", "EDDM", "LSZH", "EHAM"]


def snow_dates(apt: str) -> set:
    """UTC dates with at least one METAR observation whose present-weather code contains SN, FZRA or FZDZ."""
    r = pl.read_csv(RAW / "weather" / f"{apt}.csv", null_values=["M", "", "None"], infer_schema_length=0)
    r = r.select(C("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False).alias("t"), C("wxcodes").fill_null("").alias("wx")).drop_nulls("t")
    r = r.filter(C("wx").str.contains("SN") | C("wx").str.contains("FZRA") | C("wx").str.contains("FZDZ"))
    return set(r["t"].dt.date().to_list())


def level_layer(f: pl.DataFrame, pub: pl.DataFrame, p: np.ndarray, airports: list[str], *, lam_cap: float | None, own_bias: bool,
                guard: bool, usable: set, counters, mod30, note: list[str], dstore: dict | None = None, stop_days: str = "signature") -> np.ndarray:
    """Shift the delayed flights of each airport-day by lam x (delta - bias). Returns the new prediction aligned to f."""
    new = p.copy()
    apt_arr = f["apt"].to_numpy()
    for apt in airports:
        spec = SHIPPED.get(apt, dict(lam=LSZH_LAM, bias=None))
        ifr = apt == "LSZH"
        ltfm = apt == "LTFM"
        dt = day_table(f, pub, apt, ifr, ltfm, usable, guard)
        if dt.is_empty():
            continue
        lam = min(spec["lam"], lam_cap) if lam_cap is not None else spec["lam"]
        ok = dt.filter(C("ok") & C("delta").is_not_nan())
        bias_by_month = None
        use_own = own_bias or spec["bias"] is None
        if use_own:
            # label-free: median of the day shifts of each month's own published series
            mm = ok.with_columns(C("day").dt.month().alias("m")).group_by("m").agg(C("delta").median().alias("b"))
            bias_by_month = dict(mm.iter_rows())
            if spec["bias"] is None:
                note.append(f"{apt}: no shipped bias constant; used the median day shift of the served months")
        excl = set()
        # Stop days. Signature (validated at Istanbul, applied at every airport with its own gate parameters): a day with a METAR snow or
        # freezing report in the 3 h before take-off whose published total is more than 3 sigma below the total our file implies. Such a day is
        # left unshifted. On 2025 it fires on seven Istanbul February days and on no day at the other airports (docs/PIPELINE.md section 5,
        # docs/LAB-NOTES.md sections 9 and 11). stop_days == "metar" instead leaves every day with a snow report unshifted at
        # SNOW_AIRPORTS (never validated); "off" applies no exclusion.
        gp = gate_from_2025(apt)
        sig_f = SIG_F_APT.get(apt, SIG_F)
        sig = np.sqrt(sig_f ** 2 + gp["sigL2"] * np.maximum(1.0, ok["O"].to_numpy() / gp["O_med"]) ** 2)
        deficit = (ok["D"].to_numpy() - gp["bias"]) < -GATE_Z * sig
        snow_days = day_snow_set(f, apt)
        is_snow = np.array([d in snow_days for d in ok["day"].to_list()], dtype=bool)
        sig_ex = is_snow & deficit
        metar_all = set(snow_dates(apt)) if apt in SNOW_AIRPORTS else set()
        for d, z, sn_, ex_, wf in zip(ok["day"].to_list(), ((ok["D"].to_numpy() - gp["bias"]) / sig).tolist(), is_snow.tolist(), sig_ex.tolist(), deficit.tolist()):
            STOP_REPORT.append(dict(apt=apt, day=d, z=z, snow=sn_, signature=ex_, deficit_only=wf, metar_rule=(d in metar_all)))
        if stop_days == "signature" or (stop_days in ("ltfm", "metar") and apt == "LTFM"):
            excl = set(d for d, e in zip(ok["day"].to_list(), sig_ex) if e)
            note.append(f"{apt}: stop days excluded (snow and deficit beyond {GATE_Z:g} sigma): {sorted(str(d) for d in excl)}")
        ok = ok.with_columns(pl.Series("n_ok", (ok["n"] >= MIN_N).to_numpy() & ~np.array([d in excl for d in ok["day"].to_list()], dtype=bool)))
        dmap = {d: x for d, x in zip(ok["day"].to_list(), ok["delta"].to_list()) if d not in excl}
        if stop_days == "metar" and apt in SNOW_AIRPORTS:
            sd = metar_all
            hit = sorted(str(d) for d in dmap if d in sd)
            dmap = {d: x for d, x in dmap.items() if d not in sd}
            note.append(f"{apt}: {len(hit)} snow-report days left unshifted (METAR variant)")
        if dstore is not None:
            dstore[apt] = dmap
        mask_apt = apt_arr == apt
        days = f["day"].to_list()
        gs, yh, amb = f["gs"].to_numpy(), p, f["amb"].to_numpy()
        mons = f["mon"].to_numpy()
        rule = f["rule"].to_list()
        for k in np.flatnonzero(mask_apt):
            if amb[k] or gs[k] <= yh[k] or (ifr and rule[k] != "I"):
                continue
            d = dmap.get(days[k])
            if d is None:
                continue
            b = spec["bias"]
            if use_own:
                b = bias_by_month.get(int(mons[k]))
                if b is None:
                    continue
            new[k] = max(yh[k] + lam * (d - b), 0.0)
    return new


def _ltfm_snow_days(f: pl.DataFrame) -> set:
    """Dates (by predicted stamp) with at least one Istanbul departure under a snow report in the preceding 3 hours."""
    from agent_ltreg_feat import snow_features
    g = f.filter(C("apt") == "LTFM")
    sn = snow_features(g.select(C("id").alias("MVT_ID_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), RAW)
    flag = (sn["lt_snow_3h"] > 0).to_numpy()
    return set(g.filter(pl.Series(flag))["day"].unique().to_list())


# ---------------------------------------------------------------------------------------------------------------------
# Rome record types
# ---------------------------------------------------------------------------------------------------------------------
def _day_enumeration(row: dict, grp: pl.DataFrame, bias: float):
    """The shipped per-day enumeration: (Z, w) with Z the copy indicators of every assignment (1 = copy) and w the normalised weights (prior x minutes)."""
    k = grp.height
    pr = np.clip(grp["p_copy"].to_numpy(), 0.02, 0.98)
    e = grp["e"].to_numpy()
    sd = float(np.sqrt(ORD_FLOOR ** 2 + row["sd_noise"] ** 2))
    Z = np.array(list(itertools.product([0, 1], repeat=k)), dtype=float)
    extra = (1 - Z) @ e
    logprior = (Z * np.log(pr) + (1 - Z) * np.log(1 - pr)).sum(axis=1)
    target = row["pub"] - row["O"] - bias
    w = logprior - 0.5 * ((target - extra) / sd) ** 2
    w = np.exp(w - w.max())
    w /= w.sum()
    return Z, w


def _load_count_module():
    spec = importlib.util.spec_from_file_location("mod_36", HERE / "36_rome_count.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["mod_36"] = mod
    spec.loader.exec_module(mod)
    return mod


def _month_last(x: int) -> bool:
    d = datetime.date(1970, 1, 1) + datetime.timedelta(days=int(x))
    return (d + datetime.timedelta(days=1)).day == 1


def _count_equation_q(g: pl.DataFrame, en: pl.DataFrame, day_rows: list[dict], enums: list, days_ok: set, count_eq, note: list[str],
                      update_null_flight: bool, diag: dict | None) -> dict:
    """Second factor of the Rome record-type step: the published IFR count (final/36_rome_count.py, docs/LAB-NOTES.md).

    Returns {day: q} (copy probability of each row of the day's enumeration, in group order) for every enumerated day that holds a crossing row; the other
    days keep the minutes-only values, bit for bit.
    """
    m36 = _load_count_module()
    epoch = datetime.date(1970, 1, 1)
    Te = g["T"].dt.epoch("s").to_numpy().astype(float)
    Se = g["S"].dt.epoch("s").to_numpy().astype(float)
    taxi = g["taxi"].to_numpy()
    pin = g["pred_in"].to_numpy()
    pos_of = {int(r): i for i, r in enumerate(g["row"].to_list())}
    dr, dc, cross_ok = m36.crossing_flags(Te, Se, taxi)
    upd = (~g["hasplan"].to_numpy()) & (~g["hedged_rule"].to_numpy()) & (update_null_flight | ~g["fnull"].to_numpy())
    enum = g["enum"].to_numpy()
    inf_pos = np.zeros(len(g), bool)
    entries, info = [], []
    for row, (Z, w) in zip(day_rows, enums):
        gp = np.array([pos_of[int(r)] for r in en.filter(C("sday") == row["day"])["row"].to_list()])
        cand = np.flatnonzero(upd[gp] & cross_ok[gp])[: m36.MAX_INF]
        m = len(cand)
        if m:
            inf_pos[gp[cand]] = True
            pat = (Z[:, cand] * (1 << np.arange(m))[None, :]).sum(axis=1).astype(int)
            W = np.bincount(pat, weights=w, minlength=1 << m)
        else:
            pat, W = None, None
        entries.append(dict(day=(row["day"] - epoch).days, m=m, W=W))
        info.append((gp, cand, pat, W))
    if diag is not None:
        rl = g["row"].to_list()
        diag["crossing"] = {int(rl[i]) for i in np.flatnonzero(inf_pos)}
    if not inf_pos.any():
        return {}
    fixed = Te[~inf_pos] - pin[~inf_pos]
    extra = []
    for i in np.flatnonzero(enum & ~inf_pos & (dr != dc)):
        sdate = int(np.floor((Te[i] - pin[i]) / 86400.0))
        b = int(dr[i])
        extra.append((b, m36.AMBIG_FLIP, 0.0) if sdate == dc[i] else (b, 0.0, m36.AMBIG_FLIP))
    pp = m36.pattern_posteriors(count_eq, entries, fixed, extra, days_ok, note)
    out = {}
    for row, (Z, w), ent, (gp, cand, pat, W) in zip(day_rows, enums, entries, info):
        if ent["m"] == 0 or ent["day"] not in pp:
            continue
        marg = pp[ent["day"]]
        ratio = np.where(W > 0, marg / np.where(W > 0, W, 1.0), 0.0)
        out[row["day"]] = (w * ratio[pat]) @ Z
    return out


def rome_types_layer(f: pl.DataFrame, pub: pl.DataFrame, p: np.ndarray, guard: bool, usable: set, note: list[str],
                     update_null_flight: bool = False, count_eq=None, diag: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Posterior record types for Rome no-plan rows. Returns (new prediction, boolean mask of the rows set by the posterior).

    count_eq (a CountEq of final/36_rome_count.py, switched on by --count-equation): every assignment's weight is also multiplied by the likelihood of the
    published IFR counts. None = the shipped step. diag, if given, receives {"q": {row: copy probability}, "crossing": rows the count can see}.
    """
    g = f.with_columns(pl.Series("row", np.arange(f.height))).filter(C("apt") == "LIRF")
    if g.is_empty():
        return p, np.zeros(len(p), bool)
    g = g.with_columns(C("pred").alias("pred_in"))
    g = g.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED) & (C("pred") <= C("gs"))).alias("enum"),
                       ((C("gs") >= 1800) & (C("pred") >= 0.5 * C("taxi") + 300) & (C("pred") <= C("gs"))).alias("hedged"))
    g = g.with_columns(pl.when(C("enum")).then(C("T") - pl.duration(seconds=C("taxi"))).otherwise(C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("sday"),
                       ((C("gs") - C("taxi")) / 60.0).alias("e"),
                       pl.when(C("enum")).then(0.0).otherwise(((C("gs") - C("pred")) / 60.0).clip(lower_bound=0)).alias("dm_ord"),
                       pl.when(C("hedged") & ~C("enum")).then(C("p_copy") * (1 - C("p_copy")) * ((C("gs") - C("taxi")) / 60.0) ** 2).otherwise(0.0).alias("var_noise"))
    pb = pub.filter(C("apt") == "LIRF").select("day", "pub", "n_pub")
    days = (g.group_by(C("sday").alias("day")).agg(n=pl.len(), O=C("dm_ord").sum(), k=C("enum").sum(), sd_noise=C("var_noise").sum().sqrt())
            .join(pb, on="day", how="inner").with_columns((C("n_pub") - C("n")).alias("dn")))
    days = days.with_columns((C("pub") - C("O")).alias("D"))
    if guard:
        days = days.filter(C("day").is_in(list(usable)))
    k0 = days.filter((C("k") == 0) & (C("dn").abs() <= DN_MAX))
    bias = float(np.median((k0["pub"] - k0["O"]).to_numpy())) if k0.height > 5 else 0.0
    note.append(f"Rome types: bias {bias:+.1f} min from {k0.height} days without uncertain rows")
    en = g.filter(C("enum"))
    new = p.copy()
    setm = np.zeros(len(p), bool)
    day_rows = list(days.filter((C("k") > 0) & (C("k") <= MAXK) & (C("dn").abs() <= DN_MAX)).iter_rows(named=True))
    enums = [_day_enumeration(row, en.filter(C("sday") == row["day"]), bias) for row in day_rows]
    qcount: dict = {}
    if count_eq is not None:
        epoch = datetime.date(1970, 1, 1)
        days_ok = {(d - epoch).days for d in days.filter(C("dn").abs() <= DN_MAX)["day"].to_list()}
        days_ok = {x for x in days_ok if not _month_last(x)}
        qcount = _count_equation_q(g, en, day_rows, enums, days_ok, count_eq, note, update_null_flight, diag)
        note.append(f"Rome count equation: {len(qcount)} enumerated days carry a crossing row")
    for row, (Z, w) in zip(day_rows, enums):
        grp = en.filter(C("sday") == row["day"])
        q = qcount.get(row["day"])
        if q is None:
            q = (w[:, None] * Z).sum(axis=0)
        post = q * grp["gs"].to_numpy() + (1 - q) * grp["taxi"].to_numpy()
        for r, v, hasplan, hedged, fnull, qi in zip(grp["row"].to_list(), post, grp["hasplan"].to_list(), grp["hedged_rule"].to_list(), grp["fnull"].to_list(), q):
            if diag is not None:
                diag.setdefault("q", {})[int(r)] = float(qi)
            # as shipped, a row whose flight string is missing was never updated (the test `flight not in HEDGED` is null for it);
            # update_null_flight=True updates it like any other row
            if hasplan or hedged or (fnull and not update_null_flight):
                continue
            new[r] = np.rint(v)
            setm[r] = True
    return new, setm


def day_early_hedge(f: pl.DataFrame) -> np.ndarray:
    """Rome no-plan rows with a schedule gap of 14 h or more that took off on a later Europe/Rome day: the two record types (copy, stamp a day early) cannot be told apart. (--rule-hedge)

    These rows stay in the day's enumeration but their own value is not updated (the equation reads zero under both alternatives).
    Replaces the six hard-coded flight strings of the January/July run; the set is derived from input columns.
    """
    g = f.with_columns(pl.Series("row", np.arange(f.height)))
    slip = (g["T"].dt.convert_time_zone("Europe/Rome").dt.date() - g["S"].dt.convert_time_zone("Europe/Rome").dt.date()).dt.total_days().to_numpy()
    m = ((g["apt"] == "LIRF") & ~g["hasplan"]).to_numpy() & (g["gs"].to_numpy() >= 14 * 3600) & (slip >= 1)
    return m


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------
# cell split (v162), default off
# ---------------------------------------------------------------------------------------------------------------------
CELL_APTS = ["LFPG", "EGLL", "EDDF", "EDDM", "LIRF"]


def cell_of(cat: np.ndarray, gs: np.ndarray) -> np.ndarray:
    """Cell = coverage class (surface track matched, track without lift-off, other) x lateness bin (<15 min, 15-30, 30-60, >1 h)."""
    cls = np.where(cat == "matched", 0, np.where(cat == "points_no_liftoff", 1, 2))
    lb = np.where(gs < 900, 0, np.where(gs < 1800, 1, np.where(gs < 3600, 2, 3)))
    return cls * 4 + lb


def fit_cell_betas() -> dict:
    """Slopes per cell on the eight 2025 frame months, shrunk toward the airport's uniform lam (archive/agent_nextpools_11_buildc5.py).

    Reads the stored held-out chain values of 2025 (agent_eb136_tbl via archive/agent_dayeq2_lib.py) and the stored coverage class of the
    2025 held-out chain (agent_waithedge_tbl.cov). Cached in data/final/_gen/cell_betas_2025.json.
    """
    cache = OUT / "_gen" / "cell_betas_2025.json"
    if cache.exists():
        return {k: dict(bias=v["bias"], lam=v["lam"], beta=np.array(v["beta"])) for k, v in json.loads(cache.read_text()).items()}
    import agent_dayeq2_lib as lib
    f = lib.frame_2025(CELL_APTS)
    sv = pl.read_parquet(INTERIM / "agent_waithedge_tbl.parquet", columns=["id", "cov"]).with_columns(C("id").cast(pl.Float64))
    f = f.with_columns(C("id").cast(pl.Float64)).join(sv, on="id", how="left").with_columns(C("cov").fill_null("other")).filter(C("y").is_not_null() & C("gs").is_not_null())
    f = f.with_columns(C("day_pred").dt.month().alias("dmon")).filter(C("dmon").is_in(FRAME_MONTHS))
    f = f.with_columns(((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"))
    f = f.with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb")).alias("sh"))
    pub = lib.load_pub(2025, CELL_APTS)
    days = (f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum()).rename({"day_pred": "day"})
            .join(pub, on=["apt", "day"], how="inner").filter((C("n_pub") - C("n")).abs() <= DN_MAX))
    parts = f.partition_by("apt", "day_pred", as_dict=True)
    rows = []
    for r in days.iter_rows(named=True):
        g = parts[(r["apt"], r["day"])]
        m = g["sh"].to_numpy()
        rows.append({"apt": r["apt"], "day": r["day"], "d": solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], r["pub"] - r["fixed"]) if r["pub"] is not None and np.isfinite(r["pub"]) else np.nan})
    f = f.join(pl.DataFrame(rows), left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
    f = f.with_columns(pl.when(C("cov") == "matched").then(pl.lit("matched")).when(C("cov") == "points_no_liftoff").then(pl.lit("points_no_liftoff")).otherwise(pl.lit("other")).alias("cat"))
    out = {}
    for a in CELL_APTS:
        fa = f.filter(C("apt") == a)
        yh, y, d, gs = fa["pred"].to_numpy(), fa["y"].to_numpy(), fa["d"].to_numpy(), fa["gs"].to_numpy()
        pos = (gs > yh) & fa["sh"].to_numpy() & np.isfinite(d)
        cell = cell_of(fa["cat"].to_numpy(), gs)
        bias, lam = SHIPPED[a]["bias"], SHIPPED[a]["lam"]
        xs = np.where(pos, d - bias, 0.0)
        sxx = np.array([(xs[pos & (cell == c)] ** 2).sum() for c in range(12)])
        sxy = np.array([(xs[pos & (cell == c)] * (y - yh)[pos & (cell == c)]).sum() for c in range(12)])
        tau = 0.25 * sxx.mean()
        out[a] = dict(bias=bias, lam=lam, beta=((sxy + tau * lam) / (sxx + tau)).tolist())
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return {k: dict(bias=v["bias"], lam=v["lam"], beta=np.array(v["beta"])) for k, v in out.items()}


def cell_split_layer(f: pl.DataFrame, cur: np.ndarray, dstore: dict, census: Path | None, note: list[str]) -> np.ndarray:
    """Replace the uniform shift lam x (delta - bias) of the five airports by beta[cell] x (delta - bias) on the rows where it applied.

    From archive/agent_nextpools_11_buildc5.py (v162). `census` maps a departure id to its coverage class (ADS-B coverage census); an
    id without a class is treated as 'other', which is what the shipped build did for rows the census did not hold.
    """
    par = fit_cell_betas()
    ids = f["id"].to_numpy()
    cat = np.full(len(ids), "other", dtype=object)
    if census is not None and Path(census).exists():
        cen = pl.read_parquet(census, columns=[ID_COL, "category"]).rename({ID_COL: "id"})
        j = f.select("id").join(cen, on="id", how="left")["category"].fill_null("other").to_numpy()
        cat = j.astype(object)
        note.append("cell split: coverage classes from " + str(census) + ": " + ", ".join(f"{k} {int((cat == k).sum())}" for k in sorted(set(cat))))
    else:
        note.append("cell split: NO coverage census given; every row is class 'other' (the shipped build did the same for ids the census lacked)")
    gs = f["gs"].to_numpy()
    apt = f["apt"].to_numpy()
    days = f["day"].to_list()
    out = cur.copy()
    cell = cell_of(cat, gs)
    for a in CELL_APTS:
        if a not in dstore:
            note.append(f"cell split: no day shifts for {a} (piece not run), skipped")
            continue
        c = par[a]
        dm = dstore[a]
        for k in np.flatnonzero(apt == a):
            d = dm.get(days[k])
            if d is None or not np.isfinite(d):
                continue
            prod = c["lam"] * (d - c["bias"])
            ypre = cur[k] - prod
            amb = gs[k] >= AMB_GAP and ypre >= AMB_PRED
            if gs[k] > ypre and not amb:
                out[k] = max(cur[k] + c["beta"][cell[k]] * (d - c["bias"]) - prod, 0.0)
    return np.rint(out)


def bound_rows(f: pl.DataFrame, new: np.ndarray, touched: np.ndarray, window: float | None) -> np.ndarray:
    """After a piece: Istanbul no-plan rows at most gap + 1,200 s, plan rows inside [take-off - LOBT - W, + W], floor 0; only rows the piece touched.

    As the build steps of v159 to v161 (agent_dayeq2_04_ltfm_level.py, agent_bundle_v159.py) and the exact-edge re-application of v158.
    """
    out = new.copy()
    gs, gl = f["gs"].to_numpy(), f["gl"].to_numpy()
    noplan_ltfm = ((f["apt"] == "LTFM") & ~f["hasplan"]).to_numpy() & (gs >= 0)
    m = touched & noplan_ltfm
    out[m] = np.minimum(out[m], gs[m] + 1200.0)
    if window is not None:
        fin = np.isfinite(gl) & touched
        out[fin] = np.minimum(np.maximum(out[fin], np.maximum(gl[fin] - window, 0.0)), gl[fin] + window)
    return np.maximum(np.rint(out), 0)


def keep_offblock_date(f: pl.DataFrame, base: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Clip a shifted row so its implied off-block (take-off minus value) stays on the UTC date it had before the layer."""
    d = f.select(start=(C("T") - pl.duration(seconds=pl.Series(base))).dt.truncate("1d"), T=C("T"))
    hi = (d["T"] - d["start"]).dt.total_seconds().to_numpy().astype(float)      # latest value: off-block at 00:00:00 of that date
    lo = hi - 86400.0 + 1.0                                                  # earliest value: off-block at 23:59:59
    return np.where(cur != base, np.minimum(np.maximum(cur, lo), hi), cur)


def run(pred: pl.DataFrame, ranking: Path, *, pieces: set[str], months: list[int] | None, sample: float | None, lam_cap: float | None,
        own_bias: bool, guard: bool, level_airports: list[str], counters=None, window: float | None = 3618.0, rule_hedge: bool = False,
        update_null_flight: bool = False, cell_split: bool = False, census: Path | None = None, stop_days: str = "signature",
        count_equation: bool = False, date_guard: bool = True) -> tuple[pl.DataFrame, list[str]]:
    mod30 = load_rules_module()
    f = load_frame(ranking, pred, months, sample)
    notes: list[str] = []
    p = f["pred"].to_numpy().copy()
    allt = pl.scan_parquet(ranking).filter(C("PHASE_mvt") == "DEP").select("MVT_TIME_UTC_mvt").collect()["MVT_TIME_UTC_mvt"]
    usable = complete_days(allt if months is None else allt.filter(allt.dt.month().is_in(months)), guard)
    pub = load_published(sorted(set(f["apt"].to_list())))
    ctx_inp = f.select("id", "apt", C("mon").alias("month"))
    cnt = counters if counters is not None else mod30.Counters()

    def fire(name: str, old: np.ndarray, new: np.ndarray) -> None:
        moved = np.abs(new - old) > 0.5
        n = cnt.add(name, ctx_inp, moved)
        print(f"  {name}: {n:,} rows moved")

    base = p.copy()
    cur = p.copy()
    dstore: dict = {}
    if cell_split and (lam_cap is not None or own_bias):
        raise SystemExit("--cell-split subtracts the shipped uniform shift; it cannot be combined with --lam-cap or --own-bias")
    level_set = [a for a in level_airports if a in SHIPPED and a not in ("LTFM", "LFPG")] if "level" in pieces else []
    if level_set:
        lv = level_layer(f, pub, base, level_set, lam_cap=lam_cap, own_bias=own_bias, guard=guard, usable=usable, counters=cnt, mod30=mod30, note=notes, dstore=dstore, stop_days=stop_days)
        lv = np.rint(lv)
        cur = cur + (lv - base)
        fire("daily_level", base, lv)
    if "types" in pieces:
        hed = day_early_hedge(f) if rule_hedge else np.zeros(f.height, bool)
        notes.append(f"Rome types: rule-derived day-early hedge {'ON: ' + str(int(hed.sum())) + ' rows kept' if rule_hedge else 'OFF (as shipped; the six named flights are handled by final/31)'}")
        f2 = f.with_columns(pl.Series("hedged_rule", hed))
        ce = None
        if count_equation:
            m36 = _load_count_module()
            ce = m36.CountEq(ifr=m36.load_ifr("LIRF"), errs=m36.served_error_2025())
        nt, setm = rome_types_layer(f2, pub, base, guard, usable, notes, update_null_flight, count_eq=ce)
        # the posterior supersedes the level shift on the rows it sets
        cur = np.where(setm, nt, cur)
        fire("rome_record_types", base, np.where(setm, nt, base))
    cur = bound_rows(f, cur, np.abs(cur - base) > 0.5, window)
    for apt, piece in (("LTFM", "ltfm"), ("LSZH", "lszh"), ("LFPG", "lfpg")):
        if piece in pieces and apt in level_airports:
            f_cur = f.drop("pred").with_columns(pl.Series("pred", cur)).with_columns(
                ((C("gs") >= AMB_GAP) & (C("pred") >= AMB_PRED)).alias("amb"), ((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("p_copy"),
                (C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day"))
            lv = level_layer(f_cur, pub, cur, [apt], lam_cap=lam_cap, own_bias=own_bias, guard=guard, usable=usable, counters=cnt, mod30=mod30, note=notes, dstore=dstore, stop_days=stop_days)
            lv = bound_rows(f, lv, np.abs(lv - cur) > 0.5, window)
            fire(f"daily_level_{apt}", cur, lv)
            cur = lv
    if cell_split:
        cs = cell_split_layer(f, cur, dstore, census, notes)
        cs = bound_rows(f, cs, np.abs(cs - cur) > 0.5, window)
        fire("daily_cell_split", cur, cs)
        cur = cs
    if date_guard:
        g = keep_offblock_date(f, base, cur)
        n = int((g != cur).sum())
        if n:
            ch = pl.DataFrame({"apt": f["apt"], "mon": f["mon"], "clip": np.abs(g - cur)}).filter(pl.col("clip") > 0)
            for r in ch.group_by("apt", "mon").agg(n=pl.len(), mx=pl.col("clip").max()).sort("apt", "mon").rows():
                print(f"  date guard: {r[0]} month {r[1]}: {r[2]} rows clipped, max clip {r[3]:.0f} s")
        print(f"  date guard: {n:,} rows kept on their original off-block date")
        cur = g
    out = f.select(C("id").alias(ID_COL)).with_columns(pl.Series(TARGET_COL, cur))
    cnt.print(months)
    return out, notes


def stop_report_print() -> None:
    """Per airport and month: days excluded by the signature, days with the deficit alone (no weather condition), days the METAR rule would exclude."""
    if not STOP_REPORT:
        print("no airport-day evaluated")
        return
    r = pl.DataFrame(STOP_REPORT).with_columns(C("day").dt.month().alias("month"))
    print("\nstop days by airport and month (days evaluated by the level layer; z = (published - implied - 2025 median) / sigma)")
    for apt in sorted(set(r["apt"].to_list())):
        g = r.filter(C("apt") == apt)
        for m in sorted(set(g["month"].to_list())):
            h = g.filter(C("month") == m)
            fmt = lambda col: ", ".join(f"{d.day:02d}" + (f"({z:+.1f})" if col != "metar_rule" else "") for d, z in zip(h.filter(C(col))["day"], h.filter(C(col))["z"]))  # noqa: E731
            print(f"  {apt} m{m:02d}: days {h.height:2d} | signature [{fmt('signature')}] | deficit only [{fmt('deficit_only')}] | METAR rule [{fmt('metar_rule')}]")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--out", type=Path)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--on", dest="on", action="store_true", default=True)
    g.add_argument("--off", dest="on", action="store_false")
    ap.add_argument("--pieces", default="level,types,ltfm,lfpg", help="comma list of level, types, ltfm, lfpg, lszh (lszh is an approximation of the shipped joint solve and is off by default)")
    ap.add_argument("--window", type=float, default=3618.0, help="plan-window half-width in s applied to rows a piece touched (3,606 + 12 s jitter allowance; 0 reproduces nothing, use 3606 for the old files)")
    ap.add_argument("--months", default="")
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--shipped", action="store_true", help="the January/July form: shipped constants, six named flights only, null flight strings not updated (the defaults below are the February/June build)")
    ap.add_argument("--stop-days", dest="stop_days", choices=["signature", "metar", "ltfm", "off"], default="signature",
                    help="stop days left unshifted. signature (default): snow report AND published total more than 3 sigma below the implied total, every airport, own gate parameters; "
                         "metar: Istanbul signature plus EVERY snow-report day at EDDF EDDM LSZH EHAM (the earlier default, never validated); ltfm: Istanbul signature only (the January/July file); off: none")
    ap.add_argument("--stop-report", action="store_true", help="print per airport the days the signature, the weather-free deficit and the METAR rule exclude, then exit")
    ap.add_argument("--lam-cap", type=float, default=0.5, help="cap on the shrink factor of every airport (default 0.5, validation (a) of docs/LAB-NOTES.md)")
    ap.add_argument("--own-bias", dest="own_bias", action="store_true", default=True, help="bias of each month = median day shift of that month's own published series (label-free); default on")
    ap.add_argument("--no-own-bias", dest="own_bias", action="store_false")
    ap.add_argument("--no-edge-guard", action="store_true")
    ap.add_argument("--no-date-guard", action="store_true", help="allow a shifted row to move to another UTC off-block date (default: clipped to stay on its original date)")
    ap.add_argument("--rule-hedge", dest="rule_hedge", action="store_true", default=True, help="(default on) month-agnostic alternative: keep Rome no-plan rows with a gap of 14 h or more and a later local take-off day at their input value, instead of the six named flights of final/31")
    # The January/July file left Rome no-plan rows with a missing flight string unmoved: `flight not in HEDGED` is null for them and the
    # filter drops the row. That is a bug, not a decision; the default here updates them like every other row.
    ap.add_argument("--update-null-flight", dest="update_null_flight", action="store_true", default=True, help="(default on) update Rome no-plan rows whose flight string is missing")
    ap.add_argument("--keep-null-flight", dest="update_null_flight", action="store_false")
    ap.add_argument("--cell-split", action="store_true", help="v162 slopes per coverage class and lateness bin at EDDF EDDM EGLL LIRF LFPG")
    ap.add_argument("--census", type=Path, default=INTERIM / "agent_coverage_census_2026.parquet", help="coverage class per departure id (columns MVT_ID_mvt, category) for --cell-split")
    ap.add_argument("--count-equation", dest="count_equation", action="store_true", help="Rome record types: also use the published IFR departure count (final/36_rome_count.py, docs/LAB-NOTES.md); default off")
    a = ap.parse_args()
    if a.shipped:
        a.lam_cap, a.own_bias, a.rule_hedge, a.update_null_flight, a.stop_days = None, False, False, False, "ltfm"
    pred = pl.read_parquet(a.pred)
    if not a.on:
        print("published daily layer: OFF, input unchanged")
        out = pred
    else:
        months = [int(t) for t in a.months.split(",") if t.strip()] or None
        pieces = set(t for t in a.pieces.split(",") if t)
        la = ["EDDF", "EDDM", "EGLL", "LEBL", "LEMD", "LIRF", "LTFM", "LSZH", "LFPG"]
        out, notes = run(pred, a.ranking, pieces=pieces, months=months, sample=a.sample, lam_cap=a.lam_cap, own_bias=a.own_bias,
                         guard=not a.no_edge_guard, level_airports=la, window=a.window, rule_hedge=a.rule_hedge, update_null_flight=a.update_null_flight, cell_split=a.cell_split, census=a.census, stop_days=a.stop_days, count_equation=a.count_equation, date_guard=not a.no_date_guard)
        for n in notes:
            print("  note:", n)
        if a.stop_report:
            stop_report_print()
            return
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        out.select(ID_COL, C(TARGET_COL).cast(pred.schema[TARGET_COL])).write_parquet(a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
