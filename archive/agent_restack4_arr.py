"""experiment restack4 arr: ARRIVAL-ADS-B neighbour columns (docs/LAB-NOTES.md, ideas #11). Label-free: only
the ADS-B fixes of data/adsb/points/<day>.parquet are read for arrivals (no ARR movement row, no block time, no label);
departures contribute only their own MVT time, airport, UTC day and runway designator (all present in the scored file).

    python archive/agent_restack4_arr.py --mode 2025 --out data/interim/agent_restack4_arr_2025.parquet
    python archive/agent_restack4_arr.py --mode 2026 --out data/interim/agent_restack4_arr_2026.parquet

Touchdowns = runs_and_tracks()'s `touch` definition (archive/agent_noplan_adsb_match.py, default Gates): ground flag
alt == -1 or (|alt - field elevation| <= 800 ft and gs <= 40 kt); a ground run breaks on a silence > gap_s (1200 s); a
touchdown is a ground run whose previous fix was airborne and at most liftoff_s (600 s) earlier.
Per touchdown: land_hdg = median of `track` (unwrapped about the first value) over the run's fixes with gs >= 40 kt in
the first 120 s; in_t = first fix starting a stationary spell (consecutive fixes gs <= 1 kt, >= 2 fixes, spanning >= 30 s)
at or after td_t + 30 s, else (no such spell) the run's LAST fix if its gs <= 3 kt (reception ends at the stand);
ti = in_t - td_t kept in [60, 3600] s. (Amended 2026-09-25 00:20Z after the label-free smoke test found the first
definition -- >= 3 fixes over >= 60 s, no fallback -- on only 5.2% of touchdowns; before any fit using ARR.)
"""
import argparse
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl

C = pl.col
ARR = ["a_n15", "a_n30", "a_nr15", "a_ti30", "a_nti30", "a_tir30", "a_tiday", "a_occ0", "a_occ10"]
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]


def _ang(a, b):
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def touchdowns(day, points_dir="data/adsb/points"):
    from archive.agent_noplan_adsb_match import ELEVATION_FT, Gates
    g = Gates()
    p = (pl.read_parquet(f"{points_dir}/{day}.parquet", columns=["apt", "icao", "t", "alt", "gs", "track"])
         .filter(C("icao").is_not_null() & C("apt").is_in(APTS)).sort(["apt", "icao", "t"]))
    elev = C("apt").replace_strict(ELEVATION_FT, default=None, return_dtype=pl.Float64)
    if g.slow_low_ground:
        slow_low = ((C("alt") != -1) & ((C("alt").cast(pl.Float64) - elev).abs() <= g.slow_low_band_ft)
                    & (C("gs") <= g.slow_low_gs_kt)).fill_null(False)
        p = p.with_columns(((C("alt") == -1) | slow_low).alias("ground"))
    else:
        p = p.with_columns((C("alt") == -1).alias("ground"))
    p = p.with_columns(C("t").shift(1).over(["apt", "icao"]).alias("prev_t"),
                       C("ground").shift(1).over(["apt", "icao"]).alias("prev_ground"))
    gr = p.filter(C("ground")).with_columns(
        (C("prev_t").is_null() | ~C("prev_ground").fill_null(False) | ((C("t") - C("prev_t")) > g.gap_s)).alias("run_start"))
    gr = gr.with_columns(C("run_start").cast(pl.Int32).cum_sum().over(["apt", "icao"]).alias("run"))
    runs = (gr.group_by(["apt", "icao", "run"])
            .agg(C("t").min().alias("start_t"), C("prev_t").first().alias("before_t"),
                 C("prev_ground").first().alias("before_ground")))
    touch = runs.filter((C("before_ground") == False) & ((C("start_t") - C("before_t")) <= g.liftoff_s))  # noqa: E712
    fx = gr.join(touch.select("apt", "icao", "run", "start_t"), on=["apt", "icao", "run"], how="inner").sort(["apt", "icao", "run", "t"])
    out = {"apt": [], "td": [], "hdg": [], "ti": []}
    for (apt, _icao, _run), s in fx.partition_by(["apt", "icao", "run"], as_dict=True, maintain_order=True).items():
        t = s["t"].to_numpy().astype(float)
        gs = s["gs"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        tr = s["track"].cast(pl.Float64).fill_null(np.nan).to_numpy()
        td = t[0]
        k = np.flatnonzero((t <= td + 120) & (gs >= 40) & np.isfinite(tr))
        hdg = np.nan
        if len(k):
            ref = tr[k[0]]
            hdg = float((np.median(ref + ((tr[k] - ref + 180.0) % 360.0 - 180.0))) % 360.0)
        ti = np.nan
        slow = np.isfinite(gs) & (gs <= 1.0)
        n = len(t)
        i = int(np.searchsorted(t, td + 30, "left"))
        while i < n:
            if slow[i]:
                j = i
                while j + 1 < n and slow[j + 1]:
                    j += 1
                if j - i + 1 >= 2 and t[j] - t[i] >= 30:
                    v = t[i] - td
                    ti = v if 60 <= v <= 3600 else np.nan
                    break
                i = j + 1
            else:
                i += 1
        else:  # no stationary spell: reception ended while slow (transponder off at the stand) -> the last fix
            if n and np.isfinite(gs[-1]) and gs[-1] <= 3.0:
                v = t[-1] - td
                ti = v if 60 <= v <= 3600 else np.nan
        out["apt"].append(apt)
        out["td"].append(td)
        out["hdg"].append(hdg)
        out["ti"].append(ti)
    return pl.DataFrame(out, schema={"apt": pl.String, "td": pl.Float64, "hdg": pl.Float64, "ti": pl.Float64})


def rwy_heading(rwy):
    s = rwy.str.extract(r"^(\d{2})", 1).cast(pl.Float64)
    return (s * 10.0) % 360.0


def features_day(dep, tdf):
    """dep: one airport-day of departures (MVT_ID_mvt, t, h); tdf: that airport-day's touchdowns. Returns ARR dict arrays."""
    n = dep.height
    t = dep["t"].to_numpy().astype(float)
    h = dep["h"].to_numpy().astype(float)
    f = {c: np.full(n, np.nan) for c in ARR}
    for c in ("a_n15", "a_n30", "a_nr15", "a_nti30", "a_occ0", "a_occ10"):
        f[c] = np.zeros(n)
    if tdf.height == 0:
        return f
    td = tdf["td"].to_numpy().astype(float)
    hd = tdf["hdg"].to_numpy().astype(float)
    ti = tdf["ti"].to_numpy().astype(float)
    o = np.argsort(td, kind="stable")
    td, hd, ti = td[o], hd[o], ti[o]

    def cnt(tt, W):
        return (np.searchsorted(tt, t + W, "right") - np.searchsorted(tt, t - W, "left")).astype(float)

    f["a_n15"], f["a_n30"] = cnt(td, 900), cnt(td, 1800)
    v = np.isfinite(ti)
    tv, iv = td[v], ti[v]
    cs = np.concatenate([[0.0], np.cumsum(iv)])
    lo, hi = np.searchsorted(tv, t - 1800, "left"), np.searchsorted(tv, t + 1800, "right")
    nn = (hi - lo).astype(float)
    f["a_nti30"] = nn
    f["a_ti30"] = np.where(nn > 0, (cs[hi] - cs[lo]) / np.maximum(nn, 1), np.nan)
    f["a_tiday"] = np.full(n, float(iv.mean()) if len(iv) else np.nan)
    ins = np.sort(tv + iv)
    for c, x in (("a_occ0", t), ("a_occ10", t - 600)):
        f[c] = (np.searchsorted(tv, x, "right") - np.searchsorted(ins, x, "right")).astype(float)
    for hv in np.unique(h[np.isfinite(h)]):
        rows = h == hv
        same = np.isfinite(hd) & (_ang(hd, hv) <= 30)
        ts = td[same]
        f["a_nr15"][rows] = (np.searchsorted(ts, t[rows] + 900, "right") - np.searchsorted(ts, t[rows] - 900, "left"))
        sv = same & v
        tsv, isv = td[sv], ti[sv]
        cs2 = np.concatenate([[0.0], np.cumsum(isv)])
        lo2, hi2 = np.searchsorted(tsv, t[rows] - 1800, "left"), np.searchsorted(tsv, t[rows] + 1800, "right")
        n2 = (hi2 - lo2).astype(float)
        f["a_tir30"][rows] = np.where(n2 > 0, (cs2[hi2] - cs2[lo2]) / np.maximum(n2, 1), np.nan)
    f["a_nr15"][~np.isfinite(h)] = np.nan   # no runway designator: the same-group count is undefined
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["2025", "2026"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", default="")
    a = ap.parse_args()
    T0 = time.time()
    src = "data/raw/training_*.parquet" if a.mode == "2025" else "data/raw/ranking.parquet"
    dep = (pl.scan_parquet(src).filter((C("PHASE_mvt") == "DEP") & C("ADEP_mvt").is_in(APTS))
           .select(C("MVT_ID_mvt").cast(pl.Float64), C("ADEP_mvt").alias("apt"),
                   C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m-%d").alias("day"),
                   C("MVT_TIME_UTC_mvt").dt.epoch("s").cast(pl.Float64).alias("t"), C("RUNWAY_mvt").alias("rwy"))
           .collect().unique(subset="MVT_ID_mvt", keep="first"))
    if a.days:
        days = a.days.split(",")
    elif a.mode == "2025":
        days = sorted(pl.read_parquet("data/interim/agent_offblock_rows_c238.parquet", columns=["day"])["day"].unique().to_list())
    else:
        days = sorted(dep["day"].unique().to_list())
    dep = dep.filter(C("day").is_in(days)).with_columns(rwy_heading(C("rwy")).alias("h"))
    print(f"{a.mode}: {dep.height:,} departures on {len(days)} days", flush=True)
    parts, tds = [], []
    import os
    for i, d in enumerate(days):
        dd = dep.filter(C("day") == d)
        if not os.path.exists(f"data/adsb/points/{d}.parquet"):
            parts.append(dd.select("MVT_ID_mvt").with_columns(*[pl.lit(None, pl.Float64).alias(c) for c in ARR]))
            print(f"  {d}: NO points file", flush=True)
            continue
        tdf = touchdowns(d)
        tds.append(tdf.with_columns(pl.lit(d).alias("day")))
        for (apt,), s in dd.partition_by(["apt"], as_dict=True).items():
            f = features_day(s, tdf.filter(C("apt") == apt))
            parts.append(pl.DataFrame({"MVT_ID_mvt": s["MVT_ID_mvt"], **{c: f[c] for c in ARR}}))
        if i % 20 == 0:
            print(f"  {d}: {tdf.height:,} touchdowns, ti valid {float(tdf['ti'].is_not_nan().mean()):.2f} "
                  f"({i + 1}/{len(days)}, {time.time() - T0:.0f}s)", flush=True)
    out = pl.concat(parts, how="vertical_relaxed")
    assert out["MVT_ID_mvt"].n_unique() == out.height
    out = out.join(dep.select("MVT_ID_mvt", "apt"), on="MVT_ID_mvt", how="left")
    out.write_parquet(a.out)
    T = pl.concat(tds) if tds else pl.DataFrame(schema={"apt": pl.String, "td": pl.Float64, "hdg": pl.Float64, "ti": pl.Float64})
    print(f"touchdowns {T.height:,}; hdg known {float(T['hdg'].is_not_nan().mean()):.3f}; ti valid "
          f"{float(T['ti'].is_not_nan().mean()):.3f}; median ti {float(T['ti'].drop_nans().median()):.0f}s")
    print(out.group_by("apt").agg(pl.len().alias("n"), *[C(c).fill_nan(None).mean().round(2).alias(c) for c in ARR]).sort("apt"))
    print(f"-> {a.out} ({time.time() - T0:.0f}s)")


if __name__ == "__main__":
    main()
