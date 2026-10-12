"""agent_wxev (docs/LAB-NOTES.md): screen EUROCONTROL-style adverse-weather EVENT
flags (thunderstorm, snow/freezing precip, LVP, heavy rain, gust, lagged/cumulative-duration
compounds) against the current production control's residual. Read-only: nothing here trains a
model or touches the bucket/truthing.parquet. Bars are pre-registered in the note above this
script's own output.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "archive")

import numpy as np
import polars as pl

from agent_cong_common import Proj, partial_rho  # noqa: E402
from prc.weather import load as load_weather  # noqa: E402

C = pl.col
ID = "MVT_ID_mvt"
PLACEBO_FLOOR = 0.019
BAR = 5 * PLACEBO_FLOOR
FEATCACHE = "data/interim/train_features.parquet"
NM_COLS = ['nm_apt_fol15_mean', 'nm_apt_fol15_n', 'nm_apt_fol60_mean', 'nm_apt_fol60_n',
           'nm_apt_past15_mean', 'nm_apt_past15_n', 'nm_apt_past60_mean', 'nm_apt_past60_n',
           'nm_rwy_fol15_mean', 'nm_rwy_fol15_n', 'nm_rwy_fol60_mean', 'nm_rwy_fol60_n',
           'nm_rwy_fol60_own_minus', 'nm_rwy_next_proxy', 'nm_rwy_next_wait',
           'nm_rwy_past15_mean', 'nm_rwy_past15_n', 'nm_rwy_past60_mean', 'nm_rwy_past60_n',
           'nm_rwy_past60_own_minus']

EVENT_FLAGS = ["wxev_ts", "wxev_snowfz", "wxev_lvp", "wxev_heavyrain", "wxev_gust", "wxev_adverse"]
LAG_FLAGS = ["wxev_adverse_lag1h", "wxev_adverse_lag2h", "wxev_adverse_lag3h"]
ALL_FLAGS = EVENT_FLAGS + LAG_FLAGS + ["wxev_adverse_cumdur_today"]


def build_event_table() -> pl.DataFrame:
    """Per-observation event flags, lags and cumulative same-day duration, per station."""
    wx = load_weather()
    # weather.load() drops wxcodes after deriving its own flags -- rebuild the present-weather
    # string ourselves so we can apply the narrower event definitions (SN/FZRA/FZDZ, +RA only,
    # not the shipped wx_freezing/wx_heavy_precip which are broader).
    raw = []
    for station in ["LTFM", "LFPG", "EGLL", "EHAM", "LEMD", "EDDF", "LEBL", "LIRF", "EDDM", "LSZH"]:
        path = f"data/raw/weather/{station}.csv"
        r = pl.read_csv(path, null_values=["M", "", "None"], infer_schema_length=0)
        raw.append(r.select(
            pl.col("station").alias("wx_station"),
            pl.col("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False)
              .dt.replace_time_zone("UTC").alias("wx_time"),
            pl.col("wxcodes").fill_null("").alias("wxcodes"),
        ))
    codes_df = pl.concat(raw, how="vertical_relaxed").drop_nulls("wx_time") \
        .sort("wx_station", "wx_time").unique(subset=["wx_station", "wx_time"], keep="last") \
        .sort("wx_station", "wx_time")

    wx = wx.join(codes_df, on=["wx_station", "wx_time"], how="left")
    wxc = pl.col("wxcodes").fill_null("")

    wx = wx.with_columns(
        wxc.str.contains("TS").cast(pl.Int8).alias("wxev_ts"),
        (wxc.str.contains("SN") | wxc.str.contains("FZRA") | wxc.str.contains("FZDZ"))
            .cast(pl.Int8).alias("wxev_snowfz"),
        ((pl.col("wx_visibility_km") < 0.55) | (pl.col("wx_ceiling_ft") < 200))
            .cast(pl.Int8).alias("wxev_lvp"),
        (wxc.str.contains(r"\+RA") | wxc.str.contains(r"\+SHRA")).cast(pl.Int8).alias("wxev_heavyrain"),
        (pl.col("wx_gust_kt") >= 30).cast(pl.Int8).alias("wxev_gust"),
    ).with_columns(
        (pl.max_horizontal("wxev_ts", "wxev_snowfz", "wxev_lvp", "wxev_heavyrain", "wxev_gust"))
            .alias("wxev_adverse")
    )

    # lags (vectorised via searchsorted on the sub-array of event times -- strictly PRIOR
    # observations only, excludes the current row) + cumulative same-day duration (single
    # forward pass per station, O(n)).
    out = []
    for station, g in wx.sort("wx_time").group_by("wx_station", maintain_order=True):
        g = g.sort("wx_time")
        t = g["wx_time"].to_numpy()
        ev = g["wxev_adverse"].to_numpy().astype(bool)
        n = len(t)
        t_ev = t[ev]

        lag_cols = {}
        for hrs in (1, 2, 3):
            lo = t - np.timedelta64(hrs, "h")
            lo_idx = np.searchsorted(t_ev, lo, side="left")
            hi_idx = np.searchsorted(t_ev, t, side="left")  # excludes t[i] itself
            lag_cols[f"wxev_adverse_lag{hrs}h"] = (hi_idx > lo_idx).astype(np.int8)

        day = g["wx_time"].dt.date().to_numpy()
        cumdur = np.zeros(n, dtype=float)
        run_minutes = 0.0
        prev_t = None
        run_day = None
        for i in range(n):
            if run_day != day[i]:
                run_day = day[i]
                run_minutes = 0.0
                prev_t = None
            if ev[i] and prev_t is not None:
                run_minutes += min((t[i] - prev_t) / np.timedelta64(1, "m"), 90.0)
            if ev[i]:
                prev_t = t[i]
            cumdur[i] = run_minutes

        out.append(g.with_columns(
            pl.Series("wxev_adverse_lag1h", lag_cols["wxev_adverse_lag1h"]),
            pl.Series("wxev_adverse_lag2h", lag_cols["wxev_adverse_lag2h"]),
            pl.Series("wxev_adverse_lag3h", lag_cols["wxev_adverse_lag3h"]),
            pl.Series("wxev_adverse_cumdur_today", cumdur),
        ))
    return pl.concat(out, how="vertical_relaxed")


def attach_events(movements: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    keep = ["wx_station", "wx_time"] + ALL_FLAGS
    return (movements.sort("MVT_TIME_UTC_mvt")
            .join_asof(events.select(keep).sort("wx_time"),
                       left_on="MVT_TIME_UTC_mvt", right_on="wx_time",
                       by_left="adep", by_right="wx_station", strategy="backward")
            .drop([c for c in ("wx_time", "wx_station") if c in movements.columns]))


def load_fold_departures(ids: np.ndarray) -> pl.DataFrame:
    dep = (pl.scan_parquet("data/raw/training_*.parquet")
           .filter(C("PHASE_mvt") == "DEP")
           .select(C(ID), C("ADEP_mvt").alias("adep"), C("MVT_TIME_UTC_mvt"))
           .collect())
    idset = set(ids.tolist())
    dep = dep.filter(pl.col(ID).cast(pl.Float64).is_in(list(idset)))
    return dep


def load_board_departures() -> pl.DataFrame:
    return (pl.scan_parquet("data/raw/ranking.parquet")
            .filter(C("PHASE_mvt") == "DEP")
            .select(C(ID), C("ADEP_mvt").alias("adep"), C("MVT_TIME_UTC_mvt"))
            .collect())


def build_design(base, apts, ctrl_cols_arr, ctrl_missing_arr):
    n = len(base)
    X = [np.ones(n), base, base ** 2 / 1e3]
    for arr, miss in zip(ctrl_cols_arr, ctrl_missing_arr):
        X.append(arr)
        if miss is not None:
            X.append(miss)
    for a in np.unique(apts)[1:]:
        X.append((apts == a).astype(float))
    return np.column_stack(X)


def main():
    d0 = np.load("data/interim/agent_stagecap_RS_E9S_s0.npz", allow_pickle=True)
    dA = np.load("data/interim/agent_blend_oof_seedA.npz", allow_pickle=True)
    dV = np.load("data/interim/agent_blend_v109analogue.npz", allow_pickle=True)
    ids = d0["ids"].astype(float)
    y = d0["y"].astype(float)
    apts = np.asarray(d0["apts"]).astype(str)
    ctrl0 = d0["both_guard"].astype(float) + (dA["both_guard"].astype(float) - dV["both_guard"].astype(float))
    r = y - ctrl0
    print(f"pooled rows: {len(r):,}  RMSE(CTRL_0)={np.sqrt(np.mean(r**2)):.3f}")

    print("building event table from local METAR (no download)...")
    events = build_event_table()

    print("loading fold departures + joining events...")
    dep = load_fold_departures(ids)
    dep = attach_events(dep, events)
    pos = {m: i for i, m in enumerate(dep[ID].cast(pl.Float64).to_numpy())}
    order = np.array([pos.get(m, -1) for m in ids])
    ok = order >= 0
    print(f"matched {ok.sum():,} / {len(ids):,} fold rows to a departure record")

    flag_arr = {}
    for f in ALL_FLAGS:
        v = dep[f].fill_null(0).cast(pl.Float64).to_numpy()
        col = np.full(len(ids), np.nan)
        col[ok] = v[order[ok]]
        flag_arr[f] = col

    # controls: nm_* from the cache
    cache = pl.scan_parquet(FEATCACHE).select([ID] + NM_COLS).collect()
    cpos = {m: i for i, m in enumerate(cache[ID].cast(pl.Float64).to_numpy())}
    corder = np.array([cpos.get(m, -1) for m in ids])
    cok = corder >= 0
    ctrl_cols_arr, ctrl_missing_arr = [], []
    for c in NM_COLS:
        v = cache[c].cast(pl.Float64).to_numpy()
        col = np.full(len(ids), np.nan)
        col[cok] = v[corder[cok]]
        miss = (~np.isfinite(col)).astype(float)
        ctrl_cols_arr.append(np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0))
        ctrl_missing_arr.append(miss if miss.sum() > 0 else None)

    valid = ok & cok & np.isfinite(r)
    print(f"rows usable for the instrument: {valid.sum():,}")

    base_v = ctrl0[valid]
    r_v = r[valid]
    apt_v = apts[valid]
    X = build_design(base_v, apt_v, [a[valid] for a in ctrl_cols_arr], [m[valid] if m is not None else None for m in ctrl_missing_arr])
    proj = Proj(X)

    results = {"bar": BAR, "placebo_floor": PLACEBO_FLOOR, "n_rows": int(valid.sum()), "pooled": {}, "per_airport": {}, "top_decile": {}, "mean_residual": {}, "frequency": {}}

    for f in ALL_FLAGS:
        v = flag_arr[f][valid]
        v = np.nan_to_num(v, nan=0.0)
        rho = partial_rho(proj, v, r_v)
        results["pooled"][f] = rho
        print(f"  pooled partial_rho[{f}] = {rho:+.4f}  ({'PASS' if abs(rho) >= BAR else 'below bar'})")

    for a in np.unique(apt_v):
        m = apt_v == a
        n_ev = int(np.nan_to_num(flag_arr["wxev_adverse"][valid][m], nan=0.0).sum())
        entry = {"n_event_obs": n_ev, "n_rows": int(m.sum())}
        if n_ev >= 200:
            Xa = build_design(base_v[m], apt_v[m], [a2[valid][m] for a2 in ctrl_cols_arr], [mm[valid][m] if mm is not None else None for mm in ctrl_missing_arr])
            proja = Proj(Xa)
            for f in EVENT_FLAGS:
                v = np.nan_to_num(flag_arr[f][valid][m], nan=0.0)
                entry[f] = partial_rho(proja, v, r_v[m])
        results["per_airport"][a] = entry

    absr = np.abs(r_v)
    thresh = np.quantile(absr, 0.90)
    top = absr >= thresh
    print(f"top decile: {top.sum():,} rows, |r| >= {thresh:.1f}s")
    Xt = build_design(base_v[top], apt_v[top], [a2[valid][top] for a2 in ctrl_cols_arr], [mm[valid][top] if mm is not None else None for mm in ctrl_missing_arr])
    projt = Proj(Xt)
    for f in ALL_FLAGS:
        v = np.nan_to_num(flag_arr[f][valid][top], nan=0.0)
        rho = partial_rho(projt, v, r_v[top])
        results["top_decile"][f] = rho
        print(f"  top-decile partial_rho[{f}] = {rho:+.4f}  ({'PASS' if abs(rho) >= BAR else 'below bar'})")

    for f in EVENT_FLAGS:
        v = np.nan_to_num(flag_arr[f][valid], nan=0.0).astype(bool)
        vt = np.nan_to_num(flag_arr[f][valid][top], nan=0.0).astype(bool)
        results["mean_residual"][f] = {
            "pooled_event": float(r_v[v].mean()) if v.sum() else None,
            "pooled_nonevent": float(r_v[~v].mean()) if (~v).sum() else None,
            "pooled_n_event": int(v.sum()),
            "top_decile_event": float(r_v[top][vt].mean()) if vt.sum() else None,
            "top_decile_nonevent": float(r_v[top][~vt].mean()) if (~vt).sum() else None,
            "top_decile_n_event": int(vt.sum()),
        }
        print(f"  mean r  {f}: event={results['mean_residual'][f]['pooled_event']}  nonevent={results['mean_residual'][f]['pooled_nonevent']}  | top-decile event={results['mean_residual'][f]['top_decile_event']} nonevent={results['mean_residual'][f]['top_decile_nonevent']}")

    print("frequency: 2025 fold vs Jan/Jul 2026 board...")
    board = load_board_departures()
    board = attach_events(board, events)
    fold_n = int(valid.sum())
    board_n = board.height
    for f in EVENT_FLAGS:
        fold_rate = float(np.nan_to_num(flag_arr[f][valid], nan=0.0).mean())
        board_rate = float(board[f].fill_null(0).cast(pl.Float64).to_numpy().mean())
        results["frequency"][f] = {"fold_rate": fold_rate, "board_rate": board_rate, "ratio_board_over_fold": (board_rate / fold_rate) if fold_rate > 0 else None}
        print(f"  {f}: fold={fold_rate*100:.3f}%  board={board_rate*100:.3f}%  ratio={results['frequency'][f]['ratio_board_over_fold']}")
    results["frequency"]["_n"] = {"fold": fold_n, "board": board_n}

    with open("data/interim/agent_wxev_screen.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print("wrote data/interim/agent_wxev_screen.json")


if __name__ == "__main__":
    main()
