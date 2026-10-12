"""Snow-day exclusion rule R_snow, twelve 2025 months, at EDDF EDDM LSZH EHAM; design and gate in docs/LAB-NOTES.md section 7
(written before this was run). Same construction as final/33_daylayer_validation.py with and without the exclusion. Run once.
  bash final/2x_run.sh "why" final/36_snow_days_validation.py
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "archive"))
from agent_dayeq2_lib import C, DN_MAX, FRAME_MONTHS, I, R, SE_SCALE, load_pub, solve_delta  # noqa: E402

APTS = ["EDDF", "EDDM", "LSZH", "EHAM"]
LAMS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ROWRULES = ["R1", "R2", "R3a", "R3b"]


def snow_dates(apt: str) -> set:
    r = pl.read_csv(REPO / "data" / "raw" / "weather" / f"{apt}.csv", null_values=["M", "", "None"], infer_schema_length=0)
    r = r.select(C("valid").str.to_datetime("%Y-%m-%d %H:%M", strict=False).alias("t"), C("wxcodes").fill_null("").alias("wx")).drop_nulls("t")
    r = r.filter(C("wx").str.contains("SN") | C("wx").str.contains("FZRA") | C("wx").str.contains("FZDZ"))
    return set(r["t"].dt.date().to_list())


def one_airport(a: str, sv: pl.DataFrame, g12: pl.DataFrame):
    files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
    raw = (pl.scan_parquet(files).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == a))
           .select(C("MVT_ID_mvt").alias("id"), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"),
                   C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y")).collect())
    base = g12.join(sv, on="id", how="left").with_columns(pl.coalesce(C("served"), C("pk")).alias("pred"), C("rule").fill_null(""))
    f = raw.with_columns(C("id").cast(pl.Float64)).join(base.select("id", "pred", "rule"), on="id", how="inner")
    f = f.with_columns((C("T") - C("S")).dt.total_seconds().cast(pl.Float64).alias("gs"), (C("T") - pl.duration(seconds=C("pred"))).dt.date().alias("day_pred"))
    f = f.filter(C("gs").is_not_null() & C("y").is_not_null())
    f = f.with_columns(((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"), C("rule").is_in(ROWRULES).alias("rr"), C("day_pred").dt.month().alias("dmon"))
    f = f.with_columns(((C("gs") - C("pred")).clip(lower_bound=0) / 60.0).alias("dm"), (~C("amb") & ~C("rr")).alias("sh"))
    pub = load_pub(2025, [a])
    days = (f.group_by("apt", "day_pred").agg(n=pl.len(), fixed=C("dm").filter(~C("sh")).sum(), tot=C("dm").sum()).rename({"day_pred": "day"})
            .join(pub, on=["apt", "day"], how="inner"))
    days = days.filter(((C("n_pub") - C("n")).abs() <= DN_MAX)).with_columns((C("pub") - C("tot")).alias("D"), C("day").dt.month().alias("dm_"))
    parts = f.partition_by("apt", "day_pred", as_dict=True)
    rows = []
    for r in days.iter_rows(named=True):
        g = parts.get((r["apt"], r["day"]))
        m = g["sh"].to_numpy()
        rows.append({"apt": r["apt"], "day": r["day"], "d": solve_delta(g["gs"].to_numpy()[m], g["pred"].to_numpy()[m], r["pub"] - r["fixed"]) if (r["pub"] is not None and np.isfinite(r["pub"])) else np.nan})
    f = f.join(pl.DataFrame(rows), left_on=["apt", "day_pred"], right_on=["apt", "day"], how="left")
    sd = snow_dates(a)
    days = days.with_columns(C("day").is_in(list(sd)).alias("snow"))
    return f, days


def main() -> None:
    sv = pl.read_parquet(I + "agent_waithedge_tbl.parquet", columns=["id", "served", "rule"]).with_columns(C("id").cast(pl.Float64))
    g12 = pl.read_parquet(I + "agent_genaudit_12m.parquet", columns=["id", "pk"]).with_columns(C("id").cast(pl.Float64))
    tot = {}
    excl_rows = []
    for a in APTS:
        f, days = one_airport(a, sv, g12)
        yh, y, d, gs, mon = (f["pred"].to_numpy(), f["y"].to_numpy(), f["d"].to_numpy(), f["gs"].to_numpy(), f["dmon"].to_numpy())
        dayarr = f["day_pred"].to_list()
        pos = (gs > yh) & f["sh"].to_numpy() & np.isfinite(d)
        own = {m: float(np.median(d[(mon == m) & np.isfinite(d)])) for m in range(1, 13) if ((mon == m) & np.isfinite(d)).any()}
        bvec = np.array([own.get(int(m), np.nan) for m in mon])
        # gate per held-out month: b, sigma from non-snow days of the other months
        dm_ = days["dm_"].to_numpy()
        D = days["D"].to_numpy()
        snow = days["snow"].to_numpy()
        dlist = days["day"].to_list()
        exclude = set()
        info = []
        for m in range(1, 13):
            oth = (dm_ != m) & ~snow & np.isfinite(D)
            if oth.sum() < 20:
                continue
            b = float(np.median(D[oth]))
            sg = 1.4826 * float(np.median(np.abs(D[oth] - b)))
            te = (dm_ == m) & snow & np.isfinite(D)
            for i in np.flatnonzero(te):
                z = (D[i] - b) / sg
                if z < -3:
                    exclude.add(dlist[i])
                    info.append((str(dlist[i]), round(float(D[i]), 0), round(float(z), 1)))
        exmask = np.array([x in exclude for x in dayarr])
        gain_ex = 0.0
        gain_ex_by = {}
        for m in range(1, 13):
            te = mon == m
            if not te.any() or m not in own:
                continue
            trn = np.isin(mon, FRAME_MONTHS) & (mon != m) & pos & np.isfinite(bvec)
            best = None
            for lam in LAMS:
                sse = ((np.maximum(yh[trn] + lam * (d[trn] - bvec[trn]), 0) - y[trn]) ** 2).sum()
                if best is None or sse < best[0]:
                    best = (sse, lam)
            lam = best[1]
            new = yh.copy()
            s_ = te & pos
            new[s_] = np.maximum(yh[s_] + lam * (d[s_] - own[m]), 0)
            # gain from NOT shifting the excluded days: squared error of the shifted rows on excluded days minus that of the unshifted
            e = te & exmask
            g = float((((new - y) ** 2)[e].sum() - ((yh - y) ** 2)[e].sum()) / SE_SCALE)
            gain_ex += g
            if e.any():
                gain_ex_by[m] = round(g, 4)
        tot[a] = dict(excluded_days=len(exclude), gain_of_excluding=gain_ex, days=info, by_month=gain_ex_by, snow_days_total=int(snow.sum()), gated_days=int(len(snow)))
        print(a, tot[a])
    n_days = sum(v["excluded_days"] for v in tot.values())
    total = sum(v["gain_of_excluding"] for v in tot.values())
    print(f"\nexcluded days {n_days}; board-s gain of excluding (positive = excluding helps) {total:+.4f}")
    ok = total >= 0 and n_days >= 5
    print("RESULT:", "PASS" if ok else ("TOO FEW DAYS TO JUDGE (<5): conservative default = exclude every snow day" if n_days < 5 else "FAIL"))


if __name__ == "__main__":
    main()
