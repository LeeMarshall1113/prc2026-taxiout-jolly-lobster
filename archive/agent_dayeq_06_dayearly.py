"""day-equation 06: does any published daily series respond to a stamp HOURS BEFORE the schedule (the day-early type)?
2025 Rome day-early records (label > 50,000 s): list their days; compare every daily column (all-cause delay identity residual, ATC delay,
slot-adherence counts, flight counts) on those days against the neighbouring days (-3..+3), in units of the column's own day-to-day noise.
Also: a signed sum of (B - S) over our rows against the published columns (does any column carry the negative part?)."""
import glob
import numpy as np
import polars as pl

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(100)
C = pl.col
R = ""
E = R + "data/external/eurocontrol/"
files = sorted(glob.glob(R + "data/raw/training_*.parquet"))
tr = (pl.scan_parquet(files).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("SCHED_TIME_UTC_mvt").is_not_null())
      .select(C("MVT_TIME_UTC_mvt").alias("T"), C("SCHED_TIME_UTC_mvt").alias("S"), C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"), C("FLIGHT_mvt").alias("fs"), C("FLIGHT_ID_mvt").is_null().alias("np"))
      .with_columns((C("T") - pl.duration(seconds=C("y"))).alias("B")).with_columns(((C("B") - C("S")).dt.total_seconds() / 60.0).alias("dm"), (C("T") - C("S")).dt.total_seconds().alias("gs"))
      .with_columns(C("B").dt.date().alias("bday"), C("T").dt.date().alias("tday"), C("S").dt.date().alias("sday")).collect())
de = tr.filter(C("y") > 50000).sort("T")
print("2025 Rome day-early records (label > 50,000 s):", de.height)
print(de.select("fs", "np", "S", "B", "T", "y", "gs", "dm", "bday", "tday"))


def load(name):
    d = pl.read_csv(f"{E}{name}_2025.csv", infer_schema_length=0).filter(C("APT_ICAO") == "LIRF")
    d = d.with_columns(C("FLT_DATE").str.slice(0, 10).str.to_date().alias("day"))
    num = [c for c in d.columns if c.startswith(("FLT_", "DLY_")) and c != "FLT_DATE"]
    return d.with_columns([C(c).cast(pl.Float64, strict=False) for c in num]).select(["day"] + num)


pub = load("all_pre_departure_delays").join(load("atc_pre_departure_delays"), on="day", how="full", coalesce=True, suffix="_atc").join(load("atfm_slot_adherence"), on="day", how="full", coalesce=True, suffix="_slot")
ours = tr.group_by(C("bday").alias("day")).agg(n=pl.len(), s_pos=C("dm").clip(lower_bound=0).sum(), s_neg=C("dm").clip(upper_bound=0).sum(), s_all=C("dm").sum(), n_early15=(C("dm") < -15).sum(), n_early60=(C("dm") < -60).sum(), n_late15=(C("dm") > 15).sum())
j = pub.join(ours, on="day", how="inner").with_columns((C("DLY_ALL_PRE_2") - C("s_pos")).alias("res_pos"), (C("FLT_DEP_1") - C("n")).alias("dn")).sort("day")
cols = ["res_pos", "dn", "FLT_DEP_IFR_2", "DLY_ATC_PRE_2", "FLT_DEP_3", "DLY_ATC_PRE_3", "FLT_DEP_REG_1", "FLT_DEP_OUT_EARLY_1", "FLT_DEP_IN_1", "FLT_DEP_OUT_LATE_1"]
# day-to-day noise of each column: robust sd of the difference from the mean of days -3..-1, +1..+3
arr = {c: j[c].to_numpy().astype(float) for c in cols}; dayv = j["day"].to_list(); idx = {d: i for i, d in enumerate(dayv)}
def local_anom(c, i):
    nb = [arr[c][k] for k in range(i - 3, i + 4) if 0 <= k < len(dayv) and k != i]
    return arr[c][i] - np.nanmean(nb)
noise = {c: 1.4826 * np.nanmedian(np.abs([local_anom(c, i) for i in range(3, len(dayv) - 3)])) for c in cols}
print("\nday-to-day noise (robust sd of the local anomaly):", {c: round(v, 1) for c, v in noise.items()})
print("\nday-early records: anomaly of each column on the STAMP day (bday) and on the take-off day (tday), in noise units")
rows = []
for r in de.iter_rows(named=True):
    for which in ["bday", "tday"]:
        d = r[which]
        if d not in idx:
            continue
        i = idx[d]
        rows.append(dict(fs=r["fs"], which=which, day=d, gap_h=round(r["gs"] / 3600, 1), stamp_minus_sched_h=round(r["dm"] / 60, 1), **{c: round(local_anom(c, i) / max(noise[c], 1e-9), 1) for c in cols}))
print(pl.DataFrame(rows))
# signed sums: is any published column a function of the negative part?
print("\ncorrelation of published columns with our signed/negative sums (365 days):")
for c in ["DLY_ALL_PRE_2", "DLY_ATC_PRE_2", "FLT_DEP_3", "DLY_ATC_PRE_3", "FLT_DEP_REG_1", "FLT_DEP_OUT_EARLY_1", "FLT_DEP_IN_1", "FLT_DEP_OUT_LATE_1"]:
    y = j[c].to_numpy().astype(float)
    out = {}
    for o in ["s_pos", "s_neg", "s_all", "n_early15", "n_early60", "n_late15"]:
        x = j[o].to_numpy().astype(float); ok = np.isfinite(x) & np.isfinite(y)
        out[o] = round(float(np.corrcoef(x[ok], y[ok])[0, 1]), 3) if ok.sum() > 20 and x[ok].std() > 0 and y[ok].std() > 0 else None
    print(c, out)
# partial: residual of the positive-part identity against the negative part
x = j["s_neg"].to_numpy(); yv = j["res_pos"].to_numpy(); ok = np.isfinite(x) & np.isfinite(yv)
print("identity residual vs negative-part sum: corr", round(float(np.corrcoef(x[ok], yv[ok])[0, 1]), 3), "slope", round(float(np.polyfit(x[ok], yv[ok], 1)[0]), 4))
