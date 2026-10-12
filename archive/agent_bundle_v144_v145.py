"""Two uploads in a fixed order (plan logged in docs/LAB-NOTES.md before either was scored).
 v144 = v143 + Munich January level shift removed (agent_apt_EDDM_final - v143; docs/LAB-NOTES.md)
             + Amsterdam on the plan-clock corrector term (agent_pcdx_final - v143; docs/LAB-NOTES.md)
 v145 = v144 + the calendar condition on the Rome day-early band rule (docs/LAB-NOTES.md): a day-early record is the actual
        clock time stamped on the schedule's local date, so it needs a take-off on a later local day; a same-local-day no-plan row inside a
        day-early band (gap 50,000-100,000 s) takes the schedule gap. 2025: 63 of 63 long-gap no-plan stamps on the schedule's date; same-day rows 18 copies, 1 real, 0 day-early."""
import numpy as np, polars as pl
C = pl.col; I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
rd = lambda s: pl.read_parquet(I + s + ".parquet").sort(ID)  # noqa: E731
v = rd("agent_bundle_v143_final"); ids = v[ID].to_numpy(); cur = v[T].cast(pl.Float64).to_numpy()
def col(s):
    d = rd(s); assert (d[ID].to_numpy() == ids).all(), s; return d[T].cast(pl.Float64).to_numpy()
dH = col("agent_apt_EDDM_final") - cur; dI = col("agent_pcdx_final") - cur; assert ((dH != 0) & (dI != 0)).sum() == 0
v144 = np.rint(np.maximum(cur + dH + dI, 0))
r = (pl.scan_parquet("data/raw/ranking.parquet").filter((C("ADEP_mvt") == "LIRF") & (C("PHASE_mvt") == "DEP") & C("LOBT_flt").is_null())
     .select(ID, "FLIGHT_mvt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt").collect()
     .with_columns(gap=(C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds(),
                   slip=(C("MVT_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date() - C("SCHED_TIME_UTC_mvt").dt.convert_time_zone("Europe/Rome").dt.date()).dt.total_days())
     .filter(C("gap").is_between(50000, 100000) & (C("slip") == 0)))
pos = {int(i): k for k, i in enumerate(ids)}; v145 = v144.copy()
for x in r.iter_rows(named=True):
    k = pos[int(x[ID])]; print(f"calendar rule: {x['FLIGHT_mvt']} gap {x['gap']:.0f} same local day: {v145[k]:.0f} -> {x['gap']:.0f}"); v145[k] = round(x["gap"])
orig = pl.read_parquet("submissions/jolly-lobster_v143.parquet")
for n, a, ref in (("144", v144, cur), ("145", v145, v144)):
    out = v.with_columns(pl.Series(T, a).cast(v[T].dtype)); out.write_parquet(I + f"agent_bundle_v{n}_final.parquet")
    sub = orig.select(ID).join(out, on=ID, how="left"); assert sub[T].null_count() == 0 and (sub[ID] == orig[ID]).all()
    sub.select(orig.columns).cast(dict(orig.schema)).write_parquet(f"submissions/jolly-lobster_v{n}.parquet")
    print(f"v{n}: rows changed vs previous {int((a != ref).sum())}, max |move| {np.abs(a - ref).max():.0f}")
