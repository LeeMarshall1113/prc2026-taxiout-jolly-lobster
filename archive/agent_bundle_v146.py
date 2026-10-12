"""v146 = v145 plus six row-level bounds (51 rows). Each piece is a bound defined on input columns; the rule and its 2025 count:
 Istanbul no-plan early bound (13 rows): for a no-plan departure that took off at or after its schedule, label <= schedule gap + 1,200 s (0 exceptions in 3,585 rows).
 Madrid one-hour ceiling (4 rows): label <= 3,606 s at Madrid and Barcelona (maximum 3,598 s in 212,242 Madrid rows; 3,604 s in 179,705 Barcelona rows).
 Rome with-plan day-early band (1 row): a plan row cannot be a schedule copy outside its plan window; served at take-off minus the Network Manager off-block (0 copies in 66,665 plan rows whose schedule lies more than 3,606 s from the plan's last off-block).
 Bounds overrun by later additive terms (29 rows): gap clamp (58 of 93,993 rows above gap + 60 s), plan-clock window (0 of 2,062,577), hidden-plan lower bound (0 of 57), re-applied last.
 Plan-clock window edge (10 rows): the label lies within 3,606 s of take-off minus LOBT on 2,062,577 of 2,062,577 plan departures of 2025.
 Hidden-plan lower bound (2 rows): label > take-off minus LOBT + 3,606 s when the plan is recovered from the destination's arrival record (0 of 57 exceptions at Rome and Istanbul).
Inputs: the six piece files below, written by the builders that archive/build_v146_pieces.sh runs in order. Out: data/interim/agent_bundle_v146_final.parquet"""
import polars as pl
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
D = "data/interim/"
base = pl.read_parquet(D + "agent_bundle_v145_final.parquet")
pieces = ["agent_logicC_LTFM_noplan-early-bound_final", "agent_logicC_LEMD_ceiling3606_final", "agent_logic_planband_final",
          "agent_logic_bounds_final", "agent_logicB_window_final", "agent_logicB_hiddenlower_final"]
out = base.rename({T: "v145"})
for p in pieces:
    out = out.join(pl.read_parquet(D + p + ".parquet").rename({T: p}), on=ID, how="left")
rows = out.to_dicts(); vals = []; nconf = 0
for r in rows:
    ch = {p: r[p] for p in pieces if r[p] != r["v145"]}
    if not ch: vals.append(r["v145"]); continue
    v = set(ch.values())
    if len(v) > 1:
        nconf += 1; print("overlap", r[ID], r["v145"], ch)
        # every piece is a bound: the value farthest from v145 satisfies all of them when they point the same way
        up = [x for x in v if x > r["v145"]]; dn = [x for x in v if x < r["v145"]]
        assert not (up and dn), "pieces disagree in direction"
        vals.append(max(up) if up else min(dn))
    else: vals.append(v.pop())
res = base.with_columns(pl.Series(T, vals).cast(base.schema[T]))
res.write_parquet(D + "agent_bundle_v146_final.parquet")
for p in pieces: print(p, "rows changed", int((out[p] != out["v145"]).sum()))
d = (res[T] - base[T]); print("total rows changed", int((d != 0).sum()), "overlaps", nconf, "mean shift", float(d.mean()), "max up", int(d.max()), "max down", int(d.min()))
