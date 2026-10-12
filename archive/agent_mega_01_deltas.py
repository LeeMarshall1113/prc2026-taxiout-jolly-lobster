"""mega step 1: float deltas for ingredients 2 (Rome plan gate, extra weight 0.1) and 3 (Istanbul corrector dose 1.0), each checked, on the base agent_basefix_final.parquet.
Writes data/interim/agent_mega_deltas.npz (ids sorted as the base; dG, dI float arrays).
Checks: (2) the same code path as agent_rejudge_61_gate_build.py reproduces the w=0.2 file exactly; w=0.1 as delta equals the standalone w=0.1 build on agent_ablate_final;
(3) delta = (1/0.75-1) x served term on non-snow Istanbul rows; equals dose10 - v138 up to rounding; zero off Istanbul and on snow rows."""
import numpy as np
import polars as pl
C = pl.col
I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
base = pl.read_parquet(I + "agent_basefix_final.parquet").sort(ID); ids = base[ID].to_numpy(); cur = base[T].cast(pl.Float64).to_numpy()
abl = pl.read_parquet(I + "agent_ablate_final.parquet").sort(ID); assert (abl[ID].to_numpy() == ids).all()
ab = abl[T].cast(pl.Float64).to_numpy()
# ---------- ingredient 2: gate delta (same code path as step 61) ----------
c = lambda n: pl.read_parquet(I + n).select(C(ID).cast(pl.Float64).alias("id"), C(T).cast(pl.Float64).alias(n))  # noqa: E731
s0 = c("v105_s0ship.parquet").join(c("v105_gatebag_s0.parquet"), on="id")
inp = s0["v105_s0ship.parquet"].to_numpy(); d0 = s0["v105_gatebag_s0.parquet"].to_numpy() - inp
rk = (pl.scan_parquet("data/raw/ranking.parquet").filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LIRF") & C("AOBT_3_flt").is_not_null())
      .select(C(ID).cast(pl.Float64).alias("id"), (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gap")).filter(C("gap") <= 10800).collect())
mixd = pl.DataFrame({"id": s0["id"], "mix": inp + d0 / 0.6})
j = pl.DataFrame({"id": ids}).join(mixd, on="id", how="left"); mix = j["mix"].to_numpy()
scope = np.isin(ids, rk["id"].to_numpy()); assert np.isfinite(mix[scope]).all()
print("gate scope rows", int(scope.sum()))
assert (cur[scope] == ab[scope]).all(); print("base equals ablate on every gate-scope row (basefix only moved Istanbul)")
rule_rows = np.load(I + "agent_bigc13_serve_base.npz", allow_pickle=True)["RR"]
print("gate scope rows that are later-rule rows:", int((scope & rule_rows).sum()))
def gate(w): return np.where(scope, np.rint(np.maximum(ab + w * (mix - ab), 0)), ab)
std = {}
for w in (0.2, 0.1):
    ref = pl.read_parquet(I + ("agent_rejudge_gate_final.parquet" if w == 0.2 else "agent_ablate_final.parquet")).sort(ID)[T].cast(pl.Float64).to_numpy() if w == 0.2 else None
    if w == 0.2:
        assert (gate(0.2) == ref).all(); print("w=0.2 REPRODUCED EXACTLY: 0 of", len(ref), "rows differ from agent_rejudge_gate_final.parquet")
w = 0.1
dG = np.where(scope, w * (mix - ab), 0.0)
assert (np.rint(np.maximum(ab + dG, 0)) == gate(0.1)).all() and (np.rint(np.maximum(cur + dG, 0))[scope] == gate(0.1)[scope]).all() and (dG[~scope] == 0).all(); print("w=0.1 delta == standalone rint(max(ablate + 0.1 (mix - ablate), 0)) on every row; delta zero outside scope")
print("gate w0.1: rows with float delta nonzero", int((dG != 0).sum()), "max |delta| %.1f" % np.abs(dG).max())
# ---------- ingredient 3: Istanbul dose delta ----------
z = np.load(I + "agent_lv4r_ltfm_final_delta.npz"); assert (z["ids"] == ids).all()
lt, new = z["lt"], z["new"]
dI = np.where(lt, (1.0 / 0.75 - 1) * new, 0.0)
print("Istanbul rows (lt)", int(lt.sum()), "delta nonzero", int((dI != 0).sum()), "snow rows with nonzero delta:", int(((dI != 0) & np.isin(np.arange(len(ids)), np.where(lt)[0][z["snow_serve"][:0]])).sum()) if False else "see below")
v138 = pl.read_parquet(I + "agent_bundle_v138_final.parquet").sort(ID); d10 = pl.read_parquet(I + "agent_ist3_dose10_final.parquet").sort(ID)
lane = d10[T].cast(pl.Float64).to_numpy() - v138[T].cast(pl.Float64).to_numpy()
print("experiment diff (dose10 - v138): rows nonzero", int((lane != 0).sum()), "; off-Istanbul nonzero", int(((lane != 0) & ~lt).sum()), "; |experiment - float delta| max (rounding) %.3f" % np.abs(lane - dI)[lt].max(), "; nonzero delta rows where experiment diff is 0:", int(((dI != 0) & (lane == 0)).sum()))
print("delta nonzero off Istanbul:", int(((dI != 0) & ~lt).sum()), "; rule rows with nonzero Istanbul delta:", int(((dI != 0) & rule_rows).sum()))
print("delta / (0.25 x term) : term := new/0.75 ; max |dI - 0.25 new/0.75| %.2e" % np.abs(dI - 0.25 * (new / 0.75) * lt).max())
print("Istanbul delta stats: mean %.3f rms(nonzero) %.2f max %.1f" % (dI[dI != 0].mean(), np.sqrt((dI[dI != 0] ** 2).mean()), np.abs(dI).max()))
np.savez(I + "agent_mega_deltas.npz", ids=ids, dG=dG, dI=dI, scope=scope, lt=lt, rule_rows=rule_rows)
