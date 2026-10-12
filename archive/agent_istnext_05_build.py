"""istnext 05: 2026-vs-2025 column check and build of data/interim/agent_istnext_final.parquet = v150 with the Istanbul corrector term swapped (design term minus the shipped term, both from
archive/agent_istnext_04_serve.py, k 1.0, seeds 0-2) on Istanbul NON-SNOW rows that no row-level rule owns. No 2026 label read.
A row is "not owned" when (i) its v150 value is within 2 s (delta-build rounding) of rint(max(served Istanbul base + shipped term, 0)), and (ii) no bundle step of v150's own line (v144 -> v145 -> v146 -> v148 -> v149 -> v150; v147 is a side branch whose Istanbul moves are not in v150) moved it by more than 2 s (so the 13 no-plan rows bounded in v146 keep
their bound: they are never touched, which is the same as re-applying the bound last).
usage: python archive/agent_istnext_05_build.py DESIGN [check]   (check = column check only, no file)"""
import sys
import numpy as np
import polars as pl

I = "data/interim/"; ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"; des = sys.argv[1]; only_check = len(sys.argv) > 2; OUTN = "agent_istnext_final.parquet" if des == "D1" else "agent_istnext_%s_final.parquet" % des
c0 = np.load(I + "agent_istnext_serve_ctrl_term.npz", allow_pickle=True); c1 = np.load(I + "agent_istnext_serve_%s_term.npz" % des, allow_pickle=True)
sid = c0["sid"]; assert (c1["sid"] == sid).all() and float(c0["k"]) == 1.0 and float(c1["k"]) == 1.0 and list(c0["seeds"]) == list(c1["seeds"]) == [0, 1, 2]
snow = c0["snow_serve"]; assert (c1["snow_serve"] == snow).all() and (c1["d"][snow] == 0).all() and (c0["d"][snow] == 0).all()
# ---- column check: 2025 fold rows vs 2026 serve rows ----
names = [str(n) for n in c1["names"]]; NF, NS_ = c1["NEWF"], c1["NEWS"]; monf = c1["monf"].astype(int)
mon_s = np.load(I + "agent_istpcd_cols.npz", allow_pickle=True)["mon_s"].astype(int); assert len(mon_s) == len(sid)
assert not any(any(t in n.lower() for t in ("adsb", "m_e_", "trk", "track", "ci_", "appear")) for n in names), "a track-derived column is in the design"
print("no track-derived column in the design (asserted on the names; the builder reads prediction, keys, hour, runway, weekend flag, take-off time, plan-clock distances only)")


def auc(a, b):
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0: return float("nan")
    r = np.argsort(np.argsort(np.concatenate([a, b]), kind="stable"), kind="stable") + 1.0; u = r[:len(a)].sum() - len(a) * (len(a) + 1) / 2; v = u / (len(a) * len(b)); return max(v, 1 - v)


print("column | missing share 2025 / 2026 | January: mean 2025 / 2026, symmetric AUC | July: mean 2025 / 2026, AUC   (the prediction itself: see the plan-clock note, AUC .60 / .53)")
worst = 0.0
for j, n in enumerate(names):
    a1, b1 = NF[monf == 1, j], NS_[mon_s == 1, j]; a7, b7 = NF[monf == 7, j], NS_[mon_s == 7, j]; u1, u7 = auc(a1, b1), auc(a7, b7); worst = max(worst, u1, u7)
    const = np.nanstd(NS_[:, j]) == 0
    print(f"  {n:16s} {np.isnan(NF[:, j]).mean():.3f} / {np.isnan(NS_[:, j]).mean():.3f} | {np.nanmean(a1):+9.2f} / {np.nanmean(b1):+9.2f}  {u1:.3f} | {np.nanmean(a7):+9.2f} / {np.nanmean(b7):+9.2f}  {u7:.3f}{'  CONSTANT IN 2026' if const else ''}")
print("largest symmetric AUC %.3f" % worst)
if only_check: sys.exit(0)
# ---- build ----
rd = lambda n: pl.read_parquet(I + n + ".parquet").sort(ID)   # noqa: E731
v150 = rd("agent_bundle_v150_final"); ids = v150[ID].to_numpy(); cur = v150[T].cast(pl.Float64).to_numpy(); pos = {x: i for i, x in enumerate(ids)}; ii = np.array([pos[x] for x in sid])
base = rd("agent_combo_v129_noM4_final"); assert (base[ID].to_numpy() == ids).all(); b = base[T].cast(pl.Float64).to_numpy()[ii]
chain_now = np.rint(np.maximum(b + c0["d"], 0)); gap = cur[ii] - chain_now; same = np.abs(gap) <= 2   # v150 was built by deltas, so it sits within the rounding (2 s) of base + shipped term on rows no rule owns
moved = np.zeros(len(ids), bool)
for x, y in (("v144", "v145"), ("v145", "v146"), ("v146", "v148"), ("v148", "v149"), ("v149", "v150")):
    mv_ = np.abs(rd("agent_bundle_%s_final" % y)[T].cast(pl.Float64).to_numpy() - rd("agent_bundle_%s_final" % x)[T].cast(pl.Float64).to_numpy()) > 2; print("  step %s -> %s moved Istanbul rows by more than 2 s: %d" % (x, y, mv_[ii].sum())); moved |= mv_
elig = same & ~snow & ~moved[ii]
print("Istanbul rows %d; snow %d; non-snow rows where v150 is within 2 s of base + shipped term (dose 1.0): %d of %d; moved by more than 2 s in a bundle step v144..v150: %d; eligible (non-snow, not owned by a row rule) %d" % (len(sid), snow.sum(), (same & ~snow).sum(), (~snow).sum(), moved[ii].sum(), elig.sum()))
assert same[~snow].mean() > 0.97, "v150 does not carry the shipped Istanbul term at dose 1.0: stop"
dl = np.where(elig, np.maximum(cur[ii] + np.rint(c1["d"] - c0["d"]), 0) - cur[ii], 0.0)
cc = float(np.corrcoef(c1["d"][elig], c0["d"][elig])[0, 1]); print("corr(new term, shipped term) %.3f; mean move %+.3f; rms move %.2f; max |move| %.0f" % (cc, dl[elig].mean(), np.sqrt((dl[elig] ** 2).mean()), np.abs(dl).max()))
assert cc >= 0.7 and abs(dl[elig].mean()) <= 5 and np.abs(dl).max() <= 600, "guard failed: no file"
out = cur.copy(); out[ii] = cur[ii] + dl
assert (out[ii][snow] == cur[ii][snow]).all() and (out[ii][~elig] == cur[ii][~elig]).all() and (np.delete(out, ii) == np.delete(cur, ii)).all()
v150.with_columns(pl.Series(T, out).cast(v150.schema[T])).write_parquet(I + OUTN)
for m in (1, 7):
    s = elig & (mon_s == m); print("month %d: rows moved %d of %d; mean move %+.2f; rms %.2f" % (m, (dl[s] != 0).sum(), (mon_s == m).sum(), dl[s].mean(), np.sqrt((dl[s] ** 2).mean())))
print("sum of squared moves / 344,841 rows: %.2f  -> if the shipped term is right and the move is pure noise the cost is about %.3f board s; rows moved %d" % ((dl ** 2).sum() / 344841, (dl ** 2).sum() / 344841 / (2 * 216.67), (dl != 0).sum()))
print("wrote data/interim/" + OUTN)
