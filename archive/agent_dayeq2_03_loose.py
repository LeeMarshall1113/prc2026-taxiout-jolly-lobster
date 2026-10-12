"""day-equation 2, item 4: can the published flight set at the loose airports (EHAM, LFPG, LSZH) be reproduced from inputs?
Per day of 2025 (all 12 months, true stamps from data/interim/local/logic_rows/C_dep2025_typed): residual r = published - sum max(stamp - schedule, 0)/60 and
count residual dn = FLT_DEP_1 - n. Regress r and dn on per-day sums / counts by input category (plan, rule, segment, flight type, wake, helicopter,
duplicates, day-early stamps). A category with coefficient -1 (delay) and -1 (count) is one the portal leaves out; +1 is one it adds that we lack.
Then: residual after the best exclusion rule; number of days where the counts match exactly, in 2025 and (counts only) in the scored months."""
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(100)
APTS = ["EHAM", "LFPG", "LSZH", "EGLL"]  # EGLL as the exact reference
ck = pl.read_parquet(L + "C_dep2025_typed.parquet").filter(C("apt").is_in(APTS))
pub = load_pub(2025, APTS)
ck = ck.with_columns(((C("gap") - C("y")) / 60).clip(lower_bound=0).alias("dm"), (C("mvt") - pl.duration(seconds=C("y"))).dt.date().alias("day"))
print(ck.group_by("apt").agg(n=pl.len(), rules=C("rule").n_unique(), segs=C("seg").n_unique(), ftypes=C("ftype").n_unique(), wk=C("wk").unique().sort().str.join(","), rule_list=C("rule").unique().sort().str.join(","),
                             seg_list=C("seg").unique().sort().str.join(","), ftype_list=C("ftype").unique().sort().str.join(",")))
cats = {"noplan": ~C("plan"), "rule_V": C("rule") == "V", "rule_notI": C("rule") != "I", "wk_H": C("wk") == "H", "heli_acft": C("acft").is_in(["EC35", "EC45", "H145", "H135", "EC55", "AS50", "AS55", "H160", "A109", "A139", "AW39", "EC30", "R44", "R66", "S76", "B06", "H500"]),
        "dup": C("dup"), "dayearly": C("y") > 50000, "neg_gap": C("gap") < 0, "y_small": C("y") < 60}
for sgs in ck["seg"].unique().drop_nulls().to_list():
    cats[f"seg_{sgs}"] = C("seg") == sgs
for ft in ck["ftype"].unique().drop_nulls().to_list():
    cats[f"ftype_{ft}"] = C("ftype") == ft
aggs = [pl.len().alias("n"), C("dm").sum().alias("S")]
for k, cond in cats.items():
    aggs += [cond.fill_null(False).sum().alias(f"n_{k}"), C("dm").filter(cond.fill_null(False)).sum().alias(f"S_{k}")]
d = ck.group_by("apt", "day").agg(aggs).join(pub, on=["apt", "day"], how="inner").filter(C("pub").is_not_null() & C("n_pub").is_not_null()).with_columns((C("pub") - C("S")).alias("r"), (C("n_pub") - C("n")).alias("dn"))
for a in APTS:
    da = d.filter(C("apt") == a)
    print(f"\n===== {a}: days {da.height}; residual r med {da['r'].median():.0f} p10 {da['r'].quantile(0.1):.0f} p90 {da['r'].quantile(0.9):.0f} (min); dn med {da['dn'].median():.0f}, days with dn == 0: {int((da['dn'] == 0).sum())}, |dn| <= 2: {int((da['dn'].abs() <= 2).sum())}")
    for tgt, pre in [("r", "S_"), ("dn", "n_")]:
        cols = [c for c in da.columns if c.startswith(pre) and da[c].sum() != 0 and da[c].n_unique() > 2]
        X = np.nan_to_num(da.select(cols).to_numpy().astype(float)); y = np.nan_to_num(da[tgt].to_numpy().astype(float))
        X1 = np.column_stack([np.ones(len(y)), X])
        beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
        resid = y - X1 @ beta
        print(f"  regression of {tgt} on category {'sums' if tgt == 'r' else 'counts'}: intercept {beta[0]:+.1f}, residual sd {resid.std():.1f} (raw sd {y.std():.1f})")
        order = np.argsort(-np.abs(beta[1:]))
        print("   ", ", ".join(f"{cols[i][len(pre):]} {beta[1 + i]:+.2f} (mean {X[:, i].mean():.1f})" for i in order[:12]))
    # single-category exclusion test: residual after removing category c entirely from our sum
    best = []
    for k in cats:
        if f"S_{k}" in da.columns:
            rr = (da["r"] + da[f"S_{k}"]).to_numpy(); dd = (da["dn"] + da[f"n_{k}"]).to_numpy()
            best.append((np.median(np.abs(rr)), k, np.median(rr), np.quantile(np.abs(rr), 0.9), np.median(dd), (dd == 0).mean()))
    best.sort()
    print("  exclusion test (median |r| after removing the category; raw", f"{np.median(np.abs(da['r'].to_numpy())):.0f}):")
    for b in best[:8]:
        print(f"    {b[1]}: med|r| {b[0]:.0f}, med r {b[2]:+.0f}, p90|r| {b[3]:.0f}; dn med {b[4]:+.0f}, share dn == 0 {b[5]:.2f}")
    # days where counts match exactly: residual there
    m = da.filter(C("dn") == 0)
    print(f"  days with exact count match: {m.height}; their residual r med {m['r'].median() if m.height else float('nan'):.0f}, p90|r| {m['r'].abs().quantile(0.9) if m.height else float('nan'):.0f}")
    m2 = da.filter(C("dn").abs() <= 1)
    print(f"  days with |dn| <= 1: {m2.height}; residual med {m2['r'].median() if m2.height else float('nan'):.0f}, p90|r| {m2['r'].abs().quantile(0.9) if m2.height else float('nan'):.0f}")
# scored months: count match by day
r26 = pl.read_parquet(L + "C_dep2026.parquet").filter(C("apt").is_in(APTS)).with_columns((C("mvt") - pl.duration(seconds=C("served"))).dt.date().alias("day"))
d26 = r26.group_by("apt", "day").agg(n=pl.len(), noplan=(~C("plan")).sum()).join(load_pub(2026, APTS), on=["apt", "day"], how="inner").with_columns((C("n_pub") - C("n")).alias("dn"))
print("\nscored months, count residual by airport:")
print(d26.group_by("apt").agg(days=pl.len(), dn_med=C("dn").median(), exact=(C("dn") == 0).sum(), within1=(C("dn").abs() <= 1).sum(), within2=(C("dn").abs() <= 2).sum()).sort("apt"))
print(d26.filter(C("apt") == "EHAM").sort("day").select("day", "n", "n_pub", "dn", "pub").head(12))
