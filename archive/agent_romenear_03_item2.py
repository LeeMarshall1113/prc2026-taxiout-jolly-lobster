"""romenear 03: Item 2 tables. Rome no-plan mid-gap rows (gap 3,600-14,400 s): is 'take-off on a later local day than the schedule' a copy
signal beyond the gap and the schedule hour, or the gap in disguise? Copy rate by crossing x fine gap band x regime, by schedule local hour,
logistic on (gap, hour) with and without the indicator (month-held-out log-likelihood), the same on top of the shipped no-plan probability
P2_12, and other calendar facts (UTC-day crossing, weekday, month end)."""
import numpy as np, polars as pl
from sklearn.linear_model import LogisticRegression
C = pl.col
R = ""; I = R + "data/interim/"; OUT = R + "data/interim/local/romenear/"
f = pl.read_parquet(I + "agent_rome1_frame.parquet").filter(C("np") & (C("gs") >= 3600) & (C("gs") < 14400))
loc = lambda c: pl.from_epoch(C(c), time_unit="s").dt.replace_time_zone("UTC").dt.convert_time_zone("Europe/Rome")  # noqa: E731
f = f.with_columns(loc("S").alias("S_loc"), loc("T").alias("T_loc"), pl.from_epoch(C("S"), time_unit="s").alias("S_utc"), pl.from_epoch(C("T"), time_unit="s").alias("T_utc"))
f = f.with_columns(
    (C("T_loc").dt.date() > C("S_loc").dt.date()).alias("cross"), (C("T_utc").dt.date() > C("S_utc").dt.date()).alias("cross_utc"),
    C("S_loc").dt.hour().alias("sh_loc"), C("S_loc").dt.weekday().alias("dow_loc"), C("T_loc").dt.weekday().alias("dow_T"),
    (C("S_loc").dt.date() == C("S_loc").dt.month_end().dt.date()).alias("sched_month_end"), (C("T_loc").dt.day() == 1).alias("to_month_first"),
    ((C("S_loc").dt.date() + pl.duration(days=1)).dt.month() != C("S_loc").dt.month()).alias("sched_last_day"),
    pl.when(C("mo") <= 4).then(pl.lit("dark")).otherwise(pl.lit("lit")).alias("reg"),
    pl.when(C("gs") < 5400).then(pl.lit("3.6-5.4k")).when(C("gs") < 7200).then(pl.lit("5.4-7.2k")).when(C("gs") < 10800).then(pl.lit("7.2-10.8k")).otherwise(pl.lit("10.8-14.4k")).alias("band"),
    C("hgl2").is_not_null().alias("recovered"),
)
t = f.filter((C("yr") == 2025) & C("type").is_in(["copy", "real", "day-early"])).with_columns((C("type") == "copy").alias("copy"), (C("type") == "day-early").alias("de"))
print("2025 no-plan mid-gap rows", t.height, "recovered plan", int(t["recovered"].sum()), "| in shipped mixture scope (mid12)", int(t["mid12"].fill_null(False).sum()))
print("types:", t["type"].value_counts().sort("type").to_dicts())
print("\n--- copy rate by crossing (reproduces the audit: 71 of 104 vs 503 of 641+503) ---")
print(t.group_by("cross").agg(pl.len().alias("n"), C("copy").sum().alias("copies"), C("copy").mean().round(3).alias("rate"), C("de").sum().alias("day_early")).sort("cross"))
print("\n--- by band x crossing x regime ---")
print(t.group_by("reg", "band", "cross").agg(pl.len().alias("n"), C("copy").sum().alias("copies"), C("copy").mean().round(3).alias("rate")).sort("reg", "band", "cross"))
print("\n--- by schedule local hour band x crossing (all 2025) ---")
t = t.with_columns(pl.when(C("sh_loc") < 6).then(pl.lit("00-06")).when(C("sh_loc") < 12).then(pl.lit("06-12")).when(C("sh_loc") < 18).then(pl.lit("12-18")).when(C("sh_loc") < 21).then(pl.lit("18-21")).otherwise(pl.lit("21-24")).alias("hb"))
print(t.group_by("hb", "cross").agg(pl.len().alias("n"), C("copy").sum().alias("copies"), C("copy").mean().round(3).alias("rate")).sort("hb", "cross"))
print("\n--- within the crossing rows: by band ---")
print(t.filter(C("cross")).group_by("band").agg(pl.len().alias("n"), C("copy").mean().round(3).alias("rate"), C("sh_loc").mean().round(1).alias("sched_hour")).sort("band"))
print("same-day rows by band:"); print(t.filter(~C("cross")).group_by("band").agg(pl.len().alias("n"), C("copy").mean().round(3).alias("rate"), C("sh_loc").mean().round(1).alias("sched_hour")).sort("band"))
# matched comparison: same band and same schedule-hour band, crossing vs not
print("\n--- cells with both crossing and same-day rows (band x hour band), copy rate crossing / same-day ---")
cells = t.group_by("band", "hb", "cross").agg(pl.len().alias("n"), C("copy").mean().alias("rate")).pivot(on="cross", index=["band", "hb"], values=["n", "rate"]).sort("band", "hb")
print(cells)
# logistic: copy ~ gap (log), sched hour (cyclic), + crossing; month-held-out log-likelihood
cpy = t["copy"].to_numpy().astype(int); mo = t["mo"].to_numpy(); reg = t["reg"].to_numpy().astype(str)
gs = t["gs"].to_numpy().astype(float); sh = t["sh_loc"].to_numpy().astype(float); cr = t["cross"].to_numpy().astype(float)
base = np.column_stack([np.log(gs), np.sin(2 * np.pi * sh / 24), np.cos(2 * np.pi * sh / 24), (sh >= 18).astype(float), (sh >= 21).astype(float)])
P = np.clip(t["P2_12"].fill_null(0.5).to_numpy().astype(float), 1e-4, 1 - 1e-4); lg = np.log(P / (1 - P))
print("\nP2_12 null on these rows:", int(t["P2_12"].null_count()))


def lomo_ll(X, y, groups):
    ll = 0.0
    for g in np.unique(groups):
        tr = groups != g; te = ~tr
        if te.sum() == 0: continue
        m = LogisticRegression(C=10.0, max_iter=500).fit(X[tr], y[tr]); p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
        ll += -(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)).sum()
    return ll


rng = np.random.default_rng(1)
for nm, X0 in (("gap+hour", base), ("shipped P2_12", lg[:, None]), ("P2_12+gap+hour", np.column_stack([lg, base]))):
    for rg in ("all", "dark", "lit"):
        k = np.ones(len(cpy), bool) if rg == "all" else reg == rg
        X1 = np.column_stack([X0, cr]); coef = LogisticRegression(C=10.0, max_iter=500).fit(X1[k], cpy[k]).coef_[0][-1]
        l0 = lomo_ll(X0[k], cpy[k], mo[k]); l1 = lomo_ll(X1[k], cpy[k], mo[k])
        # shuffled control: crossing permuted within band
        ctl = []
        band = t["band"].to_numpy().astype(str)
        for _ in range(20):
            c2 = cr.copy()
            for b in np.unique(band):
                idx = np.where(k & (band == b))[0]; c2[idx] = rng.permutation(c2[idx])
            ctl.append(l0 - lomo_ll(np.column_stack([X0, c2])[k], cpy[k], mo[k]))
        print(f"{nm:16s} {rg:4s}: n {k.sum()} crossing {int(cr[k].sum())} coef {coef:+.2f}; month-held-out dLL {l0-l1:+.1f} nats (shuffled control mean {np.mean(ctl):+.1f}, p95 {np.percentile(ctl,95):+.1f})")
# temporal: fit on lit judge dark and reverse (base vs +cross)
for trr, ter in (("lit", "dark"), ("dark", "lit")):
    tr = reg == trr; te = reg == ter
    for nm, X0 in (("gap+hour", base), ("P2_12+gap+hour", np.column_stack([lg, base]))):
        X1 = np.column_stack([X0, cr])
        def ll(X):
            m = LogisticRegression(C=10.0, max_iter=500).fit(X[tr], cpy[tr]); p = np.clip(m.predict_proba(X[te])[:, 1], 1e-6, 1 - 1e-6)
            return -(cpy[te] * np.log(p) + (1 - cpy[te]) * np.log(1 - p)).sum()
        print(f"temporal {trr}->{ter} {nm:16s}: dLL {ll(X0)-ll(X1):+.1f} nats on {te.sum()} rows")
# other calendar facts, within-band observed/expected
print("\n--- other calendar facts (copy rate on / off, with the band-matched expected rate) ---")
bexp = t.group_by("band").agg(C("copy").mean().alias("bexp"))
tt = t.join(bexp, on="band")
for s in ["cross_utc", "sched_month_end", "sched_last_day", "to_month_first"]:
    print(s, tt.group_by(s).agg(pl.len().alias("n"), C("copy").mean().round(3).alias("obs"), C("bexp").mean().round(3).alias("band_expected")).sort(s).to_dicts())
print("weekday of schedule (1=Mon):", tt.group_by("dow_loc").agg(pl.len().alias("n"), C("copy").mean().round(3).alias("obs"), C("bexp").mean().round(3).alias("exp")).sort("dow_loc").to_dicts())
print("crossing x UTC crossing:", tt.group_by("cross", "cross_utc").agg(pl.len().alias("n"), C("copy").mean().round(3).alias("obs")).sort("cross", "cross_utc").to_dicts())
# scored rows
s26 = f.filter(C("yr") == 2026)
print("\nscored no-plan mid-gap rows:", s26.height, "crossing", int(s26["cross"].sum()), "by month x crossing:", s26.group_by("mo", "cross").agg(pl.len().alias("n")).sort("mo", "cross").to_dicts())
t.select("MVT_ID_mvt", "mo", "reg", "band", "gs", "sh_loc", "cross", "cross_utc", "copy", "type", "P2_12", "op").write_parquet(OUT + "item2_rows25.parquet")
print("done")
