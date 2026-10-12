"""experiment nov 01 (docs/LAB-NOTES.md): the row-novelty column, leak-free on the fold side, and its 2026 serving
values. LABEL-FREE: never reads y.

Novelty (Screen 1 of docs/LAB-NOTES.md, same 9 covariates, same per-airport z-scoring, same
"nearest row from ANOTHER calendar day" rule): gap_sched, gap_lobt, prev_stand_headway, nm_apt_past60_mean,
late_apt_past60_mean (prc.features.build columns) + hour sin/cos + day-of-week sin/cos of MVT_TIME_UTC_mvt; a row is
eligible iff all 9 are finite (else NaN, LightGBM's missing branch). Euclidean distance in the per-airport z-scored space.

Columns written:
  data/interim/agent_nov_fold.parquet   MVT_ID_mvt, nov_f0..nov_f9, nov_full   (2025 fold rows, 9 airports)
     nov_f<k>: reference set R_k = eligible rows of the airport whose day is NOT in re-stack outer fold k (the re-stack's
               own day folds: sorted days of the c238 dump, i % 10); z-score parameters from R_k; every row's value =
               distance to its nearest R_k row from another calendar day. Fold-k (test) rows are therefore scored
               against training folds only; training rows against the training folds minus their own day.
     nov_full: reference = all eligible 2025 rows of the airport, own day excluded (the serving training column).
  data/interim/agent_nov_train.parquet  MVT_ID_mvt, nov (= nov_full)             (serving: training rows)
  data/interim/agent_nov_2026.parquet   MVT_ID_mvt, nov                          (serving: 2026 rows, reference = all
     eligible 2025 rows, z-scored with the full-2025 parameters; covariates from prc.features.build(ranking.parquet),
     exactly as prc.predict builds the ranking frame; cached in data/interim/agent_nov_rank_feats.parquet)
  data/interim/agent_nov_dist.json      fold vs 2026 novelty distributions per airport (dump rows and all rows)
"""
import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, ".")
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

C = pl.col
I = "data/interim/"
ID = "MVT_ID_mvt"
T0 = time.time()
AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
FEATS = ["gap_sched", "gap_lobt", "prev_stand_headway", "nm_apt_past60_mean", "late_apt_past60_mean"]
COLS = FEATS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
KS = (8, 64, 512)  # adaptive: widen the query only for rows whose first K neighbours are all same-day


def say(m):
    print(f"[{time.time() - T0:6.0f}s] {m}", flush=True)


def preempt():
    f = os.environ.get("JOB_PREEMPT_FILE", "")
    if f and os.path.exists(f):
        say("yield requested -- exit 10 (re-run resumes)")
        sys.exit(10)


def cyc(hour, dow):
    return {"hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
            "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7)}


def nn_other_day(ref_Z, ref_day, q_Z, q_day):
    """distance from each query row to its nearest reference row from a different calendar day (NaN if none found)."""
    out = np.full(len(q_Z), np.nan)
    if len(ref_Z) == 0 or len(q_Z) == 0:
        return out
    tree = cKDTree(ref_Z)
    todo = np.arange(len(q_Z))
    for k in KS:
        if len(todo) == 0:
            break
        kk = min(k, len(ref_Z))
        dist, idx = tree.query(q_Z[todo], k=kk, workers=4)
        if kk == 1:
            dist, idx = dist[:, None], idx[:, None]
        valid = ref_day[idx] != q_day[todo][:, None]
        has = valid.any(axis=1)
        first = np.argmax(valid, axis=1)
        out[todo[has]] = dist[np.flatnonzero(has), first[has]]
        todo = todo[~has]
        if kk == len(ref_Z):
            break
    return out


# ------------------------------------------------------------------ 1. fold rows (label-free keys only)
z = np.load(I + "agent_stagecap_RS_E9S_s0.npz", allow_pickle=True)
ids = z["ids"].astype(float)
apts = z["apts"].astype(str)
days = z["days"].astype(str)
del z
N = len(ids)
pos_of = {v: i for i, v in enumerate(ids)}
dump = pl.read_parquet(I + "agent_offblock_rows_c238.parquet", columns=[ID, "day"]).with_columns(C(ID).cast(pl.Float64))
dpos = np.array([pos_of.get(v, -1) for v in dump[ID].to_list()])
assert (dpos >= 0).all(), "a c238 dump row is missing from the fold npz"
assert (days[dpos] == np.asarray(dump["day"].to_list())).all(), "dump/npz day mismatch"
ud = sorted(set(days[dpos]))
fmap = {d: i % 10 for i, d in enumerate(ud)}
fold = np.array([fmap.get(d, -1) for d in days])
in_dump = np.zeros(N, bool)
in_dump[dpos] = True
say(f"fold npz {N:,} rows, {len(set(days))} days; c238 dump {len(dpos):,} rows / {len(ud)} days "
    f"(re-stack folds = sorted dump days i % 10); npz rows on non-dump days (reference only): {int((fold < 0).sum()):,}")

# ------------------------------------------------------------------ 2. 2025 covariates (as Screen 1)
cache = pl.scan_parquet(I + "train_features.parquet").select([ID] + FEATS).collect()
cpos = {float(v): k for k, v in enumerate(cache[ID].to_numpy())}
cloc = np.array([cpos.get(v, -1) for v in ids])
X = np.full((N, len(COLS)), np.nan)
for j, c in enumerate(FEATS):
    v = cache[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
    X[cloc >= 0, j] = v[cloc[cloc >= 0]]
del cache
tt = pl.concat([pl.scan_parquet(f).filter(C("PHASE_mvt") == "DEP").select(
    C(ID).cast(pl.Float64), C("MVT_TIME_UTC_mvt").dt.hour().alias("hour"), C("MVT_TIME_UTC_mvt").dt.weekday().alias("dow"))
    for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))]).collect()
tpos = {float(v): k for k, v in enumerate(tt[ID].to_numpy())}
tloc = np.array([tpos.get(v, -1) for v in ids])
hour = np.full(N, np.nan)
dow = np.full(N, np.nan)
hour[tloc >= 0] = tt["hour"].to_numpy().astype(float)[tloc[tloc >= 0]]
dow[tloc >= 0] = tt["dow"].to_numpy().astype(float)[tloc[tloc >= 0]]
del tt
for c, v in cyc(hour, dow).items():
    X[:, COLS.index(c)] = v
fin_ok = np.isfinite(X).all(axis=1)
say(f"2025 covariates: feature-cache hit {int((cloc >= 0).sum()):,}, time hit {int((tloc >= 0).sum()):,}, "
    f"all 9 finite {int(fin_ok.sum()):,} of {N:,}; at dump rows {int(fin_ok[in_dump].sum()):,} of {int(in_dump.sum()):,}")
h_now = hashlib.sha256(open("prc/features.py", "rb").read()).hexdigest()[:16]
h_cache = open(I + "train_features.featurehash").read().strip()
say(f"prc/features.py hash {h_now} vs train_features.featurehash {h_cache}: {'SAME' if h_now == h_cache else 'DIFFERENT'}")

# ------------------------------------------------------------------ 3. 2026 covariates via prc.features.build
RF = I + "agent_nov_rank_feats.parquet"
if not os.path.exists(RF):
    from prc.features import build
    say("building 2026 features: prc.features.build(ranking.parquet, with_target=False) (as prc.predict)")
    fr = build(pl.read_parquet("data/raw/ranking.parquet"), with_target=False)
    ad = (pl.scan_parquet("data/raw/ranking.parquet").filter(C("PHASE_mvt") == "DEP")
          .select(C(ID).cast(pl.Float64), C("ADEP_mvt").alias("apt")).collect().unique(subset=ID))
    (fr.select([C(ID).cast(pl.Float64)] + FEATS + ["hour", "dow"]).join(ad, on=ID, how="left", maintain_order="left")
     .write_parquet(RF))
    del fr
rk = pl.read_parquet(RF)
say(f"2026 feature frame {rk.height:,} rows (cached {RF})")
try:
    tr = pl.read_parquet(I + "agent_trans_rank_features.parquet", columns=[ID] + FEATS[:4]).with_columns(C(ID).cast(pl.Float64))
    jj = rk.select([ID] + FEATS[:4]).join(tr, on=ID, how="inner", suffix="_old")
    for c in FEATS[:4]:
        a_, b_ = (jj[c].cast(pl.Float64).fill_null(np.nan).to_numpy(), jj[c + "_old"].cast(pl.Float64).fill_null(np.nan).to_numpy())
        both = np.isfinite(a_) & np.isfinite(b_)
        say(f"  sanity vs agent_trans_rank_features {c}: rows {jj.height:,}, equal-finite {float((a_[both] == b_[both]).mean()):.4f}, "
            f"null pattern equal {float((np.isfinite(a_) == np.isfinite(b_)).mean()):.4f}")
except Exception as e:  # informational only
    say(f"  sanity vs agent_trans_rank_features skipped: {e}")
r_ids = rk[ID].to_numpy()
r_apt = np.asarray(rk["apt"].to_list()).astype(str)
X26 = np.column_stack([rk[c].cast(pl.Float64).fill_null(np.nan).to_numpy() for c in FEATS]
                      + list(cyc(rk["hour"].cast(pl.Float64).to_numpy(), rk["dow"].cast(pl.Float64).to_numpy()).values()))
ok26 = np.isfinite(X26).all(axis=1)
say(f"2026: all 9 finite {int(ok26.sum()):,} of {len(r_ids):,}")

# ------------------------------------------------------------------ 4. novelty
nov_f = np.full((10, N), np.nan)
nov_full = np.full(N, np.nan)
nov26 = np.full(len(r_ids), np.nan)
for a in AIRPORTS:
    preempt()
    E = np.flatnonzero(fin_ok & (apts == a))
    Xa, da, fa = X[E], days[E], fold[E]
    mu, sd = Xa.mean(0), Xa.std(0)
    sd[sd < 1e-9] = 1.0
    Za = (Xa - mu) / sd
    nov_full[E] = nn_other_day(Za, da, Za, da)
    q = np.flatnonzero(ok26 & (r_apt == a))
    nov26[q] = nn_other_day(Za, da, (X26[q] - mu) / sd, np.array(["2026"] * len(q)))
    for k in range(10):
        R = fa != k
        mk, sk = Xa[R].mean(0), Xa[R].std(0)
        sk[sk < 1e-9] = 1.0
        nov_f[k, E] = nn_other_day((Xa[R] - mk) / sk, da[R], (Xa - mk) / sk, da)
    say(f"{a}: eligible 2025 {len(E):,} (NaN full {int(np.isnan(nov_full[E]).sum())}), 2026 {len(q):,}; "
        f"median full {np.nanmedian(nov_full[E]):.3f}, fold-own-test {np.nanmedian(nov_f[fa[fa >= 0], E[fa >= 0]]):.3f}, "
        f"2026 {np.nanmedian(nov26[q]):.3f}")

m9 = np.isin(apts, AIRPORTS)
pl.DataFrame({ID: ids[m9], **{f"nov_f{k}": nov_f[k, m9] for k in range(10)}, "nov_full": nov_full[m9]}).write_parquet(
    I + "agent_nov_fold.parquet")
pl.DataFrame({ID: ids[m9], "nov": nov_full[m9]}).write_parquet(I + "agent_nov_train.parquet")
m26 = np.isin(r_apt, AIRPORTS)
pl.DataFrame({ID: r_ids[m26], "nov": nov26[m26]}).write_parquet(I + "agent_nov_2026.parquet")
say("wrote agent_nov_fold / agent_nov_train / agent_nov_2026 parquets")

# ------------------------------------------------------------------ 5. distributions (fold vs 2026)
d26 = pl.read_parquet(I + "agent_offblock_rows_2026.parquet", columns=[ID]).with_columns(C(ID).cast(pl.Float64))
in26 = np.isin(r_ids, d26[ID].to_numpy())
own_test = np.full(N, np.nan)
for k in range(10):
    own_test[fold == k] = nov_f[k, fold == k]


def summ(v):
    f = v[np.isfinite(v)]
    return {"n": int(len(v)), "coverage": float(np.isfinite(v).mean()) if len(v) else float("nan"),
            **({"mean": float(f.mean()), "median": float(np.median(f)), "p90": float(np.percentile(f, 90)),
                "p99": float(np.percentile(f, 99))} if len(f) else {})}


dist = {}
print("\nNOVELTY DISTRIBUTIONS (fold = 2025 dump rows; 2026 = 2026 dump rows; share2026>p90 = 2026 rows above the fold's p90)")
print(f"{'apt':5s} {'fold cov':>8s} {'fold med':>8s} {'fold p90':>8s} {'fold p99':>8s} | {'26 cov':>6s} {'26 med':>7s} "
      f"{'26 p90':>7s} {'26 p99':>7s} | {'>p90':>6s} | {'fold-test med':>13s}")
for a in AIRPORTS + ["ALL"]:
    fm = in_dump & ((apts == a) if a != "ALL" else m9)
    qm = in26 & ((r_apt == a) if a != "ALL" else m26)
    sf, s2, st = summ(nov_full[fm]), summ(nov26[qm]), summ(own_test[fm])
    if a != "ALL":
        p90 = sf.get("p90", np.nan)
        v = nov26[qm]
        sh = float((v[np.isfinite(v)] > p90).mean()) if np.isfinite(v).any() else float("nan")
    else:
        sh = float(np.mean([dist[x]["share2026_above_fold_p90"] for x in AIRPORTS]))
    dist[a] = {"fold_dump_full": sf, "fold_dump_owntest": st, "y2026_dump": s2, "share2026_above_fold_p90": sh,
               "fold_all": summ(nov_full[(apts == a) if a != "ALL" else m9]),
               "y2026_all": summ(nov26[(r_apt == a) if a != "ALL" else m26])}
    print(f"{a:5s} {sf['coverage']:8.3f} {sf.get('median', np.nan):8.3f} {sf.get('p90', np.nan):8.3f} {sf.get('p99', np.nan):8.3f} | "
          f"{s2['coverage']:6.3f} {s2.get('median', np.nan):7.3f} {s2.get('p90', np.nan):7.3f} {s2.get('p99', np.nan):7.3f} | "
          f"{sh:6.3f} | {st.get('median', np.nan):13.3f}")
json.dump({"hash_now": h_now, "hash_cache": h_cache, "dist": dist}, open(I + "agent_nov_dist.json", "w"), indent=1)
say("wrote data/interim/agent_nov_dist.json -- FEAT DONE")
