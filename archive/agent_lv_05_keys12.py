"""lv 05: key table for MORE TRAINING MONTHS (docs/LAB-NOTES.md). Original 155,799 fold rows (months 1,2,3,7,8,9,10, same order) +
LTFM departures of the new months (4,5,6,11,12) that have a served-style REF: v94 fold bagged + (ltfmseq treat - ctrl).  fold files: f4 (4,10), f5 (5,11), f6 (6,12).
Columns: fid, yf, ref (SERVED analogue for ALL rows, original ones recomputed the same way), monf, Kall (orig Kf, new Kf, serve Ks), sid, kn, norig.
Months whose fold files are missing (f5 until the Modal v113 pair is fetched) are skipped and reported.  No 2026 labels."""
import glob
import sys

import numpy as np
import polars as pl

sys.path.insert(0, "archive")
import agent_lks_keys as KY  # noqa: E402

C = pl.col; ID = "MVT_ID_mvt"; I = "data/interim/"
K = np.load(I + "agent_lks_keys.npz", allow_pickle=True)
fid0, yf0, monf0, Kall0, kn, sid = K["fid"], K["yf"], K["monf"], K["Kall"], list(K["kn"]), K["sid"]; N0 = len(fid0)
Kf0, Ks = Kall0[:N0], Kall0[N0:]
FOLDS = {4: "4_10", 5: "5_11", 6: "6_12"}
served, newrows = {}, []
for k, t in FOLDS.items():
    try:
        b = pl.read_parquet(f"results/preds_bulkr_f{k}_{t}.parquet").select(ID, "ADEP_mvt", "bagged").filter(C("ADEP_mvt") == "LTFM").select(ID, "bagged")
        c0 = pl.read_parquet(f"results/preds_ltfmseq_ctrl_f{k}_{t}.parquet").select(ID, C("bagged").alias("c0"))
        t1 = pl.read_parquet(f"results/preds_ltfmseq_treat_f{k}_{t}.parquet").select(ID, C("bagged").alias("t1"))
    except FileNotFoundError as e:
        print(f"fold {k} ({t}) MISSING: {e}"); continue
    j = b.join(c0, on=ID).join(t1, on=ID).with_columns((C("bagged") + C("t1") - C("c0")).alias("ref"))
    served[k] = j.select(ID, "ref")
    print(f"fold {k} ({t}): {j.height:,} LTFM rows with served REF")
allref = pl.concat(list(served.values())).unique(ID)
raw = pl.concat([pl.read_parquet(f, columns=KY.RAWCOLS + ["TAXITIME_SEC_mvt", "PHASE_mvt", "ADEP_mvt"]).filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LTFM")) for f in sorted(glob.glob("data/raw/training_2025-*.parquet"))])
raw = raw.with_columns(C("MVT_TIME_UTC_mvt").dt.month().alias("_m"))
# original rows: served REF recomputed (months 1,2,3,7,8,9 from f1-f3 files, 10 from f4)
orig_ref = []
for k, t in ((1, "1_7"), (2, "2_8"), (3, "3_9"), (4, "4_10")):
    b = pl.read_parquet(f"results/preds_bulkr_f{k}_{t}.parquet").select(ID, "bagged")
    c0 = pl.read_parquet(f"results/preds_ltfmseq_ctrl_f{k}_{t}.parquet").select(ID, C("bagged").alias("c0"))
    t1 = pl.read_parquet(f"results/preds_ltfmseq_treat_f{k}_{t}.parquet").select(ID, C("bagged").alias("t1"))
    orig_ref.append(b.join(c0, on=ID).join(t1, on=ID).select(ID, (C("bagged") + C("t1") - C("c0")).alias("ref")))
oref = pl.concat(orig_ref).unique(ID)
r0 = pl.DataFrame({ID: fid0}).join(oref, on=ID, how="left", maintain_order="left")["ref"].to_numpy()
assert np.isfinite(r0).all(), int((~np.isfinite(r0)).sum())
have = set(np.unique(monf0).tolist())
new = raw.join(allref, on=ID, how="inner").filter(~C("_m").is_in(sorted(have)))
new = new.filter(C(ID).is_in(fid0).not_())
print("new rows by month", new.group_by("_m").len().sort("_m").rows())
kd, names = KY.prep(new); assert names == list(K["allnames"]), "key order"
keep = [names.index(n) for n in kn]
Knew = kd.select(names).to_numpy().astype(np.int64)[:, keep]
fid = np.concatenate([fid0, new[ID].to_numpy().astype("float64")]); yf = np.concatenate([yf0, new["TAXITIME_SEC_mvt"].to_numpy().astype("float64")])
ref = np.concatenate([r0, new.join(allref, on=ID, how="left", maintain_order="left")["ref"].to_numpy()]); monf = np.concatenate([monf0, new["_m"].to_numpy()])
Kall = np.vstack([Kf0, Knew, Ks])
assert np.isfinite(ref).all()
OUT12 = I + (sys.argv[1] if len(sys.argv) > 1 else "agent_lv_keys12.npz"); np.savez(OUT12, fid=fid, sid=sid, yf=yf, ref=ref, monf=monf, Kall=Kall, kn=np.array(kn), norig=N0)
print("wrote", OUT12, "rows", len(fid), "months", sorted(set(monf.tolist())))
