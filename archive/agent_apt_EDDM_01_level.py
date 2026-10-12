"""EDDM experiment 01: label-free level of Munich's January 2026, like with like.
Published monthly mean (EUROCONTROL additional taxi-out file) = mean taxi-out of 'reference flights': stand+runway groups with >=10 flights at or
below the group P10, MINUS flights de-iced after off-block (TF - VALID_FL). Rebuild that population on our side:
  valid = taxi<=7200, stand and runway present, group (stand,runway) valid; weight (1-w) with w = detector flag (seen) or p_full (unseen).
2025 control: label mean over the rebuilt population vs published mean, per month -> offset and month-to-month spread.
2026: served (pre-shift) prediction mean over the same population vs published mean; bias = (pred - pub) - offset.
No 2026 departure label is read (ranking is read for id and scheduled-time gap only)."""
import sys, numpy as np, polars as pl
sys.path.insert(0, ".")
C = pl.col
pl.Config.set_tbl_rows(100); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_cols(30)
I = "data/interim/"
f = pl.read_parquet(I + "agent_deice_feat.parquet", columns=["id","apt","mon","yr","stand","rwy","y","deice","seen","hour"]).filter(C("apt")=="EDDM")
pr = pl.read_parquet(I + "agent_deice_prop.parquet").select("id","p_full")
f = f.join(pr, on="id", how="left")
f = f.with_columns(pl.when(C("seen")).then(C("deice").cast(pl.Float64)).otherwise(C("p_full")).alias("w"))
print("null w", f["w"].null_count(), "null stand", f["stand"].null_count(), "null rwy", f["rwy"].null_count())
v = pl.read_parquet(I + "agent_bundle_v143_final.parquet").rename({"MVT_ID_mvt":"id","TAXITIME_SEC_mvt":"served"}).with_columns(C("id").cast(pl.Float64), C("served").cast(pl.Float64))
rk = (pl.scan_parquet("data/raw/ranking.parquet").filter((C("PHASE_mvt")=="DEP")&(C("ADEP_mvt")=="EDDM"))
      .select(C("MVT_ID_mvt").cast(pl.Float64).alias("id"), (C("MVT_TIME_UTC_mvt")-C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs")).collect())
f = f.join(v, on="id", how="left").join(rk, on="id", how="left")
f = f.with_columns(pl.when((C("yr")==2026)&(C("mon")==1)&(C("gs")<1800)).then(C("served")+38).otherwise(C("served")).alias("pre"))
eb = pl.read_parquet(I + "agent_eb136_tbl.parquet", columns=["id","pred"]); f = f.join(eb, on="id", how="left")
ec = pl.concat([pl.read_csv(f"data/external/eurocontrol/taxi_out_additional_time_{y}.csv", infer_schema_length=0) for y in (2025,2026)]).filter(C("APT_ICAO")=="EDDM")
ec = ec.select(C("YEAR").cast(int).alias("yr"), C("MONTH_NUM").cast(int).alias("mon"), C("TF").cast(int), C("VALID_FL").cast(int), C("NO_REF").cast(int), C("TOTAL_REF_NB_FL").cast(int),
   ((C("TOTAL_REF_TIME_MIN").cast(float)+C("TOTAL_ADD_TIME_MIN").cast(float))*60/C("TOTAL_REF_NB_FL").cast(float)).alias("pub")).with_columns((C("TF")-C("VALID_FL")).alias("excl"))
ok = f.filter(C("yr")==2025).filter(C("y").is_not_null() & (C("y")<=7200) & C("stand").is_not_null() & C("rwy").is_not_null())
def groups(base, extra=None):
    g = {}
    for (s, r), d in base.group_by("stand","rwy"):
        y = d["y"].to_numpy(); p10 = np.quantile(y, 0.10); g[(s, r)] = int((y <= p10).sum())
    return g
def valid_flags(df, g, add=None, k=10):
    return np.array([ (g.get((s, r), 0) + (0.1*add.get((s, r), 0) if add else 0)) >= k for s, r in zip(df["stand"].to_list(), df["rwy"].to_list())])
def wmean(x, w):
    m = ~np.isnan(x); return float((x[m]*w[m]).sum()/w[m].sum()), float(w[m].sum())
rows = []
G_all = groups(ok)
for yr, mo in [(2025,m) for m in range(1,13)] + [(2026,1),(2026,7)]:
    d = f.filter((C("yr")==yr)&(C("mon")==mo)&C("stand").is_not_null()&C("rwy").is_not_null())
    if yr == 2025:
        g = G_all; vf = valid_flags(d, g)
    elif mo == 1:
        g = groups(ok.filter(C("mon")>=2)); add = {k: v_ for k, v_ in zip(*[[tuple(a) for a in d.group_by("stand","rwy").agg(pl.len()).select("stand","rwy").iter_rows()], d.group_by("stand","rwy").agg(pl.len())["len"].to_list()])}
        vf = valid_flags(d, g, add)
    else:
        g = G_all; vf = valid_flags(d, g)
    d = d.filter(pl.Series(vf))
    w1 = 1 - d["w"].fill_null(0).to_numpy()
    r = dict(yr=yr, mon=mo, n_valid=d.height, n_pub=int(ec.filter((C("yr")==yr)&(C("mon")==mo))["TOTAL_REF_NB_FL"][0]), n_eff=float(w1.sum()),
             excl_pub=int(ec.filter((C("yr")==yr)&(C("mon")==mo))["excl"][0]), excl_det=float((1-w1).sum()), pub=float(ec.filter((C("yr")==yr)&(C("mon")==mo))["pub"][0]))
    if yr == 2025:
        yv = d["y"].to_numpy().astype(float); r["lab_all"] = float(yv.mean()); r["lab_w"], _ = wmean(yv, w1)
        hold = d.filter(C("y").is_between(0,14400)&C("pred").is_not_null())
        if hold.height: 
            ww = 1-hold["w"].fill_null(0).to_numpy(); r["lab_w_hold"],_ = wmean(hold["y"].to_numpy(), ww); r["pred_w_hold"],_ = wmean(hold["pred"].to_numpy(), ww)
            r["lab_all_hold"] = float(hold["y"].mean()); r["pred_all_hold"] = float(hold["pred"].mean())
    else:
        pv = d["pre"].to_numpy().astype(float); sv = d["served"].to_numpy().astype(float)
        r["pre_w"], _ = wmean(pv, w1); r["served_w"], _ = wmean(sv, w1); r["pre_all"] = float(np.nanmean(pv))
        sn = d.filter(C("seen")); wn = 1-sn["deice"].cast(pl.Float64).to_numpy(); r["pre_seen_noflag"], _ = wmean(sn["pre"].to_numpy().astype(float), wn)
    rows.append(r)
t = pl.DataFrame(rows, infer_schema_length=None)
print(t)
t.write_parquet(I + "agent_apt_EDDM_level.parquet")
t25 = t.filter(C("yr")==2025).with_columns((C("lab_all")-C("pub")).alias("d_all"), (C("lab_w")-C("pub")).alias("d_w"), (C("lab_w_hold")-C("pub")).alias("d_w_hold"))
print(t25.select("mon","n_valid","n_pub","excl_pub","excl_det","pub","lab_all","lab_w","d_all","d_w","d_w_hold"))
for nm, mm in [("all 12", list(range(1,13))), ("winter 1,2,11,12", [1,2,11,12]), ("deice-heavy 1,2,12", [1,2,12]), ("non-summer 1,2,3,10,11,12",[1,2,3,10,11,12])]:
    s = t25.filter(C("mon").is_in(mm))
    print(f"{nm}: d_all mean {s['d_all'].mean():.1f} sd {s['d_all'].std():.1f}   d_w mean {s['d_w'].mean():.1f} sd {s['d_w'].std():.1f}")
t26 = t.filter(C("yr")==2026)
print(t26.select("mon","n_valid","n_pub","excl_pub","excl_det","pub","pre_all","pre_w","served_w","pre_seen_noflag"))
