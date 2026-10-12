import polars as pl, numpy as np, lightgbm as lgb
C = pl.col
R = ""
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
m0 = 1025.5
W = 3606


def garble(x):
    if x is None:
        return 0.0
    x = x.strip().upper()
    if len(x) < 6 or not x[:3].isalpha():
        return 0.0
    p = x[:3]
    for k in (2, 1):
        if len(x) > 3 + k and x[3:3 + k] == p[3 - k:3] and x[3 + k].isdigit():
            return 1.0
    return 0.0


S = pl.read_parquet(R + "data/interim/agent_rome2_sensor.parquet").filter(C("gs") >= 1800)
dp = pl.read_parquet(R + "data/interim/agent_rome2_depall.parquet").filter(C("ADEP_mvt") == "LIRF").select("MVT_ID_mvt", C("ADES_mvt").alias("dest"))
S = S.join(dp, on="MVT_ID_mvt", how="left")
S = S.with_columns(pl.Series("garb", [garble(x) for x in S["FLIGHT_mvt"].to_list()]),
                   pl.Series("dest_hc", [1.0 if x in ("LTFJ", "LMML", "LTFM", "LLBG") else 0.0 for x in S["dest"].to_list()]),
                   ((C("S") % 300) == 0).cast(pl.Float64).alias("round5"), ((C("S") % 900) == 0).cast(pl.Float64).alias("round15"),
                   (C("S") // 3600 % 24).alias("sh"), ((C("S") // 86400 + 3) % 7).alias("dow"),
                   C("STAND_mvt").str.slice(0, 1).cast(pl.Float64, strict=False).alias("apron"),
                   C("STAND_mvt").cast(pl.Float64, strict=False).alias("stand_num"), C("np").cast(pl.Float64).alias("npf"),
                   C("copy_raw").cast(pl.Int32).alias("c"))
tr = S.filter(C("yr") == 2025)
te = S.filter(C("yr") == 2026)
mo = tr["mo"].to_numpy()
c = tr["c"].to_numpy()
n = tr.height
p0 = c.mean()


def enc(col):
    v = tr[col].to_list()
    u = sorted({x for x in v if x is not None})
    ix = {k: i for i, k in enumerate(u)}
    ii = np.array([ix.get(x, -1) for x in v])
    K = np.zeros((len(u), 13))
    N = np.zeros((len(u), 13))
    for i in range(n):
        if ii[i] >= 0:
            K[ii[i], mo[i]] += c[i]
            N[ii[i], mo[i]] += 1
    return ix, ii, K, N


def prior(K, N, ex):
    keep = [m for m in range(1, 13) if m not in ex]
    k = K[:, keep].sum(1)
    nn = N[:, keep].sum(1)
    return (k + 5 * p0) / (nn + 5), nn


base = ["gs", "round5", "round15", "sh", "dow", "npf", "apron", "stand_num", "gl", "ga", "ge", "gi"]
arr = ["arr_n_w1h", "arr_copy_w1h", "arr_np_w1h", "arr_n_w3h", "arr_copy_w3h", "arr_np_w3h", "arr_copy_w12h", "arr_np_w12h", "op6_n", "op6_copy", "op6_np", "st6_n", "st6_copy"]
link = ["L1_exists", "L1_copy", "L1_np", "L1_age", "L1_sameop", "L2_exists", "L2_copy", "L2_np", "L2_age", "L2_fdiff", "Nx_exists", "Nx_gap", "Nx_copy"]
cols = base + arr + link
ixo, iio, Ko, No = enc("op")
ixd, iid, Kd, Nd = enc("dest")


def build(df, train):
    Xb = np.column_stack([df[x].cast(pl.Float64).to_numpy() for x in cols])
    Xn = np.column_stack([df[x].cast(pl.Float64).to_numpy() for x in ["garb", "dest_hc"]])
    if train:
        pr = np.zeros(n); nn = np.zeros(n); pd = np.zeros(n); nd = np.zeros(n)
        for k in range(1, 13):
            sel = mo == k
            po, no = prior(Ko, No, {k})
            pde, nde = prior(Kd, Nd, {k})
            io = iio[sel]
            idd = iid[sel]
            pr[sel] = np.where(io >= 0, po[np.maximum(io, 0)], p0)
            nn[sel] = np.where(io >= 0, no[np.maximum(io, 0)], 0)
            pd[sel] = np.where(idd >= 0, pde[np.maximum(idd, 0)], p0)
            nd[sel] = np.where(idd >= 0, nde[np.maximum(idd, 0)], 0)
    else:
        po, no = prior(Ko, No, set())
        pde, nde = prior(Kd, Nd, set())
        io = np.array([ixo.get(x, -1) for x in df["op"].to_list()])
        idd = np.array([ixd.get(x, -1) for x in df["dest"].to_list()])
        pr = np.where(io >= 0, po[np.maximum(io, 0)], p0)
        nn = np.where(io >= 0, no[np.maximum(io, 0)], 0)
        pd = np.where(idd >= 0, pde[np.maximum(idd, 0)], p0)
        nd = np.where(idd >= 0, nde[np.maximum(idd, 0)], 0)
    X1 = np.column_stack([Xb, pr, np.log1p(nn)])
    return X1, np.column_stack([X1, Xn, pd, np.log1p(nd)])


X1, X2 = build(tr, True)
T1, T2 = build(te, False)
PO = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, verbose=-1, num_threads=4)
p1 = np.zeros(te.height)
p2 = np.zeros(te.height)
for sd in range(3):
    q = {**PO, "seed": sd, "bagging_seed": sd, "feature_fraction_seed": sd}
    p1 += lgb.train(q, lgb.Dataset(X1, c), 250).predict(T1) / 3
    p2 += lgb.train(q, lgb.Dataset(X2, c), 250).predict(T2) / 3
te = te.with_columns(pl.Series("Pn1", p1), pl.Series("Pn2", p2))
F = pl.read_parquet(R + "data/interim/agent_rome2_footprint26b.parquet").select(ID, "mo", "FLIGHT_mvt", "gs", "v132", "P")
F = F.join(te.select(ID, "Pn1", "Pn2", "dest"), on=ID, how="left")
print("footprint", F.height, "corr(old P, refit P1)", np.corrcoef(F["P"], F["Pn1"])[0, 1], "mean P1 %.3f P2 %.3f" % (F["Pn1"].mean(), F["Pn2"].mean()), "dest null", F["dest"].null_count())
v133 = pl.read_parquet(R + "data/interim/agent_bundle_v133_final.parquet")
Af = pl.read_parquet(R + "data/interim/agent_rome2_A_final.parquet")
chk = F.select(ID).join(v133, on=ID).join(Af.rename({T: "A"}), on=ID)
print("v133 == A_final on footprint rows:", int((chk[T] == chk["A"]).sum()), "of", chk.height)
z = np.load(R + "data/interim/agent_rome2_incr53.npz", allow_pickle=True)
a = z["a"]; y = z["y"]; gs = z["gs"]; rule = z["rule"]; P2 = z["P2"]; P1 = z["P1"]
mid = (gs >= 3600) & (gs < 14400) & (rule == "none")
GR = np.linspace(0, 1, 21)
wf = {}
for nm, lo, hi in [("b1", 3600, 7200), ("b2", 7200, 14400)]:
    sc = mid & (gs >= lo) & (gs < hi)
    mx = P2 * gs + (1 - P2) * m0
    best = min(GR, key=lambda w: ((y[sc] - (a[sc] + w * (mx[sc] - a[sc]))) ** 2).sum())
    wf[nm] = float(best)
    mx1 = P1 * gs + (1 - P1) * m0
    b1 = min(GR, key=lambda w: ((y[sc] - (a[sc] + w * (mx1[sc] - a[sc]))) ** 2).sum())
    print(nm, "fitted all-month weight P2", best, "(P1:", b1, ")")
Gs = F["gs"].to_numpy()
Pn = F["Pn2"].to_numpy()
A = F["v132"].to_numpy().astype(float)
w = np.where(Gs < 7200, wf["b1"], wf["b2"])
newv = np.maximum(np.rint(A + w * (Pn * Gs + (1 - Pn) * m0 - A)), 0)
base133 = F.join(v133.select(ID, C(T).alias("v133v")), on=ID, how="left")["v133v"].to_numpy().astype(float)
dpm = newv - base133
print("increment vs v133 on footprint: rms move %.0f, max %.0f, sum dp^2/row %.1f, mean %+.1f; rows moving >300: %d" % (np.sqrt((dpm ** 2).mean()), np.abs(dpm).max(), (dpm ** 2).sum() / 344841, dpm.mean(), (np.abs(dpm) > 300).sum()))
upd = pl.DataFrame({ID: F[ID], "_n": newv})
out = v133.join(upd, on=ID, how="left").with_columns(pl.coalesce(C("_n"), C(T).cast(pl.Float64)).round(0).cast(v133.schema[T]).alias(T)).drop("_n").select(v133.columns)
out.write_parquet(R + "data/interim/agent_rome3_final.parquet")
# Lever B
H = pl.read_parquet(R + "data/interim/agent_rome2_hidden2.parquet").filter(C("yr") == 2026)
H = H.with_columns(pl.when((C("gl") >= W) & ((C("gap") - C("gl")).abs() < W)).then(pl.lit("R1")).when(C("gl") < W).then(pl.lit("R2")).when(C("a_l").abs() < W).then(pl.lit("R3a")).otherwise(pl.lit("R3b")).alias("rule"))
B = H.filter(C("rule") == "R3b").select(ID, "FLIGHT_mvt", C("gap").alias("gs"), C("t_aobt").clip(0, None).alias("tgt")).join(v133, on=ID)
print(B)
g = B["gs"].to_numpy().astype(float)
cur_ = B[T].to_numpy().astype(float)
tg = B["tgt"].to_numpy().astype(float)
Nn = 344841
Rr = 221.65
for p in [0.0, 0.05, 0.12, 0.2, 0.26, 0.4, 1.0]:
    ev = ((1 - p) * ((cur_ - tg) ** 2) + p * (-(g - tg) ** 2 + (g - cur_) ** 2)).sum() / (2 * Nn * Rr)
    print("P(copy)=%.2f expected dRMSE (positive=gain) %+.2f" % (p, ev))
print("if real: gain %.2f ; if all copies: %.2f ; per row if copy:" % (((cur_ - tg) ** 2).sum() / (2 * Nn * Rr), (-(g - tg) ** 2 + (g - cur_) ** 2).sum() / (2 * Nn * Rr)), np.round((-(g - tg) ** 2 + (g - cur_) ** 2) / (2 * Nn * Rr), 2))
upd = pl.DataFrame({ID: B[ID], "_n": tg})
outB = v133.join(upd, on=ID, how="left").with_columns(pl.coalesce(C("_n"), C(T).cast(pl.Float64)).round(0).cast(v133.schema[T]).alias(T)).drop("_n").select(v133.columns)
outB.write_parquet(R + "data/interim/agent_rome3B_final.parquet")
# dose evidence (served-style fold, as computed before v133; no refit)
d = pl.read_parquet(R + "data/interim/agent_rome2_midgap_fold.parquet")
a_ = d["pred"].to_numpy(); y_ = d["y"].to_numpy(); g_ = d["gs"].to_numpy(); P_ = d["P"].to_numpy(); mo_ = d["mo"].to_numpy()
mx = P_ * g_ + (1 - P_) * m0
Nm = {1: 130045, 2: 143732, 3: 164449, 7: 190713, 8: 191182, 9: 183950, 10: 185674, 11: 140142}
Sm = {1: (251.9 ** 2) * 130045, 2: (247.6 ** 2) * 143732, 3: (167.5 ** 2) * 164449, 7: (259.9 ** 2) * 190713, 8: (212.2 ** 2) * 191182, 9: (217.4 ** 2) * 183950, 10: (183.6 ** 2) * 185674, 11: (173.9 ** 2) * 140142}
for w in [0.5, 0.746]:
    gg = (y_ - a_) ** 2 - (y_ - (a_ + w * (mx - a_))) ** 2

    def f(Ms, x=None):
        x = gg[np.isin(mo_, Ms)].sum() if x is None else x
        Ns = sum(Nm[M] for M in Ms)
        Ss = sum(Sm[M] for M in Ms)
        return np.sqrt(Ss / Ns) - np.sqrt((Ss - x) / Ns)
    s17 = np.sort(gg[np.isin(mo_, [1, 7])])[::-1]
    tot17 = gg[np.isin(mo_, [1, 7])].sum()
    print("dose w=%.3f: Jan %+.2f Jul %+.2f pooled %+.2f all8 %+.2f ; minus10 pooled %+.2f minus50 pooled %+.2f ; top10 share %.2f" % (w, f([1]), f([7]), f([1, 7]), f([1, 2, 3, 7, 8, 9, 10, 11]), f([1, 7], tot17 - s17[:10].sum()), f([1, 7], tot17 - s17[:50].sum()), s17[:10].sum() / tot17))
    print("   by month:", {M: round(float(f([M])), 2) for M in [1, 2, 3, 7, 8, 9, 10, 11]})
print("DIAG")
dP=(F["Pn2"]-F["Pn1"]).to_numpy(); print("2026 footprint dP: mean %.3f sd %.3f absmean %.3f"%(dP.mean(),dP.std(),np.abs(dP).mean()))
zz=np.load(R+"data/interim/agent_rome2_incr53.npz",allow_pickle=True); m_=(zz["gs"]>=3600)&(zz["gs"]<14400)&(zz["rule"]=="none"); d25=(zz["P2"]-zz["P1"])[m_]
print("2025 fold dP: mean %.3f sd %.3f absmean %.3f"%(d25.mean(),d25.std(),np.abs(d25).mean()))
print(F.with_columns((C("Pn2")-C("Pn1")).alias("dP")).sort("dP").select("FLIGHT_mvt","dest","gs","P","Pn1","Pn2","dP").head(6))
print(F.with_columns((C("Pn2")-C("Pn1")).alias("dP")).sort("dP",descending=True).select("FLIGHT_mvt","dest","gs","P","Pn1","Pn2","dP").head(6))
print("increment pieces: mean|dp| old-w .5 mix1:",np.abs(0.5*(F["Pn1"].to_numpy()*Gs+(1-F["Pn1"].to_numpy())*m0-A)).mean()," new:",np.abs(w*(Pn*Gs+(1-Pn)*m0-A)).mean())
print("base133 vs A+0.5(mix-A):",np.abs(base133-(A+0.5*(F["P"].to_numpy()*Gs+(1-F["P"].to_numpy())*m0-A))).max())
