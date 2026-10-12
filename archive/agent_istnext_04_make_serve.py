"""istnext 04 (generator; pattern of agent_istpcd_04_make_serve.py): writes archive/agent_istnext_04_serve.py = agent_istjul_03_serve.py (read only, unchanged; it reproduces the shipped
Istanbul term to 0.000) with (a) a --cols switch (none | D1 | D2 | D3 | GY = the six day / window columns of D1, post hoc) appending the design's columns after the view columns, built for fold AND 2026 serve rows by the same builder
(serve rows = an extra pseudo-month outside the pool: their cell statistics come from all eight 2025 months, non-snow rows; fold rows leave out their own month), and
(b) the tail replaced by: save the served term (k x clipped seed mean, snow rows 0) in serve-row order + the fold / serve column matrices for the distribution check. Run once (no fit)."""
src = open("archive/agent_istjul_03_serve.py", encoding="utf-8").read()
head = src.split("d = a.k * d; d[snow_s] = 0")[0]
head = head.replace('ap.add_argument("--arm", default="AQC");', 'ap.add_argument("--cols", default="none", help="none | D1 | D2 | D3"); ap.add_argument("--arm", default="AQC");', 1)
assert "--cols" in head
needle = "Ftr2 = np.hstack(parts_tr).astype(np.float32); Fs2 = np.hstack(parts_sv).astype(np.float32)"
assert needle in head
add = '''# ---- istnext: design columns for fold + serve rows (one builder call; serve rows are pseudo-month 99, outside the pool) ----
NEWF = NEWS = None; NEWN = []
if a.cols != "none":
    import agent_arith3_02_cols as C3
    import agent_istnext_lib as LB
    KN8 = [str(x) for x in K["kn"]]; Ns = len(rl); assert a.anom == "A" and not a.exclude and len(sr) == Ns
    ZC = np.load(I + "agent_istpcd_cols.npz", allow_pickle=True); assert (ZC["fid"] == fid).all() and (ZC["sid"] == rl[ID].to_numpy()).all()
    rk = pl.read_parquet("data/raw/ranking.parquet", columns=[ID, "MVT_TIME_UTC_mvt"]).unique(subset=[ID], keep="first").with_columns(C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"))
    tsv = rl.select(ID).join(rk.select(ID, "t"), on=ID, how="left", maintain_order="left")["t"]; assert tsv.null_count() == 0
    rows_all = np.concatenate([np.arange(Nf), sr]); kcA = lambda k: Kall[rows_all, KN8.index(k)]  # noqa: E731
    rw_all = np.unique(np.concatenate([rwy_s, rl["RUNWAY_mvt"].fill_null("?").to_numpy()]).astype(str), return_inverse=True)[1].astype(float)
    hr_all = np.concatenate([Ftr[:, cols.index("hour")], Fs[:, cols.index("hour")]]); dw_all = np.concatenate([Ftr[:, cols.index("dow")], Fs[:, cols.index("dow")]])
    mon_all = np.concatenate([monf.astype(np.int64), np.full(Ns, 99, np.int64)]); pv_all = np.concatenate([ref, v137]).astype(np.float64); y_all = np.concatenate([yf, np.zeros(Ns)]).astype(np.float64)
    r_all = np.concatenate([yf - ref, np.zeros(Ns)]).astype(np.float64)
    use = np.concatenate([~snow_f, np.zeros(Ns, bool)]); mlist = sorted(set(mon_all.tolist())); pmask = np.array([m != 99 for m in mlist]); MIa = np.searchsorted(mlist, mon_all)
    Dd = dict(pv=pv_all.astype(np.float32), y=y_all, mon=mon_all, apt=np.array(["LTFM"] * (Nf + Ns)), hour=hr_all, RUNWAY_mvt=rw_all, is_weekend=(dw_all >= 6).astype(float),
              Kf=np.column_stack([kcA("op"), kcA("op_stand"), C3.factor(kcA("type"), rw_all), kcA("rwy_stand"), kcA("cty")]).astype(np.int64), KN=["op_apt", "op_stand", "ty_rwy", "st_rwy", "cty"],
              Dpcd=np.vstack([ZC["D"][:, :4], ZC["Ds"][:, :4]]).astype(np.float32), t=np.concatenate([_J["ts"].to_numpy(), tsv.to_numpy()]).astype(np.int64))
    POS, pn = LB.position_block(Dd, use, pmask)
    if a.cols in ("D1", "GY"):
        A_KEYS = ["rwy_stand", "op", "op_rwy", "op_stand", "rwy_hb", "ty_stand", "op_hb"]; A = []
        for k in A_KEYS:
            G = np.unique(kcA(k), return_inverse=True)[1]
            ym = C3.loo_cell(G, y_all, use, MIa, len(mlist), pmask)[0]; sdk = C3.loo_cell(G, r_all, use, MIa, len(mlist), pmask)[1]
            A += [pv_all - ym, (pv_all - ym) / np.maximum(sdk, 30.0)]
        NEW = np.column_stack([np.column_stack(A).astype(np.float32), POS]); lo_, hi_ = LB.limits(NEW, np.arange(Nf)); NEW = np.clip(NEW, lo_, hi_); NEWN = [f"A_{p}_{k}" for k in A_KEYS for p in ("lvl", "z")] + pn
        if a.cols == "GY":   # post-hoc subset of D1 (the six day / window columns), same winsorising
            keep_ = [i for i, n in enumerate(NEWN) if n.startswith("Y_")]; NEW = NEW[:, keep_]; NEWN = [NEWN[i] for i in keep_]
    elif a.cols == "D3":
        NEW = np.clip(POS[:, [pn.index(n) for n in LB.Z9]], -3.0, 0.0); NEWN = list(LB.Z9)
    elif a.cols == "D2":
        L9 = np.load(I + "agent_istnext_pool/limits.npz", allow_pickle=True); Xp = np.clip(POS, L9["lo"], L9["hi"]); NEW = np.full((Nf + Ns, 1), np.nan, np.float32)
        for m in mlist:
            rr = np.where(mon_all == m)[0]; NEW[rr, 0] = lgb.Booster(model_file=I + "agent_istnext_pool/" + ("all" if m == 99 else "m%d" % m) + ".txt").predict(Xp[rr])
        NEWN = ["D2_pooled_response"]
    NEWF, NEWS = NEW[:Nf].astype(np.float32), NEW[Nf:].astype(np.float32); parts_tr.append(NEWF); parts_sv.append(NEWS)
print("extra columns", a.cols, len(NEWN), flush=True)
'''
head = head.replace(needle, add + needle, 1)
tail = '''d = a.k * d; d[snow_s] = 0
np.savez(I + a.out + "_term.npz", sid=rl[ID].to_numpy(), d=d, snow_serve=snow_s, k=a.k, seeds=np.array(SEEDS), names=np.array(NEWN),
         **({"NEWF": NEWF, "NEWS": NEWS, "monf": monf, "snow_f": snow_f} if NEWF is not None else {}))
print("served term (k %.2f, seeds %s): mean %.3f sd %.3f, snow rows %d at 0" % (a.k, SEEDS, float(d.mean()), float(d.std()), int(snow_s.sum())))
_o = np.load(I + "agent_lv4r_ltfm_final_delta.npz", allow_pickle=True); _p = {x: i for i, x in enumerate(_o["ids"])}; _old = _o["new"][[_p[x] for x in rl[ID].to_numpy()]]
print("this term at k 0.75 scale minus the shipped lv4r term (k 0.75): mean %.4f rms %.4f max abs %.4f (0 = reproduces; meaningful for --cols none)" % (float((d * 0.75 / a.k - _old).mean()), float(np.sqrt(((d * 0.75 / a.k - _old) ** 2).mean())), float(np.abs(d * 0.75 / a.k - _old).max())))
print("wrote", a.out + "_term.npz")
'''
open("archive/agent_istnext_04_serve.py", "w", encoding="utf-8").write(head + tail)
print("written archive/agent_istnext_04_serve.py", len(head + tail))
