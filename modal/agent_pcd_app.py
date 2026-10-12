"""agent_pcd (docs/LAB-NOTES.md): COPY of modal/agent_bigc13_app.py (the shipped refit recipe, config C4ff7V5) with NINE extra input columns appended after the 52 variant-view columns:
d_SCHED, d_LOBT, d_IOBT, d_EOBT, d_AOBT3, clamp, wmin, d_AOBT3_ind, d_AOBT3_der (prediction minus take-off-minus-clock; file data/interim/agent_pcd_cols_app.npz = fold D, serve Ds + the id vectors asserted here).
Everything else is identical. OWN app name (prc-pcd) and OWN volume folders (/vol/pcd/out, /vol/pcd/serve, /vol/pcd/in): nothing of /vol/bigc13/out or /vol/bigc13/serve is read as output or overwritten
(the bigc13 INPUT files in /vol/bigc13/in and /vol/bigc12/in are read-only fallbacks, as in the bigc13 app).
    $SYS modal/agent_pcd_app.py upload                                                      # agent_pcd_cols_app.npz (91 MB) -> prc-stage1:/pcd/in
    $SYS -m modal run --detach modal/agent_pcd_app.py::launch --cfgs C4ff7V5 --seeds 0 --months 1,7             # step 1: 2 fits
    $SYS -m modal run --detach modal/agent_pcd_app.py::launch --cfgs C4ff7V5 --seeds 0,1,2                      # step 2: 36 fits (finished ones are skipped)
    $SYS -m modal run --detach modal/agent_pcd_app.py::launch_serve --cfgs C4ff7V5 --seeds 0,1,2,3,4,5          # step 3: 6 serve fits (needs the 12+ month fits above for the round count)
    $SYS modal/agent_pcd_app.py status | fetch_months | fetch | fetch_serve                                       # fetch_* write to data/interim/agent_pcd_modal/
"""
import os
import sys

import modal

image = (modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.2", "lightgbm==4.7.0", "polars==1.44.1")
         .add_local_file("archive/agent_bigc_te.py", "/repo/agent_bigc_te.py"))
vol = modal.Volume.from_name("prc-stage1", create_if_missing=True)
app = modal.App("prc-pcd")
IN, OUT = "/vol/pcd/in", "/vol/pcd/out"
SERVE = "/vol/pcd/serve"
IN13 = "/vol/bigc13/in"                                   # bigc13 inputs, read-only
IN12 = "/vol/bigc12/in"                                   # unchanged inputs, read-only fallback
PFX = os.environ.get("BIGC13_PFX", "agent_bigc13")        # agent_bigc13 = new fit base; agent_bigc12 = shipped inputs (local control arm)
NEW3 = ("fold", "keys2", "serve_in")


def pj(name):
    """path of an input file: searched in IN, then IN12, then data/interim (local runs)."""
    for d in (IN, IN13, IN12, "data/interim"):
        q = f"{d}/{name}"
        if os.path.exists(q):
            return q
    return f"{IN}/{name}"


def pcd_cols():
    import numpy as np
    return np.load(pj("agent_pcd_cols_app.npz"), allow_pickle=True)


def pn(kind):
    return pj(f"{PFX}_{kind}.npz")
CFG = {  # leaves, min_data, l2, max rounds, extra params
    "R1": (15, 500, 10.0, 600, {}), "C1": (63, 300, 20.0, 1200, {}), "C2": (127, 300, 50.0, 1200, {}),
    "C3": (255, 300, 50.0, 1500, {}), "C3b": (255, 150, 50.0, 1500, {}), "C4": (511, 300, 100.0, 1500, {}), "C4b": (511, 150, 100.0, 1500, {}),
    "C3h": (255, 300, 50.0, 1500, {"objective": "huber", "alpha": 150.0}), "C3f": (255, 300, 50.0, 1500, {"objective": "fair", "fair_c": 150.0}),
    "C4h": (511, 300, 100.0, 1500, {"objective": "huber", "alpha": 150.0}),
    "C4ff3": (511, 150, 100.0, 1500, {"feature_fraction": 0.3}), "C4ff7": (511, 150, 100.0, 1500, {"feature_fraction": 0.7}),
    "C3ff3": (255, 300, 50.0, 1500, {"feature_fraction": 0.3}), "C3ff7": (255, 300, 50.0, 1500, {"feature_fraction": 0.7}),
}
FILES = ("agent_bigc12_serve_in.npz", "agent_bigc12_maps.json", "agent_bigc12_fold.npz", "agent_bigc12_extra.npz", "agent_bigc12_extra2.npz", "agent_bigc12_keys.npz", "agent_bigc12_keys2.npz", "agent_bigc12_vfac_clean3.npz", "agent_bigc12_vfac_clean4.npz", "agent_bigc12_vfac_clean5.npz", "agent_bigc12_vfac_clean6.npz", "agent_bigc12_vfac_clean6a.npz")
# suffix V = W + variant-factory columns; modifiers after "+": D (fit y directly, pv a feature), M (monotone in pv), P (per-coverage-class models), e.g. C4ff7W+DM -- docs/LAB-NOTES.md
# suffix W = Y + 5 pair keys (16 keys x 7 stats) + 8 label-free drift columns; suffix L = Y + LTFM rows pooled (snow rows excluded from fitting; judge zeroes them) -- ladder Z, docs/LAB-NOTES.md
# a cfg name ending in "Y" = X inputs + the rich key statistics (archive/agent_bigc_te.py Te2; keys from archive/agent_bigc_v2_15_keys.py), e.g. C3Y, C4bY
# a cfg name ending in "X" = same params as the base name, plus step-3 inputs (d_NEW, d_ET, out-of-month operator/callsign/stand/type target stats), e.g. C3X


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=28672, timeout=60 * 60 * 2, retries=1)
def fit_one(cfg: str, seed: int, month: int) -> dict:
    import json
    import time
    import numpy as np
    import lightgbm as lgb
    vol.reload()
    path = f"{OUT}/{cfg}_s{seed}_m{month}.npz"
    if os.path.exists(path):
        return {"cfg": cfg, "seed": seed, "month": month, "skipped": True}
    t0 = time.time()
    z = np.load(pn("fold"), allow_pickle=True); ex = np.load(pj("agent_bigc12_extra.npz"), allow_pickle=True)
    cfg0, *mods = cfg.split("+")   # modifiers after "+": D = fit y directly (pv is a feature), M = monotone-increasing in pv, P = one model per coverage class
    import re
    mv = re.match(r"^(.*?)V([0-9a-z]*)$", cfg0)   # versioned vfac input: C4ff7V -> agent_bigc12_vfac_clean.npz, C4ff7V2a -> agent_bigc12_vfac_clean2a.npz, C4ff7V3 -> ...clean3.npz
    if mv:
        suf, base_cfg, vtag = "V", mv.group(1), mv.group(2)
    else:
        suf = cfg0[-1] if cfg0[-1] in "XYWL" else ""; base_cfg = cfg0[:-1] if suf else cfg0; vtag = ""
    x3 = bool(suf); y3 = suf in ("Y", "W", "L", "V"); w3 = suf in ("W", "V"); pool = suf == "L"
    X = np.hstack([z["X"], ex["Xf"]]); catidx = [int(i) for i in z["cat"]]; y = z["y"]; pv = z["pv"]; mon = z["mon"]
    cols = list(z["cols"]) + list(ex["cols"]); apts = None
    if x3:
        sys.path.insert(0, "/repo"); sys.path.insert(0, "archive"); import agent_bigc_te as TE
        e2 = np.load(pj("agent_bigc12_extra2.npz"), allow_pickle=True); Xe2 = e2["Xf"]
        if y3:
            kk = np.load(pn("keys2") if (w3 or pool) else pj("agent_bigc12_keys.npz"), allow_pickle=True); K = kk["Kf"][:, :11] if pool else kk["Kf"]
            apts = np.load(pj("agent_bigc12_keys.npz"), allow_pickle=True)["apt_f"]
        if pool:  # LTFM rows (v113-analogue REF, months 1,2,3,7,8,9,10), pooled with the nine airports; ADS-B / chain-variant inputs are NaN there
            zl = np.load(pj("agent_bigc_ltfm_fold.npz"), allow_pickle=True); nL = len(zl["y"])
            XL = np.hstack([zl["X"], np.full((nL, ex["Xf"].shape[1]), np.nan, np.float32)]); X = np.vstack([X, XL]); Xe2 = np.vstack([Xe2, np.full((nL, 2), np.nan, np.float32)])
            y = np.concatenate([y, zl["y"]]); pv = np.concatenate([pv, zl["pv"]]); mon = np.concatenate([mon, zl["mon"]]); K = np.vstack([K, kk["Kl"][:, :11]]); apts = np.concatenate([apts, np.array(["LTFM"] * nL)])
            snow = np.zeros(len(y), bool); snow[-nL:] = (zl["X"][:, cols.index("wx_snow")] == 1) | (zl["X"][:, cols.index("wx_freezing")] == 1); del XL
        X = np.hstack([X, Xe2, np.zeros((len(X), 4), np.float32)]); cols = cols + ["d_NEW", "d_ET"]
    r = np.clip(y - pv, -900, 900).astype(np.float32); months = sorted(set(mon.tolist()))
    if x3:
        X[:, -4:] = TE.te_fold(TE.keys_from(X, cols + ["x"] * 4), r.astype(float), mon, month, months)
        if y3:  # rich out-of-month key statistics, nested: train rows exclude {held, own month}; held rows use the other 7
            ymu_apt = {a: y[apts == a].mean() for a in set(apts.tolist())}; ymu = np.array([ymu_apt[a] for a in apts])
            F2 = TE.Te2(K).fold(r.astype(float), y.astype(float), ymu, mon, month, months); X = np.hstack([X, F2]); del F2
        if w3:  # label-free drift features (own-month mean of pv / ADS-B span minus other months), precomputed on the fold
            X = np.hstack([X, kk["Df"]])
        if suf == "V":  # variant-factory difference inputs (agent_bigc12_vfac_clean.npz = agent_vfac_extra.npz with out-of-band airports NaN-ed; NaN allowed)
            X = np.hstack([X, np.load(pj(f"agent_bigc12_vfac_clean{vtag}.npz"), allow_pickle=True)["Xf"]])
            pc = pcd_cols(); assert pc["D"].shape == (len(X), 9), (pc["D"].shape, len(X))
            assert np.array_equal(pc["ids"].astype(float), z["ids"].astype(float)), "pcd columns are not in fold row order"
            X = np.hstack([X, pc["D"]]); del pc
    nxt = [k for k in months if k != month]; val = min([k for k in nxt if k > month] or nxt)
    tr = (mon != month) & (mon != val); te = mon == month; va = mon == val
    if pool:
        tr &= ~snow; va &= ~snow   # LTFM snow rows belong to the v117 snow model; no training / early stopping on them
    nl, md, l2, nr, extra = CFG[base_cfg]
    P = dict(objective="l2", learning_rate=0.03, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.5, max_bin=63, num_threads=int(os.environ.get("BIGC_THREADS", 8)), verbose=-1,
             seed=seed, num_leaves=nl, min_data_in_leaf=md, lambda_l2=l2)
    P.update(extra)
    direct = "D" in mods; tgt = (y.astype(np.float32) if direct else r)
    if "M" in mods:
        mc = [0] * X.shape[1]; mc[cols.index("pv")] = 1; P["monotone_constraints"] = mc; P["monotone_constraints_method"] = "basic"
    def one(tr_, va_, te_):
        dt = lgb.Dataset(X[tr_], tgt[tr_], categorical_feature=catidx)
        dv = lgb.Dataset(X[va_], tgt[va_], categorical_feature=catidx, reference=dt)
        bst = lgb.train(P, dt, nr, valid_sets=[dv], callbacks=[lgb.early_stopping(50, verbose=False)])
        return bst.predict(X[te_], num_iteration=bst.best_iteration), bst.best_iteration
    if "P" in mods:
        cm = json.load(open(pj("agent_bigc12_maps.json")))["category"]; cov = X[:, cols.index("cov_cat")]
        cls = np.full(len(y), 3); cls[cov == cm["matched"]] = 0; cls[cov == cm["points_no_liftoff"]] = 1; cls[cov == cm["no_adsb_points"]] = 2
        pred = np.zeros(int(te.sum())); bis = []; te_idx = np.where(te)[0]
        for c in range(4):
            m_te = cls[te_idx] == c
            if m_te.sum() == 0 or (tr & (cls == c)).sum() < 1000 or (va & (cls == c)).sum() < 100: continue
            p_, bi_ = one(tr & (cls == c), va & (cls == c), te_idx[m_te]); pred[m_te] = p_; bis.append(bi_)
        best_it = int(np.mean(bis)) if bis else 0
    else:
        pred, best_it = one(tr, va, te)
    if direct: pred = np.clip(pred - pv[te], -900, 900)   # saved as a correction on pv, like every other arm
    os.makedirs(OUT, exist_ok=True)
    np.savez(path + ".tmp.npz", pred=pred, rows=np.where(te)[0], bi=best_it); os.replace(path + ".tmp.npz", path)
    vol.commit()
    return {"cfg": cfg, "seed": seed, "month": month, "bi": best_it, "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def launch(cfgs: str = "C4ff7V5", seeds: str = "0", months: str = "1,2,3,4,5,6,7,8,9,10,11,12"):
    for cfg in cfgs.split(","):
        for s in seeds.split(","):
            for m in [int(q) for q in months.split(",")]:
                c = fit_one.spawn(cfg, int(s), m)
                print("spawned", cfg, s, m, c.object_id)


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=28672, timeout=60 * 60 * 3, retries=1)
def serve_one(cfg: str, seed: int) -> dict:
    """SERVING fit: all fold months (leave-own-month-out key statistics), config's inputs, predicts the January/July 2026 rows. rounds = mean best_iter of the cfg's LOMO fits on the volume.
    Needs agent_bigc12_serve_in.npz (built locally by archive/agent_bigc_v2_36_serve_prep.py: serve bulk/extra/d_NEW,d_ET columns, drift columns, month, pv). Writes /vol/bigc13/serve/<cfg>_s<seed>.npz (pred, imp, names)."""
    import glob
    import re
    import time
    import numpy as np
    import lightgbm as lgb
    vol.reload()
    path = f"{SERVE}/{cfg}_s{seed}.npz"
    if os.path.exists(path):
        return {"cfg": cfg, "seed": seed, "skipped": True}
    t0 = time.time(); sys.path.insert(0, "/repo"); sys.path.insert(0, "archive"); import agent_bigc_te as TE
    assert "+" not in cfg, "structural modifiers are not supported in serve_one"
    mv = re.match(r"^(.*?)V([0-9a-z]*)$", cfg)
    if mv:
        suf, base_cfg, vtag = "V", mv.group(1), mv.group(2)
    else:
        suf = cfg[-1] if cfg[-1] in "XYW" else ""; base_cfg = cfg[:-1] if suf else cfg; vtag = ""
    assert suf in ("X", "Y", "W", "V"), "serve_one supports X/Y/W/V configs"
    y3 = suf in ("Y", "W", "V"); w3 = suf in ("W", "V")
    z = np.load(pn("fold"), allow_pickle=True); ex = np.load(pj("agent_bigc12_extra.npz"), allow_pickle=True); e2 = np.load(pj("agent_bigc12_extra2.npz"), allow_pickle=True)
    si = np.load(pn("serve_in"), allow_pickle=True)
    X = np.hstack([z["X"], ex["Xf"], e2["Xf"]]); cols = list(z["cols"]) + list(ex["cols"]) + ["d_NEW", "d_ET"]; Xs = si["Xs_pre"]
    assert list(map(str, si["cols_pre"])) == [str(c) for c in cols], "serve/fold column mismatch"
    y = z["y"]; pv = z["pv"]; mon = z["mon"]; catidx = [int(i) for i in z["cat"]]; months = sorted(set(mon.tolist()))
    r = np.clip(y - pv, -900, 900); names = list(map(str, cols))
    kf = TE.keys_from(X, cols); ks = TE.keys_from(Xs, cols)
    X = np.hstack([X, TE.te_loo(kf, r, mon, months)]); Xs = np.hstack([Xs, TE.te_serve(kf, r, ks)]); names += ["te_" + c for c in TE.KEYCOLS]
    if y3:
        kk = np.load(pn("keys2") if w3 else pj("agent_bigc12_keys.npz"), allow_pickle=True); k0 = np.load(pj("agent_bigc12_keys.npz"), allow_pickle=True)
        yy = y.astype(float); ymu_apt = {a: yy[k0["apt_f"] == a].mean() for a in set(k0["apt_f"].tolist())}
        T2 = TE.Te2(kk["Kf"])
        F2f = T2.fold(r, yy, np.array([ymu_apt[a] for a in k0["apt_f"]]), mon, None, months)
        F2s = T2.serve(r, yy, mon, months, kk["Ks"], np.array([ymu_apt.get(a, yy.mean()) for a in k0["apt_s"]]))
        X = np.hstack([X, F2f]); Xs = np.hstack([Xs, F2s]); names += [str(n) + "_" + q for n in kk["names"] for q in TE.STATS]; del F2f, F2s
        if w3:
            X = np.hstack([X, kk["Df"]]); Xs = np.hstack([Xs, si["Dsv"]]); names += ["drift_%s_%s" % (a, b) for a in ("op_apt", "op_stand", "st_rwy", "st_hb") for b in ("pv", "span")]
        if suf == "V":
            vf = np.load(pj(f"agent_bigc12_vfac_clean{vtag}.npz"), allow_pickle=True); X = np.hstack([X, vf["Xf"]]); Xs = np.hstack([Xs, vf["Xs"]]); names += ["vf_" + str(c) for c in vf["cols"]]
            pc = pcd_cols(); assert pc["D"].shape == (len(X), 9) and pc["Ds"].shape == (len(Xs), 9), (pc["D"].shape, pc["Ds"].shape, len(X), len(Xs))
            assert np.array_equal(pc["ids"].astype(float), z["ids"].astype(float)), "pcd fold columns are not in fold row order"
            assert np.array_equal(pc["ids_s"].astype(float), si["ids"].astype(float)), "pcd serve columns are not in serve_in row order"
            X = np.hstack([X, pc["D"]]); Xs = np.hstack([Xs, pc["Ds"]]); names += ["pcd_" + str(c) for c in pc["names"]]; del pc
    assert X.shape[1] == Xs.shape[1], (X.shape, Xs.shape)
    bis = [int(np.load(f)["bi"]) for f in glob.glob(f"{OUT}/{cfg}_s*_m*.npz")]; assert len(bis) >= 12, "no LOMO fits for this cfg on the volume"
    nr = int(round(float(np.mean(bis)))); nl, md, l2, _, extra = CFG[base_cfg]
    P = dict(objective="l2", learning_rate=0.03, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.5, max_bin=63, num_threads=int(os.environ.get("BIGC_THREADS", 8)), verbose=-1, seed=seed, num_leaves=nl, min_data_in_leaf=md, lambda_l2=l2)
    P.update(extra)
    b = lgb.train(P, lgb.Dataset(X, r.astype(np.float32), categorical_feature=catidx, params=dict(max_bin=63, verbose=-1)), nr)
    pred = b.predict(Xs); os.makedirs(SERVE, exist_ok=True)
    np.savez(path + ".tmp.npz", pred=pred, imp=b.feature_importance("gain"), names=np.array(names), rounds=nr); os.replace(path + ".tmp.npz", path); vol.commit()
    return {"cfg": cfg, "seed": seed, "rounds": nr, "features": X.shape[1], "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def launch_serve(cfgs: str = "C4ff7V5", seeds: str = "0,1,2,3,4,5"):
    for cfg in cfgs.split(","):
        for s in seeds.split(","):
            c = serve_one.spawn(cfg, int(s)); print("spawned serve", cfg, s, c.object_id)


def upload():
    """the nine-column file only"""
    with vol.batch_upload(force=True) as up:
        up.put_file("data/interim/agent_pcd_cols_app.npz", "/pcd/in/agent_pcd_cols_app.npz")
        print("uploaded agent_pcd_cols_app.npz")


def fetch_months():
    """finished per-month files -> data/interim/agent_pcd_modal/C4ff7V5_s<seed>_m<month>.npz"""
    os.makedirs("data/interim/agent_pcd_modal", exist_ok=True)
    for e in vol.listdir("pcd/out"):
        n = os.path.basename(e.path)
        if n.endswith(".npz") and ".tmp." not in n:
            open(f"data/interim/agent_pcd_modal/{n}", "wb").write(b"".join(vol.read_file(e.path))); print("fetched", n)


def fetch_serve():
    os.makedirs("data/interim/agent_pcd_modal/serve", exist_ok=True)
    for e in vol.listdir("pcd/serve"):
        n = os.path.basename(e.path)
        if n.endswith(".npz") and ".tmp." not in n:
            open(f"data/interim/agent_pcd_modal/serve/{n}", "wb").write(b"".join(vol.read_file(e.path))); print("fetched", n)


def status():
    for e in vol.listdir("pcd/out"):
        print(os.path.basename(e.path))


def fetch():
    import numpy as np
    got = {}
    for e in vol.listdir("pcd/out"):
        n = os.path.basename(e.path)
        if not n.endswith(".npz") or ".tmp." in n:
            continue
        cfg, s, m = n[:-4].split("_")[0], int(n[:-4].split("_")[1][1:]), int(n[:-4].split("_")[2][1:])
        buf = b"".join(vol.read_file(e.path))
        import io
        d = np.load(io.BytesIO(buf)); got.setdefault((cfg, s), {})[m] = (d["rows"], d["pred"], int(d["bi"]))
    N = len(np.load("data/interim/agent_bigc13_fold.npz", allow_pickle=True)["y"])
    for (cfg, s), ms in got.items():
        if len(ms) < 12:
            print("incomplete", cfg, s, sorted(ms)); continue
        arr = np.zeros(N)
        for m, (rows, pred, bi) in ms.items():
            arr[rows] = pred
        os.makedirs("data/interim/agent_pcd_modal", exist_ok=True); out = f"data/interim/agent_pcd_modal/agent_pcd_r1x_s{s}_{cfg}.npz"
        np.savez(out, **{cfg: arr, cfg + "_bi": np.array([ms[m][2] for m in sorted(ms)])}); print("wrote", out)


if __name__ == "__main__":
    {"upload": upload, "fetch_months": fetch_months, "status": status, "fetch": fetch, "fetch_serve": fetch_serve}[sys.argv[1]]()
