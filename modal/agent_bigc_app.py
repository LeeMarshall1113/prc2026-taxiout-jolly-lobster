"""bigc ladder on Modal (docs/LAB-NOTES.md): one container per (config, seed, held month) LOMO fit of the month-grouped residual corrector.
Same data and recipe as archive/agent_bigc_v2_02_lomo2.py mode r1x (192 features, target clip +-900, inner-month early stopping, lr .03, bagging .5).
Control and treatment both here: the LOMO control is the already-saved local R1/C2 predictions (identical recipe, identical data) -- the judge compares
Modal-built arms with each other and with the local C2 arm re-run as config C2 on Modal (include it in --cfgs for the pairing rule).

    $SYS modal/agent_bigc_app.py upload                                  # fold matrix + extras -> prc-stage1:/bigc/in (about 1.0 GB, once)
    $SYS -m modal run --detach modal/agent_bigc_app.py::launch --cfgs C2,C3,C3b,C4,C4b --seeds 0     # spawns cfgs x seeds x 8 months
    $SYS modal/agent_bigc_app.py status                                  # which fits are done
    $SYS modal/agent_bigc_app.py fetch                                   # assembles data/interim/agent_bigc_v2_r1x_s{seed}_{cfg}.npz (key = cfg)
    (local) .venv/Scripts/python.exe archive/agent_bigc_v2_03_judge.py r1x 0 C3 _C3

Cost: 8 vCPU container ~ $0.38/h; one fit 3-10 min -> ~ $0.04 per fit; 5 cfgs x 8 months x 1 seed = 40 fits ~ $1.5-2; x3 seeds ~ $5. Per-fit result is
committed to the volume the moment it finishes, so a preempted container loses at most one fit; re-launching skips finished fits.
"""
import os
import sys

import modal

image = (modal.Image.debian_slim(python_version="3.13").pip_install("numpy==2.5.2", "lightgbm==4.7.0", "polars==1.44.1")
         .add_local_file("archive/agent_bigc_te.py", "/repo/agent_bigc_te.py"))
vol = modal.Volume.from_name("prc-stage1", create_if_missing=True)
app = modal.App("prc-bigc")
IN, OUT = "/vol/bigc/in", "/vol/bigc/out"
CFG = {  # leaves, min_data, l2, max rounds, extra params
    "R1": (15, 500, 10.0, 600, {}), "C1": (63, 300, 20.0, 1200, {}), "C2": (127, 300, 50.0, 1200, {}),
    "C3": (255, 300, 50.0, 1500, {}), "C3b": (255, 150, 50.0, 1500, {}), "C4": (511, 300, 100.0, 1500, {}), "C4b": (511, 150, 100.0, 1500, {}),
    "C3h": (255, 300, 50.0, 1500, {"objective": "huber", "alpha": 150.0}), "C3f": (255, 300, 50.0, 1500, {"objective": "fair", "fair_c": 150.0}),
    "C4h": (511, 300, 100.0, 1500, {"objective": "huber", "alpha": 150.0}),
    "C4ff3": (511, 150, 100.0, 1500, {"feature_fraction": 0.3}), "C4ff7": (511, 150, 100.0, 1500, {"feature_fraction": 0.7}),
    "C3ff3": (255, 300, 50.0, 1500, {"feature_fraction": 0.3}), "C3ff7": (255, 300, 50.0, 1500, {"feature_fraction": 0.7}),
}
FILES = ("agent_bigc_serve_in.npz", "agent_bigc_maps.json", "agent_bigc_fold.npz", "agent_bigc_extra.npz", "agent_bigc_extra2.npz", "agent_bigc_keys.npz", "agent_bigc_keys2.npz", "agent_bigc_ltfm_fold.npz")
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
    z = np.load(f"{IN}/agent_bigc_fold.npz", allow_pickle=True); ex = np.load(f"{IN}/agent_bigc_extra.npz", allow_pickle=True)
    cfg0, *mods = cfg.split("+")   # modifiers after "+": D = fit y directly (pv is a feature), M = monotone-increasing in pv, P = one model per coverage class
    import re
    mv = re.match(r"^(.*?)V([0-9a-z]*)$", cfg0)   # versioned vfac input: C4ff7V -> agent_bigc_vfac_clean.npz, C4ff7V2a -> agent_bigc_vfac_clean2a.npz, C4ff7V3 -> ...clean3.npz
    if mv:
        suf, base_cfg, vtag = "V", mv.group(1), mv.group(2)
    else:
        suf = cfg0[-1] if cfg0[-1] in "XYWL" else ""; base_cfg = cfg0[:-1] if suf else cfg0; vtag = ""
    x3 = bool(suf); y3 = suf in ("Y", "W", "L", "V"); w3 = suf in ("W", "V"); pool = suf == "L"
    X = np.hstack([z["X"], ex["Xf"]]); catidx = [int(i) for i in z["cat"]]; y = z["y"]; pv = z["pv"]; mon = z["mon"]
    cols = list(z["cols"]) + list(ex["cols"]); apts = None
    if x3:
        sys.path.insert(0, "/repo"); sys.path.insert(0, "archive"); import agent_bigc_te as TE
        e2 = np.load(f"{IN}/agent_bigc_extra2.npz", allow_pickle=True); Xe2 = e2["Xf"]
        if y3:
            kk = np.load(f"{IN}/{'agent_bigc_keys2.npz' if (w3 or pool) else 'agent_bigc_keys.npz'}", allow_pickle=True); K = kk["Kf"][:, :11] if pool else kk["Kf"]
            apts = np.load(f"{IN}/agent_bigc_keys.npz", allow_pickle=True)["apt_f"]
        if pool:  # LTFM rows (v113-analogue REF, months 1,2,3,7,8,9,10), pooled with the nine airports; ADS-B / chain-variant inputs are NaN there
            zl = np.load(f"{IN}/agent_bigc_ltfm_fold.npz", allow_pickle=True); nL = len(zl["y"])
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
        if suf == "V":  # variant-factory difference inputs (agent_bigc_vfac_clean.npz = agent_vfac_extra.npz with out-of-band airports NaN-ed; NaN allowed)
            X = np.hstack([X, np.load(f"{IN}/agent_bigc_vfac_clean{vtag}.npz", allow_pickle=True)["Xf"]])
    nxt = [k for k in months if k != month]; val = min([k for k in nxt if k > month] or nxt)
    tr = (mon != month) & (mon != val); te = mon == month; va = mon == val
    if pool:
        tr &= ~snow; va &= ~snow   # LTFM snow rows belong to the v117 snow model; no training / early stopping on them
    nl, md, l2, nr, extra = CFG[base_cfg]
    P = dict(objective="l2", learning_rate=0.03, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.5, max_bin=63, num_threads=8, verbose=-1,
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
        cm = json.load(open(f"{IN}/agent_bigc_maps.json"))["category"]; cov = X[:, cols.index("cov_cat")]
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
def launch(cfgs: str = "C3", seeds: str = "0"):
    for cfg in cfgs.split(","):
        for s in seeds.split(","):
            for m in (1, 2, 3, 7, 8, 9, 10, 11):
                c = fit_one.spawn(cfg, int(s), m)
                print("spawned", cfg, s, m, c.object_id)


@app.function(image=image, volumes={"/vol": vol}, cpu=8.0, memory=28672, timeout=60 * 60 * 3, retries=1)
def serve_one(cfg: str, seed: int) -> dict:
    """SERVING fit: all fold months (leave-own-month-out key statistics), config's inputs, predicts the January/July 2026 rows. rounds = mean best_iter of the cfg's LOMO fits on the volume.
    Needs agent_bigc_serve_in.npz (built locally by archive/agent_bigc_v2_36_serve_prep.py: serve bulk/extra/d_NEW,d_ET columns, drift columns, month, pv). Writes /vol/bigc/serve/<cfg>_s<seed>.npz (pred, imp, names)."""
    import glob
    import re
    import time
    import numpy as np
    import lightgbm as lgb
    vol.reload()
    path = f"/vol/bigc/serve/{cfg}_s{seed}.npz"
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
    z = np.load(f"{IN}/agent_bigc_fold.npz", allow_pickle=True); ex = np.load(f"{IN}/agent_bigc_extra.npz", allow_pickle=True); e2 = np.load(f"{IN}/agent_bigc_extra2.npz", allow_pickle=True)
    si = np.load(f"{IN}/agent_bigc_serve_in.npz", allow_pickle=True)
    X = np.hstack([z["X"], ex["Xf"], e2["Xf"]]); cols = list(z["cols"]) + list(ex["cols"]) + ["d_NEW", "d_ET"]; Xs = si["Xs_pre"]
    assert list(map(str, si["cols_pre"])) == [str(c) for c in cols], "serve/fold column mismatch"
    y = z["y"]; pv = z["pv"]; mon = z["mon"]; catidx = [int(i) for i in z["cat"]]; months = sorted(set(mon.tolist()))
    r = np.clip(y - pv, -900, 900); names = list(map(str, cols))
    kf = TE.keys_from(X, cols); ks = TE.keys_from(Xs, cols)
    X = np.hstack([X, TE.te_loo(kf, r, mon, months)]); Xs = np.hstack([Xs, TE.te_serve(kf, r, ks)]); names += ["te_" + c for c in TE.KEYCOLS]
    if y3:
        kk = np.load(f"{IN}/{'agent_bigc_keys2.npz' if w3 else 'agent_bigc_keys.npz'}", allow_pickle=True); k0 = np.load(f"{IN}/agent_bigc_keys.npz", allow_pickle=True)
        yy = y.astype(float); ymu_apt = {a: yy[k0["apt_f"] == a].mean() for a in set(k0["apt_f"].tolist())}
        T2 = TE.Te2(kk["Kf"])
        F2f = T2.fold(r, yy, np.array([ymu_apt[a] for a in k0["apt_f"]]), mon, None, months)
        F2s = T2.serve(r, yy, mon, months, kk["Ks"], np.array([ymu_apt.get(a, yy.mean()) for a in k0["apt_s"]]))
        X = np.hstack([X, F2f]); Xs = np.hstack([Xs, F2s]); names += [str(n) + "_" + q for n in kk["names"] for q in TE.STATS]; del F2f, F2s
        if w3:
            X = np.hstack([X, kk["Df"]]); Xs = np.hstack([Xs, si["Dsv"]]); names += ["drift_%s_%s" % (a, b) for a in ("op_apt", "op_stand", "st_rwy", "st_hb") for b in ("pv", "span")]
        if suf == "V":
            vf = np.load(f"{IN}/agent_bigc_vfac_clean{vtag}.npz", allow_pickle=True); X = np.hstack([X, vf["Xf"]]); Xs = np.hstack([Xs, vf["Xs"]]); names += ["vf_" + str(c) for c in vf["cols"]]
    assert X.shape[1] == Xs.shape[1], (X.shape, Xs.shape)
    bis = [int(np.load(f)["bi"]) for f in glob.glob(f"{OUT}/{cfg}_s*_m*.npz")]; assert len(bis) >= 8, "no LOMO fits for this cfg on the volume"
    nr = int(round(float(np.mean(bis)))); nl, md, l2, _, extra = CFG[base_cfg]
    P = dict(objective="l2", learning_rate=0.03, bagging_fraction=0.5, bagging_freq=1, feature_fraction=0.5, max_bin=63, num_threads=8, verbose=-1, seed=seed, num_leaves=nl, min_data_in_leaf=md, lambda_l2=l2)
    P.update(extra)
    b = lgb.train(P, lgb.Dataset(X, r.astype(np.float32), categorical_feature=catidx, params=dict(max_bin=63, verbose=-1)), nr)
    pred = b.predict(Xs); os.makedirs("/vol/bigc/serve", exist_ok=True)
    np.savez(path + ".tmp.npz", pred=pred, imp=b.feature_importance("gain"), names=np.array(names), rounds=nr); os.replace(path + ".tmp.npz", path); vol.commit()
    return {"cfg": cfg, "seed": seed, "rounds": nr, "features": X.shape[1], "seconds": round(time.time() - t0)}


@app.local_entrypoint()
def launch_serve(cfgs: str = "C4ff7V3", seeds: str = "0,1,2"):
    for cfg in cfgs.split(","):
        for s in seeds.split(","):
            c = serve_one.spawn(cfg, int(s)); print("spawned serve", cfg, s, c.object_id)


def upload():
    only = sys.argv[2:] or FILES   # `upload agent_bigc_keys.npz` pushes just the new file
    with vol.batch_upload(force=True) as up:
        for f in only:
            up.put_file(f"data/interim/{f}", f"/bigc/in/{f}")
            print("uploaded", f)


def status():
    for e in vol.listdir("bigc/out"):
        print(os.path.basename(e.path))


def fetch():
    import numpy as np
    got = {}
    for e in vol.listdir("bigc/out"):
        n = os.path.basename(e.path)
        if not n.endswith(".npz") or ".tmp." in n:
            continue
        cfg, s, m = n[:-4].split("_")[0], int(n[:-4].split("_")[1][1:]), int(n[:-4].split("_")[2][1:])
        buf = b"".join(vol.read_file(e.path))
        import io
        d = np.load(io.BytesIO(buf)); got.setdefault((cfg, s), {})[m] = (d["rows"], d["pred"], int(d["bi"]))
    N = len(np.load("data/interim/agent_bigc_fold.npz", allow_pickle=True)["y"])
    for (cfg, s), ms in got.items():
        if len(ms) < 8:
            print("incomplete", cfg, s, sorted(ms)); continue
        arr = np.zeros(N)
        for m, (rows, pred, bi) in ms.items():
            arr[rows] = pred
        out = f"data/interim/agent_bigc_v2_r1x_s{s}_{cfg}.npz"
        np.savez(out, **{cfg: arr, cfg + "_bi": np.array([ms[m][2] for m in sorted(ms)])}); print("wrote", out)


if __name__ == "__main__":
    {"upload": upload, "status": status, "fetch": fetch}[sys.argv[1]]()
