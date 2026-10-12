"""day-equation 2, step 1: ONE joint solve per airport-day (record types + level) from the published daily delay; design in
docs/LAB-NOTES.md section 2. Usage:
  python agent_dayeq2_01_joint.py APT USE MODE [BASE]
APT in LIRF LEBL LTFM EGLL EDDM EDDF; USE in rome_np rome_wp_i rome_wp_ii rome_wp_iii copy hold; MODE 2025 | 2026 (BASE = pre-equation base file).
2025: nested validation (parameters and the multiplier c from the other months), control = published shifted 7 days.
2026: row table data/interim/agent_dayeq2_rows_{APT}_{USE}.parquet (id, pred, new, cand, q, delta) for the build step."""
import itertools
import sys
import numpy as np
import polars as pl
sys.path.insert(0, "archive")
from agent_dayeq2_lib import *  # noqa

pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_tbl_rows(120)
APT, USE, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
BASE = sys.argv[4] if len(sys.argv) > 4 else None
SIG_F = 33.0 if APT == "LIRF" else 5.0
CGRID = [0.25, 0.5, 1.0, 2.0, 4.0]
MAXK_ENUM = 18
EMIN = 10.0
SNOW_DAYS_LTFM = [f"2025-02-{d:02d}" for d in (6, 7, 11, 19, 20, 21, 22, 23, 24)]

IFR_ONLY = APT in ("EHAM", "LSZH")  # item 4: the published set at Amsterdam and Zurich is the IFR departures (archive/agent_dayeq2_03_loose.py)
f = frame_2025([APT]) if MODE == "2025" else frame_2026([APT], BASE)
if IFR_ONLY:
    print("IFR-only frame:", f.height, "->", f.filter(C("rule") == "I").height); f = f.filter(C("rule") == "I")
pub = load_pub(2025 if MODE == "2025" else 2026, [APT])
SNOW_GATE = USE == "snow"
GATE_Z = 3.0


def snow_rows(fr):
    """refit definition: snow observation in [take-off - 3 h, take-off] (archive/agent_ltreg_feat.py), same function both years."""
    from pathlib import Path
    from agent_ltreg_feat import snow_features
    d = snow_features(fr.select(C("id").alias("MVT_ID_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), Path(R + "data/raw"))
    return d.select(C("MVT_ID_mvt").alias("id"), (C("lt_snow_3h") > 0).alias("snow"))


if APT == "LTFM":
    if SNOW_GATE:
        sn = snow_rows(f)
    elif MODE == "2025":
        sn = pl.read_parquet(I + "agent_eb136_tbl.parquet", columns=["id", "snow"]).with_columns((C("snow") > 0).alias("snow")).select("id", "snow")
    else:
        sn = pl.read_parquet(I + "agent_basefix_ltfm_snow26.parquet").select(C("MVT_ID_mvt").alias("id"), (C("lt_snow_3h") > 0).alias("snow"))
    f = f.join(sn, on="id", how="left").with_columns(C("snow").fill_null(False))
elif USE == "ehamw":
    from agent_eusnow_feat import snow_features_apts
    from pathlib import Path as _P
    _yr = 2025 if MODE == "2025" else 2026
    _dw = pl.read_parquet(I + f"agent_multipush_dwells_{_yr}.parquet").filter(C("apt") == APT).select(C("MVT_ID_mvt").alias("id"), "est_t")
    f = f.join(_dw, on="id", how="left").with_columns((C("T").dt.epoch("s").cast(pl.Float64) - C("est_t")).alias("W"), C("T").dt.hour().alias("hour"))
    _sn = snow_features_apts(f.select(C("id").alias("MVT_ID_mvt"), pl.lit(APT).alias("ADEP_mvt"), C("T").alias("MVT_TIME_UTC_mvt")), _P(R + "data/raw"), [APT]).select(C("MVT_ID_mvt").alias("id"), (C("lt_snow_3h") > 0).alias("snowrow"))
    f = f.join(_sn, on="id", how="left").with_columns(C("snowrow").fill_null(False))
    _sd = f.group_by("day_pred").agg(C("snowrow").any().alias("snow")); f = f.join(_sd, on="day_pred", how="left").with_columns(C("snow").fill_null(False))
else:
    f = f.with_columns(pl.lit(False).alias("snow"))
f = f.with_columns(C("day_pred").alias("day"), C("hasplan").alias("plan"), (C("gl") - 3606).alias("lo"), (C("gl") + 3606).alias("hi"))
f = f.with_columns(((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("pos"),
                   ((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"),
                   ((C("gs") >= 1800) & (C("pred") >= 0.5 * C("taxi") + 300) & (C("pred") <= C("gs"))).alias("hedged"))
enum_row = (C("gs") >= 5400) & (C("pred") >= 1800) & (C("pred") <= C("gs")) & ~C("fs").is_in(HEDGED)

# ---- candidate definition, type value t, ordinary value o, class for the prior
if USE.startswith("rome"):
    tval = pl.when(C("plan")).then(pl.min_horizontal(C("gs"), C("hi"))).otherwise(C("gs"))
    oval = pl.when(C("plan")).then(pl.max_horizontal(C("taxi"), C("lo"))).otherwise(C("taxi"))
    np_c = enum_row & ~C("plan")
    if USE == "rome_np":
        cand = np_c
    elif USE == "rome_wp_i":
        cand = enum_row
    elif USE == "rome_wp_ii":
        cand = np_c | (enum_row & C("plan") & C("copy_ok"))
    else:
        cand = np_c | (enum_row & C("plan") & C("copy_ok") & (C("pos") >= 0.1) & (C("pos") <= 0.9))
    f = f.with_columns(cand.alias("cand"), tval.alias("t"), oval.alias("o"), C("pos").clip(0.02, 0.98).alias("q0"), pl.lit("rome").alias("cls"))
    truth = ((C("y") - C("gs")).abs() <= 90)
elif USE == "copy":  # Barcelona schedule copies
    cand = C("plan") & (C("gs") >= 1800) & C("copy_ok") & ~C("fs").is_in(HEDGED)
    f = f.with_columns(cand.alias("cand"), pl.min_horizontal(C("gs"), C("hi")).alias("t"), C("pred").alias("o"), pl.lit("copy").alias("cls"))
    truth = ((C("y") - C("gs")).abs() <= 90)
elif USE in ("hold", "snow"):
    cls = pl.when(C("plan")).then(pl.lit("plan")).when(C("ades") == "LCEN").then(pl.lit("np_lcen")).otherwise(pl.lit("np_other"))
    cand = (C("gs") > 3600) & ~C("fs").is_in(HEDGED)
    f = f.with_columns(cand.alias("cand"), C("pred").alias("o"), cls.alias("cls"))
    truth = (C("y") > 3600)
elif USE == "ehamw":  # section 14: Amsterdam parked / follows-W rows
    cand = C("W").is_not_null() & (C("W") < 20000) & (C("pred") < C("W") + 258 - 300) & (C("gs") > 0) & ~C("fs").is_in(HEDGED)
    cls = pl.when(C("plan")).then(pl.lit("plan")).otherwise(pl.lit("noplan"))
    f = f.with_columns(cand.fill_null(False).alias("cand"), (C("W") + 258).alias("t"), cls.alias("cls"))
    truth = (C("y") >= C("W") + 258 - 600)
else:
    raise SystemExit("unknown USE")
if MODE == "2025":
    f = f.with_columns(truth.alias("truth"))
else:
    f = f.with_columns(pl.lit(None, pl.Boolean).alias("truth"))


def class_params(fo):
    """from a frame of OTHER months: prior rate per class (hold / copy) and h (hold: median stamp - schedule of hold rows) per plan state."""
    out = {}
    if USE in ("copy", "hold", "snow"):
        r = fo.filter(C("cand")).group_by("cls").agg(n=pl.len(), k=C("truth").sum())
        for row in r.iter_rows(named=True):
            out[("q0", row["cls"])] = float(np.clip((row["k"] + 1) / (row["n"] + 2), 0.002, 0.98))
    if USE in ("hold", "snow"):
        h = fo.filter(C("cand") & C("truth")).group_by("plan").agg(h=(C("gs") - C("y")).median())
        for row in h.iter_rows(named=True):
            out[("h", row["plan"])] = float(np.clip(row["h"], 0, 3600))
    if USE == "ehamw":
        pk = fo.filter(C("cand") & ~C("truth"))
        out[("parked_cell",)] = pk.group_by("plan", "hour").agg(C("y").median().alias("o_cell"), pl.len().alias("n_cell"))
        out[("parked_plan",)] = pk.group_by("plan").agg(C("y").median().alias("o_plan"))
        out[("rate",)] = float(fo.filter(C("cand"))["truth"].mean())
    return out


def apply_class_params(fr, cp):
    if USE in ("copy", "hold", "snow"):
        q0 = pl.lit(0.01)
        for (kind, k), v in cp.items():
            if kind == "q0":
                q0 = pl.when(C("cls") == k).then(pl.lit(v)).otherwise(q0)
        fr = fr.with_columns(q0.alias("q0"))
    if USE in ("hold", "snow"):
        h = pl.when(C("plan")).then(pl.lit(cp.get(("h", True), 900.0))).otherwise(pl.lit(cp.get(("h", False), 900.0)))
        t = C("gs") - h
        t = pl.when(C("plan")).then(t.clip(C("lo"), C("hi"))).otherwise(t)
        fr = fr.with_columns(t.alias("t"))
    if USE == "ehamw":
        fr = fr.drop([c for c in ("o_cell", "n_cell", "o_plan", "o") if c in fr.columns])
        fr = fr.join(cp[("parked_cell",)], on=["plan", "hour"], how="left").join(cp[("parked_plan",)], on="plan", how="left")
        fr = fr.with_columns(pl.when(C("n_cell") >= 10).then(C("o_cell")).otherwise(C("o_plan")).fill_null(C("pred")).alias("o"))
        fr = fr.with_columns(pl.when(C("t") > C("o") + 60).then(((C("pred") - C("o")) / (C("t") - C("o"))).clip(0.02, 0.98)).otherwise(0.02).alias("q0"),
                             (C("cand") & (C("t") > C("o") + 60)).alias("cand"))
    fr = fr.with_columns(((C("gs") - C("o")).clip(lower_bound=0) / 60).alias("a"), ((C("gs") - C("t")).clip(lower_bound=0) / 60).alias("b"))
    fr = fr.with_columns((C("a") - C("b")).fill_null(0.0).alias("e")).with_columns((C("cand") & (C("e") >= EMIN)).fill_null(False).alias("cand"))
    fr = fr.with_columns(pl.when(C("cand")).then(C("a")).otherwise(((C("gs") - C("pred")).clip(lower_bound=0) / 60)).alias("c0"),
                         pl.when(~C("cand") & C("hedged") & pl.lit(APT == "LIRF")).then(C("pos") * (1 - C("pos")) * ((C("gs") - C("taxi")) / 60) ** 2).otherwise(0.0).alias("vh"),
                         (~C("amb") & (C("gs") > C("pred")) & ~(C("snow") & pl.lit(APT == "LTFM")) & ~C("fs").is_in(HEDGED) & pl.lit(not SNOW_GATE)).fill_null(False).alias("shift"),
                         pl.when(C("cand")).then(C("q0") * C("e")).otherwise(0.0).alias("q0e"),
                         pl.when(C("cand")).then(C("q0") * (1 - C("q0")) * C("e") ** 2).otherwise(0.0).alias("vc"))
    return fr


def day_table(fr, pubcol, pub=pub):
    d = fr.group_by("day").agg(n=pl.len(), O0=C("c0").sum(), Vh=C("vh").sum(), Vc=C("vc").sum(), k=C("cand").sum(), pe=C("q0e").sum()).join(pub, on="day", how="inner")
    d = d.with_columns((C("n_pub") - C("n")).alias("dn")).filter((C("dn").abs() <= DN_MAX) & (C("n") >= MIN_N) & (C("n_pub") >= MIN_N)).with_columns(C(pubcol).alias("P")).filter(C("P").is_not_null())
    return d.with_columns((C("P") - C("O0") + C("pe")).alias("T0"), C("day").dt.month().alias("mon"))


NU = 3.0  # Student-t degrees of freedom of the day-level prior (heavy tails: large ordinary-level days exist)
MIN_N = 100  # a published day must describe a full day of ours
DMAX = 600.0  # level shift cap per flight (seconds); round one's largest applied shift was 119 s
STOP_P, STOP_Q = 0.02, 0.5  # Istanbul two-regime prior: a stop day has prior probability 0.02 and a hold rate 0.5 in every candidate class


def t_logpdf(z):
    return -0.5 * (NU + 1) * np.log1p(z * z / NU)


def posterior_types(T, e, q0, V):
    """marginals q_i of Z_i under prior Bernoulli(q0) and likelihood t_nu((T + sum Z e) / sqrt(V)); enumeration for k <= MAXK_ENUM,
    otherwise the self-consistent tilt (one common multiplier theta: q_i = q0 e^{theta e} / (1 - q0 + q0 e^{theta e}), theta chosen by the
    mean-field free energy on a grid). Returns (q, log evidence estimate)."""
    k = len(e)
    if k <= MAXK_ENUM:
        Z = np.array(list(itertools.product([0, 1], repeat=k)), dtype=np.float32); extra = Z @ e.astype(np.float32)
        w = (Z * np.log(q0, dtype=np.float32) + (1 - Z) * np.log(1 - q0, dtype=np.float32)).sum(axis=1) + t_logpdf((T + extra) / np.sqrt(V))
        mx = w.max(); w = np.exp(w - mx); ev = float(mx + np.log(w.sum())); w /= w.sum()
        return (w[:, None] * Z).sum(axis=0).astype(float), ev
    lo0 = np.log(q0 / (1 - q0))
    esc = max(float(np.median(e[e > 0])) if (e > 0).any() else 1.0, 1.0); grid = np.linspace(-30.0 / esc, 30.0 / esc, 1601)
    best = (-np.inf, 0.0)
    for th in grid:
        q = 1 / (1 + np.exp(-np.clip(lo0 + th * e, -60, 60))); qc = np.clip(q, 1e-9, 1 - 1e-9)
        M = float((q * e).sum()); Vs = V + float((q * (1 - q) * e ** 2).sum())
        elbo = float((qc * (np.log(q0) - np.log(qc)) + (1 - qc) * (np.log(1 - q0) - np.log(1 - qc))).sum()) + float(t_logpdf((T + M) / np.sqrt(Vs)))
        if elbo > best[0]:
            best = (elbo, th)
    th = best[1]
    return 1 / (1 + np.exp(-np.clip(lo0 + th * e, -60, 60))), best[0]


def posterior_regimes(T, e, q0, V):
    """Istanbul: two-regime prior (ordinary day: class rates q0; stop day: hold rate STOP_Q, prior STOP_P); the regimes are mixed by their evidence."""
    q_a, ev_a = posterior_types(T, e, q0, V)
    if APT != "LTFM":
        return q_a
    q_b, ev_b = posterior_types(T, e, np.full_like(q0, STOP_Q), V)
    la = np.log(1 - STOP_P) + ev_a; lb = np.log(STOP_P) + ev_b; m = max(la, lb)
    wb = np.exp(lb - m) / (np.exp(la - m) + np.exp(lb - m))
    return (1 - wb) * q_a + wb * q_b


def level_shrink(r, sL, sn):
    """posterior mean of the level x (minutes of day total) under prior t_nu(0, sL) and noise N(0, sn), given residual r = x + noise; returns x_hat / r."""
    if abs(r) < 1e-9:
        return 1.0
    x = np.linspace(-8 * max(sL, sn, 1.0) - abs(r), 8 * max(sL, sn, 1.0) + abs(r), 1201)
    lp = t_logpdf(x / sL) - 0.5 * ((r - x) / sn) ** 2; w = np.exp(lp - lp.max()); w /= w.sum()
    return float((w * x).sum() / r)


def gated_days(fm, bias, sigL2, O0_med, pub_df):
    """snow day AND T0 - bias < -GATE_Z sigma_day, on the REAL series. Returns dict day -> (T0 - bias, sigma)."""
    dr = day_table(fm, "pub", pub_df)
    sd = fm.group_by("day").agg(C("snow").any().alias("snowday"))
    dr = dr.join(sd, on="day", how="left").with_columns((C("T0") - bias).alias("Tb"), (SIG_F ** 2 + sigL2 * pl.max_horizontal(pl.lit(1.0), C("O0") / O0_med) ** 2).sqrt().alias("sig"))
    dr = dr.filter(C("snowday") & (C("Tb") < -GATE_Z * C("sig")))
    return {r["day"]: (r["Tb"], r["sig"]) for r in dr.iter_rows(named=True)}


def solve_day(g, P, bias, V_lvl, O0_med=None):
    """g: rows of one day (polars). Returns (new values array aligned to g, q array for candidates, delta_hat, lam).
    Two passes: types at delta = 0, level given the types, types again at the shifted ordinary values, level again."""
    cand = g["cand"].to_numpy(); gs = g["gs"].to_numpy(); pred = g["pred"].to_numpy(); shift = g["shift"].to_numpy()
    o = g["o"].to_numpy(); t = g["t"].to_numpy(); q0 = np.clip(g["q0"].to_numpy(), 0.002, 0.98)
    Vh = float(g["vh"].sum()); sn2 = SIG_F ** 2 + Vh
    O0 = float(np.clip(gs - pred, 0, None).sum() / 60.0)
    sL2 = V_lvl * (max(1.0, O0 / O0_med) ** 2 if O0_med else 1.0)
    delta = 0.0; q = np.zeros(cand.sum()); lam = 1.0
    for _ in range(2):
        osh = o + np.where(shift, delta, 0.0)  # ordinary values at the current level
        a = np.clip(gs - osh, 0, None) / 60.0; b = np.clip(gs - t, 0, None) / 60.0
        c_non = np.clip(gs - pred - np.where(shift, delta, 0.0), 0, None) / 60.0  # non-candidate contribution at the current level
        base = float(c_non[~cand].sum() + a[cand].sum())
        T = P - bias - base
        if cand.any():
            e = np.maximum(a[cand] - b[cand], 0.0)
            q = posterior_regimes(T, e, q0[cand], sn2 + sL2)
        else:
            e = np.zeros(0)
        # level: exact solve of the shifted sum to the remaining target, then shrink by the t-prior posterior mean
        wgt = np.ones(len(cand)); wgt[cand] = 1.0 - q
        fixed = float((b[cand] * q).sum() + a[cand & ~shift].dot(1 - q[~shift[cand]]) if cand.any() else 0.0) + float(c_non[~cand & ~shift].sum())
        tgt = P - bias - fixed
        S = shift
        if S.any():
            ov = np.where(cand, o, pred)[S]; w = wgt[S]
            def Ssum(d):
                return float((w * np.clip(gs[S] - ov - d, 0, None)).sum() / 60.0)
            lo_, hi_ = -DMAX, DMAX
            if Ssum(lo_) < tgt:
                dstar = lo_
            elif Ssum(hi_) > tgt:
                dstar = hi_
            else:
                for _i in range(40):
                    m = 0.5 * (lo_ + hi_)
                    if Ssum(m) > tgt:
                        lo_ = m
                    else:
                        hi_ = m
                dstar = 0.5 * (lo_ + hi_)
            r = Ssum(0.0) - tgt  # residual the level would explain (minutes)
            lam = level_shrink(r, np.sqrt(sL2), np.sqrt(sn2))
            delta = lam * dstar
        else:
            delta = 0.0
    new = pred.copy()
    new[shift & ~cand] = np.maximum(pred[shift & ~cand] + delta, 0)
    if cand.any():
        osh = o + np.where(shift, delta, 0.0)
        new[cand] = q * t[cand] + (1 - q) * np.maximum(osh[cand], 0)
    return new, q, delta, lam


def run_months(fr, pubcol, months_eval, cp_fn, verbose=False):
    """nested: for each month m in months_eval, parameters from the other months; returns per-row frame with new value for every c in CGRID."""
    outs = []
    for m in months_eval:
        fo = fr.filter(C("mon") != m); fm = fr.filter(C("mon") == m)
        cp = cp_fn(fo); fo = apply_class_params(fo, cp); fm = apply_class_params(fm, cp)
        do = day_table(fo, pubcol); dm = day_table(fm, pubcol)
        bias = float(do["T0"].median()); rsd = 1.4826 * float((do["T0"] - bias).abs().median())
        sigL2 = max(rsd ** 2 - SIG_F ** 2 - float(do["Vh"].median()) - float(do["Vc"].median()), 100.0)
        if verbose:
            print(f"  month {m}: bias {bias:+.0f} min, rsd {rsd:.0f}, sigma_L {np.sqrt(sigL2):.0f} min, days {dm.height}, cand rows {int(fm['cand'].sum())}, h/q0 {cp}")
        pmap = {r["day"]: r["P"] for r in dm.iter_rows(named=True)}; O0_med = float(do["O0"].median())
        if SNOW_GATE:
            if pubcol != "pub":  # control: gate and parameters from the REAL series, only the published value swapped
                dr = day_table(fo, "pub"); bias = float(dr["T0"].median()); rsd = 1.4826 * float((dr["T0"] - bias).abs().median())
                sigL2 = max(rsd ** 2 - SIG_F ** 2 - float(dr["Vh"].median()) - float(dr["Vc"].median()), 100.0); O0_med = float(dr["O0"].median())
            gd = gated_days(fm, bias, sigL2, O0_med, pub)
            pmap = {d: P for d, P in pmap.items() if d in gd}
            if verbose:
                print(f"    gated days: {[(str(d), round(v[0]), round(v[1])) for d, v in sorted(gd.items())]}")
        for c in CGRID:
            V_lvl = c * sigL2
            parts = []
            for d, g in fm.partition_by("day", as_dict=True, maintain_order=True).items():
                d = d[0] if isinstance(d, tuple) else d
                if d not in pmap:
                    continue
                new, q, dh, lam = solve_day(g, pmap[d], bias, V_lvl, O0_med)
                qq = np.full(g.height, np.nan); qq[g["cand"].to_numpy()] = q
                parts.append(g.select("id", "day", "mon", "fs", "plan", "cls", "gs", "pred", "y", "cand", "shift", "q0", "t", "o", "e", "truth", "snow").with_columns(
                    pl.Series("new", new), pl.Series("q", qq), pl.lit(dh).alias("delta"), pl.lit(lam).alias("lam"), pl.lit(c).alias("c"), pl.lit(pubcol).alias("pubcol")))
            if parts:
                outs.append(pl.concat(parts))
    return pl.concat(outs)


def score(t, label):
    t = t.with_columns(((C("pred") - C("y")) ** 2 - (C("new") - C("y")) ** 2).alias("g"))
    tot = t["g"].sum() / SE_SCALE
    byday = t.group_by("day").agg(C("g").sum()).sort("day"); gd = byday["g"].to_numpy()
    lo, hi, p = day_bootstrap(gd)
    top5 = np.sort(gd)[-5:].sum() / SE_SCALE
    gc = t.filter(C("cand"))["g"].sum() / SE_SCALE; gl_ = t.filter(C("shift"))["g"].sum() / SE_SCALE
    bym = t.group_by("mon").agg((C("g").sum() / SE_SCALE).round(3).alias("g")).sort("mon")
    mv = int((t["new"] - t["pred"]).abs().gt(1000).sum())
    print(f"{label}: gain {tot:+.3f} board-s (cand {gc:+.3f}, level {gl_:+.3f}); bootstrap [{lo:+.3f}, {hi:+.3f}] P>0 {p:.2f}; without top-5 days {tot - top5:+.3f} (top-5 share {top5 / tot if tot else 0:.2f}); "
          f"moved >1000 s: {mv}; by month {dict(zip(bym['mon'].to_list(), bym['g'].to_list()))}")
    return tot


if MODE == "2025":
    months = sorted(f["mon"].unique().to_list())
    print(f"{APT} {USE} 2025: rows {f.height}, months {months}, sigma_f {SIG_F}")
    res = {}
    for pc in ["pub", "pub7"]:
        print(f"\n=== series {pc}")
        t = run_months(f, pc, months, class_params, verbose=(pc == "pub"))
        res[pc] = t
    t = res["pub"]
    # nested choice of c: for held-out month m, c_m = argmax of the gain over the other months (every month's rows were solved with its own nested parameters)
    tg = t.with_columns(((C("pred") - C("y")) ** 2 - (C("new") - C("y")) ** 2).alias("g")).group_by("mon", "c").agg(C("g").sum()).to_pandas()
    chosen = {}
    for m in months:
        oth = tg[tg.mon != m].groupby("c").g.sum(); chosen[m] = float(oth.idxmax()) if len(oth) else 1.0  # no other-month rows (gated use): the design default c = 1
    print("\nnested c per held-out month:", chosen)
    print("gain by c (all months, for information, NOT the selection):", {c: round(float(tg[tg.c == c].g.sum() / SE_SCALE), 3) for c in CGRID})
    sel = pl.concat([t.filter((C("mon") == m) & (C("c") == c)) for m, c in chosen.items()])
    sel7 = pl.concat([res["pub7"].filter((C("mon") == m) & (C("c") == c)) for m, c in chosen.items()])
    print()
    score(sel, "REAL   ")
    score(sel7, "CONTROL (published shifted 7 days)")
    if USE.startswith("rome"):
        score(sel.filter(~C("plan") | ~C("cand")), "REAL no-plan candidates + level")
        if USE != "rome_np":
            score(sel.filter(C("plan") & C("cand")), "REAL with-plan candidates only")
            score(sel7.filter(C("plan") & C("cand")), "CONTROL with-plan candidates only")
            sel.filter(C("plan") & C("cand")).write_parquet(I + f"agent_dayeq2_val_{APT}_{USE}_wp.parquet")
    if USE == "ehamw":
        score(sel.filter(C("snow")), "REAL snow days")
        score(sel.filter(~C("snow")), "REAL ordinary days")
        score(sel.filter(C("cand")), "REAL candidate rows only")
        score(sel.filter(C("cand") & C("plan")), "REAL candidates with a plan")
        score(sel.filter(C("cand") & ~C("plan")), "REAL candidates without a plan")
        score(sel.filter(C("cand") & C("snow")), "REAL candidates on snow days")
        score(sel.filter(C("cand") & ~C("snow")), "REAL candidates on ordinary days")
        _c = sel.filter(C("cand")); _q0 = np.clip(_c["q0"].to_numpy(), 1e-4, 1 - 1e-4); _q = np.clip(_c["q"].to_numpy(), 1e-4, 1 - 1e-4); _tr = _c["truth"].cast(pl.Float64).to_numpy()
        for nm, v in (("prior", _q0), ("posterior", _q)):
            lo_b = v < 0.1; hi_b = v > 0.9
            ll = lambda m: float(-(_tr[m] * np.log(v[m]) + (1 - _tr[m]) * np.log(1 - v[m])).mean()) if m.any() else float("nan")
            print(f"  {nm}: extreme bins: q < 0.1: n {int(lo_b.sum())}, truth {float(_tr[lo_b].mean()) if lo_b.any() else float('nan'):.3f}, log-loss {ll(lo_b):.4f}; q > 0.9: n {int(hi_b.sum())}, truth {float(_tr[hi_b].mean()) if hi_b.any() else float('nan'):.3f}, log-loss {ll(hi_b):.4f}")
    if SNOW_GATE:
        gdays = sorted(sel["day"].unique().to_list())
        ev = []
        for d in gdays:
            if ev and (d - ev[-1][-1]).days <= 1:
                ev[-1].append(d)
            else:
                ev.append([d])
        print(f"gated days {len(gdays)}: {[str(d) for d in gdays]}; events {len(ev)}: {[(str(e[0]), str(e[-1])) for e in ev]}")
        gg = sel.with_columns(((C("pred") - C("y")) ** 2 - (C("new") - C("y")) ** 2).alias("g")); g7 = sel7.with_columns(((C("pred") - C("y")) ** 2 - (C("new") - C("y")) ** 2).alias("g"))
        tot = gg["g"].sum() / SE_SCALE
        for e in ev:
            ge = gg.filter(C("day").is_in(e))["g"].sum() / SE_SCALE; ce = g7.filter(C("day").is_in(e))["g"].sum() / SE_SCALE
            nb = gg.filter(C("day").is_in(e)).filter((C("new") - C("pred")).abs() > 1000).height
            print(f"  event {e[0]}..{e[-1]} ({len(e)} days): gain {ge:+.3f} (share {ge / tot if tot else 0:.2f}), control {ce:+.3f}, rows moved >1000 s {nb}")
        bd = gg.group_by("day").agg(C("g").sum()).sort("g", descending=True)
        print(f"  best day {bd['day'][0]} {bd['g'][0] / SE_SCALE:+.3f}; gain without it {(gg['g'].sum() - bd['g'][0]) / SE_SCALE:+.3f}; gated days outside February: {[str(d) for d in gdays if d.month != 2]}")
    if APT == "LTFM" and not SNOW_GATE:
        sd = pl.Series(SNOW_DAYS_LTFM).str.to_date()
        score(sel.filter(C("day").is_in(sd)), "REAL snow event days only")
        score(sel.filter(~C("day").is_in(sd)), "REAL without the snow event days")
        score(sel.filter(C("snow")) if "snow" in sel.columns else sel.filter(C("mon") == 2), "REAL February")
    # calibration of q on candidates
    cc = sel.filter(C("cand")).with_columns(pl.col("truth").cast(pl.Float64))
    if cc.height:
        qv = cc["q"].to_numpy(); tr = cc["truth"].to_numpy(); q0 = cc["q0"].to_numpy()
        bins = np.digitize(qv, [0.05, 0.2, 0.4, 0.6, 0.8])
        cal = [(round(float(qv[bins == b].mean()), 3), round(float(tr[bins == b].mean()), 3), int((bins == b).sum())) for b in range(6) if (bins == b).sum() > 0]
        ll0 = -(tr * np.log(np.clip(q0, 1e-4, 1 - 1e-4)) + (1 - tr) * np.log(np.clip(1 - q0, 1e-4, 1))).mean(); ll1 = -(tr * np.log(np.clip(qv, 1e-4, 1 - 1e-4)) + (1 - tr) * np.log(np.clip(1 - qv, 1e-4, 1))).mean()
        print(f"candidates {cc.height}, type rate {tr.mean():.3f}, log-loss prior {ll0:.4f} -> posterior {ll1:.4f}; calibration (q, truth, n): {cal}")
        print(cc.group_by("cls").agg(n=pl.len(), rate=C("truth").mean(), q0=C("q0").mean(), q=C("q").mean(), gain=(((C("pred") - C("y")) ** 2 - (C("new") - C("y")) ** 2).sum() / SE_SCALE)).sort("cls").with_columns(pl.col(pl.Float64).round(4)))
    if sel.filter(C("shift")).height:
        print("level: mean |applied shift| on shifted rows", float((sel.filter(C("shift"))["new"] - sel.filter(C("shift"))["pred"]).abs().mean()), "mean lam", float(sel["lam"].mean()))
    sel.write_parquet(I + f"agent_dayeq2_val_{APT}_{USE}.parquet")
    import json
    json.dump({"c": chosen}, open(I + f"agent_dayeq2_c_{APT}_{USE}.json", "w"))
else:
    # scored months: parameters from all 8 frame months of 2025; c = the median of the nested choices stored by the 2025 run
    import json
    chosen = json.load(open(I + f"agent_dayeq2_c_{APT}_{USE}.json"))["c"]; c = float(np.median(list(chosen.values())))
    f25 = frame_2025([APT])
    if IFR_ONLY:
        f25 = f25.filter(C("rule") == "I")
    if APT == "LTFM":
        f25 = f25.join(pl.read_parquet(I + "agent_eb136_tbl.parquet", columns=["id", "snow"]).with_columns((C("snow") > 0).alias("snow")).select("id", "snow"), on="id", how="left").with_columns(C("snow").fill_null(False))
    else:
        f25 = f25.with_columns(pl.lit(False).alias("snow"))
    f25 = f25.with_columns(C("day_pred").alias("day"), C("hasplan").alias("plan"), (C("gl") - 3606).alias("lo"), (C("gl") + 3606).alias("hi"),
                           ((C("pred") - C("taxi")) / (C("gs") - C("taxi"))).clip(0.0, 1.0).alias("pos"), ((C("gs") >= 5400) & (C("pred") >= 1800)).alias("amb"),
                           ((C("gs") >= 1800) & (C("pred") >= 0.5 * C("taxi") + 300) & (C("pred") <= C("gs"))).alias("hedged"))
    if USE == "ehamw":
        _dw25 = pl.read_parquet(I + "agent_multipush_dwells_2025.parquet").filter(C("apt") == APT).select(C("MVT_ID_mvt").alias("id"), "est_t")
        f25 = f25.join(_dw25, on="id", how="left").with_columns((C("T").dt.epoch("s").cast(pl.Float64) - C("est_t")).alias("W"), C("T").dt.hour().alias("hour"))
        f25 = f25.with_columns(cand.fill_null(False).alias("cand"), (C("W") + 258).alias("t"), cls.alias("cls"))
    elif USE.startswith("rome"):
        f25 = f25.with_columns(cand.alias("cand"), tval.alias("t"), oval.alias("o"), C("pos").clip(0.02, 0.98).alias("q0"), pl.lit("rome").alias("cls"))
    elif USE == "copy":
        f25 = f25.with_columns(cand.alias("cand"), pl.min_horizontal(C("gs"), C("hi")).alias("t"), C("pred").alias("o"), pl.lit("copy").alias("cls"))
    else:
        f25 = f25.with_columns(cand.alias("cand"), C("pred").alias("o"), cls.alias("cls"))
    f25 = f25.with_columns(truth.alias("truth"))
    pub25 = load_pub(2025, [APT])
    cp = class_params(f25); fo = apply_class_params(f25, cp); do = day_table(fo, "pub", pub25)
    bias = float(do["T0"].median()); rsd = 1.4826 * float((do["T0"] - bias).abs().median()); sigL2 = max(rsd ** 2 - SIG_F ** 2 - float(do["Vh"].median()) - float(do["Vc"].median()), 100.0)
    pub25 = load_pub(2025, [APT])
    print(f"{APT} {USE} 2026 on {BASE}: c {c}, bias {bias:+.0f}, sigma_L {np.sqrt(sigL2):.0f} min, class params {cp}")
    fm = apply_class_params(f, cp); dm = day_table(fm, "pub")
    print("2026 gated days", dm.height, "of", fm["day"].n_unique(), "; T0 median", float(dm["T0"].median()), "rsd", 1.4826 * float((dm["T0"] - dm["T0"].median()).abs().median()))
    pmap = {r["day"]: r["P"] for r in dm.iter_rows(named=True)}; O0_med = float(do["O0"].median())
    if SNOW_GATE:
        gd = gated_days(fm, bias, sigL2, O0_med, pub)
        pmap = {d: P for d, P in pmap.items() if d in gd}
        print("2026 gated days (date, T0 - bias min, sigma min, z):", [(str(d), round(v[0]), round(v[1]), round(v[0] / v[1], 1)) for d, v in sorted(gd.items())])
    parts = []
    for d, g in fm.partition_by("day", as_dict=True, maintain_order=True).items():
        d = d[0] if isinstance(d, tuple) else d
        if d not in pmap:
            continue
        new, q, dh, lam = solve_day(g, pmap[d], bias, c * sigL2, O0_med)
        qq = np.full(g.height, np.nan); qq[g["cand"].to_numpy()] = q
        parts.append(g.select("id", "day", "mon", "fs", "plan", "cls", "gs", "gl", "pred", "cand", "shift", "q0", "t", "o", "e").with_columns(pl.Series("new", new), pl.Series("q", qq), pl.lit(dh).alias("delta"), pl.lit(lam).alias("lam")))
        if USE == "ehamw" and g["cand"].any():
            _k = int(g["cand"].sum()); print(f"  {d}: candidates {_k} (plan {int((g['cand'] & g['plan']).sum())}), q mean {float(np.mean(q)):.2f}, level {dh:+.0f} s")
    t = pl.concat(parts).with_columns((C("new") - C("pred")).alias("move"))
    assert t.filter(C("fs").is_in(HEDGED) & (C("move") != 0)).height == 0
    print(f"rows {t.height}, moved {int((t['move'] != 0).sum())}, candidates {int(t['cand'].sum())}, |move| > 1000: {int((t['move'].abs() > 1000).sum())}, sum dp^2/scored row {float((t['move'] ** 2).sum()) / 344841:.1f}, mean move {float(t['move'].mean()):+.2f}")
    print("by month:", t.group_by("mon").agg(n=pl.len(), cand=C("cand").sum(), q_mean=C("q").mean(), delta_mean=C("delta").mean(), move_cand=C("move").filter(C("cand")).sum(), move_lvl=C("move").filter(C("shift")).sum()).sort("mon"))
    print("largest moves:"); print(t.sort(C("move").abs(), descending=True).head(25).select("day", "fs", "plan", "cls", "gs", "pred", "q0", "q", "t", "o", "new", "move", "delta").with_columns(pl.col(pl.Float64).round(1)))
    print("days with the largest total candidate movement:")
    print(t.filter(C("cand")).group_by("day").agg(k=pl.len(), q=C("q").mean(), move=C("move").sum(), delta=C("delta").first()).sort(C("move").abs(), descending=True).head(12).with_columns(pl.col(pl.Float64).round(1)))
    t.write_parquet(I + f"agent_dayeq2_rows_{APT}_{USE}.parquet")
