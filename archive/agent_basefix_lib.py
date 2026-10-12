"""basefix lib: held-out chain frame (nine airports x 8 held months) with the audit's chain function and a day-bootstrap judge.
Everything is read from data/interim/agent_ablate_frame.parquet + agent_basefix_aux.parquet. Not run on its own."""
import numpy as np
import polars as pl

I = "data/interim/"
df = pl.read_parquet(I + "agent_ablate_frame.parquet").filter(pl.col("M1").is_not_nan() & pl.col("M1").is_not_null()).filter(pl.col("apt") != "LTFM")
aux = pl.read_parquet(I + "agent_basefix_aux.parquet")
df = df.join(aux, on="id", how="left", maintain_order="left")
N = df.height
col = lambda n: df[n].to_numpy().astype(float)  # noqa: E731
ids = col("id"); y = col("y"); p = col("p"); apt = df["apt"].to_numpy().astype(str); mon = col("mon").astype(int)
day = df["day"].to_numpy().astype(str); cov = df["cov"].to_numpy().astype(str); plan = df["plan"].to_numpy().astype(bool)
AIR = sorted(set(apt))
Z = lambda n: np.nan_to_num(col(n))  # noqa: E731
R = Z("r_full"); DN = Z("dnbr"); M1, M2, M3 = Z("M1"), Z("M2"), Z("M3"); SIL = Z("silent"); META = Z("meta")
F5 = Z("F5"); K5 = Z("k5"); KF = K5 * F5; TAIL = Z("tailB")
PVF = col("pv_b")
rule = df["rule"].to_numpy().astype(str)
mid = Z("eb_mid") > 0.5
P2 = Z("eb_P2"); gsr = Z("eb_gs_r"); ebpred = col("eb_pred")
W3 = np.where(gsr < 7200, 0.5, 0.85)
MIX = P2 * gsr + (1 - P2) * 1025.5
JJ = np.isin(mon, [1, 7]); JAN = mon == 1; JUL = mon == 7
OTHER = np.isin(mon, [2, 3, 8, 9, 10, 11])
ALL = np.ones(N, bool)


def chain(r=1.0, nb=1.0, m=(1.0, 1.0, 1.0), sil=1.0, meta=1.0, corr=1.0, tail=1.0, rome=True, rnd=True, extra=None, w3=None, romew=1.0):
    """the audit's chain: p -> rules -> neighbour -> M -> silent, meta -> corrector -> tail -> Rome rules. `extra` = additive column added with the corrector."""
    x = np.maximum(p + r * R, 0)
    x = np.maximum(x + nb * DN, 0)
    x = np.maximum(x + m[0] * M1 + m[1] * M2 + m[2] * M3, 0)
    x = x + sil * SIL + meta * META
    x = np.maximum(x + corr * KF + (0 if extra is None else extra), 0)
    x = np.maximum(x + 0.5 * tail * TAIL, 0)
    out = x.copy()
    if rome:
        ww = W3 if w3 is None else w3
        out[mid] = x[mid] + romew * ww[mid] * (MIX[mid] - x[mid])
        for rr in ("R1", "R2", "R3b"):
            s = rule == rr
            out[s] = ebpred[s]
    out = np.maximum(out, 0)
    return np.rint(out) if rnd else out


dayid = {d: i for i, d in enumerate(sorted(set(day[JJ])))}
did = np.array([dayid.get(d, -1) for d in day]); nd = len(dayid)
dmon = np.zeros(nd, int)
for d, i in dayid.items():
    dmon[i] = int(d[5:7])
rng = np.random.default_rng(7); B = 2000
CNT = np.zeros((B, nd))
for m_ in (1, 7):
    idx = np.where(dmon == m_)[0]
    draw = rng.integers(0, len(idx), size=(B, len(idx)))
    for b in range(B):
        np.add.at(CNT[b], idx[draw[b]], 1)


def per_day(e2):
    return np.bincount(did[JJ], weights=e2[JJ], minlength=nd), np.bincount(did[JJ], minlength=nd).astype(float)


def rmse(e2, m):
    return float(np.sqrt(e2[m].mean()))


SHIP = chain()
E2_SHIP = (SHIP - y) ** 2
S0, N0 = per_day(E2_SHIP)


def judge(pred, base_e2=None, mask=None):
    """variant minus reference (reference = shipped chain unless base_e2 given); negative = variant better. Returns dict Jan, Jul, JJ, all8, P*, top50, JJ_m50, per-airport."""
    b2 = E2_SHIP if base_e2 is None else base_e2
    e2 = (pred - y) ** 2
    d = {}
    for nm, m in (("Jan", JAN), ("Jul", JUL), ("JJ", JJ), ("all8", ALL)):
        d[nm] = rmse(e2, m) - rmse(b2, m)
    s, n = per_day(e2); sb, nb_ = per_day(b2)
    for nm, msk in (("Jan", dmon == 1), ("Jul", dmon == 7), ("JJ", np.ones(nd, bool))):
        c = CNT * msk[None, :] if nm != "JJ" else CNT
        dd = np.sqrt((c @ s) / (c @ n)) - np.sqrt((c @ sb) / (c @ nb_))
        d["P" + nm] = float((dd < 0).mean())
    g = (b2 - e2)[JJ]; tot = g.sum()
    d["top50"] = float(np.sort(g)[::-1][:50].sum() / tot) if tot > 0 else float("nan")
    keep = np.ones(JJ.sum(), bool); keep[np.argsort(-g)[:50]] = False
    d["JJ_m50"] = float(np.sqrt(e2[JJ][keep].mean()) - np.sqrt(b2[JJ][keep].mean()))
    d["worstapt"] = max((rmse(e2, JJ & (apt == a)) - rmse(b2, JJ & (apt == a))) for a in AIR)
    d["rms_move"] = float(np.sqrt(((pred - SHIP) ** 2).mean()))
    return d


def fmt(d):
    return (f"Jan {d['Jan']:+.3f} Jul {d['Jul']:+.3f} JJ {d['JJ']:+.3f} all8 {d['all8']:+.3f} | P {d['PJan']:.2f}/{d['PJul']:.2f}/{d['PJJ']:.2f} "
            f"| m50 {d['JJ_m50']:+.3f} top50 {d['top50']:.2f} worst-apt {d['worstapt']:+.3f} rms-move {d['rms_move']:.2f}")
