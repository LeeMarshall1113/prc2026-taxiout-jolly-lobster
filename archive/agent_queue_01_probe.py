"""FIRST CHECK for the explicit departure-queue model. Read-only, no slot, no training.

THE QUESTION. Every congestion feature we ship is a CORRELATIONAL summary of the take-off
sequence (trailing counts, rolling means, headways, per-runway rates). A structural model says
a runway is a server and taxi-out = travel to the queue + wait in it, so the quantity that
matters is QUEUE POSITION: how many aircraft the runway served between this flight joining the
queue and its own lift-off. MVT (wheels-up) is given for every scored row, so the take-off
SEQUENCE is fully observed in both years -- but a flight's queue position depends on when it
pushed back, which is the thing we predict.

ITERATION CHOICE, stated up front. One iteration, seeded from the production chained prediction
(q from yhat), PLUS the ORACLE queue computed from the true y. The oracle is the fixed point any
EM / iterate-to-convergence scheme could at best converge to, so it bounds the whole family: if
the oracle queue adds nothing to the chained residual, no iteration scheme can, and the
one-iteration/oracle ratio sizes what iterating would buy. That is strictly stronger than
iterating blind, and it costs one pass instead of five.

WHY THE CHAINED RESIDUAL IS THE RIGHT INSTRUMENT. base = the production chain (bagged ->
clip0 -> gap clamp -> LOBT box) applied to OUT-OF-FOLD predictions from a model holding all 73
shipped features. So y - base is already orthogonalised against everything we ship. A queue
construction that the model already holds through its proxies must leave no trace there. The
extra residualisation against dep_rwy_30min / dep_rwy_headway / yhat / airport guards the two
specific ways this project has fooled itself before: a feature that is really a nonlinear
transform of a shipped count, and a feature that is really just "the model predicted a long
taxi" (q scales with the window, and the window is yhat).

PRE-DECLARED PASS RULE, written before any number was looked at:
  P1  best construction reaches |partial rho| >= 0.05 against the chained residual on ORDINARY
      rows (y <= 3600s, 99.69 per cent of the file), after the controls, with the SAME SIGN on
      both gate-inclusive folds (1,7) and (2,8).
  P2  the implied ordinary-row RMSE reduction from a linear use of it is >= 1.0s pooled.
  P3  the PLACEBO construction (identical arithmetic, window length taken from another flight in
      the same airport/runway/month) must come in at less than half the real construction's
      partial rho. If placebo ~= real, the "signal" is window width, not queue position.
  FAIL P1                -> the line is DEAD. Report and stop. No slot.
  P1 only                -> real but too small. Report, no slot.
  P1+P2+P3               -> file a slot with an honest eta and run folds.
  The ORACLE row is diagnostic, not a pass condition: it is the ceiling, so oracle failing P1 is
  a harder kill than the one-iteration version failing it.

SIZING ARITHMETIC used for P2, from the plan: scored rows, 1.67e8 s^2 of squared error
per board second, bulk-model changes survive to the board at 0.38x. Ordinary rows hold 59.6 per
cent of squared error at RMSE 223. So an ordinary-row RMSE cut of d seconds is worth roughly
0.38 * d * 223 * 0.9969 / RMSE_pooled board seconds -- about 0.07 board seconds per 1.0s of
ordinary-row RMSE at the pooled fold RMSE. P2 at 1.0s is therefore a LOW bar deliberately: it is
"is there anything here at all", not "is it worth shipping".

    .venv/Scripts/python.exe archive/agent_queue_01_probe.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import polars as pl

C = pl.col
ID, W = "MVT_ID_mvt", 3606.0
MAX_HEADWAY_S = 6 * 3600  # features.py _MAX_HEADWAY_S
BURST_S = 300             # a "bank" of departures: consecutive lift-offs closer than 5 minutes

FOLDS = [("1_7", "results/preds_q_gate_1_7.parquet", False),
         ("2_8", "results/preds_q_gate_2_8.parquet", False),
         ("3_9", "results/preds_n3_bulk_3_9.parquet", True),
         ("4_10", "results/preds_s4_ctrl_4_10.parquet", True),
         ("5_11", "results/preds_c_ctrl_5_11.parquet", True)]
GATE_FOLDS = {"1_7", "2_8"}


def load_departures() -> pl.DataFrame:
    """Every 2025 departure with the columns the chain and the sequence need."""
    return (pl.scan_parquet("data/raw/training_*.parquet")
            .filter(C("PHASE_mvt") == "DEP")
            .select(C(ID),
                    C("ADEP_mvt").alias("adep"),
                    C("RUNWAY_mvt").cast(pl.Utf8).fill_null("?").alias("rwy"),
                    C("MVT_TIME_UTC_mvt").dt.epoch("s").alias("t"),
                    C("MVT_TIME_UTC_mvt").dt.strftime("%Y-%m").alias("ym"),
                    C("AOBT_3_flt").is_not_null().alias("plan"),
                    (C("MVT_TIME_UTC_mvt") - C("SCHED_TIME_UTC_mvt")).dt.total_seconds().alias("gs"),
                    (C("MVT_TIME_UTC_mvt") - C("LOBT_flt")).dt.total_seconds().alias("gl"))
            .collect())


def chain(bagged, gs, gl, plan):
    """production's chain, lifted verbatim from archive/screen_rule.py."""
    p = np.clip(bagged, 0, None)
    p = np.where(np.isfinite(gs) & (gs >= 5000) & (gs < 50000) & (p > gs), gs, p)
    box = plan & np.isfinite(gl)
    p = np.where(box, np.clip(p, np.where(box, gl - W, -np.inf), np.where(box, gl + W, np.inf)), p)
    return np.clip(p, 0, None)


def group_slices(keys: np.ndarray):
    """Yield (indices sorted by key) boundaries for each distinct key."""
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    if len(sk) == 0:
        return
    edges = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    edges = np.r_[edges, len(sk)]
    for a, b in zip(edges[:-1], edges[1:]):
        yield order[a:b]


def build_queue_features(t, key, yhat, ytrue, rng):
    """Every construction below reads ONLY the observed lift-off sequence t within key.

    q_pred    one iteration: departures served on my key while I was taxiing, window from yhat
    q_oracle  the ceiling: same, window from the true y -- the fixed point of any EM scheme
    q_placebo same arithmetic, window length borrowed from another flight in the same key
    srv_k     inverse view: how long the key took to serve its previous k departures
    burst_*   position within, and length of, the contiguous bank of lift-offs I sit in
    """
    n = len(t)
    out = {k: np.full(n, np.nan) for k in
           ("q_pred", "q_oracle", "q_placebo", "srv5", "srv10", "srv20",
            "burst_pos", "burst_len", "q_fixed30")}
    for idx in group_slices(key):
        o = idx[np.argsort(t[idx], kind="stable")]
        st = t[o].astype(np.float64)
        m = len(o)
        hi = np.searchsorted(st, st, side="left")          # rows strictly before me (ties excluded)
        for name, win in (("q_pred", yhat[o]), ("q_oracle", ytrue[o]),
                          ("q_placebo", yhat[o][rng.permutation(m)]),
                          ("q_fixed30", np.full(m, 1800.0))):
            w = np.where(np.isfinite(win), np.clip(win, 0, MAX_HEADWAY_S), 0.0)
            lo = np.searchsorted(st, st - w, side="right")
            out[name][o] = (hi - lo).astype(np.float64)
        for k in (5, 10, 20):
            v = np.full(m, np.nan)
            if m > k:
                d = st[k:] - st[:-k]
                d[d > MAX_HEADWAY_S] = np.nan
                v[k:] = d
            out[f"srv{k}"][o] = v
        # contiguous bank: break the sequence wherever the headway exceeds BURST_S
        gap = np.r_[np.inf, np.diff(st)]
        newbank = gap > BURST_S
        bank = np.cumsum(newbank)
        pos = np.arange(m) - np.maximum.accumulate(np.where(newbank, np.arange(m), 0)) + 1
        _, cnt = np.unique(bank, return_counts=True)
        out["burst_pos"][o] = pos.astype(np.float64)
        out["burst_len"][o] = cnt[bank - bank.min()].astype(np.float64)
    return out


def residualise(X: np.ndarray, v: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


def partial_rho(X, a, b):
    ra, rb = residualise(X, a), residualise(X, b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.mean(ra * rb) / (sa * sb))


def main():
    rng = np.random.default_rng(20260920)
    dep = load_departures()
    print(f"2025 departures: {dep.height:,}")

    t = dep["t"].to_numpy().astype(np.int64)
    adep = dep["adep"].to_numpy().astype(str)
    key_rwy = np.char.add(np.char.add(np.char.add(adep, "|"), dep["rwy"].to_numpy().astype(str)),
                          np.char.add("|", dep["ym"].to_numpy().astype(str)))
    key_apt = np.char.add(adep, np.char.add("|", dep["ym"].to_numpy().astype(str)))
    ids = dep[ID].to_numpy()
    gs = dep["gs"].fill_null(np.nan).to_numpy().astype(float)
    gl = dep["gl"].fill_null(np.nan).to_numpy().astype(float)
    plan = dep["plan"].fill_null(False).to_numpy()

    # shipped controls, recomputed exactly as features.py does (month-scoped keys, 6h headway cap)
    ctrl30 = np.zeros(len(t))
    headway = np.full(len(t), np.nan)
    for idx in group_slices(key_rwy):
        o = idx[np.argsort(t[idx], kind="stable")]
        st = t[o].astype(np.float64)
        ctrl30[o] = (np.searchsorted(st, st + 900, "right") - np.searchsorted(st, st - 900, "left") - 1)
        h = np.r_[np.nan, np.diff(st)]
        h[h > MAX_HEADWAY_S] = np.nan
        headway[o] = h

    rows = {}
    for label, path, drop_lirf in FOLDS:
        P = pl.read_parquet(path).select(ID, "y", "bagged")
        rows[label] = (P[ID].to_numpy(), P["y"].to_numpy().astype(float),
                       P["bagged"].to_numpy().astype(float), drop_lirf)

    pos = {m: i for i, m in enumerate(ids)}
    results = {}
    for label, (mid, y, bag, drop_lirf) in rows.items():
        loc = np.array([pos[m] for m in mid])
        base = chain(bag, gs[loc], gl[loc], plan[loc])
        if drop_lirf:
            keep = adep[loc] != "LIRF"
            loc, y, base = loc[keep], y[keep], base[keep]
        results[label] = (loc, y, base)
        print(f"  fold {label:5}: {len(loc):,} rows, chained RMSE {np.sqrt(((y - base) ** 2).mean()):.3f}")

    # yhat / ytrue defined on the whole 2025 departure frame: fold rows get their own values,
    # everything else gets the population median so the window arithmetic stays defined. Only
    # the WINDOW of a fold row matters; the SOURCE events are lift-off times, always observed.
    print("\nbuilding queue constructions from the observed take-off sequence...")
    feats = {}
    for scope, key in (("rwy", key_rwy), ("apt", key_apt)):
        yhat_all = np.full(len(t), np.nan)
        ytrue_all = np.full(len(t), np.nan)
        for label, (loc, y, base) in results.items():
            yhat_all[loc] = base
            ytrue_all[loc] = y
        med = np.nanmedian(yhat_all)
        f = build_queue_features(t, key, np.where(np.isfinite(yhat_all), yhat_all, med),
                                 np.where(np.isfinite(ytrue_all), ytrue_all, med), rng)
        for k, v in f.items():
            feats[f"{k}_{scope}"] = v

    names = [k for k in feats if k.endswith("_rwy")] + [k for k in feats if k.endswith("_apt")]

    print("\n" + "=" * 108)
    print("PARTIAL CORRELATION WITH THE CHAINED RESIDUAL, ordinary rows (y <= 3600s)")
    print("  raw   = corr(feature, y - base)          [the trap: this project has seen +0.27..+0.38 mean nothing]")
    print("  part  = after controls: dep_rwy_30min, dep_rwy_headway(+missing flag), yhat, yhat^2, airport one-hots")
    print("  dRMSE = ordinary-row RMSE change a single linear term would buy, pooled over the folds")
    print("=" * 108)
    hdr = f"{'construction':<16}" + "".join(f"{l:>17}" for l, _, _ in FOLDS) + f"{'pooled dRMSE':>14}"
    print(hdr)
    print(f"{'':<16}" + "".join(f"{'raw / partial':>17}" for _ in FOLDS))

    summary = {}
    for name in names:
        cells, num, den = [], 0.0, 0.0
        for label, _, _ in FOLDS:
            loc, y, base = results[label]
            v = feats[name][loc]
            r = y - base
            ord_ = (y <= 3600) & np.isfinite(v)
            if ord_.sum() < 1000:
                cells.append(f"{'--':>17}")
                continue
            vv, rr = v[ord_], r[ord_]
            apt = adep[loc][ord_]
            h = headway[loc][ord_]
            X = [np.ones(len(vv)), ctrl30[loc][ord_], np.nan_to_num(h, nan=0.0),
                 (~np.isfinite(h)).astype(float), base[ord_], base[ord_] ** 2 / 1e3]
            for a in np.unique(apt)[1:]:
                X.append((apt == a).astype(float))
            X = np.column_stack(X)
            raw = float(np.corrcoef(vv, rr)[0, 1])
            pr = partial_rho(X, vv, rr)
            cells.append(f"{raw:+.3f} / {pr:+.3f}".rjust(17))
            # linear-probe RMSE gain on these rows: var explained by the residualised feature
            rv = residualise(X, vv)
            rr2 = residualise(X, rr)
            if rv.std() > 1e-12:
                b = float(np.dot(rv, rr2) / np.dot(rv, rv))
                sse0 = float(np.dot(rr, rr))
                sse1 = sse0 - b ** 2 * float(np.dot(rv, rv))
                num += sse0 - sse1
                den += len(vv)
                summary.setdefault(name, []).append((label, pr))
        # pooled ordinary-row RMSE change
        tot_sse = sum(float(((results[l][1][results[l][1] <= 3600]
                              - results[l][2][results[l][1] <= 3600]) ** 2).sum()) for l, _, _ in FOLDS)
        tot_n = sum(int((results[l][1] <= 3600).sum()) for l, _, _ in FOLDS)
        d = np.sqrt(max(tot_sse - num, 0.0) / tot_n) - np.sqrt(tot_sse / tot_n)
        print(f"{name:<16}" + "".join(cells) + f"{d:>+14.3f}")
        summary[name] = (summary.get(name, []), d)

    print("\n" + "=" * 108)
    print("PRE-DECLARED RULE")
    best = None
    for name in names:
        if name.startswith(("q_oracle", "q_placebo", "q_fixed30")):
            continue
        prs = dict(summary[name][0])
        if "1_7" not in prs or "2_8" not in prs:
            continue
        same = np.sign(prs["1_7"]) == np.sign(prs["2_8"])
        mag = min(abs(prs["1_7"]), abs(prs["2_8"]))
        if same and (best is None or mag > best[1]):
            best = (name, mag, summary[name][1])
    if best is None:
        print("  P1 FAIL: no construction has the same sign on both gate-inclusive folds.")
    else:
        name, mag, d = best
        print(f"  best construction: {name}   min|partial rho| over (1,7),(2,8) = {mag:.4f}   dRMSE {d:+.3f}s")
        print(f"  P1  |partial rho| >= 0.05, same sign  : {'PASS' if mag >= 0.05 else 'FAIL'}")
        print(f"  P2  ordinary-row dRMSE <= -1.0s       : {'PASS' if d <= -1.0 else 'FAIL'}")
        pl_ = dict(summary.get(name.replace('q_pred', 'q_placebo'), ([], 0))[0])
        if pl_:
            pm = min(abs(pl_.get("1_7", 0)), abs(pl_.get("2_8", 0)))
            print(f"  P3  placebo < half the real rho       : "
                  f"{'PASS' if pm < 0.5 * mag else 'FAIL'}  (placebo {pm:.4f} vs real {mag:.4f})")
    print("=" * 108)


if __name__ == "__main__":
    main()
