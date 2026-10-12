"""experiment lens 09: PRE-UPLOAD BOARD PREDICTOR for an ADS-B-experiment change (docs/LAB-NOTES.md).

F-class lens: board dMSE = sum over classes c (airport x census category x plan) of
    [fold dSSE_c / fold sum dp_c^2]  x  [2026 sum dp_c^2 from the two serving submissions] / (number of rows of the file)
i.e. the fold says how EFFICIENT each unit of prediction change is in each class, the 2026 serving build says how MUCH
change there is. Zero fitted parameters. Backtest (v97, v100, v103 in-sample; v104 out-of-sample) in the note.
Also prints L0 (plain fold) and L1 (2026 class-mix weights) for comparison.

    ./.venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- ./.venv/Scripts/python.exe archive/agent_lens_09_predict.py \
        --fold-ctrl data/interim/modal/NP1.npz --fold-treat data/interim/modal/TX.npz --fold-splice EDDF,EGLL,LEMD,LSZH,LFPG \
        --sub-ctrl submissions/jolly-lobster_v104.parquet --sub-treat submissions/jolly-lobster_v105.parquet --board-ctrl 234.5083

--fold-splice / --sub-splice A,B: take the treatment only at those airports (control elsewhere).
"""
import argparse

import numpy as np
import polars as pl

C = pl.col
I = "data/interim/"
N26 = 344_841
KEEP = {"matched", "points_no_liftoff", "no_adsb_points"}
ap = argparse.ArgumentParser()
ap.add_argument("--fold-ctrl", required=True)
ap.add_argument("--fold-treat", required=True)
ap.add_argument("--fold-splice", default="")
ap.add_argument("--sub-ctrl", required=True)
ap.add_argument("--sub-treat", required=True)
ap.add_argument("--sub-splice", default="")
ap.add_argument("--board-ctrl", type=float, required=True)
args = ap.parse_args()


def load(path, keys):
    z = np.load(path, allow_pickle=True)
    o = np.argsort(z["ids"], kind="stable")
    return {k: z[k][o] for k in keys}


def cat_of(df):
    return df.with_columns(pl.when(C("category").is_in(list(KEEP))).then(C("category")).otherwise(pl.lit("other")).alias("cat"))


c25 = cat_of(pl.read_parquet(I + "agent_coverage_census_2025.parquet")).sort("MVT_ID_mvt")
c26 = cat_of(pl.read_parquet(I + "agent_coverage_census_2026.parquet")).sort("MVT_ID_mvt")
cls25 = (c25["apt"] + "|" + c25["cat"] + "|" + c25["plan"].cast(pl.Utf8)).to_numpy()
cls26 = (c26["apt"] + "|" + c26["cat"] + "|" + c26["plan"].cast(pl.Utf8)).to_numpy()
id26, apt26 = c26["MVT_ID_mvt"].to_numpy(), c26["apt"].to_numpy()

a = load(args.fold_ctrl, ["ids", "y", "both_guard", "apts"])
b = load(args.fold_treat, ["ids", "both_guard"])
assert (a["ids"] == b["ids"]).all()
pt = b["both_guard"]
if args.fold_splice:
    pt = np.where(np.isin(a["apts"].astype(str), args.fold_splice.split(",")), pt, a["both_guard"])
pos = np.searchsorted(c25["MVT_ID_mvt"].to_numpy(), a["ids"])
assert (c25["MVT_ID_mvt"].to_numpy()[pos] == a["ids"]).all()
cls = cls25[pos]
y = a["y"]
dp = pt - a["both_guard"]
dsse = (pt - y) ** 2 - (a["both_guard"] - y) ** 2
nf = len(y)


def sub(path):
    s = pl.read_parquet(path).sort("MVT_ID_mvt")
    assert (s["MVT_ID_mvt"].cast(pl.Float64).to_numpy() == id26).all()
    return s["TAXITIME_SEC_mvt"].cast(pl.Float64).to_numpy()


dp26 = sub(args.sub_treat) - sub(args.sub_ctrl)
if args.sub_splice:
    dp26 = np.where(np.isin(apt26, args.sub_splice.split(",")), dp26, 0.0)
rc = args.board_ctrl
brd = lambda dm: float(np.sqrt(rc ** 2 + dm) - rc)  # noqa: E731

keys, inv = np.unique(cls, return_inverse=True)
k26, n26 = np.unique(cls26, return_counts=True)
P26 = dict(zip(k26, n26 / len(cls26)))
p25 = np.bincount(inv) / nf
w1 = np.array([P26.get(k, 0.0) / p for k, p in zip(keys, p25)])[inv]
L0 = float(dsse.mean())
L1 = float((w1 * dsse).sum() / w1.sum())
Fc, Fa, unexplained = 0.0, 0.0, 0.0
pooled = dsse.sum() / (dp ** 2).sum() if (dp ** 2).sum() > 0 else 0.0
for k in sorted(set(cls) | set(cls26)):
    D, S, D6 = float((dp[cls == k] ** 2).sum()), float(dsse[cls == k].sum()), float((dp26[cls26 == k] ** 2).sum())
    if D > 1e3:
        Fc += S / D * D6 / N26
    else:
        unexplained += D6 / N26
apts = a["apts"].astype(str)
for A in sorted(set(apt26)):
    m, m6 = apts == A, apt26 == A
    D, S, D6 = float((dp[m] ** 2).sum()), float(dsse[m].sum()), float((dp26[m6] ** 2).sum())
    Fa += (S / D if D > 1e3 else pooled) * D6 / N26
print(f"fold: dMSE {L0:+.1f} per row; sum dp^2/row {float((dp ** 2).mean()):.1f}   2026: sum dp^2/row {float((dp26 ** 2).mean()):.1f} "
      f"(rows moved {int((np.abs(dp26) > 0.5).sum()):,}); dp^2 in classes the fold never moved: {unexplained:.1f}/row")
print(f"PREDICTED board delta vs {rc}:  L0 {brd(L0):+.3f}   L1 {brd(L1):+.3f}   F-airport {brd(Fa):+.3f}   F-class {brd(Fc):+.3f}")
print("backtest of F-class (pred/realised): v97 1.21, v100 0.78, v103 0.56 (in-sample); v104 0.90 (predicted before reading its score)")
