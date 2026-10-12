"""Score saved fold-member predictions for the "members" full-scale test.

Answers: does lightgbm_huber earn a place in the shipped blend, at full scale,
on held-out predictions -- rather than on the 3-member-blend-on-a-subsample
screen (archive/member_screen.py, logs/member_screen.log) that motivated this
test? That screen measured (3-member blend RMSE 241.84 on a subsample):
lightgbm_huber alone 252.91 against plain lightgbm's 265.56, error correlation
with the blend 0.855 against 0.901 for plain lightgbm, and adding it as a 4th
equal member -4.13s. Caveat, from this project's own record: scaled-down
screens here have overstated gains 3-7x (see
docs/LAB-NOTES.md's "scaled-down harness inflates" pattern), so
those numbers are upper bounds, not a forecast.

Reads one or more per-fold parquet files written by

    python -m prc.crossval ... --blend-lightgbm 0.70 --blend-cat-depth 10 \\
        --lgb-objective huber --lgb-huber-alpha 600 \\
        --save-fold-members results/members_fold.parquet

(columns: MVT_ID_mvt, ADEP_mvt, y, has_flight_plan, cat_bulk, cat_d3,
lightgbm, lightgbm_huber -- prc/crossval.py's run_fold, see the "BLEND
MEMBERS" block added 2026-09-22), and reports, per fold and pooled:

  * each member alone (RMSE, all / blind / covered, error-corr with the
    production blend)
  * the PRODUCTION blend: (cat_bulk + cat_d3 + lightgbm) / 3, equal thirds
    -- blend-weight refitting is CLOSED (docs/LAB-NOTES.md
    ~1565-1587), so every arm below is equal-weight, never a fitted blend
  * production PLUS lightgbm_huber as a 4th EQUAL member
  * lightgbm_huber REPLACING lightgbm (production stays 3 members)

split by BLIND airports (LTFM, LIRF, LFPG -- no ADS-B coverage) vs COVERED
(the other seven), with an expected board delta from the measured transfer
rates in docs/LAB-NOTES.md (lines 5138, 5374): a fold-RMSE
point at a blind airport is worth ~0.77s of board movement, ~0.31s at a
covered one. See `expected_board_delta` for the exact formula and its check
against the ledger's own worked example.

    .venv/Scripts/python.exe archive/agent_members_blend_eval.py \\
        results/members_fold_3_9.parquet results/members_fold_4_10.parquet

Self-test, no repo data touched (writes a synthetic tiny parquet per fold to
a scratch dir and scores those):

    .venv/Scripts/python.exe archive/agent_members_blend_eval.py --self-test

Never reads truthing.parquet, never touches the prc-2026-testsets bucket --
this script only reads files --save-fold-members already wrote locally.
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np
import polars as pl

BLIND = {"LTFM", "LIRF", "LFPG"}
BLIND_TRANSFER = 0.77
COVERED_TRANSFER = 0.31

# Order matters only for display. lightgbm_huber is last because it is the
# candidate, not a shipped member.
MEMBER_COLS = ["cat_bulk", "cat_d3", "lightgbm", "lightgbm_huber", "lightgbm_routed"]
PRODUCTION = ("cat_bulk", "cat_d3", "lightgbm")


def rmse(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return float(np.sqrt(np.mean((p - y) ** 2)))


def present_members(frame: pl.DataFrame) -> list[str]:
    """Member columns that exist AND are fully populated (no nulls from a
    diagonal concat of fold files that did not all save the same members)."""
    out = []
    for m in MEMBER_COLS:
        if m in frame.columns and int(frame[m].null_count()) == 0:
            out.append(m)
    return out


def arms_for(have: list[str]) -> dict[str, tuple[str, ...]]:
    """Which of the three composite arms can be built from what is present."""
    have_set = set(have)
    arms: dict[str, tuple[str, ...]] = {}
    if set(PRODUCTION) <= have_set:
        arms["production (cat_bulk+cat_d3+lgb, eq. thirds)"] = PRODUCTION
    if set(PRODUCTION) | {"lightgbm_huber"} <= have_set:
        arms["production + huber-lgb, 4th equal member"] = PRODUCTION + ("lightgbm_huber",)
    if {"cat_bulk", "cat_d3", "lightgbm_huber"} <= have_set:
        arms["huber-lgb REPLACES lgb, eq. thirds"] = ("cat_bulk", "cat_d3", "lightgbm_huber")
    # 2026-09-22: per-airport routed LightGBM (crossval --lgb-routes).
    if {"cat_bulk", "cat_d3", "lightgbm_routed"} <= have_set:
        arms["routed-lgb REPLACES lgb, eq. thirds"] = ("cat_bulk", "cat_d3", "lightgbm_routed")
    if {"cat_bulk", "cat_d3", "lightgbm_routed", "lightgbm_huber"} <= have_set:
        arms["routed-lgb REPLACES lgb + huber-lgb 4th"] = ("cat_bulk", "cat_d3", "lightgbm_routed", "lightgbm_huber")
    return arms


def blend(frame: pl.DataFrame, members: tuple[str, ...]) -> np.ndarray:
    stack = np.vstack([frame[m].to_numpy().astype(float) for m in members])
    return stack.mean(axis=0)


def expected_board_delta(y, pred_before, pred_after, blind_mask):
    """First-order estimate of a candidate's board delta from full held-out
    fold predictions for BOTH arms (not the linear-proxy LOFO instrument the
    idea ledger uses when a full retrain per candidate is not affordable --
    here it is, since --save-fold-members already saved real predictions).

    RMSE is not additive across a row split, but summed-squared-error is:
    SSE = SSE_blind + SSE_covered exactly, for any partition of the rows. A
    first-order (small-delta) expansion of RMSE = sqrt(SSE/n) around
    `pred_before` gives

        d(RMSE) ~= d(SSE) / (2 * n * RMSE_before)

    which is linear in SSE and so is separable the same way: the part of the
    whole-fold RMSE delta attributable to the blind rows, plus the part
    attributable to the covered rows, sums (to first order) to the actual
    pooled delta. That expresses each subset's contribution in the same
    "whole-fold seconds" units the board transfer rates are stated in.

    Derivation checked against docs/LAB-NOTES.md line 5374's
    worked example: "Blind LOFO ... pooled -0.044s ... Covered rows -0.028s
    ... Board: -0.034s blind-only; -0.042s everywhere (0.77 x blind + 0.31 x
    covered)". 0.77*(-0.044) = -0.0339 ~= -0.034; 0.31*(-0.028) = -0.00868;
    sum -0.0427 ~= -0.042 -- i.e. the ledger's "blind"/"covered" LOFO figures
    ARE whole-fold-equivalent contributions in exactly this sense, not raw
    within-subset RMSE (which would need no n or RMSE_before rescaling).

    Returns (board_delta, blind_contrib, covered_contrib, actual_pooled_delta).
    `actual_pooled_delta` is the real, non-approximated RMSE(after) -
    RMSE(before) -- printed alongside the first-order sum so a reader can see
    directly if the linear approximation is breaking down (it should not, for
    a member-swap-sized delta, but nothing here assumes that silently).
    """
    y = np.asarray(y, dtype=float)
    e_before = np.asarray(pred_before, dtype=float) - y
    e_after = np.asarray(pred_after, dtype=float) - y
    n = len(y)
    rmse_before = float(np.sqrt(np.mean(e_before ** 2)))
    if rmse_before == 0 or n == 0:
        return 0.0, 0.0, 0.0, 0.0
    d_sse_blind = float(np.sum(e_after[blind_mask] ** 2 - e_before[blind_mask] ** 2))
    d_sse_covered = float(np.sum(e_after[~blind_mask] ** 2 - e_before[~blind_mask] ** 2))
    denom = 2 * n * rmse_before
    blind_contrib = d_sse_blind / denom
    covered_contrib = d_sse_covered / denom
    board = BLIND_TRANSFER * blind_contrib + COVERED_TRANSFER * covered_contrib
    actual_pooled = rmse(y, pred_after) - rmse_before
    return board, blind_contrib, covered_contrib, actual_pooled


def _self_check_expected_board_delta() -> None:
    """Reproduces the ledger's own worked example algebraically (not from
    saved predictions, which the ledger entry does not include) as a unit
    check on the formula itself, independent of any parquet I/O. Raises if
    the formula's constants or arithmetic ever drift from that example."""
    blind_wholefold, covered_wholefold = -0.044, -0.028
    board = BLIND_TRANSFER * blind_wholefold + COVERED_TRANSFER * covered_wholefold
    assert abs(board - (-0.04256)) < 1e-9, board  # exact, from the two lines above
    # The ledger rounds its own inputs and output to ~2-3 significant figures
    # ("-0.034s blind-only", "-0.042s everywhere"), so the check against its
    # printed numbers uses a looser tolerance than the exact recomputation above.
    assert abs(BLIND_TRANSFER * blind_wholefold - (-0.034)) < 1e-3
    assert abs(board - (-0.042)) < 1e-3


def report_frame(frame: pl.DataFrame, label: str) -> None:
    if frame.height == 0:
        print(f"\n=== {label}: EMPTY, skipping ===")
        return
    y = frame["y"].to_numpy().astype(float)
    apt = frame["ADEP_mvt"].to_numpy()
    blind_mask = np.isin(apt, list(BLIND))
    n_blind, n_cov = int(blind_mask.sum()), int((~blind_mask).sum())
    print(f"\n=== {label}: n={len(y):,} (blind {n_blind:,}, covered {n_cov:,}) ===")

    have = present_members(frame)
    missing = [m for m in MEMBER_COLS if m not in have]
    if missing:
        print(f"  (not present / partially null, skipped: {missing})")

    arms = arms_for(have)
    prod_pred = blend(frame, PRODUCTION) if "production (cat_bulk+cat_d3+lgb, eq. thirds)" in arms else None
    prod_e = (prod_pred - y) if prod_pred is not None else None

    print(f"  {'member':34} {'all':>9} {'blind':>9} {'covered':>9} {'err-corr(prod)':>15}")
    for m in have:
        p = frame[m].to_numpy().astype(float)
        row_rmse = rmse(y, p)
        b_rmse = rmse(y[blind_mask], p[blind_mask]) if n_blind else float("nan")
        c_rmse = rmse(y[~blind_mask], p[~blind_mask]) if n_cov else float("nan")
        corr = float(np.corrcoef(p - y, prod_e)[0, 1]) if prod_e is not None else float("nan")
        print(f"  {m:34} {row_rmse:9.2f} {b_rmse:9.2f} {c_rmse:9.2f} {corr:15.4f}")

    if not arms:
        print("  (no composite arm buildable -- need at least cat_bulk, cat_d3, lightgbm)")
        return

    print(f"\n  {'arm':42} {'all':>9} {'blind':>9} {'covered':>9} {'vs production':>14}")
    prod_all = rmse(y, prod_pred) if prod_pred is not None else float("nan")
    for name, members in arms.items():
        p = blend(frame, members)
        row_rmse = rmse(y, p)
        b_rmse = rmse(y[blind_mask], p[blind_mask]) if n_blind else float("nan")
        c_rmse = rmse(y[~blind_mask], p[~blind_mask]) if n_cov else float("nan")
        delta = row_rmse - prod_all if prod_pred is not None else float("nan")
        print(f"  {name:42} {row_rmse:9.2f} {b_rmse:9.2f} {c_rmse:9.2f} {delta:+14.3f}")

    if prod_pred is None:
        return
    print("\n  expected board delta vs production (0.77x blind + 0.31x covered; see docstring):")
    for name, members in arms.items():
        if name.startswith("production ("):
            continue
        p = blend(frame, members)
        board, bc, cc, actual = expected_board_delta(y, prod_pred, p, blind_mask)
        print(f"    {name}")
        print(f"      board ~= {board:+.4f}s   (blind contrib {bc:+.4f}, covered contrib {cc:+.4f})"
              f"   actual pooled delta {actual:+.4f}s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*",
                    help="results/members_fold_*.parquet files from --save-fold-members")
    ap.add_argument("--self-test", action="store_true",
                    help="write a synthetic tiny parquet per fold to --out-dir and score those "
                         "instead of reading real results files")
    ap.add_argument("--out-dir", default="",
                    help="--self-test only: where to write the synthetic parquet files "
                         "(default: a fresh dir under the system temp directory)")
    args = ap.parse_args()

    _self_check_expected_board_delta()

    if args.self_test or not args.paths:
        if not args.self_test:
            print("no paths given -- running --self-test instead\n")
        _self_test(args.out_dir)
        return

    frames = []
    for p in args.paths:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"missing {path}")
        f = pl.read_parquet(path)
        frames.append((path.stem, f))
        report_frame(f, path.stem)

    if len(frames) > 1:
        pooled = pl.concat([f for _, f in frames], how="diagonal_relaxed")
        report_frame(pooled, "POOLED (" + " + ".join(n for n, _ in frames) + ")")


def _synthetic_members_frame(seed: int, n: int = 4000) -> pl.DataFrame:
    """A small frame with EXACTLY the schema --save-fold-members writes
    (MVT_ID_mvt, ADEP_mvt, y, has_flight_plan, cat_bulk, cat_d3, lightgbm,
    lightgbm_huber), built from synthetic numbers only -- no repo data is
    read. Not a real model: just member predictions correlated with a
    right-skewed "taxi time" target in a way that lets the scoring/printing
    code above be exercised end to end, with lightgbm_huber constructed to be
    a bit worse alone and noticeably decorrelated at the blind airports,
    matching the SHAPE (not the size) of what archive/member_screen.py found.
    """
    rng = np.random.default_rng(seed)
    airports = np.array(["LTFM", "LIRF", "LFPG", "EGLL", "EHAM", "EDDF", "LEMD"])
    apt = rng.choice(airports, size=n)
    y = rng.gamma(shape=3.0, scale=300.0, size=n)  # right-skewed, seconds-shaped

    def noise(scale):
        return rng.normal(0.0, scale, size=n)

    cat_bulk = y + noise(60)
    cat_d3 = y + noise(62) + 0.7 * (cat_bulk - y)     # correlated with cat_bulk (same family)
    lightgbm = y + noise(70) + 0.3 * (cat_bulk - y)   # some decorrelation (different library)
    is_blind = np.isin(apt, list(BLIND))
    # A bit worse alone everywhere, but its errors move independently of the
    # other members, especially at the blind airports (where the real screen
    # reported its own biggest gains).
    lightgbm_huber = y + noise(85) + np.where(is_blind, noise(20) - 15.0, noise(10))

    return pl.DataFrame({
        "MVT_ID_mvt": np.arange(n),
        "ADEP_mvt": apt,
        "y": y,
        "has_flight_plan": rng.integers(0, 2, size=n).astype(np.int8),
        "cat_bulk": cat_bulk,
        "cat_d3": cat_d3,
        "lightgbm": lightgbm,
        "lightgbm_huber": lightgbm_huber,
    })


def _self_test(out_dir: str) -> None:
    d = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="prc_members_eval_selftest_"))
    d.mkdir(parents=True, exist_ok=True)

    f3 = _synthetic_members_frame(seed=1)
    f4 = _synthetic_members_frame(seed=2)
    p3 = d / "members_fold_3_9.parquet"
    p4 = d / "members_fold_4_10.parquet"
    f3.write_parquet(p3)
    f4.write_parquet(p4)
    print(f"wrote synthetic members parquet -> {p3}\n"
          f"wrote synthetic members parquet -> {p4}\n")

    frames = [("members_fold_3_9", pl.read_parquet(p3)), ("members_fold_4_10", pl.read_parquet(p4))]
    for name, f in frames:
        report_frame(f, name + " (SYNTHETIC)")
    pooled = pl.concat([f for _, f in frames], how="diagonal_relaxed")
    report_frame(pooled, "SYNTHETIC POOLED")

    print("\nself-test complete: exercised member-alone scoring, all three blend "
          "arms, the blind/covered split, error correlation and "
          "expected_board_delta (plus its algebraic check against the ledger's "
          "own worked example) end to end, reading real parquet files back off "
          f"disk from {d}. No file under data/, results/ or submissions/ was read.")


if __name__ == "__main__":
    main()
