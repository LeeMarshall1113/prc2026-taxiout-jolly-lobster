"""Stage 50: assembly of the final four-month file from the outputs of the stages 10 to 43, in the shipped order.

Every join takes the prediction of the served departures (array aligned to the departures in id order) and returns it. Each join
asserts that the row set is unchanged and prints, per airport and month, the rows it changed and the mean and rms change, so a
stage that does nothing on February or June rows is visible at once.

Order of the joins (the shipped order of docs/BUILD-TRACE.md section 1, with the data dependencies of final/README.md):

  generic months (default 2,6; built from the stage outputs)
   1 base           data/final/J2_v94model.parquet (final/10)
   2 pre-rules      Rome day-early bands and plan gate, before the ADS-B stage       (stage 21, J5_base.parquet; report only)
   3 adsb_stage     ADS-B off-block stage, stream 105                                (stage 21, J5_v105_s1.parquet; report only)
   4 restack        pooled re-stack, post-chain, splices, monster rule               (stage 22, J7_restack.parquet; replaces)
   5 blend          neighbour blend                                                  NOT PORTED (needs older lineages), no-op
   6 istanbul_rows  Istanbul bulk and snow refits                                    (stage 43, J3_ist.parquet; Istanbul rows)
   (chain value after join 6 is written by --stop-after-chain: it is the --base of stage 42)
   7 small          M1, M2, M3, M4 (stage 42, J8_small.parquet); nudge and meta stay zero
   8 correctors     plan-clock corrector (LIRF, LSZH) or round-three arithmetic corrector (other seven airports) at the shipped doses,
                    and the tail component at half dose on the nine airports (stages 41, 42)
   9 istanbul_corr  Istanbul corrector with the nine-airport column lists, non-snow rows (stage 43 term file)
  carried months (default 1,7; January and July)
   0 carried        stored lineage: submissions/lineage v152 file + Istanbul corrector delta (see CARRIED below). The base, ADS-B stage,
                    re-stack, blend, small stages, correctors, Rome mixtures and the rule layers of the post-chain are inside that file.
  all months
  10 rules          general row rules and the bounds applied last (final/30, stages rows,last)
  11 mixtures       Rome no-plan copy-or-real mixture (final/34), generic months only (the carried file already contains it)
  -- variants --
   A                10 + 11, then the plan-clock window last           (general model only)
   B                A with the published-daily layer inserted before the window   (--lam-cap 0.5 --own-bias, final/32)
   C                B with the named-row fixes of final/31 (January/July), before the window

Placeholders: with --placeholder a missing generic-month input is replaced by the fallback file F0 (base) or skipped (later joins),
the line says PLACEHOLDER, and every output is written under data/final/_ph_* and never under a final_* name.

    python final/50_assemble.py --months 1,2,6,7 --carried 1,7                 # A, B, C -> data/final/final_{A,B,C}.parquet
    python final/50_assemble.py --months 1,7 --carried 1,7                     # January/July only -> data/final/J11_jj_{A,B,C}.parquet
    python final/50_assemble.py --months 1,2,6,7 --carried 1,7 --placeholder --sample  # plumbing test on three days of each month

Memory: a few GB at the rule and daily steps (each runs as its own process); run under final/5x_run.sh.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fcommon import ID_COL, OUT, RAW, REPO, TARGET_COL, parse_months, template_ids  # noqa: E402

HERE = Path(__file__).resolve().parent
INTERIM = REPO / "data" / "interim"
C = pl.col
AIR9 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
# corrector doses per airport and the model each airport is served by (archive/agent_arith3_07_build.py: DOSE, DEFAULT)
DOSE = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25, "EHAM": 1.25, "LFPG": 0.75}
ARITH_AIRPORTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LFPG", "LEMD"]
PCD_AIRPORTS = ["LIRF", "LSZH"]
TAIL_DOSE = 0.5   # archive/agent_tailc_04_assemble.py: K

# Stored January/July lineage: which file each stage output comes from (see the report of experiment 5x).
CARRIED = [
    ("base, Rome pre-rules, ADS-B stage, re-stack, post-chain, neighbour blend, Istanbul bulk and snow refits, small correctors, tail, "
     "corrector terms, Rome mixtures, hidden-plan and Paris rules, six bounds, Rome hard-binding window", "data/interim/agent_bundle_v152_final.parquet (upload v152)"),
    ("Istanbul corrector with the nine-airport column lists (non-snow Istanbul rows)",
     "agent_istnext_final.parquet minus agent_bundle_v150_final.parquet, added to the v152 file (as archive/agent_bundle_v156_v157.py)"),
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------------
# frame helpers
# ------------------------------------------------------------------------------------------------------------------------------
class Ctx:
    """Served departures in id order with airport and month; `p` arrays always follow this order."""

    def __init__(self, ranking: Path, months: list[int], sample_days: list[int] | None) -> None:
        d = (pl.scan_parquet(ranking).filter(C("PHASE_mvt") == "DEP")
             .select(C(ID_COL).cast(pl.Float64), C("ADEP_mvt").alias("apt"), C("MVT_TIME_UTC_mvt").dt.month().alias("month"),
                     C("MVT_TIME_UTC_mvt").dt.day().alias("dom"))
             .filter(C("month").is_in(months)).collect().sort(ID_COL))
        if sample_days:
            d = d.filter(C("dom").is_in(sample_days))
        self.meta = d.select(ID_COL, "apt", "month")
        self.ids = self.meta[ID_COL].to_numpy()
        self.apt = self.meta["apt"].to_numpy()
        self.month = self.meta["month"].to_numpy()
        self.ranking = ranking
        self.tag = ""            # "" real, "_ph" placeholder run, "_sample" sample run
        self.placeholders: list[str] = []
        self.steps: list[dict] = []
        self.stats: list[pl.DataFrame] = []
        self.warnings: list[str] = []

    def months_mask(self, months: list[int]) -> np.ndarray:
        return np.isin(self.month, months)


def read_values(path: Path, ctx: Ctx, mask: np.ndarray | None = None, col: str | None = None) -> np.ndarray:
    """Values of a file keyed by the id column for the rows of the context (NaN where the file has no row, unless mask is given: then all rows
    under the mask must be present)."""
    df = pl.read_parquet(path)
    if ID_COL not in df.columns and "id" in df.columns:
        df = df.rename({"id": ID_COL})
    col = col or (TARGET_COL if TARGET_COL in df.columns else [c for c in df.columns if c != ID_COL][0])
    j = ctx.meta.select(ID_COL).join(df.select(C(ID_COL).cast(pl.Float64), C(col).cast(pl.Float64).alias("v")), on=ID_COL, how="left", maintain_order="left")
    v = j["v"].to_numpy().astype(float)
    if mask is not None and np.isnan(v[mask]).any():
        raise SystemExit(f"{path.name}: {int(np.isnan(v[mask]).sum())} of the {int(mask.sum()):,} rows needed have no value")
    return v


def write_values(path: Path, ctx: Ctx, p: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({ID_COL: ctx.ids, TARGET_COL: np.rint(p)}).write_parquet(path)


def write_template(path: Path, ctx: Ctx, p: np.ndarray, ranking: Path) -> None:
    """Variant file in the template's row order, id type and answer type (rows of the template not in the context are an error when the
    context covers all template ids; a partial run keeps its own id order)."""
    ids, dtype = template_ids(RAW / "final_submitting.parquet")
    got = pl.DataFrame({ID_COL: ctx.ids, TARGET_COL: np.rint(p)})
    if got.height == ids.height:
        out = ids.join(got.with_columns(C(ID_COL).cast(ids.schema[ID_COL])), on=ID_COL, how="left", maintain_order="left")
        if out[TARGET_COL].null_count():
            raise SystemExit("template ids without a prediction")
        out = out.with_columns(C(TARGET_COL).cast(dtype))
    else:
        out = got
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)


def report(ctx: Ctx, name: str, before: np.ndarray, after: np.ndarray, mask: np.ndarray | None = None, note: str = "") -> None:
    """Rows changed and mean / rms change per airport-month for the rows under `mask`."""
    m = np.ones(len(before), bool) if mask is None else mask
    d = (after - before)
    df = (pl.DataFrame(dict(apt=ctx.apt[m], month=ctx.month[m], d=d[m]))
          .group_by("apt", "month").agg(rows=pl.len(), changed=(C("d") != 0).sum(), mean=C("d").mean(), rms=(C("d") ** 2).mean().sqrt()).sort("apt", "month"))
    tot_ch = int((d[m] != 0).sum())
    ctx.stats.append(df.with_columns(pl.lit(name).alias('step')))
    print(f"\n-- join {name}: rows {int(m.sum()):,}, changed {tot_ch:,}, rms change {float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else 0:.2f} s {note}")
    if m.any():
        with pl.Config(tbl_rows=60, tbl_cols=8):
            print(df.with_columns(C("mean").round(2), C("rms").round(2)))
    ctx.steps.append(dict(step=name, rows=int(m.sum()), changed=tot_ch, rms=float(np.sqrt(np.mean(d[m] ** 2))) if m.any() else 0.0))
    if m.any() and tot_ch == 0 and "no-op" not in note and "report only" not in note:
        print(f"   WARNING: join {name} changed no row")


def need(ctx: Ctx, path: Path, how: str, placeholder: bool) -> bool:
    """True if the input exists. Missing: stop with the command, or (placeholder mode) note it and return False."""
    if path.exists():
        return True
    if placeholder:
        ctx.placeholders.append(path.name)
        ctx.tag = "_ph"
        print(f"   PLACEHOLDER: {path} does not exist ({how})")
        return False
    raise SystemExit(f"missing {path}; produce it with: {how}")


# ------------------------------------------------------------------------------------------------------------------------------
# generic-month joins
# ------------------------------------------------------------------------------------------------------------------------------
def join_base(ctx, p, g, a):
    f = OUT / "J2_v94model.parquet"
    if need(ctx, f, "bash final/10_base.sh (writes J2_v94model.parquet)", a.placeholder):
        new = read_values(f, ctx, g)
    else:
        f0 = OUT / "F0.parquet"
        log("base placeholder = F0 (airport-month mean of 2025 labels for February and June)")
        new = read_values(f0, ctx, g)
    p = p.copy()
    p[g] = new[g]
    return p


def join_report_only(ctx, p, g, a, label, path, how):
    """Rome pre-rules and the ADS-B stage are inside the re-stack file; their own outputs are compared with the base value for the report."""
    if not path.exists():
        print(f"\n-- {label}: {path.name} not present ({how}); not reported")
        return
    v = read_values(path, ctx)
    ok = g & ~np.isnan(v)
    report(ctx, label, p, np.where(ok, v, p), g, "(report only, replaced by the re-stack file in the next join)")


def join_restack(ctx, p, g, a):
    f = OUT / "J7_restack.parquet"
    if not need(ctx, f, "bash final/2x_run.sh 'why' final/22_restack.py --months 2,6", a.placeholder):
        return p
    new = read_values(f, ctx, g)
    q = p.copy()
    q[g] = new[g]
    report(ctx, "restack (Rome pre-rules + ADS-B stage + re-stack + post-chain)", p, q, g)
    return q


def join_blend(ctx, p, g, a):
    print("\n-- join blend: NOT PORTED (the neighbour blend needs the serve values of about ten older lineages for the served months); no-op. "
          "Size in the ablation: -0.087 s on the board scale (docs/PIPELINE.md row C6).")
    return p


def join_istanbul_rows(ctx, p, g, a):
    f = OUT / "J3_ist.parquet"
    if not need(ctx, f, "final/43_ist_build.py build ... --out data/final/J3_ist.parquet (bulk and snow refits first, see README)", a.placeholder):
        return p
    new = read_values(f, ctx, g)
    m = g & (ctx.apt == "LTFM")
    q = p.copy()
    q[m] = new[m]
    report(ctx, "istanbul_rows (bulk + snow refit rows)", p, q, m)
    return q


def join_small(ctx, p, g, a):
    f = OUT / "J8_small.parquet"
    if not need(ctx, f, "bash final/4x_run.sh 'why' final/42_small_stages.py --base <chain value file> --months 2,6", a.placeholder):
        return p
    new = read_values(f, ctx, col="value_small")
    ok = g & ~np.isnan(new)
    if ok.sum() != g.sum():
        raise SystemExit(f"J8_small.parquet covers {int(ok.sum()):,} of {int(g.sum()):,} generic-month rows")
    q = p.copy()
    q[ok] = np.rint(new[ok])
    report(ctx, "small (M1 + M2 + M3 + M4)", p, q, g)
    return q


def _serve_mean(pattern: str, ctx: Ctx, min_files: int, key: str = "pred") -> tuple[np.ndarray, int]:
    fs = sorted(f for f in glob.glob(pattern) if ".tmp" not in f and "_wire" not in f)
    if len(fs) < min_files:
        return np.full(len(ctx.ids), np.nan), len(fs)
    acc, n = None, 0
    for f in fs:
        z = np.load(f, allow_pickle=True)
        ids = z["ids"].astype(float)
        s = pl.DataFrame({ID_COL: ids, "v": z[key].astype(float)})
        v = ctx.meta.select(ID_COL).join(s, on=ID_COL, how="left", maintain_order="left")["v"].to_numpy()
        acc = v if acc is None else acc + v
        n += 1
    return acc / n, n


def join_correctors(ctx, p, g, a):
    """Corrector term at the shipped dose, then the tail at half dose.

    Shipped: archive/agent_arith3_07_build.py (value = rint(max(chain value + dose x round-three term, 0)) at the seven airports; LIRF and LSZH
    stay on the plan-clock term, archive/agent_bundle_v143.py) and archive/agent_tailc_04_assemble.py (value + 0.5 x ehat at the nine airports).
    The shipped build replaced an older corrector term inside the chain value; here the chain value carries none, so the term is added.
    """
    nf = a.seeds
    ar, na = _serve_mean(str(OUT / "J10_stage" / "out" / "arith" / "KYQSR-pw-s0-r0_serve_s*.npz"), ctx, nf)
    pc, npc = _serve_mean(str(OUT / "J10_stage" / "out" / "pcd" / "C4ff7V5_s*.npz"), ctx, nf)
    if na < nf or npc < nf:
        msg = f"corrector serve fits found: arith {na}, plan-clock {npc} (need {nf} each)"
        if not a.placeholder:
            raise SystemExit(msg + "; run final/41_corrector_serve.py arith / pcd (see README)")
        ctx.placeholders.append("corrector serve fits")
        ctx.tag = "_ph"
        print(f"   PLACEHOLDER: {msg}; corrector term not added")
        return p
    k = np.array([DOSE.get(x, 0.0) for x in ctx.apt])
    term = np.zeros(len(p))
    ma = g & np.isin(ctx.apt, ARITH_AIRPORTS)
    mp = g & np.isin(ctx.apt, PCD_AIRPORTS)
    term[ma] = k[ma] * ar[ma]
    term[mp] = k[mp] * pc[mp]
    if np.isnan(term[g]).any():
        raise SystemExit("corrector serve fits do not cover every nine-airport row of the generic months")
    q = p.copy()
    m = ma | mp
    q[m] = np.rint(np.maximum(p[m] + term[m], 0))
    report(ctx, "correctors (arith round three: 7 airports; plan-clock: LIRF, LSZH)", p, q, g)
    dz = OUT / "J8_deltas.npz"
    if dz.exists():
        z = np.load(dz, allow_pickle=True)
        e = ctx.meta.select(ID_COL).join(pl.DataFrame({ID_COL: z["ids"].astype(float), "e": z["ehat"].astype(float)}), on=ID_COL, how="left", maintain_order="left")["e"].fill_null(0).to_numpy()
        mt = g & np.isin(ctx.apt, AIR9)
        r = q.copy()
        r[mt] = np.rint(np.maximum(q[mt] + TAIL_DOSE * e[mt], 0))
        report(ctx, "tail component (half dose, nine airports)", q, r, mt, "(zero unless 42 was run with --v5-glob)")
        q = r
    return q


def join_istanbul_corrector(ctx, p, g, a):
    fs = sorted(glob.glob(str(OUT / "*ist*_term.npz")))
    if not fs and a.allow_missing_ist:
        msg = "WARNING: Istanbul corrector term (nine-airport column lists, non-snow rows) NOT applied: its serve fit has not been run"
        ctx.warnings.append(msg)
        print("\n-- join istanbul_corrector: " + msg)
        return p
    if not fs:
        if not a.placeholder:
            raise SystemExit("no Istanbul corrector term file data/final/*ist*_term.npz (final/ist_corrector_serve.py)")
        ctx.placeholders.append("Istanbul corrector term")
        ctx.tag = "_ph"
        print("   PLACEHOLDER: no Istanbul corrector term file; not added")
        return p
    z = np.load(fs[-1], allow_pickle=True)
    d = ctx.meta.select(ID_COL).join(pl.DataFrame({ID_COL: z["sid"].astype(float), "d": z["d"].astype(float)}), on=ID_COL, how="left", maintain_order="left")["d"].fill_null(0).to_numpy()
    m = g & (ctx.apt == "LTFM")
    q = p.copy()
    q[m] = np.rint(np.maximum(p[m] + d[m], 0))
    report(ctx, f"istanbul_corrector ({Path(fs[-1]).name})", p, q, m)
    return q


# ------------------------------------------------------------------------------------------------------------------------------
# carried months
# ------------------------------------------------------------------------------------------------------------------------------
def join_carried(ctx, p, c, a):
    if not c.any():
        return p
    rd = lambda n: pl.read_parquet(INTERIM / n)  # noqa: E731
    v152 = read_values(INTERIM / "agent_bundle_v152_final.parquet", ctx, c)
    v150 = read_values(INTERIM / "agent_bundle_v150_final.parquet", ctx, c)
    ist = read_values(INTERIM / "agent_istnext_final.parquet", ctx, c)
    q = p.copy()
    q[c] = v152[c] + (ist[c] - v150[c])
    print("\nCarried January/July stages and where each comes from:")
    for what, src in CARRIED:
        print(f"  - {what}\n      <- {src}")
    report(ctx, "carried (Istanbul corrector delta on the v152 file)", np.where(c, v152, p), q, c)
    return q


# ------------------------------------------------------------------------------------------------------------------------------
# sub-script joins (each its own process, file in / file out)
# ------------------------------------------------------------------------------------------------------------------------------
def run_script(script: str, args: list[str]) -> None:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage")
    cmd = [sys.executable, str(HERE / script)] + args
    log("run " + " ".join(Path(x).name if os.sep in x or "/" in x else x for x in cmd[1:]))
    r = subprocess.run(cmd, env=env, cwd=str(REPO))
    if r.returncode:
        raise SystemExit(f"{script} failed with exit {r.returncode}")


def via_script(ctx, p, name, script, args, tmp: Path, mask=None, note=""):
    src, dst = tmp.with_suffix(".in.parquet"), tmp.with_suffix(".out.parquet")
    write_values(src, ctx, p)
    if dst.exists():
        dst.unlink()
    run_script(script, ["--pred", str(src), "--ranking", str(ctx.ranking), "--out", str(dst)] + args)
    q = read_values(dst, ctx, np.ones(len(p), bool))
    if len(q) != len(p):
        raise SystemExit(f"{script} changed the row set")
    report(ctx, name, np.rint(p), q, mask, note)
    return q


def join_rules(ctx, p, a, tmp):
    return via_script(ctx, p, "rules (rows, last)", "30_general_rules.py",
                      ["--stage", "rows,last", "--jitter-allow", str(a.jitter_allow)], tmp / "J11_rules")


def join_mixtures(ctx, p, g, a, tmp):
    if not g.any():
        print("\n-- join mixtures: no generic month, skipped (the carried file contains the January/July mixtures)")
        return p
    months = sorted(set(int(x) for x in ctx.month[g]))
    return via_script(ctx, p, "mixtures (Rome no-plan copy-or-real, generic months)", "34_rome_mixtures.py",
                      ["--months", ",".join(map(str, months))], tmp / "J11_mix", g)


def join_last(ctx, p, a, tmp, label):
    return via_script(ctx, p, label, "30_general_rules.py", ["--stage", "last", "--jitter-allow", str(a.jitter_allow)], tmp / "J11_last")


def join_daily(ctx, p, a, tmp):
    args = ["--lam-cap", "0.5", "--own-bias", "--stop-days", a.stop_days]
    if a.cell_split:
        args += ["--cell-split"]
    return via_script(ctx, p, "published-daily layer (lam cap 0.5, own-month bias)", "32_published_daily.py", args, tmp / "J11_daily")


def join_fixes(ctx, p_before, p_daily, a, tmp):
    src0, src1, dst = tmp / "J11_fix0.parquet", tmp / "J11_fix1.parquet", tmp / "J11_fix.out.parquet"
    write_values(src0, ctx, p_before)
    write_values(src1, ctx, p_daily)
    if dst.exists():
        dst.unlink()
    run_script("31_scored_row_fixes.py", ["--before", str(src0), "--pred", str(src1), "--ranking", str(ctx.ranking), "--out", str(dst), "--with-row-fixes"])
    q = read_values(dst, ctx, np.ones(len(p_daily), bool))
    report(ctx, "named-row fixes (January/July flights only)", np.rint(p_daily), q)
    return q


# ------------------------------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ranking", type=Path, default=RAW / "final_ranking.parquet")
    ap.add_argument("--months", default="1,2,6,7", help="months of the output")
    ap.add_argument("--carried", default="1,7", help="months taken from the stored lineage (the others are built from the stage outputs)")
    ap.add_argument("--variants", default="A,B,C")
    ap.add_argument("--placeholder", action="store_true", help="missing generic-month inputs are replaced by placeholders; outputs go to _ph_*")
    ap.add_argument("--sample", action="store_true", help="plumbing: only the departures of days 10 to 12 of each month; outputs go to _sample_*")
    ap.add_argument("--seeds", type=int, default=12, help="serve fits per corrector that must exist")
    ap.add_argument("--jitter-allow", type=float, default=12.0)
    ap.add_argument("--stop-days", choices=["signature", "metar", "ltfm", "off"], default="signature", help="stop-day rule of the daily layer (final/32 --stop-days)")
    ap.add_argument("--cell-split", action="store_true", help="v162 cell split in the daily layer (needs a coverage census of the served months)")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--allow-missing-ist", action="store_true", help="assemble without the Istanbul corrector term (prints a warning, output keeps its real name)")
    ap.add_argument("--no-checks", action="store_true")
    ap.add_argument("--stop-after-chain", action="store_true",
                    help="write data/final/J11_chain.parquet (generic months after join 6: the value the small stages 42 and the corrector inputs 40 are served on) and stop")
    a = ap.parse_args()

    months, carried = parse_months(a.months), parse_months(a.carried)
    carried = [m for m in carried if m in months]
    ctx = Ctx(a.ranking, months, [10, 11, 12] if a.sample else None)
    if a.sample:
        ctx.tag = "_sample"
    c = ctx.months_mask(carried)
    g = ctx.months_mask(months) & ~c
    log(f"{len(ctx.ids):,} departures, months {months}, carried {carried}, generic {sorted(set(months) - set(carried))}")
    tmp = a.out_dir / "_t5x"
    tmp.mkdir(parents=True, exist_ok=True)

    p = np.zeros(len(ctx.ids))
    if g.any():
        p = join_base(ctx, p, g, a)
        base = p.copy()
        join_report_only(ctx, base, g, a, "pre-rules (Rome day-early bands and plan gate)", OUT / "J5_base.parquet", "final/21_adsb_stage.py")
        join_report_only(ctx, base, g, a, "ADS-B stage, stream 105", OUT / "J5_v105_s1.parquet", "final/21_adsb_stage.py")
        p = join_restack(ctx, p, g, a)
        p = join_blend(ctx, p, g, a)
        p = join_istanbul_rows(ctx, p, g, a)
        if a.stop_after_chain:
            dst = a.out_dir / "J11_chain.parquet"
            pl.DataFrame({ID_COL: ctx.ids[g], TARGET_COL: np.rint(p[g])}).write_parquet(dst)
            log(f"wrote {dst} ({int(g.sum()):,} rows): input of final/42_small_stages.py --base")
            return
        p = join_small(ctx, p, g, a)
        p = join_correctors(ctx, p, g, a)
        p = join_istanbul_corrector(ctx, p, g, a)
    p = join_carried(ctx, p, c, a)
    p = np.rint(p)
    assert np.isfinite(p).all() and len(p) == len(ctx.ids)

    p = join_rules(ctx, p, a, tmp)
    p = join_mixtures(ctx, p, g, a, tmp)
    pre = p.copy()
    write_values(a.out_dir / f"J11{ctx.tag}_pre_variants.parquet", ctx, pre)

    out = {}
    out["A"] = join_last(ctx, pre, a, tmp, "A: plan-clock window last")
    raw_b = join_daily(ctx, pre, a, tmp)
    out["B"] = join_last(ctx, raw_b, a, tmp, "B: plan-clock window last (after the daily layer)")
    out["C"] = join_last(ctx, join_fixes(ctx, pre, raw_b, a, tmp), a, tmp, "C: plan-clock window last (after the fixes)")

    if ctx.stats:
        pl.concat(ctx.stats).write_csv(a.out_dir / f"J11{ctx.tag}_join_stats.csv")
    for w in ctx.warnings:
        print(w)
    full = set(months) == {1, 2, 6, 7} and not a.sample
    for v in a.variants.split(","):
        q = out[v]
        if full:
            name = f"final_{v}.parquet" if not ctx.tag else f"_ph_{v}.parquet"
        elif a.sample:
            name = f"_sample_{v}.parquet"
        else:
            name = f"J11_jj_{v}.parquet" if not ctx.tag else f"_ph_part_{v}.parquet"
        dst = a.out_dir / name
        write_template(dst, ctx, q, a.ranking)
        log(f"wrote {dst}  ({len(q):,} rows)" + ("  [PLACEHOLDER INPUTS: " + ", ".join(sorted(set(ctx.placeholders))) + "]" if ctx.placeholders else ""))
        if full and not a.no_checks and not ctx.tag:
            subprocess.run([sys.executable, str(HERE / "check_file.py"), str(dst)], check=False)
            subprocess.run([sys.executable, str(HERE / "51_roundtrip_check.py"), str(dst)] + (["--pre", str(a.out_dir / "J11_pre_variants.parquet")] if v in ("B", "C") else []), check=False)
    for v in ("A", "B", "C"):
        print(f"variant {v}: rms vs A {float(np.sqrt(np.mean((out[v] - out['A']) ** 2))):.2f} s, mean vs A {float(np.mean(out[v] - out['A'])):+.3f} s")


if __name__ == "__main__":
    main()
