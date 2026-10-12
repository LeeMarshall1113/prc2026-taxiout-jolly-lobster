"""Shared helpers for the final-phase corrector stages (final/40 to 44). Nothing here reads a 2026 departure label.

Each function is a parameterised copy of code that ran for January and July 2026; the shipped script is named in the
comment above it. Fold-side files (2025) are read as stored and never rebuilt.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

os.environ.setdefault("POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR", "load_as_storage")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "archive"))
INTERIM = REPO / "data" / "interim"
OUT = REPO / "data" / "final"
RAW = REPO / "data" / "raw"
ID = "MVT_ID_mvt"
AIR9 = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LSZH"]
C = pl.col


def parse_months(text: str) -> list[int]:
    return [int(t) for t in text.split(",") if t.strip()]


def departures(ranking: Path, months: list[int], columns: list[str] | None = None) -> pl.DataFrame:
    """Departure rows of the chosen calendar months (UTC month of the take-off time), sorted by id.
    The month is the take-off month, as everywhere in final/."""
    cols = columns or [ID, "ADEP_mvt", "MVT_TIME_UTC_mvt"]
    need = list(dict.fromkeys(cols + [ID, "PHASE_mvt", "MVT_TIME_UTC_mvt"]))
    d = pl.read_parquet(ranking, columns=need).filter(C("PHASE_mvt") == "DEP")
    d = d.filter(C("MVT_TIME_UTC_mvt").dt.month().is_in(months)).sort(ID)
    assert d[ID].n_unique() == d.height, "duplicate departure ids"
    return d.select(list(dict.fromkeys(cols + ["MVT_TIME_UTC_mvt"])))


# ---- key hashes. Copy of the prep() in archive/agent_bigc_v2_20_keys2.py (same names, same hash).
KEY_INPUT_COLS = [ID, "CALLSIGN_flt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt",
                  "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt", "ADEP_mvt"]


def key_table(df: pl.DataFrame, with_fn: bool = False) -> pl.DataFrame:
    s = lambda c: C(c).cast(pl.Utf8).fill_null("NA")  # noqa: E731
    hb = (C("MVT_TIME_UTC_mvt").dt.hour() // 3).cast(pl.Utf8)
    A, op, st, rw, ty, ad = s("ADEP_mvt"), s("AIRCRAFT_OPERATOR_flt"), s("STAND_mvt"), s("RUNWAY_mvt"), s("AIRCRAFT_TYPE_mvt"), s("ADES_mvt")
    ex = {"fn": [A, s("CALLSIGN_flt")]} if with_fn else {}
    ex |= {"ades": [A, ad], "cty": [A, ad.str.slice(0, 2)], "op_apt": [A, op], "op_stand": [A, op, st], "op_hb": [A, op, hb],
          "op_rwy": [A, op, rw], "st_rwy": [A, st, rw], "st_hb": [A, st, hb], "ty_st": [A, ty, st], "mkt": [A, s("MARKET_SEGMENT_flt")],
          "wk": [A, s("WK_TBL_CAT_flt")], "op_ades": [A, op, ad], "op_ty": [A, op, ty], "ty_rwy": [A, ty, rw], "ades_hb": [A, ad, hb],
          "st_op_rwy": [A, st, op, rw]}
    return df.select([C(ID)] + [pl.concat_str(v, separator="|").hash(seed=7).floordiv(2).cast(pl.Int64).alias(k) for k, v in ex.items()])


# ---- AOBT_3 default flag. Copy of apply() in archive/agent_aobtflag_01_flag.py; modes are the stored 2025 modes (not refitted).
def aobt_flags(df: pl.DataFrame, modes25: pl.DataFrame, box: int = 35) -> pl.DataFrame:
    """df: id (float), apt, rwy, stand, rd (take-off minus AOBT_3, seconds, null when AOBT_3 is missing), aobt.
    Returns df with flag (0 no AOBT_3, 1 independent, 2 derived), K."""
    tgt = df.with_row_index("ri")
    fl = np.zeros(tgt.height, bool)
    K1 = np.full(tgt.height, np.nan)
    keys = ["sr", "stand", "rwy"]
    tgt = tgt.with_columns((C("stand") + "|" + C("rwy")).alias("sr"))
    for kc in keys[::-1]:   # coarse first, specific overrides
        m = modes25.filter(C("kc") == kc).select(["apt", C("k").alias(kc), "K"])
        j = tgt.select(["ri", "apt", kc, "rd"]).join(m, on=["apt", kc], how="left").sort("ri")
        K = j["K"].to_numpy().astype(float)
        rd = j["rd"].cast(pl.Float64).to_numpy()
        has = np.isfinite(K)
        fl |= has & (np.abs(rd - K) <= box)
        K1 = np.where(has, K, K1)
    flag = np.where(df["aobt"].is_null().to_numpy(), 0, np.where(fl, 2, 1)).astype(np.int8)
    return df.with_columns(pl.Series("flag", flag), pl.Series("K", K1))


def flag_inputs(ranking: Path, months: list[int]) -> pl.DataFrame:
    d = departures(ranking, months, [ID, "ADEP_mvt", "RUNWAY_mvt", "STAND_mvt", "AOBT_3_flt"])
    return d.select(C(ID).cast(pl.Float64).alias("id"), C("ADEP_mvt").cast(pl.Utf8).alias("apt"), C("RUNWAY_mvt").cast(pl.Utf8).alias("rwy"),
                    C("STAND_mvt").cast(pl.Utf8).alias("stand"),
                    (C("MVT_TIME_UTC_mvt").dt.epoch("s") - C("AOBT_3_flt").dt.epoch("s")).alias("rd"),
                    C("AOBT_3_flt").dt.epoch("s").alias("aobt"))


# ---- plan-clock distance columns. Copy of build() and gaps() in archive/agent_pcd_01_cols.py.
PCD_NAMES = ["d_SCHED", "d_LOBT", "d_IOBT", "d_EOBT", "d_AOBT3", "clamp", "wmin", "d_AOBT3_ind", "d_AOBT3_der"]


def pcd_gaps(X, cols):
    g = lambda n: X[:, cols.index(n)].astype(np.float64)  # noqa: E731
    return dict(sched=g("gap_sched"), lobt=g("gap_lobt"), iobt=g("gap_lobt") - g("clk_iobt_minus_lobt"), eobt=g("gap_eobt"), aobt=g("gap_aobt")), g("pv")


def pcd_build(pv, gap, flag):
    d = {k: pv - gap[k] for k in ("sched", "lobt", "iobt", "eobt", "aobt")}
    gs = gap["sched"]
    clamp = np.where((gs >= 5000) & (gs < 50000), gs - pv, np.nan)
    wmin = 3606.0 - np.abs(d["lobt"])
    ind = np.where(flag == 1, d["aobt"], np.nan)
    der = np.where(flag == 2, d["aobt"], np.nan)
    return np.column_stack([d["sched"], d["lobt"], d["iobt"], d["eobt"], d["aobt"], clamp, wmin, ind, der]).astype(np.float32)


def load_json(name: str):
    return json.load(open(INTERIM / name))


def id_index(ids_have: np.ndarray, ids_want: np.ndarray) -> np.ndarray:
    """position of each wanted id in ids_have, -1 where absent"""
    order = np.argsort(ids_have, kind="stable")
    pos = np.searchsorted(ids_have[order], ids_want)
    pos = np.clip(pos, 0, len(ids_have) - 1)
    hit = ids_have[order][pos] == ids_want
    return np.where(hit, order[pos], -1)
