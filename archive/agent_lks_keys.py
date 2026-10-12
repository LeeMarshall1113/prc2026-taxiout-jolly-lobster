"""LTFM key hashing shared by lks build/crossval/final (identical to agent_lks_01_build.prep; verified by agent_lks_03_keycheck.py)."""
import polars as pl
C = pl.col; ID = "MVT_ID_mvt"
RAWCOLS = [ID, "CALLSIGN_flt", "ADES_mvt", "AIRCRAFT_OPERATOR_flt", "STAND_mvt", "RUNWAY_mvt", "AIRCRAFT_TYPE_mvt", "MARKET_SEGMENT_flt", "WK_TBL_CAT_flt", "MVT_TIME_UTC_mvt"]
s = lambda c: C(c).cast(pl.Utf8).fill_null("NA")
def prep(df):
    cs = s("CALLSIGN_flt"); st = s("STAND_mvt"); hb = (C("MVT_TIME_UTC_mvt").dt.hour() // 3).cast(pl.Utf8)
    pre = cs.str.extract(r"^([A-Z]{3})", 1).fill_null("NA"); num = cs.str.extract(r"^[A-Z]{3}(\d+)$", 1)
    csfn = pl.when(num.is_not_null()).then(pre + pl.lit("#") + (num.cast(pl.Int64, strict=False) // 100).cast(pl.Utf8)).otherwise(pre + pl.lit("@") + cs.str.len_chars().cast(pl.Utf8))
    area = st.str.extract(r"^([A-Z]+)", 1).fill_null(st.str.slice(0, 1) + pl.lit("n") + st.str.len_chars().cast(pl.Utf8))
    pier = st.str.extract(r"^([A-Z]?\d)", 1).fill_null(st.str.slice(0, 2))
    op, ty, rw, ad = s("AIRCRAFT_OPERATOR_flt"), s("AIRCRAFT_TYPE_mvt"), s("RUNWAY_mvt"), s("ADES_mvt")
    ex = {"op": [op], "csp": [pre], "csfn": [csfn], "ades": [ad], "cty": [ad.str.slice(0, 2)], "reg": [ad.str.slice(0, 1)], "stand": [st], "area": [area], "pier": [pier],
          "type": [ty], "wk": [s("WK_TBL_CAT_flt")], "mkt": [s("MARKET_SEGMENT_flt")], "rwy_stand": [rw, st], "rwy_area": [rw, area], "op_hb": [op, hb], "ty_stand": [ty, st],
          "op_stand": [op, st], "op_rwy": [op, rw], "rwy_hb": [rw, hb], "op_ty": [op, ty]}
    return df.select([C(ID)] + [pl.concat_str(v, separator="|").hash(seed=7).floordiv(2).cast(pl.Int64).alias(k) for k, v in ex.items()]), list(ex)
