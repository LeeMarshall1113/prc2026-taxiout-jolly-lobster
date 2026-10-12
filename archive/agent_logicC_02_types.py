"""logicC 02: classify 2025 large (>3600 s) and small (<60 s) labels at nine airports by record type, from input columns.
Types for large labels, first match wins:
 DAY    : y in [82800, 95000] (a stamp one calendar day early)
 DUP    : another record, same airport + flight string, take-off within 1800 s
 CSCHED : gap=(mvt-sched)>=1800 and |y-gap|<=90 (off-block equals schedule)
 CLOBT  : plan row, |y-gl|<=90 (off-block equals last plan clock)  [gl = mvt-LOBT]
 CEOBT  : |y-ge|<=90 ; CIOBT: |y-gi|<=90
 REAL   : plan row and |y-ga|<=900 (the network manager's own off-block agrees: a real wait after off-block)
 GA     : flight type G/M/X or segment Military/Business/Other/Non-Scheduled or rule V   (not otherwise explained)
 OTHER
Small labels: HELI (runway H or heli-type), LATE (plan row and ga-y>=300: off-block recorded after the plan's off-block),
 NEG (y<=0), GA, OTHER.
"""
import os, sys
import polars as pl
C = pl.col
pl.Config.set_tbl_rows(300); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(250)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pl.read_parquet(os.path.join(ROOT, ".local", "logic_rows", "C_dep2025.parquet"))
d = d.sort(["apt", "flight", "mvt"]).with_columns(
    (C("mvt") - C("mvt").shift(1).over(["apt", "flight"])).dt.total_seconds().abs().alias("dprev"),
    (C("mvt").shift(-1).over(["apt", "flight"]) - C("mvt")).dt.total_seconds().abs().alias("dnext"))
d = d.with_columns(((C("dprev") <= 1800) | (C("dnext") <= 1800)).fill_null(False).alias("dup"))
GAFLAG = C("ftype").is_in(["G", "M", "X"]) | C("seg").is_in(["Military", "Business", "Other", "Non-Scheduled"]) | (C("rule") == "V")
d = d.with_columns(
    pl.when(C("y").is_between(82800, 95000)).then(pl.lit("DAY"))
    .when(C("dup")).then(pl.lit("DUP"))
    .when((C("gap") >= 1800) & ((C("y") - C("gap")).abs() <= 90)).then(pl.lit("CSCHED"))
    .when(C("plan") & ((C("y") - C("gl")).abs() <= 90)).then(pl.lit("CLOBT"))
    .when(C("eobt1").is_not_null() & ((C("y") - C("ge")).abs() <= 90)).then(pl.lit("CEOBT"))
    .when(C("iobt").is_not_null() & ((C("y") - C("gi")).abs() <= 90)).then(pl.lit("CIOBT"))
    .when(C("plan") & ((C("y") - C("ga")).abs() <= 900)).then(pl.lit("REAL"))
    .when(GAFLAG).then(pl.lit("GA")).otherwise(pl.lit("OTHER")).alias("tbig"),
    pl.when(C("runway") == "H").then(pl.lit("HELI"))
    .when(C("plan") & ((C("ga") - C("y")) >= 300)).then(pl.lit("LATE"))
    .when(C("y") <= 0).then(pl.lit("NEG"))
    .when(GAFLAG).then(pl.lit("GA")).otherwise(pl.lit("OTHER")).alias("tsmall"))
d.write_parquet(os.path.join(ROOT, ".local", "logic_rows", "C_dep2025_typed.parquet"))
big = d.filter(C("y") > 3600)
print("LARGE labels (>3600 s): type by airport")
print(big.group_by(["apt", "tbig"]).agg(pl.len().alias("n"), C("y").median().alias("med_y"), C("y").max().alias("max_y"),
      (C("y") ** 2).sum().alias("ss")).sort(["apt", "n"], descending=[False, True]))
print("LARGE by month")
print(big.group_by(["apt", "tbig", "month"]).len().sort(["apt", "tbig", "month"]).pivot(on="month", index=["apt", "tbig"], values="len").sort(["apt", "tbig"]).select(["apt", "tbig"] + [str(m) for m in range(1, 13) if str(m) in big.group_by(["apt", "tbig", "month"]).len().pivot(on="month", index=["apt", "tbig"], values="len").columns]))
sm = d.filter(C("y") < 60)
print("SMALL labels (<60 s): type by airport")
print(sm.group_by(["apt", "tsmall"]).agg(pl.len().alias("n"), C("y").median().alias("med_y"), C("y").min().alias("min_y")).sort(["apt", "n"], descending=[False, True]))
print("acft of heli rows", sm.filter(C("runway") == "H").group_by(["apt", "acft"]).len().sort("len", descending=True).head(15))
