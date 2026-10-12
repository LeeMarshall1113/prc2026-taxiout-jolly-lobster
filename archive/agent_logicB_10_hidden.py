import polars as pl, numpy as np
pl.Config.set_tbl_rows(150); pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(260); pl.Config.set_fmt_str_lengths(14)
C = pl.col; W=3606
D="data/interim/"
F = pl.read_parquet(D+"agent_detrules_frame.parquet"); print([c for c in F.columns])
cs = pl.read_parquet(D+"agent_detrules_cs.parquet").select("MVT_ID_mvt","yr","hc_lobt","hc_aobt","hc_eobt","arr_t").rename({"arr_t":"arr_c"})
F = F.filter(C("lobt").is_null()).join(cs, on=["MVT_ID_mvt","yr"], how="left")
F = F.with_columns(L=pl.coalesce("h_lobt","hc_lobt"), A=pl.coalesce("h_aobt","hc_aobt"), route=pl.when(C("h_lobt").is_not_null()).then(pl.lit("str")).otherwise(pl.lit("cs")), AR=pl.coalesce("arr_t","arr_c"))
F = F.filter(C("L").is_not_null()).with_columns(gl=C("mvt")-C("L"), gs=C("mvt")-C("sched"), l_s=C("L")-C("sched"), ft=C("AR")-C("mvt"), fp=C("h_arvt3")-C("A"))
F = F.with_columns(cls=pl.when(C("gl")<W).then(pl.lit("lowW")).when((C("gl")>=W)&(C("l_s").abs()<W)).then(pl.lit("imp")).otherwise(pl.lit("other")))
F = F.with_columns(link=(C("AR")-C("h_arvt3")))   # arrival movement time minus plan's arrival estimate
print("== link residual (arrival mvt - plan ARVT_3) quantiles by year, class")
print(F.filter(C("h_arvt3").is_not_null()).group_by("yr","cls").agg(n=pl.len(), p01=C("link").quantile(.01), p50=C("link").median(), p99=C("link").quantile(.99), mx=C("link").abs().max()).sort("yr","cls"))
F.write_parquet("data/interim/local/logic_rows/hidden_frame.parquet")
a=F.filter(C("yr")==2025)
print("== 2025: violations per class, by adep (imp: y>=gl-W ; lowW: y<=gl+W), n")
a=a.with_columns(y=C("TAXITIME_SEC_mvt"))
print(a.group_by("adep","cls").agg(n=pl.len(), viol=pl.when(C("cls")=="imp").then(C("y")>=C("gl")-W).when(C("cls")=="lowW").then(C("y")<=C("gl")+W).otherwise(False).sum()).sort("adep","cls").pivot(on="cls",index="adep",values=["n","viol"]))
print("== 2025 violators of 'imp' (y>=gl-W) with link residual:")
print(a.filter((C("cls")=="imp")&(C("y")>=C("gl")-W)).select("adep","flight","route","gl","gs","y","link","ft","fp"))
print("== 2025 violators of 'lowW' (y<=gl+W):")
print(a.filter((C("cls")=="lowW")&(C("y")<=C("gl")+W)).select("adep","flight","route","gl","gs","y","link"))
F.filter(C("yr")==2026).write_parquet("data/interim/local/logic_rows/hidden26.parquet")
