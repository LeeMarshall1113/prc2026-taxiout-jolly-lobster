import polars as pl, numpy as np
pl.Config.set_tbl_cols(40); pl.Config.set_tbl_width_chars(220); pl.Config.set_tbl_rows(80)
C=pl.col
R=""
dep=pl.read_parquet(R+"data/interim/agent_rome2_dep.parquet").filter((C("ADEP_mvt")=="LIRF")&C("np"))
H=pl.read_parquet(R+"data/interim/agent_rome2_hidden.parquet")
def fix(s):
    if s is None: return None
    s=s.strip().upper()
    if len(s)>=6 and s[:3].isalpha():
        p=s[:3]
        for k in (2,1):
            if len(s)>3+k and s[3:3+k]==p[3-k:3] and s[3+k].isdigit(): return p+s[3+k:]
    return s
dep=dep.with_columns(pl.Series("k",[fix(x) for x in dep["FLIGHT_mvt"].to_list()]))
pr=pl.read_parquet(R+"data/interim/agent_rome2_planrows.parquet")
allfid=pl.read_parquet(R+"data/interim/agent_rome2_dep.parquet").filter((C("ADEP_mvt")=="LIRF")&C("FLIGHT_ID_mvt").is_not_null())["FLIGHT_ID_mvt"].unique().to_list()
o=pr.filter((C("PHASE_mvt")=="ARR")&(C("ADEP_flt")=="LIRF")&~C("FLIGHT_ID_mvt").is_in(allfid)).select(C("MVT_ID_mvt").alias("arr_id"),"yr",C("MVT_TIME_UTC_mvt").alias("arr_t"),C("LOBT_flt").alias("LOBT_o"),C("AOBT_3_flt").alias("AOBT_o"),C("ADES_mvt").alias("ades_o"),
   C("CALLSIGN_flt").str.strip_chars().str.to_uppercase().alias("k1"),C("FLIGHT_mvt").str.strip_chars().str.to_uppercase().alias("k2"))
out=[]
for key in ["k1","k2"]:
    j=dep.join(o,left_on=["k","yr"],right_on=[key,"yr"],how="inner")
    out.append(j.drop(["k1","k2"],strict=False))
j=pl.concat(out).unique(subset=["MVT_ID_mvt","arr_id"])
j=j.with_columns(((C("arr_t")-C("MVT_TIME_UTC_mvt")).dt.total_seconds()).alias("ft"),((C("MVT_TIME_UTC_mvt")-C("AOBT_o")).dt.total_seconds()).alias("t_aobt"),((C("MVT_TIME_UTC_mvt")-C("LOBT_o")).dt.total_seconds()).alias("gl"),((C("AOBT_o")-C("LOBT_o")).dt.total_seconds()).alias("a_l"))
j=j.filter((C("ft")>=1500)&(C("ft")<=20000)&(C("t_aobt")>=-120)&(C("t_aobt")<=7200))
nc=j.group_by("MVT_ID_mvt").agg(pl.len().alias("ncand")); j=j.join(nc,on="MVT_ID_mvt").sort("MVT_ID_mvt","ft").unique(subset=["MVT_ID_mvt"],keep="first")
print("recovered with de-garbled keys:",j.group_by("yr").agg(pl.len(),(C("ncand")>1).sum().alias("multi")).to_dicts(),"| previously",H.group_by("yr").len().to_dicts())
new=j.join(H.select("MVT_ID_mvt").with_columns(pl.lit(1).alias("old")),on="MVT_ID_mvt",how="left").filter(C("old").is_null())
print(new.select("yr","mo","FLIGHT_mvt","k","gap","gl","a_l","t_aobt","TAXITIME_SEC_mvt","ctrl","prod").sort("yr","gap"))
j=j.with_columns(C("TAXITIME_SEC_mvt").cast(pl.Float64).alias("y"))
j.write_parquet(R+"data/interim/agent_rome2_hidden2.parquet")
# rule classification (2025 evidence)
W=3606
j=j.with_columns(pl.when((C("gl")>=W)&((C("gap")-C("gl")).abs()<W)).then(pl.lit("R1 copy-impossible")).when(C("gl")<W).then(pl.lit("R2 low-gl")).when(C("a_l").abs()<W).then(pl.lit("R3a copy-likely")).otherwise(pl.lit("R3b real-likely")).alias("rule"))
q=j.filter(C("yr")==2025).with_columns(((C("y")-C("gap")).abs()<=60).alias("copy"),((C("y")-C("t_aobt")).abs()<=600).alias("real"))
print(q.group_by("rule").agg(pl.len(),C("copy").mean().alias("copy"),C("real").mean().alias("real")).sort("rule"))
s=j.filter(C("yr")==2026)
print(s.group_by("rule").agg(pl.len(),((C("prod")-C("t_aobt")).abs()<1.5).sum().alias("prod==t_aobt"),((C("prod")-C("gap")).abs()<1.5).sum().alias("prod==gs")).sort("rule"))
