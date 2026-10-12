"""aobtflag 01: per-row flag 'AOBT_3 is a stand default' built from AOBT_3 and take-off ONLY (no recorded off-block, no label). 2025 and 2026 identical definitions.
Modes learned on 2025 inputs (take-off minus AOBT_3 per stand+runway, stand, runway), applied to both years. Peak must be sharp against its own shoulders (label-free)."""
import numpy as np, polars as pl
pl.Config.set_tbl_rows(200); pl.Config.set_tbl_cols(30)
I = "data/interim/"
d = pl.read_parquet(I + "agent_long_dep2.parquet", columns=["id", "yr", "mon", "adep", "rwy", "stand", "mvt", "aobt"])
d = d.with_columns(pl.col("adep").cast(pl.Utf8).alias("apt"), pl.col("rwy").cast(pl.Utf8), pl.col("stand").cast(pl.Utf8)).drop("adep")
d = d.with_columns((pl.col("stand") + "|" + pl.col("rwy")).alias("sr"), (pl.col("mvt") - pl.col("aobt")).alias("rd"))
KEYS = ["sr", "stand", "rwy"]
KS = np.arange(120, 3600, 10)
BOX = 35


def fit_modes(src, minn=100):
    rows = []
    for kc in KEYS:
        for (ap, k), g in src.filter(pl.col("aobt").is_not_null() & pl.col(kc).is_not_null()).group_by(["apt", kc]):
            if g.height < minn:
                continue
            rd = g["rd"].to_numpy()
            c = np.array([(np.abs(rd - K) <= BOX).sum() for K in KS])
            j = c.argmax(); K = int(KS[j])
            sh = [(np.abs(rd - (K + o)) <= BOX).sum() for o in (-240, -160, 160, 240)]
            sho = max(np.mean(sh), 1.0)
            rows.append((ap, k, K, c[j] / len(rd), c[j] / sho, len(rd), kc))
    m = pl.DataFrame(rows, schema=["apt", "k", "K", "share", "sharp", "n", "kc"], orient="row")
    return m.filter((pl.col("share") >= 0.3) & (pl.col("sharp") >= 6))


def apply(tgt, modes):
    tgt = tgt.with_row_index("ri")
    fl = np.zeros(tgt.height, bool); K1 = np.full(tgt.height, np.nan); SH = np.full(tgt.height, np.nan)
    for kc in KEYS[::-1]:  # coarse first, specific overrides
        m = modes.filter(pl.col("kc") == kc).select(["apt", pl.col("k").alias(kc), "K", "share"])
        j = tgt.select(["ri", "apt", kc, "rd"]).join(m, on=["apt", kc], how="left").sort("ri")
        K = j["K"].to_numpy().astype(float); rd = j["rd"].to_numpy().astype(float)
        has = np.isfinite(K)
        fl |= has & (np.abs(rd - K) <= BOX)
        K1 = np.where(has, K, K1); SH = np.where(has, j["share"].to_numpy().astype(float), SH)
    return tgt.with_columns(pl.Series("derived", fl), pl.Series("K", K1), pl.Series("sshare", SH)).drop("ri")


if __name__ == "__main__":
    a25 = d.filter(pl.col("yr") == 2025); a26 = d.filter(pl.col("yr") == 2026)
    m25 = fit_modes(a25); m26 = fit_modes(a26, 60)
    m25.write_parquet(I + "agent_aobtflag_modes25.parquet")
    print("modes 2025", m25.group_by("kc").len().to_dicts(), " modes 2026", m26.group_by("kc").len().to_dicts())
    # mode drift: same key, K difference
    j = m25.join(m26, on=["apt", "k", "kc"], suffix="_26")
    print("keys in both:", j.height, " K equal:", int((j["K"] == j["K_26"]).sum()), " |dK|<=40:", int(((j["K"] - j["K_26"]).abs() <= 40).sum()))
    print(j.filter((j["K"] - j["K_26"]).abs() > 40).select(["apt", "k", "kc", "K", "K_26", "share", "share_26", "n", "n_26"]).head(20))
    out = pl.concat([apply(a25, m25), apply(a26, m26.filter(False) if False else m25)])
    out = out.with_columns(pl.when(pl.col("aobt").is_null()).then(0).when(pl.col("derived")).then(2).otherwise(1).cast(pl.Int8).alias("flag"),
                           pl.when(pl.col("K").is_finite() & pl.col("aobt").is_not_null()).then(pl.col("rd") - pl.col("K")).otherwise(None).alias("dist"))
    out.select(["id", "yr", "mon", "apt", "rwy", "stand", "rd", "flag", "dist", "sshare", "K"]).write_parquet(I + "agent_aobtflag_flags.parquet")
    print("\nshares (derived / independent / no AOBT_3) per airport, 2025 all | 2025 Jan+Jul | 2026 Jan+Jul")
    rows = []
    for ap in sorted(out["apt"].unique().to_list()):
        r = [ap]
        for nm, f in (("25", (pl.col("yr") == 2025)), ("25JJ", (pl.col("yr") == 2025) & pl.col("mon").is_in([1, 7])), ("26", pl.col("yr") == 2026)):
            g = out.filter(f & (pl.col("apt") == ap)); n = g.height; nn = g.filter(pl.col("flag") > 0).height
            r += [n, round(g.filter(pl.col("flag") == 2).height / max(nn, 1), 3), round(g.filter(pl.col("flag") == 0).height / max(n, 1), 3)]
        rows.append(r)
    print(pl.DataFrame(rows, schema=["apt", "n25", "der25", "none25", "nJJ25", "derJJ25", "noneJJ25", "n26", "der26", "none26"], orient="row"))
    # derived share on rows that have AOBT_3 only for 2026 Jan vs Jul
    print(out.filter(pl.col("flag") > 0).group_by(["apt", "yr", "mon"]).agg((pl.col("flag") == 2).mean().round(3).alias("der")).filter(pl.col("mon").is_in([1, 7])).sort(["apt", "yr", "mon"]).pivot(on=["yr", "mon"], index="apt", values="der"))
