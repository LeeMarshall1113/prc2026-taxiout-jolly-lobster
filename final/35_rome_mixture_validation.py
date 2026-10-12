"""Validation (b): Rome no-plan mixture weight scaled by the day's copy mass, twelve 2025 months, as written in
docs/LAB-NOTES.md section 6 before it was run. Run once:  bash final/2x_run.sh "why" final/35_rome_mixture_validation.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

R = Path(__file__).resolve().parents[1]
I = R / "data" / "interim"
C = pl.col
M0 = 1025.5
SE = 1.5e8
COOL = [1, 2, 3, 11, 12]
SUMMER = [5, 6, 7, 8, 9]


def main() -> None:
    s = (pl.read_parquet(I / "agent_rome2_sensor.parquet", columns=["MVT_ID_mvt", "yr", "mo", "np", "gs", "y", "ctrl", "T", "copy_raw"])
         .filter((C("yr") == 2025) & (C("gs") >= 1800) & C("np")))
    z = np.load(I / "agent_rome2_clf52.npz", allow_pickle=True)
    p = pl.DataFrame({"MVT_ID_mvt": z["ids"], "P": z["P2"], "mo_z": z["mo"]})
    s = s.join(p, on="MVT_ID_mvt", how="inner")
    pk = pl.read_parquet(I / "agent_genaudit_12m.parquet", columns=["id", "pk"]).rename({"id": "MVT_ID_mvt"}).with_columns(C("MVT_ID_mvt").cast(pl.Float64))
    s = s.with_columns(C("MVT_ID_mvt").cast(pl.Float64)).join(pk, on="MVT_ID_mvt", how="left")
    s = s.with_columns(pl.coalesce(C("ctrl"), C("pk")).alias("a"), (C("T") // 86400).alias("day"))
    hid = pl.read_parquet(I / "agent_rome2_hidden2.parquet").filter(C("yr") == 2025)["MVT_ID_mvt"].cast(pl.Float64).to_list()
    print("informative no-plan Rome rows", s.height, "with a base", s["a"].is_not_null().sum())
    day = s.group_by("day").agg(C("P").mean().alias("m_d"))
    s = s.join(day, on="day", how="left")
    fp = s.filter((C("gs") >= 3600) & (C("gs") < 14400) & ~C("MVT_ID_mvt").is_in(hid) & C("a").is_not_null())
    print("footprint rows", fp.height, fp.group_by("mo").len().sort("mo").rows())
    rows = []
    for m in range(1, 13):  # months 4, 5, 6, 12 have no footprint rows: the 12-month table holds no base value for them
        oth = s.filter(C("mo") != m)
        m_ref = float(oth["copy_raw"].cast(pl.Float64).mean())
        t = fp.filter(C("mo") == m)
        if t.is_empty():
            continue
        a, y, gs, P, md = (t["a"].to_numpy().astype(float), t["y"].to_numpy().astype(float), t["gs"].to_numpy(), t["P"].to_numpy(), t["m_d"].to_numpy())
        w = np.where(gs < 7200, 0.5, 0.85)
        mix = P * gs + (1 - P) * M0
        flat = np.maximum(np.rint(a + w * (mix - a)), 0)
        scal = np.maximum(np.rint(a + w * np.minimum(1.0, md / m_ref) * (mix - a)), 0)
        g = lambda v: float((((a - y) ** 2) - ((v - y) ** 2)).sum() / SE)  # noqa: E731
        rows.append(dict(mo=m, n=t.height, m_ref=round(m_ref, 3), copy_rate=round(float(t["copy_raw"].cast(pl.Float64).mean()), 3), flat=g(flat), scaled=g(scal)))
    res = pl.DataFrame(rows)
    print(res.with_columns(pl.col(pl.Float64).round(3)))
    out = {}
    for form in ("flat", "scaled"):
        worst = float(res[form].min())
        cool = float(res.filter(C("mo").is_in(COOL))[form].sum())
        summer = float(res.filter(C("mo").is_in(SUMMER))[form].sum())
        ok = worst >= -0.05 and cool >= 0
        out[form] = ok
        print(f"{form}: total {float(res[form].sum()):+.3f}; worst month {worst:+.3f}; cool months {COOL} sum {cool:+.3f}; months {SUMMER} sum {summer:+.3f}; "
              f"months present {res['mo'].to_list()}; PASS={ok}")
    res.write_csv(R / "data" / "final" / "_gen" / "rome_mixture_validation_b.csv")


if __name__ == "__main__":
    main()
