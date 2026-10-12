"""Re-fetch METAR wind direction (drct) + speed for the 10 airports from the same IEM ASOS endpoint prc/weather.py uses
(run 2026-09-30). Writes data/raw/weather_wind/<station>.csv; prc/weather.py and its cache are untouched."""
import urllib.parse
import urllib.request
from pathlib import Path

from prc.weather import AIRPORTS, ASOS

OUT = Path("data/raw/weather_wind")
OUT.mkdir(parents=True, exist_ok=True)
for st in AIRPORTS:
    out = OUT / f"{st}.csv"
    if out.exists() and out.stat().st_size > 10_000:
        print(st, "cached"); continue
    q = {"station": st, "data": "drct,sknt,gust", "year1": "2025", "month1": "01", "day1": "01",
         "year2": "2026", "month2": "08", "day2": "01", "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no",
         "missing": "M", "trace": "T", "direct": "no", "report_type": ["3", "4"]}
    with urllib.request.urlopen(ASOS + "?" + urllib.parse.urlencode(q, doseq=True), timeout=300) as r:
        t = r.read().decode("utf-8", "replace")
    out.write_text(t, encoding="utf-8")
    print(st, len(t.splitlines()) - 1, "obs", f"{len(t) / 1e6:.1f} MB")
