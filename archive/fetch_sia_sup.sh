#!/usr/bin/env bash
# Fetch French AIP supplements (SUP AIP, metropolitan series) for 2025 and 2026 from SIA.
# Licence Ouverte / Open Licence (SIA): free reuse incl. commercial, attribution required
# (source "SIA" + date of last update), no alteration of meaning. Fetched 2026-09-21.
# Each number exists as EITHER an AIRAC file (lf_sup_a_) or a non-AIRAC one (lf_sup_). Polite:
# sequential, with a delay. Skips anything already on disk.
OUT="data/external/sia_sup"
BASE="https://www.sia.aviation-civile.gouv.fr/media/store/documents/file/l/f"
UA="Mozilla/5.0 (research; PRC Data Challenge 2026)"
MAN="$OUT/manifest.tsv"
[ -f "$MAN" ] || printf "year\tnum\tkind\tbytes\n" > "$MAN"
for Y in 2025 2026; do
  MISS=0
  for N in $(seq -w 1 240); do
    NNN=$(printf "%03d" $((10#$N)))
    got=""
    for K in a ""; do
      F="lf_sup_${K:+${K}_}${Y}_${NNN}_fr.pdf"
      if [ -f "$OUT/$F" ]; then got="$F"; break; fi
      code=$(curl -s -o "$OUT/$F.part" -w "%{http_code}" -A "$UA" "$BASE/$F")
      sleep 0.4
      if [ "$code" = "200" ] && head -c 4 "$OUT/$F.part" | grep -q "%PDF"; then
        mv "$OUT/$F.part" "$OUT/$F"; got="$F"
        printf "%s\t%s\t%s\t%s\n" "$Y" "$NNN" "${K:-regular}" "$(stat -c %s "$OUT/$F")" >> "$MAN"
        break
      fi
      rm -f "$OUT/$F.part"
    done
    if [ -z "$got" ]; then MISS=$((MISS+1)); else MISS=0; fi
    # 12 consecutive misses past the known range => end of that year's series
    if [ "$MISS" -ge 12 ] && [ $((10#$N)) -gt 150 ]; then echo "$Y: series ends near $NNN"; break; fi
  done
done
echo "DONE: $(ls "$OUT"/*.pdf 2>/dev/null | wc -l) PDFs, $(du -sh "$OUT" | cut -f1)"
