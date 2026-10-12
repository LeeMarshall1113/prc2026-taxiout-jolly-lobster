# For people who may use this

Five minutes. The long version of everything below is in `docs/DATA-DEFECTS.md` (the defects), `docs/PIPELINE.md` (the model) and
`docs/REPRODUCE.md` (re-running it).

The one thing to know first: the target of the challenge is the difference between the recorded take-off and the recorded off-block
time, and the recorded off-block is often not a pushback. A few hundred records a year (584 of 2,085,047 departures at the ten
airports in 2025) carry about half of the spread of the number. Finding those records mattered more to us than the model did.

## If you maintain or publish this data

What we found in the 2025 file, at the ten airports:

- **Schedule copies at Rome.** On 49% of the departures without a flight plan that left at least half an hour late, and on 8% of those with one, the off-block stamp
  is the scheduled time. A few hours late, same stamp. (Barcelona has a rarer, summer-heavy version of the same.)
- **Stamps dated a day early** (15 in 2025: Rome 13, Paris 1, Zurich 1). The clock time is right, the date is the schedule's.
- **Duplicate movement records at Paris** (115 pairs): one record with a flight plan and an obsolete early stamp, one without a plan and an ordinary stamp.
- **Late stamps** (label near zero): Zurich 305, Paris 136, Amsterdam 105, Barcelona 65.
- **Hold records**: stamp near the schedule, take-off one to three hours later. At Istanbul 62% fall on five snow days in February 2025; at Heathrow and Paris they are spread over 100 to 200 days (Heathrow 562 records, Paris 256).
- **Helicopter and general-aviation movements** recorded as departures (Zurich runway H, Amsterdam helicopter stands).
- **Network manager off-block that is a stand default** on 43% of plan rows (87% at Barcelona, 80% at Madrid, 79% at Munich, 75% at Zurich; 0% at Heathrow).

How to find them in your own table: `python tools/check_offblock_stamps.py your_movements.parquet` (or a CSV). It needs the challenge's column names or a
`--map key=COLUMN` for each (the required ones are airport, flight, take-off and scheduled off-block; the off-block stamp gives the label-based flags, the plan's last off-block time
the plan-window flags). It writes one row per flight with a flag per defect and prints, per airport and month, how many rows are flagged and how much of the spread of the stamp they carry.
On a month of the 2025 file it takes about a second.

What they do to published numbers:

- The portal's daily all-cause pre-departure delay is a sum over the stamps. A copied stamp counts as an on-time departure, and so does a stamp a day early. At Rome 429 flights in 2025
  left on average four hours after a stamp that said they were on time, so the published delay on those days is lower by about 458 minutes a day than if those flights were counted from schedule to take-off (their taxi included).
- Monthly taxi-out figures move with a few records: the squared deviation of the stamp is dominated by them, so a mean or an RMSE is fragile and a median or a trimmed mean is not.
- De-icing counts do not mean the same thing everywhere: the stamp is taken when the aircraft leaves its stand, so de-icing at a remote pad (Munich, Zurich, Amsterdam, Madrid, Heathrow) is inside
  the taxi-out and de-icing on the stand (Frankfurt) is not.
- At the seven airports where the daily delay matches the stamps to a minute or two a day (Rome, Heathrow, Munich, Istanbul, Madrid, Frankfurt, Barcelona), the identity is a free consistency check on your records. At Amsterdam and
  Zurich it matches once VFR departures are dropped; at Paris the portal is about 69 minutes a day higher than our rows and we do not know why.

## If you analyse taxi-out or ground delay from open data

Labels not to trust without a look: Rome (schedule copies, day-early stamps); Paris (duplicate records); anything with runway H or a helicopter stand; Istanbul departures without a flight plan
(the ones to Ercan behave differently from the rest); hold records on disruption days; near-zero labels at Zurich, Paris and Amsterdam.

Accuracy to expect, per airport, from a model like this one. These are held-out values (each month predicted by a model that never saw it; 2025 months 1-3 and 7-11, Istanbul months 1-3 and 7-10), in seconds of RMSE.
"Ordinary" means flights with a flight plan and a label of 60-3,600 s; "all" is every row with a label of 0-14,400 s. They do not include the step that uses published daily figures.

| airport | ordinary | all rows |
|---|---:|---:|
| Munich | 77 | 88 |
| Amsterdam | 79 | 93 |
| Zurich | 82 | 104 |
| Frankfurt | 88 | 95 |
| Madrid | 140 | 141 |
| Barcelona | 153 | 155 |
| Heathrow | 172 | 195 |
| Paris | 200 | 221 |
| Istanbul | 217 | 274 |
| Rome | 270 | 398 |

On the two scored months of 2026 the leaderboard RMSE was 213.3 s over all ten airports (v162; 216.2 s without the published daily figures). The 2026 files differ from 2025 (reception of the open ADS-B network changed between the two years at several airports, and January 2026 had snow),
so expect worse on a new year than on a held-out month of the same year.

What it needs as inputs: the scheduled and the recorded take-off time (it is an input, not a target); the flight plan's clocks (last, first and estimated off-block, the network manager's off-block field) where there is a plan, and for departures without a plan
the plan recovered from the destination airport's arrival record; stand, runway, aircraft type and operator; METAR weather; and, where the open ADS-B network hears aircraft on the ground, ground positions
from the adsb.lol archive. Without ground positions the base model and the corrector still run; the track stages do not.

It works after the fact. Because the take-off time is an input, it finds when the airport recorded the off-block, in hindsight. It cannot tell you at 08:00 how long a 09:00 departure will taxi.

## If you want to reuse the code

Four pieces are worth lifting. All are GPLv3.

1. **The stamp checker**, `tools/check_offblock_stamps.py`. Needs polars and numpy and nothing else. One file, no dependency on the rest of the repository. Needs a table with airport, flight, take-off and schedule; the more optional columns it gets
   (off-block stamp, plan's last off-block, network manager's off-block, stand, runway, type, flight rule) the more flags it can set. Thresholds are constants at the top of the file. Will not work as-is on a table with other time zones or
   a different plan join: test `label_outside_window` first.
2. **The ground-track off-block stage**: `prc/adsb.py` (extract ground positions from the adsb.lol daily archive), `prc/adsb_match.py` (match tracks to movements, find the stand-leave time), `archive/adsb_apply.py` (apply). Needs the adsb.lol daily
   files (multi-GB each; the ones for the scored days are not in the repository), a stand map built from the tracks and OpenStreetMap, and calibration days. Serving it took about two hours and 11 GB. Works only where the receiver network covers the apron
   (see D14 in `docs/DATA-DEFECTS.md`); there are no ground positions at Istanbul and almost none at Paris, and the feed around Rome went dark in July 2026.
3. **The de-icing detector**: `agent_deice_01_pads.py` (pad cells from cold-enriched ground-track dwells), `agent_deice_02_flag.py` (per-flight flag). Label-free; needs the stationary-period table built by `scripts/agent_multipush_*.py` and METAR temperature. Pad cells are
   airport-specific; it found Munich, Zurich, Amsterdam, Madrid, Heathrow and Frankfurt and misses where the treatment is on the stand.
4. **The residual corrector**: a leave-one-month-out gradient-boosted correction of a base prediction, with key statistics computed out of month so a held month never influences its own features (`archive/agent_bigc_te.py`, judged by
   `archive/agent_bigc_v2_03_judge.py`). Its first five versions were worth 2.08 s on the leaderboard and the arithmetic columns a further 0.83 s; it is the part most tied to this competition's folds. The fits ran on rented CPU containers (`modal/agent_bigc_app.py`, about $0.04 a fit).

What will not work outside the competition: the fitted constants (Rome's copy rates, Paris's duplicate offset, the stand modes), anything keyed on the ten airports' stand names, the plan-window width if your database joins differently, and the scripts that call the
rented-compute containers or read the organisers' files (`data/raw/`, which are not in the repository).

## What this is not

- Not a forecast. It uses the take-off time of the flight it predicts.
- Not an operational tool. Nothing here was run in real time.
- Not tested beyond these ten airports, and trained on 2025 only; the final phase scores four months of 2026 once.
- Not free of external inputs. `final/run_all.sh` runs the stages from raw inputs in order, but some stages need rented or free cloud machines, outside files (the ADS-B archive, METAR, the organisers' data) and many hours (`docs/REPRODUCE.md`). The upstream fits are seeded, not bit-exact.
- Not a claim that the last step is a model. The step that uses the published daily delay is the reconciliation layer (`docs/PIPELINE.md` I.3); the file is reported with and without it.

## Commands

Re-assemble the v156 file from its stored pieces and compare it with the uploaded one (needs the intermediate files named in `docs/REPRODUCE.md`, which are not in the repository):

```
.venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- .venv/Scripts/python.exe archive/reproduce_final.py
```

Run the stamp checker on a month of the 2025 file (needs the organisers' training file; `--selftest` needs no data):

```
python tools/check_offblock_stamps.py data/raw/training_2025-07-01_2025-08-01.parquet
```
