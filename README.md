# prc-taxiout-2026

Team **jolly-lobster**'s entry to the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/): predict the
taxi-out time in seconds (`MVT_TIME - BLOCK_TIME`) of departures at ten European airports (Istanbul LTFM, Paris CDG LFPG,
London Heathrow EGLL, Amsterdam EHAM, Madrid LEMD, Frankfurt EDDF, Barcelona LEBL, Rome LIRF, Munich EDDM, Zurich LSZH).
Training data: 2025 movements. The final phase is one blind submission over January, February, June and July 2026
(670,790 departures), scored by RMSE. All code is released under the GNU GPLv3 ([LICENSE](LICENSE)).

The submitted file is a general model plus one step that reconciles its predictions with EUROCONTROL's published daily delay
totals of the same days (section 3; the model alone is variant A). February and June are built from raw inputs by `final/`;
January and July reuse stored results of the leaderboard phase. The leaderboard history, with its exceptions, is in
docs/SUBMISSION-LOG.md.

## The problem

The take-off time of every departure is an input, so the task is not to forecast a duration: it is to locate the off-block
time the airport recorded, `taxi-out = take-off - off-block`. The label is the operator's stamp, and the stamp differs from the
physical pushback by airport-specific habits. At Heathrow (2025, the 9,524 departures whose track is seen leaving the stand)
the first movement comes a median 484 s after the recorded off-block (5th to 95th percentile 347 to 742 s); at Paris CDG the
first fix near the stand comes a median 401 s after it.

Error in this target is concentrated in records, not in ordinary flights. At Rome the off-block is often a copy of the
scheduled time, and a few hundred departures a year (copies of the schedule, stamps dated a calendar day early, duplicate
records) have labels of hours. Rows missed by more than 1,800 s are 0.16% of the rows and 38% of the held-out squared error.
Any model of this label has to treat those rows by mechanism, defined on input columns and judged on held-out data.

## 1. What the data work found

the team contribution is the catalogue of defects in recorded off-block stamps, how each is detected, what each does to
published statistics, and a standalone checker. The model is the instrument that found and measured them.

- **Fifteen kinds of defect and recording convention** are catalogued in **[docs/DATA-DEFECTS.md](docs/DATA-DEFECTS.md)**: schedule
  copies (Rome, rarely Barcelona), stamps dated a day early, stamps at the plan clock, duplicate movement records, late stamps,
  hold records, a second population inside "no plan", helicopter and general-aviation movements, stand-default fields of the
  network manager, de-icing inside or outside the stamp, missing ground tracks, and the portal's daily delay against the stamps.
- **Concentration.** In 2025, 584 of the 2,085,047 departures at the ten airports (0.028%) have a label above 7,200 s and carry
  46% of the squared deviation of the label from its airport-month median. Fifteen stamps dated a day early carry 20% by
  themselves. At Rome, 693 no-plan departures whose off-block is a copy of the schedule carry 48% of the airport's total.
- **An observed window that bounds the label.** In 2025 every one of the 2,062,577 plan departures has its recorded off-block
  within 3,606 s of the plan's last off-block time. We read this as the tolerance of the join that attaches a movement to its plan.
- **The portal's daily pre-departure delay is an identity on the recorded stamps.** EUROCONTROL's performance portal publishes
  a daily all-cause pre-departure delay per airport. On 2025 it equals, to 1 to 2 minutes a day at seven airports, the sum over
  the day's departures of max(off-block minus schedule, 0) on the recorded stamps. A copied stamp, and a stamp dated a day
  early, count as zero delay: at Rome on 224 days of 2025, 429 flights carry a copied stamp, and the published delay of those
  days is about 458 minutes a day lower than if those flights were counted from schedule to take-off (their taxi included). The portal understates delay wherever such records exist.
- **Reception is a feed state.** The share of Rome plan departures seen on the ground by the open ADS-B network was 0% in
  January to April 2025, 18 to 92% in May to December 2025, 65% in January 2026 and 0.1% in July 2026. The same swap happened at
  Heathrow, Madrid and Munich. Any component that uses tracks has to be judged on months in the same reception regime.
- **The checker.** `tools/check_offblock_stamps.py` flags these records in any table with the challenge's columns (polars and
  numpy only, `--selftest`). Run label-free on February and June 2026, months it was not built on, it can evaluate only the input-defined flags and populations: 37 airport cells fall outside the range of the reference months,
  and the only defect-family count among them is Paris duplicate records in June ([docs/CHECKER-ON-NEW-MONTHS.md](docs/CHECKER-ON-NEW-MONTHS.md)).

## 2. The model

`final/` takes any set of months from raw inputs to predictions. In the submitted file February and June are built this way; January and
July are carried from stored results (section 3). Stages, in the order they are applied:

```
 1 base blend  ->  2 ADS-B ground-track  ->  3 pooled    ->  4 residual corrector  ->  5 record rules
   (GBM members,     stage (where heard)      re-stack        (leave-one-month-out,       (plan-clock window, gap clamp,
   no-plan model)                             (trust track)   out-of-month keys)          Rome / Paris / Istanbul handling)
```

1. **Base blend**: CatBoost, LightGBM and extra-trees members on flight-plan, Network Manager clock, runway, stand, congestion
   and METAR features, with a separate model for departures without a flight plan.
2. **ADS-B stage**: the off-block time estimated from where and when the aircraft is first seen moving on open ADS-B data.
3. **Pooled re-stack** (LightGBM) that absorbs the stage output and learns when to trust the track.
4. **Residual corrector**, fitted leave-one-month-out, with out-of-month key statistics, plan-clock distance columns and
   arithmetic columns, per-airport doses; a separate Istanbul corrector.
5. **Record rules**: the plan-clock window, the schedule-gap clamp, hidden-plan bounds, the Rome copy mixtures, the Paris
   duplicate rule, the Istanbul and Madrid bounds. Each is defined on input columns and counted on 2025 ([docs/ROW-RULES.md](docs/ROW-RULES.md)).
   The method is general; the constants are fitted to these ten airports.

**What each stage is worth on held-out 2025 months.** Nine airports (Istanbul is fitted separately); every month is predicted by
models that never saw it. Unit: board seconds (change in summed squared error / 1.5e8; one board second on a month's rows is about
one second of RMSE on a two-month score and half on the four-month score; positive = the stage helps). The constants were fitted on
2025 and judged first on held-out January and July 2025, the calendar months the leaderboard scored.

| stage | Jan | Feb | Mar | Jul | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|---|---|---|
| stage + re-stack + blend | 9.84 | 7.95 | 10.44 | 17.72 | 11.10 | 13.06 | 12.93 | 12.33 |
| residual corrector (eight-month fit) | 0.47 | 0.36 | 0.56 | 1.20 | 0.97 | 0.89 | 0.65 | 0.75 |
| tail component | 0.01 | 0.04 | 0.03 | 0.14 | 0.03 | 0.01 | 0.10 | 0.00 |
| Rome mixtures and rules | 1.13 | -0.08 | -0.41 | 3.43 | 0.71 | 0.19 | 0.39 | 0.14 |

The base, stage, re-stack and corrector are positive in all eight months, and February keeps about 80% of January's gain (81% and 77%). The
Rome mixtures are a summer gain and lose in February and March, so the final pipeline switches the no-plan mixture off in February.
On June 2025, which the stage never saw, base 209.8 s becomes 178.8 s after stage and re-stack (12.39 board-s, inside the January
to November range of 7.9 to 17.7) and 177.2 s after the corrector. The corrector is positive in all twelve 2025 months (0.31 to 1.45).
In the twelve-month fit the upstream predictions of April, May, June and December came from models fitted with the other months'
labels, so those four figures are not fully nested (`docs/LAB-NOTES.md` section 3.5).
Details, the selection history and what the month-agnostic alternatives cost: [docs/GENERALISATION.md](docs/GENERALISATION.md).

## 3. Three layers, and the score each way

The code and this text keep three layers apart.

1. **The general model**: stages 1 to 5 above, retrained on any airport's history.
2. **A reconciliation layer** that uses EUROCONTROL's published daily pre-departure delay of the same days. It has two parts. The day
   shift moves the delayed flights of an airport-day together toward the published daily total, at half strength (shrink factor capped
   at 0.5, each month's own offset), at eight airports; snow days whose published total is far below what the file implies are left
   unchanged. The Rome part updates the probability that a long-gap Rome departure without a flight plan is a schedule copy; it moves
   few rows a long way (168 rows, 1,861 s rms, in the January/July uploads). The day shift passed a test written before it was run, at
   seven airports on all twelve 2025 months: worst month -0.004 board-s, total +1.38, the four months outside the fitting frame +0.15
   (on a weaker base). Istanbul's shift was tested on seven 2025 months at the uploaded strength, not with the cap; the Rome part on
   eight (`docs/PIPELINE.md` I.3). The organisers answered on 2026-09-17 that the portal's daily airport files "can be considered open
   data for the challenge", on 2026-09-27 that the published monthly taxi-out figures of January and July 2026 may be used as model
   input, and on 2026-10-08, after the final was announced, that shifting predictions toward a published EUROCONTROL aggregate of the
   scored months was still acceptable. None of these answers names the daily delay file ([docs/PIPELINE.md](docs/PIPELINE.md) section I.3).
3. **Fixes that name specific January and July flights**: six Rome flight strings left at their pre-layer value and one
   single-row decision. They are isolated in `final/31_scored_row_fixes.py` and are off in the submitted file.

The submitted file is the general model plus layer 2 (**variant B**). Variants A (general model only) and C (B plus the named
fixes) are built from the same run.

**Which months come from where.** February and June come entirely from `final/`. January and July are carried from the stored
lineage for the expensive stages: their base predictions come from the same refit, the ADS-B stage, the re-stack and the correctors
(with their 52 view columns, the small blend and nudge layers and Istanbul's own corrector term) are the results of the earlier runs
that served those months ([docs/BUILD-TRACE.md](docs/BUILD-TRACE.md), `archive/`), and the rule layers and the published-daily layer are
rebuilt by `final/` on top of them. Istanbul's own corrector term is therefore in the submitted file for January and July only, inside
that lineage; it is not in February or June.

**All four months through `final/`.** We also ran January and July from raw inputs (`final_A/B/C_rebuilt`, `final/50_assemble_rebuilt.py`). That file passes
every label-free check. Row by row it differs from the submitted January and July by 37.5 s rms (January 34.1, July 40.0). About an eighth of the
squared difference is a fresh random realisation of the ADS-B stage and re-stack, about a fifth is the correctors refitted without their 52 view
columns and two variant columns, and about half cannot be attributed from the stored files. We submit the carried version because the rebuild lacks
those columns, two small blending steps and Istanbul's own corrector. On 2025 months the corrector's held-out gain is 0.82 board-s on average in
months that have the view columns and 0.38 in months that do not; the months differ in other ways too, so the gap is not a measurement of what the
columns are worth. The per-airport table and the comparison by stage are in [docs/REPRODUCE.md](docs/REPRODUCE.md) section A.7.

No final-phase score exists: the final has no leaderboard and no 2026 label is read, so the worth of each layer is shown on the 2025 months
held out from all fitting, next to the mean shift each layer makes to the February and June 2026 predictions.

| layer | worth on 2025 held-out months, February / June (board-s) | mean shift it makes to the 2026 predictions, February / June |
|---|---|---|
| 1 general model (variant A) | stage, re-stack and blend 7.95 / 12.39 (June: stage and re-stack only); residual corrector (twelve-month fit) 0.37 / 0.55; six record rules 0.14 / not measured; Rome mixtures -0.08 / not testable (off in February) | A against the same airport-month mean of 2025, ten airports: -20.0 s / +9.2 s |
| 2 reconciliation (B minus A) | +0.088 / +0.116 (June on the weaker base of section 8.2 of docs/GENERALISATION.md) | -0.39 s / -0.20 s (by airport -2.9 to +3.5 s / -1.9 to +0.9 s); 47% / 54% of the rows move |
| 3 named fixes (C minus B) | no 2025 analogue by construction | 0 s / 0 s (the fixes move 28 January and July rows) |

Sources: [docs/GENERALISATION.md](docs/GENERALISATION.md) sections 8 and 10 and [docs/REPRODUCE.md](docs/REPRODUCE.md) section A.4.
For January and July, the two reference files of the earlier run were scored on the public ranking set: the file without any published-figure step
(v152) 216.24, and the file with the published-figure steps at full strength (v162) 213.29; the published-figure steps account for 2.90 s of the 2.95 s between them.

Label-free checks of the submitted file (variant B; A and C pass the same hard checks), details in [docs/REPRODUCE.md](docs/REPRODUCE.md) section A.4:
670,790 rows, ids equal to the template in order, no nulls, Int32, no negatives, no plan-window violation (0 of 661,713 plan rows), no gap-clamp
violation; 19 of 80 airport-month mean or median statistics lie outside the 2025 twelve-month range of the airport; the daily IFR count lies within
+-2 of the published count on 1,118 of 1,173 days; no airport-day has a delay-total residual that differs from half of its pre-layer residual by more
than 6 sigma.

## 4. Running it

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements-final.txt   # Windows, Git Bash; on Linux .venv/bin/python
bash final/run_all.sh --dry-run          # the stage table, the inputs, which outputs exist
bash final/run_all.sh --run              # every out-of-date stage in order, stops at the first failure
python final/check_file.py data/final/final_B.parquet
```

Stage commands, memory, runtime, the stages that need rented or free cloud machines and their free alternatives:
[docs/REPRODUCE.md](docs/REPRODUCE.md). The stages and the three layers: [docs/PIPELINE.md](docs/PIPELINE.md). The data is not in
the repository; each source has a fetch script.

## 5. Limits

- **Rome in June and July 2026 has no ground tracks** (0.1% of plan departures were heard in July). Rome rests on the base, the
  record rules and the mixtures.
- **June could not be tested on 2025 for the Rome mixtures**: no base predictions exist for those rows in April to June or December,
  so the Rome no-plan mixture is judged on eight months and applied to June as it stands in the summer months (July to September
  gain 9.7 board-s with the flat weights, measured on the 532 no-plan rows of that test; inside the full chain the Rome mixtures and rules are worth 4.3
  over those months, table in section 2).
- **The 52 view columns and the two variant columns of the correctors are empty for February and June** (they are present in the carried
  January and July): the columns that come from two further re-stack variants are not
  rebuilt there (over 3 hours and 11 GB together), and are NaN in the serve data. The 2025 fold has NaN views for four months, so NaN is
  inside what the fitted corrector has seen.
- **February is mostly Istanbul and June is mostly Rome**: on the 2025 analogue Istanbul is 63% of February's squared error and Rome
  41% of June's. February 2026 has fewer snow hours at Istanbul and more at Frankfurt, Munich and Zurich than February 2025.
- **Frankfurt in February 2026 has about 10 surface points per aircraft**, against 35 to 110 in the other months, so the ADS-B stage
  rests on less there.
- **11 June 2026 is a partial day** in the open ADS-B feed.
- **Istanbul's own corrector term is in the submitted file only inside the carried January and July lineage.** For February and June, and
  in the all-four-months rebuild, Istanbul rows carry the bulk and snow refits and the small stages only.
- **Named-row fixes cannot be re-judged**: they have no 2025 analogue by construction.
- **The model is a reconstruction, not an operational forecast**: the ranking set is retrospective and whole-day, so same-day
  neighbours and arrivals are inputs.
- **Reproduction**: the upstream fits are seeded but not bit-exact; seed noise is about 0.03 s of RMSE for the corrector term and under 0.1 s for
  the chain. January and July in the submitted file are stored results; a from-raw rebuild differs by 37.5 s rms per row (section 3).

The public leaderboard history (111 scored uploads, the last on 2026-10-08, the exceptions to the practice of writing
the plan before the upload, the uploads that changed few rows, and the end of uploads on 2026-10-08) is in [docs/SUBMISSION-LOG.md](docs/SUBMISSION-LOG.md).

## Data sources and licences

Read from the official pages on 2026-10-02; open risks are in [docs/DATA-SOURCES.md](docs/DATA-SOURCES.md).

| source | licence as stated | needed to reproduce? |
|---|---|---|
| Challenge data (organisers' OpenSky bucket) | none stated on the challenge pages; not redistributed | yes |
| adsb.lol historical ADS-B (`adsblol/globe_history_2025`, `_2026`) | Open Database License 1.0 (the repositories also list CC0-1.0); attribution to adsb.lol and its feeders | stage 2 onward; without it the base blend alone is served |
| OpenStreetMap via Overpass | ODbL 1.0; "(c) OpenStreetMap contributors" | stand and way statics |
| IEM ASOS METAR archive (Iowa Environmental Mesonet) | no formal licence; the site states public domain for any lawful purpose | yes |
| EUROCONTROL performance portal daily airport file | free to copy with EUROCONTROL cited, non-commercial use; no formal open-licence text found | layer 2 only; layers 1 and 3 do not read it |

Evaluated and not used: OPDI flight lists, ADSBiq, adsb.lol ACARS, the ERA5 snow reanalysis (no estimator passed), a re-fetch of IEM
wind observations, public-holiday and AIP-supplement files. The reference taxi time in `prc/reference.py` follows the indicator
document's recipe but is computed from our own training data.

## Documentation

| File | What it is |
|---|---|
| [docs/FOR-USERS.md](docs/FOR-USERS.md) | Five-minute guide for people who maintain or analyse this kind of data, and what this is not |
| [docs/DATA-DEFECTS.md](docs/DATA-DEFECTS.md) | The catalogue: what each defect is, where and how often, how to detect it, what the entry does, how it generalises |
| [docs/CHECKER-ON-NEW-MONTHS.md](docs/CHECKER-ON-NEW-MONTHS.md) | The checker run on February and June 2026 inputs, months it was not built on |
| [docs/PIPELINE.md](docs/PIPELINE.md) | The stages of `final/`, the three layers, the ablation table of the January/July lineage |
| [docs/GENERALISATION.md](docs/GENERALISATION.md) | Results on months never used for selection, what was weighted toward January and July, the half-strength day layer |
| [docs/ROW-RULES.md](docs/ROW-RULES.md) | The rules that act on a few rows: fact or observed bound, condition, 2025 counts and exceptions, held-out evidence |
| [docs/REPRODUCE.md](docs/REPRODUCE.md) | Inputs, one command per stage, expected outputs, runtime and memory, pinned versions |
| [docs/BUILD-TRACE.md](docs/BUILD-TRACE.md) | The January/July lineage from raw data to the last uploaded file, stage by stage |
| [docs/SUBMISSION-LOG.md](docs/SUBMISSION-LOG.md) | Every scored upload with predicted against realised gain, the exceptions, the end of uploads |
| [docs/DATA-SOURCES.md](docs/DATA-SOURCES.md) | Every external dataset: URL, licence, dates, fetch script, what it feeds, licence risks |
| [docs/LAB-NOTES.md](docs/LAB-NOTES.md) | The consolidated research record, negative results included |
| [docs/DATA-ACCESS.md](docs/DATA-ACCESS.md) | How organiser bucket credentials are handled (no secrets in the tree) |

## Repository layout

- `final/`: the pipeline for any month set (`final/README.md` lists every stage, input and output).
- `prc/`: library modules (features, training, ADS-B extraction and matching, METAR, validation, submission checks).
- `tools/`: `check_offblock_stamps.py` (the stamp checker) and `defect_counts_2025.py` (re-runs the counts of the defect catalogue).
- `archive/` (release tree) or `archive/` (working tree): the January/July lineage, one script per step, the historical record.
- `modal/`: Modal apps for the heavy fits. `docs/`: the documents above and `docs/lists/` (ADS-B day lists).

## Credits

Ideas from other participants' publicly released work (public repository or public write-up). We took ideas, not code.

- A public write-up by another participant (licence recorded as GPLv3): estimating off-block time from adsb.lol surface tracks. Used in the ADS-B stage (`prc/adsb.py`, `prc/adsb_match.py`).
- https://github.com/ChandanHegde07/OpenAir (GPL-3.0): the Rome schedule-copy gate (predict the gate delay and subtract it from the schedule-to-take-off gap). Used in `prc/crossval.py` (`_GateDelay`).
- https://github.com/Phoenix-Ops-LTD/prc2026-taxiout (GPL-3.0): features from other departures' Network Manager taxi proxy around a flight and the next departure on the same runway. Used in `prc/features.py` (`_nm_neighbours`).
- https://github.com/EnioAguiar/prc-taxiout-2026 (GPL-3.0): the idea of giving the residual corrector the prediction minus the taxi time implied by each plan clock, and its distance to the plan-clock window (our nine plan-clock distance columns; columns, controls and judging are ours).
- https://github.com/javidmardanov/PRC-Data-Challenge-2026 (GPL-3.0) and the public archive of the 2024 edition's winners, https://github.com/prc-data-challenge-2024: ideas examined and not adopted.
- Other work that informed smaller choices is listed in docs/LAB-NOTES.md (Credits).

## Acknowledgement

AI tools were used as an aid during development. The project was led and directed by the author, who made the design, modelling and submission decisions.
