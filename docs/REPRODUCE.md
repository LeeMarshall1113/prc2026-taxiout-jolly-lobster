# Reproducing the entry

**Part A** reproduces the submitted file with `final/`: raw inputs, one command per stage, the expected outputs, runtime and memory,
what needs rented or free cloud machines, and the pinned versions; section A.7 states which months of the submitted file are built by `final/` and which are carried, and shows all four months built by `final/`. **Part B** is the ordered rebuild of the January/July lineage
of 111 scored uploads that `final/` ports, kept as the historical record; its January/July final assembly is the reproduction
target of `final/52_jj_verify.py`. Read `docs/PIPELINE.md` first for what each stage is.

# Part A. Reproducing the submitted file with `final/`

## A.1 Inputs

| input | where it comes from | used by |
|---|---|---|
| `data/raw/training_2025-*.parquet`, `final_ranking.parquet`, `final_submitting.parquet` | the organisers' bucket (credentials from the organisers; `python -m prc.data pull`; `docs/DATA-ACCESS.md`) | every stage |
| `data/raw/weather/*.csv` (METAR, 2025-01-01 to 2026-07-31) | `python -m prc.weather fetch` (Iowa Environmental Mesonet) | stages 10, 32 |
| `data/external/osm/stands.parquet`, `ways.parquet` | `archive/fetch_osm_stands.py`, `fetch_osm_ways.py` (Overpass; a snapshot of 2026-09-17/20) | stages 20, 40 |
| `data/adsb/points/*.parquet`: 354 days of 2025 (lists in `docs/lists/`), and every day of the served months | `final/05_adsb_download.sh` for 2026-02-01..28 and 2026-06-01..30 (58 days, about 153 GB streamed from adsb.lol, 0.9 GB kept); the January and July 2026 days and the 2025 days with `python -m prc.adsb extract --dates ... --out data/adsb/points` | stages 20 to 22 |
| `data/external/eurocontrol/all_pre_departure_delays_2025.csv`, `..._2026.csv` (10.5 MB together) | `https://www.eurocontrol.int/performance/data/download/csv/` | layer 2 only |
| the stored January/July lineage (`data/interim/agent_bundle_v152_final.parquet` and the Istanbul corrector files, named in `final/50_assemble.py`) | derived from the competition data by Part B; not in the repository | January and July rows of the assembly |

`docs/DATA-SOURCES.md` has the licence of each source. Without the EUROCONTROL files, layer 2 cannot run and variant A is the file.
Without the adsb.lol days the stages 2 and 3 cannot run: the base blend alone is served (the `untracked` regime of
`docs/GENERALISATION.md` section 8.3, RMSE 258 to 788 against 87 to 198 with tracks).

## A.2 Environment

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install -r requirements-final.txt     # Python 3.14.0, pinned
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage          # set by final/run_all.sh
```

Pinned: numpy 2.5.2, pandas 3.0.5, polars 1.44.1, pyarrow 25.0.1, scikit-learn 1.9.0, scipy 1.18.1, lightgbm 4.7.0, catboost 1.2.10,
xgboost-cpu 3.4.1, boto3 1.43.88, requests 2.34.2. The Modal images (Python 3.13) pin `numpy==2.5.2 lightgbm==4.7.0 polars==1.44.1`. The
wrappers `final/*_run.sh` request a slot from a local job scheduler and a hard memory cap (`archive/capped_run.py --cap-gb N`);
both are specific to the build machine, and a reproducer can drop the slot and keep the cap. Scripts are run from the repository root.

## A.3 One command per stage

`bash final/run_all.sh --dry-run` prints the table below with the inputs, which outputs exist and which are out of date;
`bash final/run_all.sh --run [--only ID] [--from ID]` runs every out-of-date stage in order and stops at the first failure
(stages marked MANUAL print their command). The memory column is the peak the stage was run under (the slot), the time column is
the estimate filed with the slot unless a measured value is given. Status is as of 2026-10-09 (EDT).

| ID | stage | command | memory | time | status (2026-10-09 08:06 EDT) | output |
|---|---|---|---|---|---|---|
| 05 | ADS-B day files | `bash final/05_adsb_download.sh` | 3 GB | about 1 h 15 min for 58 days | executed, 58 of 58 day files (11 June is partial in the feed) | `data/adsb/points/2026-02-*`, `2026-06-*` |
| 10 | base refit (stage 1) | `bash final/10_base.sh` | peak commit 11.5 GB | main fit 14:59Z on 2026-10-08 to 00:34Z on 2026-10-09 (9 h 35 min), last step 35 min; one CatBoost seed 1.5 to 2 h; resumable | executed | `J2_base.parquet`, `J2_members.npz`, `J2_v94model.parquet` |
| 20 | ADS-B dumps | `bash final/2x_run.sh "dumps" final/20_adsb_dumps.py --months 2,6` | 3 GB | 60 min | executed | `J6_*.parquet` |
| 21 | Rome pre-rules, ADS-B stage (stage 2) | `bash final/2x_run.sh "ADS-B stage" final/21_adsb_stage.py --months 2,6 --workers 3` | 11 GB locally (5.3 GB peak per process, 3 workers) | locally about 5 to 6 h; run in the cloud as 33 jobs, about 50 min wall, about 6.5 USD | executed (cloud) | `J5_v103_s1`, `J5_v105_s1`, `J5_v105sc_s1` |
| 22 | re-stack, post-chain, splices (stage 3) | `bash final/2x_run.sh "re-stack" final/22_restack.py --months 2,6` | 3.4 GB | 12 min | executed | `J7_restack.parquet` |
| 43a | Istanbul bulk and snow refits | `final/43_kaggle_prepare.py`, then three kernels; `python final/43_ist_build.py build ...` | Kaggle kernels: control 4.2 h at 9.65 GB; snow arms 2.25 h and about 3 h | about 4 h on free cloud CPU kernels | executed | `J3_ist.parquet` |
| 50a | chain value | `bash final/5x_run.sh "chain value" final/50_assemble.py --months 1,2,6,7 --carried 1,7 --stop-after-chain` | 3 GB | 10 min | executed | `J11_chain.parquet` |
| 42 | small stages, tail | `bash final/4x_run.sh "small stages" final/42_small_stages.py --base data/final/J11_chain.parquet --months 2,6` | 3 GB | 60 min | executed | `J8_*` |
| 40 | corrector serve inputs | `bash final/4x_run.sh "serve inputs" final/40_serve_inputs.py --months 2,6 --pv data/final/J8_pv.parquet --profile final` | 3 GB | 60 min | executed | `J9_serve/` |
| 41s | corrector stage directory | `bash final/4x_run.sh "stage" final/41_corrector_serve.py stage --serve-dir data/final/J9_serve --tag J10` | 3 GB | 10 min | executed | `J10_stage/` |
| 41a | arithmetic corrector, 12 seeds | `SLOT_GB=10 bash final/4x_run.sh "arith serve fits" final/41_corrector_serve.py arith --stage data/final/J10_stage --seeds 0-11` | 10 GB locally | about 40 min in the cloud for both fits, about 2.4 USD | executed (cloud) | `J10_stage/out/arith/*.npz` |
| 41p | plan-clock corrector, 12 seeds | the same with `pcd` | 10 GB locally | (with the line above) | executed (cloud) | `J10_stage/out/pcd/*.npz` |
| 43b | Istanbul corrector term | `final/43_gen_lv.py`, `43_ist_serve_inputs.py`, `ist_corrector_serve.py` (`final/README.md`) | local | see command | not executed for February and June (Istanbul's own corrector term is not in those months; it is inside the carried January and July lineage) | Istanbul term `.npz` |
| 50 | assembly, variants A, B, C | `bash final/5x_run.sh "assembly" final/50_assemble.py --months 1,2,6,7 --carried 1,7` | 3 GB (peak 1.6 GB) | about 10 min | executed | `final_A/B/C.parquet` |
| 50r | assembly of all four months from `final/` (demonstration, not submitted; section A.7) | `bash final/5x_run.sh "rebuilt" final/50_assemble_rebuilt.py` | 3 GB | about 10 min | executed | `final_A/B/C_rebuilt.parquet` |

The rule scripts (`30_general_rules.py`, `32_published_daily.py`, `34_rome_mixtures.py`, `31_scored_row_fixes.py`) are called by
`50_assemble.py`; each runs alone with `--pred IN.parquet --out OUT.parquet` (`31` also takes `--before`), prints the rows it moves per
airport-month and writes nothing else. A `--sample` mode on each script takes a random fraction of the departures.

## A.4 Checks on the submitted file

All are label-free.

```bash
python final/check_file.py data/final/final_B.parquet                 # row count, ids, nulls, negatives, template type;
                                                                      # per airport-month level against the 2025 range; plan window; gap clamp
python final/51_roundtrip_check.py data/final/final_B.parquet         # published-count identity and delay-total residual by airport-month
python final/52_jj_verify.py                                          # January and July rows against the last uploaded file
bash final/run_all.sh --dry-run
```

Results for the submitted file (variants A, B and C pass the same hard checks). `check_file.py` on `final_B`: 670,790 rows, ids equal to the template in
order and unique, no nulls, Int32, no negatives, 0 of 661,713 plan rows outside the plan window, 0 gap-clamp violations; 19 of 80 airport-month
mean or median statistics lie outside the 2025 twelve-month range (January: Frankfurt, Munich, Amsterdam, Zurich, Istanbul; February: Rome, Istanbul;
June: Frankfurt, Barcelona, Madrid, Zurich; July: Frankfurt, Rome, Zurich). `51_roundtrip_check.py`: the daily IFR count by stamp date lies within +-2
of the published count on 1,118 of 1,173 days (Paris on about 50 to 70% of days, the weakest series); with `--pre`, no airport-day has a delay-total
residual that differs from half of its pre-layer residual by more than 6 sigma (850 airport-days exceed 6 sigma on the raw residual, which a
half-strength layer leaves in place). Limits for February and June
from the inputs alone (`docs/CHECKER-ON-NEW-MONTHS.md`): the predicted mean per airport-month should sit inside the 2025
twelve-month range of that airport (Heathrow 1310 to 1431 s, Frankfurt 829 to 895, Munich 766 to 947, Amsterdam 716 to 847, Barcelona
887 to 1008, Madrid 982 to 1043, Paris 932 to 1106, Rome 1079 to 1406, Zurich 722 to 775, Istanbul 986 to 1261 s; February Istanbul
below its February 2025 value because the snow is absent), no prediction outside the plan window plus 12 s of jitter allowance, none
negative, none above the gap plus 1,200 s at Istanbul no-plan rows; the daily IFR count by stamp date should lie within +-2 of the
published count on the days where the inputs bound it to that width. In the submitted file six airport-month means lie outside the range (January: Frankfurt,
Munich, Amsterdam, Zurich; February: Rome, 1,069.8 s against 1,078.8 to 1,405.7; June: Madrid, 1,051.1 against 981.9 to 1,042.8); the range is a
reference that `check_file.py` reports, not a pass rule.
`52_jj_verify.py` on the January and July rows (213,225 of the January and July rows differ from v162): variant B has rms 21.8 s and mean +3.2 s against v162,
20.8 s of the rms from the day-layer decisions and 5.7 s from everything else; variant C has rms 18.5 s. By airport (rms, B): Frankfurt 25.1, Munich 2.7,
Heathrow 3.3, Amsterdam 0, Barcelona 15.3, Madrid 9.4, Paris 11.8, Rome 63.7 (C: 48.4, the named Rome flights), Zurich 12.7, Istanbul 15.5 s.
v162 used the full-strength day layer; variant B uses the half-strength form, so the two differ by design.

## A.5 What needs rented or shared compute, and the free alternative

| step | paid or shared | free alternative |
|---|---|---|
| stage 1 refit (10) | local, peak commit 11.5 GB, main fit 9 h 35 min | the same; no GPU needed |
| stage 2 ADS-B stage (21) | cloud: 33 jobs, about 50 min wall, about 6.5 USD | local, 11 GB with 3 workers, about 5 to 6 h (5.3 GB peak per process); fewer workers (`--workers 1`) if memory is short |
| Istanbul refits (43a) | `modal/final_istanbul_app.py` (rented CPU, not launched) | free Kaggle CPU kernels prepared by `final/43_kaggle_prepare.py`, about 4 h in all (used for the submitted file) |
| corrector serve fits (41a, 41p) | `final_corrector_app.py` (rented CPU; about 40 min and about 2.4 USD for the final fits) | local with a 10 GB slot: 12 seeds of each, minutes per fit at 6 GB in the earlier work |
| Istanbul corrector (43b) | Modal for the views members (about USD 2 for seven arms) | the corrector itself runs locally; the members can be fitted locally |

Nothing needs a GPU. The earlier lineage also used rented CPU for about 140 corrector fits (Part B section 0.3).

## A.6 Randomness and limits

Seeds are set per script. Run-to-run variation, as measured on the earlier work: two builds of the same residual corrector that
differ only in where they ran differ by about 7 s rms per row; means over 6 seeds differ by about 5.5 s rms and over 9 seeds by
about 4 s; in score terms the seed noise of the served corrector term is about 0.03 s of RMSE at infinitely many seeds and under 0.1 s
for the whole chain. A CatBoost fit moved a paired comparison by about 0.9 s when the thread count changed between two arms, so the
thread counts in the scripts have to be kept. A rebuild reproduces the submitted file to that precision, not bit for bit. January and July in the
submitted file are stored results; a from-raw rebuild differs by 37.5 s rms per row (section A.7).

What is exact: the January/July stages that `final/` ports reproduce the stored lineage (the correctors' serve-side inputs: all
rows of `agent_bigc13_serve_in.npz`, 194 columns, 0 differing values; the Istanbul build against the shipped v137 step: 0 of
45,183 rows differing; `archive/reproduce_final.py` assembles the stored pieces into the v156 file with maximum absolute
difference 0 s, Part B section 0.1).

What is not reproduced: the neighbour blend, the silent-start nudge and the meta layer (`docs/PIPELINE.md` section I.4); the two further
re-stack variant streams, so the corrector's view columns that rest on them are NaN for February and June; the Istanbul treatment arm
(telescoped into one step). The upstream fits of the January/July lineage were run once, over about 140 versions, on a mix of the
local machine, Kaggle and Modal; their exact command lines are partly not recorded (Part B section 3).

## A.7 The same pipeline on all four months (demonstration), and why the submitted January and July are carried

**Which months come from where in the submitted file (variant B).** February and June come entirely from `final/`. January and July
are carried from the stored lineage for the expensive stages: the base predictions are those of the same refit; the ADS-B stage, the
re-stack and the correctors (with their 52 view columns, the small blend and nudge layers and Istanbul's own corrector term) are the
stored results of the earlier runs that served those months; the rule layers and the published-daily layer are rebuilt by `final/` on top.
The reduced archive of the old scripts (`archive/`) ships for that reason. Istanbul's own corrector term is in the submitted file only
inside that carried lineage; it is not in February or June.

**What was rebuilt.** January and July were also run through every stage of `final/` on `final_ranking.parquet` (tags `J5jj`, `J6jj`,
`J7jj`, `J3jj`, `J8jj`, `J9jj`, `J10jj`; stage 21 with one worker under a 9 GB slot, `final/2x_jj.sh`), and `final/50_assemble_rebuilt.py` assembled
all four months with `--carried ""` and the same layer settings (capped day layer with each month's own offset, date guard, stop-day
signature, flat Rome mixtures in January, June and July and off in February, named-row fixes in C only, Istanbul's own corrector term not
applied). Outputs `data/final/final_{A,B,C}_rebuilt.parquet`; the submitted `final_{A,B,C}.parquet` were not touched. The comparison is
`final/54_rebuilt_compare.py`.

**Checks on `final_B_rebuilt`.** Every hard check of `check_file.py` passes (670,790 rows, ids equal to the template in order, no nulls,
Int32, no negatives), and so for A and C; 0 plan-window violations and 0 gap-clamp violations; 18 airport-month level statistics outside the
2025 range (submitted B: 19); the daily IFR count within +-2 of the published count on 1,115 of 1,173 days (B: 1,118); with `--pre`, one
airport-day flagged (Rome, 2026-07-01, z -6.4).

**Difference from the carried January and July** (B_rebuilt minus B, all January and July rows; 336,025 of them differ, mean -0.88 s,
rms 37.45 s; January 34.1, July 40.0; the 20 largest rows carry 17.8% of the squared difference and the largest 1,000 carry 39.6%):

| airport | January: mean / rms (s) | July: mean / rms (s) | rms against the last uploaded file: carried B / rebuilt (s) |
|---|---|---|---|
| Frankfurt EDDF | +4.6 / 28.9 | +1.5 / 27.1 | 25.1 / 35.8 |
| Munich EDDM | -3.0 / 23.7 | -3.1 / 24.1 | 2.7 / 24.0 |
| Heathrow EGLL | -0.8 / 36.0 | +1.7 / 30.8 | 3.3 / 33.6 |
| Amsterdam EHAM | -2.0 / 20.9 | +2.2 / 31.7 | 0.0 / 27.6 |
| Barcelona LEBL | +1.8 / 22.4 | -1.2 / 28.5 | 15.3 / 30.0 |
| Madrid LEMD | -0.9 / 21.7 | -2.9 / 29.2 | 9.4 / 27.1 |
| Paris LFPG | +1.1 / 38.0 | -7.6 / 27.9 | 11.8 / 33.4 |
| Rome LIRF | +1.5 / 54.2 | -5.8 / 97.4 | 63.7 / 105.9 |
| Zurich LSZH | -1.2 / 30.1 | -1.2 / 27.1 | 12.7 / 31.2 |
| Istanbul LTFM | -1.7 / 44.3 | -0.1 / 36.0 | 15.5 / 42.8 |
| all ten | | | 21.8 / 43.3 |

The last column is the distance from the last uploaded file (`52_jj_verify.py`), not error: the carried B shares that file's lineage by construction.
The largest single rows are Rome July rows with labels of 1,000 to 3,600 s that the day layer and the Rome path move, Amsterdam 7 July (+3,467 s on one
row) and Frankfurt January (+1,478 s).

**Where the difference enters** (from the stored files; share of the squared difference between the two files, adding to 1; the rms of the
difference before the layers is 35.1 s and after them 37.5 s):

| stage | share | rms by airport-month |
|---|---|---|
| ADS-B stage and re-stack (a new draw of the realisations and of the take-off jitter; no Munich January shift) | 0.125 (Amsterdam 0.57, Paris 0.26) | 8 to 40 s |
| correctors (new serve fits, dose applied, view and variant columns empty, against the stored fits) | 0.215 (Zurich 0.57, Munich 0.55, Barcelona 0.54) | 8.6 to 31.5 s; correlation with the stored term 0.63 to 0.97 |
| small stages M1 to M4 | -0.001 | 0.2 to 2.0 s |
| tail component | -0.001 | 0.2 to 3.1 s |
| Istanbul's own corrector term (in the carried version, not in the rebuild) | 0.007 overall, 0.043 of Istanbul | 9.7 to 11.4 s |
| other: not separable from the stored files (Istanbul bulk and snow rows from old-frame arms, neighbour blend and nudge and meta layers not ported, earlier corrector pieces, Rome pre-rules, rounding) | 0.529 (0.95 of Istanbul's own difference) | 8.6 to 68 s |
| layers after the chain (the day layer and the window react to the level) | 0.126 (Rome 0.33, July 0.20, January 0.004) | Rome July 61.6 s, others 0 to 4.5 s |

Per join on the January and July rows: ADS-B stage 237,346 rows changed (rms change 114.8 s, report only), re-stack 286,447 (141.9 s), Istanbul rows 46,292
(54.0 s), small stages 296,434 (10.4 s), correctors 287,422 (29.0 s), tail 60,159 (5.8 s), rules 102 (32.7 s), mixtures 260 (33.2 s), day layer 192,001
(34.6 s), fixes 29 (11.3 s). The difference is spread over every stage and is not carried by a few rows; about half of it cannot be attributed from the
stored files.

**Why the carried January and July are the ones submitted.** The rebuilt months are a different draw of the same model, not a better one.
They leave out the 52 view columns and the two variant columns of the correctors, the small blend and nudge layers (about 0.24 s in the ablation, `docs/PIPELINE.md`
Part II section 2, row C6) and Istanbul's own corrector term, and Istanbul rows still use the stored arms of the old frame, so they are not all new either. What
they have in their favour is take-off times and realisations consistent with `final_ranking.parquet`. An average of the two would reduce variance but carries the missing pieces at half weight and has no tested precedent.

**2025 evidence on the view columns.** The 12-month leave-one-month-out corrector's held-out gain per month (board seconds, `docs/GENERALISATION.md` section 8.2)
is 0.53, 0.37, 0.59, 1.45, 1.08, 1.01, 0.75 and 0.81 in the eight months whose fold has the view columns (January, February, March, July to November), mean 0.82, and
0.33, 0.31, 0.55 and 0.34 in the four months without them (April, May, June, December), mean 0.38. It is positive in all twelve, so the corrector with the columns
empty still helps, at roughly half the size. The two groups differ in more than the columns (the months without them have a weaker base chain with no rules, blend or Rome
handling, and fall in other seasons; July's 1.45 is a July effect), so the gap of 0.44 is an upper bound on what the columns are worth, not a measurement.

---

# Part B. The January/July lineage (historical record)

Items marked UNVERIFIED were not checked against code or an artefact. Section numbers below are those of the lineage rebuild. Terms: the submitted file is variant B of Part A; the last upload is v162; the v156 file is upload v156. `archive/reproduce_final.py` re-assembles upload v156 exactly; v158 to v162 are not assembled by it.

## 0. Verify without re-training, and what can be re-run

### 0.1 Verify the v156 file in minutes

With the file of upload v156, `submissions/jolly-lobster_v156.parquet` (a data file; it is not part of the public repository), and the organisers' `data/raw/ranking.parquet` (inputs only; the
check never reads a departure label):

```bash
.venv/Scripts/python.exe archive/reproduce_final.py --check-file submissions/jolly-lobster_v156.parquet     # seconds, under 1 GB
```

It reports the row count and id set against the ranking file, negative predictions (none), predictions above the schedule gap in
the clamp's scope (0 for v156; the v143 file had 21, maximum excess 246 s) and plan predictions outside the plan-clock window (6 for v156, 15 to 25 s outside the window, so 1 to 11 s beyond the 14 s slack; 0 for v146; the v143 file had 6, at most 252 s beyond the slack). The additive terms of the late stages were applied after the bounds; v146
re-applied them and the later corrector terms (v150 to v156) moved six plan rows slightly out again (`docs/BUILD-TRACE.md` section 1a).

With the intermediate prediction files in `data/interim/` (they are derived from the competition data and are not in the
repository; the earlier stages write them):

```bash
.venv/Scripts/python.exe archive/capped_run.py --cap-gb 3 -- .venv/Scripts/python.exe archive/reproduce_final.py   # about a minute, 1.1 GB peak
```

It performs the final assembly steps in order, from v133 to v156 (v133, v135, v136, v137 as replacements of the rows a component changed; v141
as base plus component minus reference; v142, v143 and v144 as the base plus a sum of float deltas, rounded once, with the same arithmetic as
`archive/agent_bundle_v142.py`, `agent_bundle_v143.py` and `agent_bundle_v144_v145.py`; v145 as the calendar rule on the ranking file's input
columns; v146 as the six bound pieces of `archive/agent_bundle_v146.py`; then v148 as the Rome hard-binding piece, v150 as the arithmetic-column corrector piece on v148, v152 as the round-three corrector piece, v153 and v154 as the two published-daily-delay pieces on v152, v155 as both of them, and v156 as the Istanbul corrector piece on v155, with the arithmetic of `archive/agent_bundle_v150_v151.py`, `agent_bundle_dayeq.py` and `agent_bundle_v156_v157.py`), names the role of every file in its header, prints the rows each
step moves (270, 259, 5, 22, 191,725, 200,541, 190,957, 38,316, 1, 51, 14, 189,076, 209,078, 139,724, 168, 139,892 and 42,538), compares each intermediate (v133 to v156) with the stored file of the same
upload by id, and compares the assembled files of v143 to v146, v148, v150 and v152 to v156 with `data/interim/agent_bundle_v<V>_final.parquet` and
`submissions/jolly-lobster_v<V>.parquet`, printing the maximum absolute difference and the rows differing per airport. **Result on
the build machine (2026-10-07, under `archive/capped_run.py --cap-gb 3`, peak commit 1.29 GB, exit 0): every intermediate equals its stored file; for all eleven assembled versions the maximum absolute difference is 0 s at all ten airports, 0
rows differing among all rows (22 comparisons), and the label-free checks on the v156 file pass except for the six plan rows noted above. Reported by the script as "identical to the stored final file".** The script needs these files in `data/interim/` besides the earlier ones: `agent_basefix_final`,
`agent_regime_gatejul_final`, `agent_mega_refit_delta.npz`, `agent_mega_refit12_delta.npz`, `agent_pcd_refit12_delta.npz`,
`agent_rome1c_final`, `agent_apt_EDDM_final`, `agent_pcdx_final`, the six v146 piece files named in section 2.8, and, for the later steps, `agent_romenear_hardbind_final`, `agent_arith_final`, `agent_arith3_final_noG`, `agent_dayeq_level_final`, `agent_dayeq_types_final`, `agent_istnext_final` and the stored `agent_bundle_v150_final`.

To see what each rule does, run its script on a stage output (every script prints the rows it moves, e.g.
`archive/apply_eddm_january.py` prints 7,062 rows moved by -38 s; that rule is undone in v144); the counts for the shipped lineage are in `docs/BUILD-TRACE.md`
section 1a and the conditions in `docs/ROW-RULES.md`.

### 0.2 What can be re-run, how long it takes, where

| Step | Command (details in section 2) | Time and size | Where | Stochastic? |
|---|---|---|---|---|
| Final assembly and checks | `archive/reproduce_final.py` | a few minutes, 1.2 GB peak | local | no; exact |
| Post-chain rules (stage 4) | the seven `scripts/apply_*.py` in order (section 2.5) | a few minutes in all, under 3 GB | local | no |
| Competition data and external data | `python -m prc.data pull`, `prc.weather`, `prc.adsb extract` | hours; the ADS-B days are multi-GB each | local, needs organiser credentials | no |
| Stage 1, base blend | `python -m prc.final ...` (section 2.3) | duration not recorded | local CPU | yes: seeds and thread count |
| Stage 2, ADS-B stage serving | `archive/adsb_apply.py` | about 2 h, 11 GB | local | yes: 10 realisations averaged |
| Stage 2 calibration, stage 3 re-stack, stage 8 corrector, Istanbul members | `modal run` apps in `modal/` | about $0.04 per corrector fit; about $2 for a round of Istanbul member arms | rented CPU | yes: seeded, 3 to 9 seeds averaged |
| Round-three corrector of v152 (stage 22) | `modal/agent_arith_app.py` with the job lists `scripts/agent_arith3_jobs_cloud*.txt` | about 140 fits (section 0.3) | rented CPU | yes: 3 seeds held out, 12 serve seeds |
| Round-two corrector (v150), Istanbul corrector (v156), the published-daily-delay pieces (v153 to v155) | `scripts/agent_arith_*.py`, `agent_istnext_*.py`, `agent_dayeq_*.py` | serve fits about six minutes each at 9 GB; about 330 Istanbul fits of 15 to 35 s; the daily-delay pieces need no fit | local | yes for the fits; the daily-delay pieces are deterministic |
| Istanbul refits (stage 6) | `archive/agent_kg_ltfm_kaggle.py`, `agent_ltreg_kaggle.py` | about 8.5 h per snow serving fit | free CPU kernels | yes |
| Small correctors, tail component, Rome and Paris rows | scripts named in `docs/BUILD-TRACE.md` section 1 | minutes to an hour each | local | seeded; the row rules are deterministic given their inputs |

### 0.3 What a reviewer cannot rerun without paid compute, and what needs external files

- **Needs paid (rented CPU) compute.** The round-three corrector of v152 (stage 22): the held-out fits of the control, the candidate and the shuffled-column control (3 seeds x 8 months each: 72 fits), the 24 held-out and 6 temporal fits of the variant that shipped (without the view-geometry group), 12 temporal fits of the first variant, and the 12 serve fits of each variant: about 140 fits in all, run on Modal containers (`modal/agent_arith_app.py`; job lists `archive/agent_arith3_jobs_cloud1.txt`, `agent_arith3_jobs_cloud_noG.txt`, `agent_arith3_jobs_cloud_temp.txt`, `agent_arith3_jobs_cloud_serve.txt`). A fit needs about 10 GB at the serve step and runs minutes; the same fits can be run locally with `archive/agent_arith3_02_fit.py` and `agent_arith3_06_fit_serve.py` (several hours in all on a 15 GB machine); the cost of the rented run is not recorded in total. Earlier rented fits: the p0 rebuild, the stage-1 and re-stack fits, the 12-month corrector and the plan-clock corrector (`docs/PIPELINE.md` section 7).
- **What is provided instead.** The stored serve outputs of those fits are the files `data/interim/agent_arith3_serve/*.npz` (24 files) and the stored piece `agent_arith3_final_noG.parquet`; `archive/reproduce_final.py` uses the piece. They are derived from the competition data and are not in the repository; a reviewer with the competition data regenerates them by running the job lists above (rented or local), after which `archive/agent_arith3_07_build.py build --arm KYQSR-pw-s0-r0 --suffix _noG` rebuilds the piece (its self-test `selftest` reproduces v150 exactly).
- **Ran locally, no paid compute.** The round-two corrector of v150 (40 held-out fits and 12 serve fits), the Istanbul corrector of v156 (about 330 fits of 15 to 35 s each), the published-daily-delay pieces (no fit), the Rome rules and the row-level bounds.
- **Needs external files** (`docs/DATA-SOURCES.md`): the organisers' bucket for the competition data (credentials from the organisers); adsb.lol day archives (multi-GB each, streamed); IEM METAR; OpenStreetMap through Overpass; and, for stages 23 and 24, the portal's daily files `all_pre_departure_delays_2025.csv` and `..._2026.csv` from `https://www.eurocontrol.int/performance/data/download/csv/` (10.5 MB together). Without those two files the best file that can be rebuilt is v152; with them, v156.

Run-to-run variation, as measured: two builds of the same residual corrector that differ only in where they ran differ by about
7 s rms per row; the means over 6 seeds that are served differ by about 5.5 s rms and over 9 seeds by about 4 s; in score terms the
seed noise of the served corrector term is about 0.03 s of RMSE at infinitely many seeds and under 0.1 s for the whole chain. A
CatBoost fit moved a paired comparison by about 0.9 s when the thread count changed between two arms, so the thread counts in the
scripts have to be kept. A rebuild of the upstream fits therefore reproduces the submitted file to that precision, not bit for bit;
the final assembly from the intermediate files is exact.

Templates. Three files are templates with placeholders that the launching script fills in at run time, so they are not valid
Python on their own and will not compile or run directly: `agent_bulkr_kernel_template.py`,
`archive/agent_kg_ltfm_kernel_template.py.tmpl` and `agent_ltreg_kernel_template.py` (filled in by `agent_bulkr_kaggle.py`,
`agent_kg_ltfm_kaggle.py` and `agent_ltreg_kaggle.py` respectively). `agent_ltreg_dryrun.sh` contains one placeholder (`SP=<scratch>`, a scratch directory you
choose); replace it before running.

Conventions: repo root as working directory; `PY=.venv/Scripts/python.exe` (Python 3.14; Modal client in `.venv-modal`,
Python 3.13 images in `modal/`). Most `archive/*.sh` set `PYTHONIOENCODING=utf-8`,
`POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`, `PYTHONPATH=<repo>` and thread caps; copy those. Run every wrapper from the repo root (or set `PRC_ROOT` to it). Many wrappers originally ran under a local job scheduler; in this release those calls go to two pass-through stubs, `archive/run_job.sh` and `archive/job_slots.py`, which accept the original scheduling options and ignore them.
Everything heavy used `archive/capped_run.py --cap-gb N -- <command>` as a memory cap.

## 1. Environment

See Part A, section A.2 and `requirements-final.txt` (pinned). Modal client in `.venv-modal` (Python 3.13 images in `modal/`). Conventions: repo root as working directory; `PY=.venv/Scripts/python.exe`. Most `archive/*.sh` set `PYTHONIOENCODING=utf-8`, `POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage`, `PYTHONPATH=<repo>` and thread caps; copy those.

## 2. Ordered steps

### 2.1 Competition data (needs organiser access)
```bash
export PRC_TEAM_NAME=<team>  PRC_BUCKET_CREDENTIALS=<path outside the repo>
python -m prc.data list && python -m prc.data pull      # -> data/raw/training_2025-*.parquet, ranking.parquet, submitting.parquet
python -m prc.schema data/raw
```

### 2.2 External data (see docs/DATA-SOURCES.md)
```bash
python -m prc.weather fetch                                  # IEM METAR -> data/raw/weather/<ICAO>.csv  (2025-01-01..2026-08-01)
.venv/Scripts/python.exe archive/fetch_osm_stands.py         # Overpass -> data/external/osm/stands.parquet
.venv/Scripts/python.exe archive/fetch_osm_ways.py           # Overpass -> data/external/osm/ways.parquet (if the osm-ways columns are kept, see gaps)
python -m prc.adsb extract --dates <comma list> --out data/adsb/points   # adsb.lol days; the lists are in scripts/extract_*_days.sh and agent_md_dumps.sh
```
Days actually used: 354 days of 2025, plus all of 2026-01 and 2026-07. The exact 354-day list is
`docs/lists/calib_days_264_octnov.txt` (264 days) plus `docs/lists/agent_md_newdays.txt` (116 more days); these are copies of lists that otherwise sit in the git-ignored `data/interim/`.

### 2.3 Stage 1, bulk base (v94)
```bash
.venv/Scripts/python.exe -u -m prc.final --seeds 5 --iterations 2500 --threads 6 --noplan-seeds 20 --lgb-seeds 10 --d3-seeds 3 \
    --save-members data/interim/v93_members.npz $COMMON --keep "$KEEP" --team jolly-lobster --version 93
```
`$COMMON` and `$KEEP` are defined inside `archive/build_v93model.sh` (it also prints the model smoke test); v94 =
that model plus the member delta of `archive/auto_v94.sh` / `member_delta.py`. Fold p0 (month-pair-out): `prc.crossval`,
`run_members_fold.sh`, Modal `modal/agent_bulkr_bulk.py` (`python -m modal run modal/agent_bulkr_bulk.py ...`).

### 2.4 Stage 2, ADS-B stage
```bash
.venv/Scripts/python.exe archive/adsb_apply.py --in submissions/<base>.parquet --out submissions/<out>.parquet ...   # see its docstring
bash run_adsb_stage1.sh ; bash archive/run_adsb_stage2.sh      # calibration (earlier rounds: run_adsb_round*.sh)
```
Stage-calibration on the 264-day list and the bigger S1/E9S stacks: `modal/stage1.py`, `agent_bulkr_stage1.py`,
`scripts/agent_stagecap_*`. LIRF pre-rules: `archive/apply_rome_dayearly.py`, `apply_rome_plan_gate.py`.

### 2.5 Stage 3, re-stack and the post chain
```bash
bash archive/build_combo.sh 1 est_v2 cap,evt,trk OUT            # T=1 taxi-phase world; the base re-stack build
bash archive/agent_stagecap_bxbuild.sh                          # E9S build (env SC_TRAIN, SC_P105, SC_PFX required)
# post chain, in this order (copied from the chain() function of agent_bulkr_build.sh):
python archive/apply_gap_clamp.py --in A --out B
python archive/apply_lobt_box.py --hard-binding-blend 0.5 --in B --out C
python archive/apply_min_floor.py --in C --out D
python archive/apply_v35_rules.py --in D --out E
python archive/apply_upper_bound.py --in E --out F
python archive/apply_hidden_reading.py --in F --out G
python archive/apply_eddm_january.py --in G --out H            # removed again in v144 (stage 17): the files of v143 and earlier carry it
```
Fold re-stack fits run on Modal (`modal/agent_md_*`, `agent_bulkr_restack.py`, `agent_sb2_app.py`).

### 2.6 Stages 5-7, blend, LTFM refits, small correctors
```bash
# neighbour blend: archive/agent_blend_run.sh, agent_blend_03_build.py   (v112)
# LTFM seq refit (v113): archive/agent_kg_ltfm_kaggle.py -> Kaggle kernel, then agent_kg_ltfm_final.py / agent_kg_ltfm_delta.py
# LTFM snow refit (v117): archive/agent_ltreg_buildall.sh, agent_ltreg_kaggle.py, agent_ltreg_final.py
# small correctors M1-M3: archive/agent_bndl_01_bundle.py, agent_bndl_02_serve.py, agent_bndl_03_foldnpz.py
# silent-start nudge: scripts/agent_nxt_01..03; meta NNLS: scripts/agent_meta_*; the combined file is `data/interim/agent_combo_v125_final.parquet` (the script that merged the nudge and the blend into it is not identified: UNVERIFIED)
```

### 2.7 Stage 8, residual corrector (bigc)
```bash
$SYS modal/agent_bigc_app.py upload                             # fold matrix + extras -> Modal volume prc-stage1:/bigc/in (~1 GB)
$SYS -m modal run --detach modal/agent_bigc_app.py::launch --cfgs C4ff7V4 --seeds 0,1,2   # config name per the staged file
$SYS modal/agent_bigc_app.py fetch                              # -> data/interim/agent_bigc_v2_r1x_s{seed}_{cfg}.npz
.venv/Scripts/python.exe archive/agent_bigc_v2_03_judge.py r1x 0 <cfg> _<cfg>      # gate
.venv/Scripts/python.exe archive/agent_bigc_v2_36_serve_prep.py ; $SYS -m modal run modal/agent_bigc_app.py::launch_serve ...
.venv/Scripts/python.exe archive/agent_bigc_v2_37_serve_assemble.py                # -> serving parquet on the previous best
```
Inputs for the corrector: `archive/agent_bigc_01_build.py` (+ `01b_dropbx.py`) fold matrix, `agent_bigc_v2_15_keys.py`
keys, `agent_bigc_te.py` key statistics, variant views via `scripts/agent_vfac_*` (`modal/agent_vfac_app.py`) and
`agent_vfac_03_assemble.py`. Final guards before an upload: `archive/agent_smallbundle_06_check.py <new> <prior>`
(ids, row count, correlation > 0.99, |mean shift| < 5 s, no negatives) and `archive/check_submission.py`.

### 2.8 Stages 9 to 25 and the assembly of the v156 file
The Istanbul views corrector, the tail component, the Rome and Paris rows, the corrector refit, the plan-clock distance columns of the
corrector, the Rome with-plan mixture, the Munich removal with the Amsterdam plan-clock term, the calendar condition, the v146 bounds, the Rome hard-binding rule, the arithmetic-column correctors, the published-daily-delay pieces and the Istanbul corrector are built by the scripts named in
`docs/BUILD-TRACE.md` section 1. The last steps combine files of earlier stages; with the component files in `data/interim/`:
```bash
.venv/Scripts/python.exe archive/agent_bundle_v133.py     # v132 + Rome no-plan mixture (259 rows) + callsign reach (11 rows)
.venv/Scripts/python.exe archive/agent_bundle_v136.py     # v135 + hard rule for five hidden-plan rows
.venv/Scripts/python.exe archive/agent_bundle_v137.py     # v136 + Paris duplicate rule (16) + callsign cap (6)
.venv/Scripts/python.exe archive/agent_v14z_build.py 141 137 agent_mega_final agent_mega_no4_final   # v141 = v137 + (mega_final - mega_no4_final)
.venv/Scripts/python.exe archive/agent_bundle_v142.py     # v142 = v141 + small pieces (bounds last, plan-gate weights, Istanbul dose, 12 serve seeds)
.venv/Scripts/python.exe archive/agent_bundle_v143.py     # v143 = v142 + plan-clock distance corrector + Rome with-plan mixture (uploaded together)
.venv/Scripts/python.exe archive/agent_bundle_v144_v145.py   # v144 = v143 + Munich shift removed + Amsterdam plan-clock term; v145 = v144 + Rome calendar condition (one row)
.venv/Scripts/python.exe archive/agent_bundle_v146.py     # v146 = v145 + six row-level bounds (51 rows) from the piece files below
.venv/Scripts/python.exe archive/agent_romenear_05_build_item1.py   # the v148 piece (agent_romenear_hardbind_final = v146 + 14 rows)
.venv/Scripts/python.exe archive/agent_bundle_v150_v151.py 150 data/interim/agent_bundle_v148_final.parquet   # v150 = v148 + arithmetic-column corrector (189,076 rows)
.venv/Scripts/python.exe archive/agent_arith3_07_build.py build --arm KYQSR-pw-s0-r0 --suffix _noG   # v152 file (needs the 12 serve fits)
.venv/Scripts/python.exe archive/agent_bundle_dayeq.py data/interim/agent_bundle_v152_final.parquet 153 154 155   # v153, v154, v155 from the two daily-delay pieces
.venv/Scripts/python.exe archive/agent_bundle_v156_v157.py   # v156 = v155 + Istanbul corrector (also writes v157, which lost)
```
The two daily-delay pieces come from `archive/agent_dayeq_01_identity.py` to `agent_dayeq_06_dayearly.py` (`agent_dayeq_04_level.py` writes `agent_dayeq_level_final`, `agent_dayeq_05_rome_types.py` writes `agent_dayeq_types_final`) and read the two daily files named in section 0.3; the Istanbul piece comes from `archive/agent_istnext_01_pool.py` to `agent_istnext_05_build.py`; the arithmetic-column pieces from `scripts/agent_arith_*.py` and `agent_arith3_*.py`. The assembly in `archive/reproduce_final.py` reproduces v148 to v156 from those stored pieces.
One step rebuilds the six piece files and the v146 file: `bash archive/build_v146_pieces.sh` (from the repository root; each step runs under
`archive/capped_run.py --cap-gb 3`, `NOCAP=1` runs without the cap). It writes the row extracts first, then the piece builders, then `agent_bundle_v146.py`,
and prints the rows each piece moves (13, 4, 1, 29, 10 and 2; 51 rows in all; the pieces overlap on 6 rows, where the value farthest from v145 is used; peak commit 3.0 GB at its largest step, about 16 s in all on the build machine). It needs `data/raw/`,
`data/interim/agent_bundle_v145_final.parquet` and, for `agent_logicB_10_hidden.py`, `data/interim/agent_detrules_frame.parquet` and
`agent_detrules_cs.parquet`. The piece files and the scripts that write them (the other scripts of the same families are the censuses and
counts that the 2025 numbers of `docs/ROW-RULES.md` come from):

| Piece file in `data/interim/` | Built by | Rule (`docs/ROW-RULES.md`) |
|---|---|---|
| `agent_logicC_LTFM_noplan-early-bound_final` | `archive/agent_logicC_10_ltfm_noplan_bound.py` (inputs: `agent_logicC_01_extract.py`, `agent_logicC_02_types.py`, `agent_logicC_03_ltfm_snow.py`) | Istanbul no-plan early bound, 13 rows |
| `agent_logicC_LEMD_ceiling3606_final` | `archive/agent_logicC_12_madrid_ceiling.py` (counts: `agent_logicC_05_bounds.py`, `agent_logicC_06_counts.py`) | Madrid one-hour ceiling, 4 rows |
| `agent_logic_planband_final`, `agent_logic_bounds_final` | `archive/agent_logic_06_build.py` (inputs: `agent_logic_01_extract.py` to `agent_logic_05_hiddenplan.py`) | Rome with-plan day-early band, 1 row; bounds re-applied last, 29 rows |
| `agent_logicB_window_final`, `agent_logicB_hiddenlower_final` | `archive/agent_logicB_20_build.py` (inputs: `agent_logicB_00_schema.py` to `agent_logicB_17_link.py`; checks `agent_logicB_21_gapchk.py`, `agent_logicB_22_verify.py`) | exact plan-window edge, 10 rows; hidden-plan lower bound, 2 rows |

These builders read compact per-row extracts of the 2025 and 2026 departures that the extract scripts write into a git-ignored local folder named `logic_rows/` (its location is set by the extract scripts)
(`agent_logic_01_extract.py` and `agent_logic_05_hiddenplan.py`; `agent_logicB_01_extract.py` and `agent_logicB_10_hidden.py`; `agent_logicC_01_extract.py` and
`agent_logicC_02_types.py`); `archive/build_v146_pieces.sh` runs them first. The `inputs:` lists in the table name the scripts of each family; only the extracts above are needed to rebuild.

The Munich removal and the Amsterdam term: `archive/agent_apt_EDDM_01_level.py` to `agent_apt_EDDM_07_build.py` write `agent_apt_EDDM_final`;
`archive/agent_pcdx_01_judge.py`, `agent_pcdx_02_eham.py`, `agent_pcdx_03_build.py` write `agent_pcdx_final`. The calendar rule of v145 has
no piece file: `agent_bundle_v144_v145.py` reads the ranking file's schedule, take-off and plan columns.
The corrector term with the plan-clock distance columns comes from `archive/agent_pcd_09_refit12.py` and `modal/agent_pcd_app.py` (rented
CPU; about USD 2.5, an estimate, for the month fits and the 12 serve fits).
The v141 delta is the change of the residual-corrector term at seven airports; the roles of the three files are in
`docs/BUILD-TRACE.md` section 2. `archive/reproduce_final.py` performs the steps from v133 to v156 in one go and compares the result with the v156 file (section 0).
The v132 merge (Istanbul rows and the nine other airports) was a command that was not saved as a script; its result is checked in
BUILD-TRACE by exact equality with its components, and the file it produced is the starting point of `reproduce_final.py`.

### 2.9 Submission
`python archive/ship_helpers.py upload submissions/<file>.parquet` (needs the team bucket credentials), or `prc.submit` for
the local rule check only. The file that `reproduce_final.py` re-assembles is `submissions/jolly-lobster_v156.parquet` (v156; see section 2.8); the last upload is v162.

## 3. Gaps (steps that exist only as ad-hoc commands or hand-edited state)
1. **No single driver for the lineage** (`final/run_all.sh` is the driver of the pipeline of Part A). ~1,500 files in `archive/` were written as separate experiments, each a COPY of its predecessor (only the scripts the documented chain calls, transitively, are shipped in this release) ("COPY of agent_X.sh adding ONLY ..."). The lineage of the uploaded files is a chain of
   "this build on top of that parquet", not a pipeline run from raw data.
2. **Intermediate artefacts are not in the repo** and are large: `data/interim/` (served parquets, fold `.npz`
   composites such as `agent_bulkr_fold_SA_s0.npz`, `agent_bndl_members.npz`, `agent_bigc_fold.npz`,
   `agent_vfac_extra.npz`, keys `.npz`). The scripts cite them by path; a rebuilder must regenerate each.
3. **Stage 1 shell variables.** `$COMMON` and `$KEEP` for the v93/v94 fit are inside the shell wrapper, with ~80 feature
   names listed by hand (`build_v92model.sh`); the exact v94 command is UNVERIFIED.
4. **Paths and job wrappers.** The original scripts hard-coded absolute repo paths and called a local job scheduler. In this release the paths are relative or `$PRC_ROOT` and the scheduler calls go to the stubs named above; `archive/capped_run.py --cap-gb N` (a memory cap) was sized for the original machine, so adjust N.
5. **Modal and Kaggle state.** Fits run in containers against a Modal volume `prc-stage1` and Kaggle private datasets
   (`prc-ltfm-kg`, `prc-ltreg-kg`); their contents and the exact `launch` arguments of the final runs are not recorded in
   this repository. Kaggle/Modal accounts are private.
6. **Calibration day lists.** Scripts read `data/interim/calib_days_264_octnov.txt` and `agent_md_newdays.txt` (git-ignored
   directory); copies are in `docs/lists/` (copy them back to `data/interim/`). The two lists hold 264 and 116 days and overlap on
   26 days; their union is exactly the 354 days of 2025 extracted on the build machine (checked 2026-10-04), so the 354-day set is
   fully specified by the two files.
7. **Ad-hoc commands.** The 12-month fold extension (`scripts/agent_m12_*`, `modal/agent_m12_*`) and the
   v129/v130 serve assemblies were driven by ad-hoc commands that were not saved as scripts. The v132 merge and
   the E9S delta are not saved as scripts (docs/BUILD-TRACE.md section 3, item 6); the v135 application is now part of
   `archive/reproduce_final.py`.
8. **Randomness.** Seeds are set per script; thread count matters (a CatBoost thread-count nondeterminism of ~0.9 s on a
   paired delta was observed). Served terms are means over 3 to 9 seeds; two builds of the same corrector differ by about 7 s rms
   per row from seed noise (about 4 s for a 9-seed mean). Bit-exact reproduction is not expected, a close one is.
9. A Python 3.14 + Modal 3.13 split (versions pinned in `requirements-final.txt`).
10. **OSM ways**: which corrector columns need `ways.parquet` is UNVERIFIED (docs/DATA-SOURCES.md 2.4).
11. **Data availability.** The competition files need organiser-issued bucket credentials; adsb.lol day archives are
    multi-GB per day and streamed; Overpass results change over time (OSM stands/ways are a 2026-09-17/20 snapshot).
