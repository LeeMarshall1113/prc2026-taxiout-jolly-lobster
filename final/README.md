# final/ : scripts for the final phase (January, February, June, July 2026)

One blind submission is due 2026-10-14 10:00 UTC: `data/raw/final_submitting.parquet` is the template
(`MVT_ID_mvt`, `TAXITIME_SEC_mvt`, 670,790 departures) and `data/raw/final_ranking.parquet` holds the inputs.
The stages, the three layers and the commands: `docs/PIPELINE.md` Part I and `docs/REPRODUCE.md` Part A.

Rules the scripts follow: they take the input file and the month set as arguments, hard-code no row counts, ids or flight
strings (except `31_scored_row_fixes.py`, layer 3, off in the submitted file), never read the label of a 2026 departure, and write only to `data/final/` (plus logs and `data/adsb/points`).
Existing modules in `prc/` and `archive/` are imported or copied, never edited. A departure's month is the UTC month of its
take-off time.

Run order (files are numbered):

| step | file | what it does | output |
|---|---|---|---|
| 00 | `00_fallback.py` | Fallback file F0. January and July rows = the values of the last uploaded file (`submissions/jolly-lobster_v162.parquet`) by `MVT_ID`; February and June rows = airport x calendar-month mean taxi time of the 2025 training labels. Template row order and dtype. | `data/final/F0.parquet` |
| check | `check_file.py FILE` | Label-free checks every candidate must pass: row count, ids equal to the template and unique, no nulls, no negatives, template integer type (these fail the run); per airport-month mean and median against the 2025 twelve-month range, plan-window violations, gap-clamp violations (reported). | prints; exit 1 on a hard failure |
| 05 | `05_adsb_download.sh` | ADS-B day files for 2026-02-01..28 and 2026-06-01..30 (58 days, about 153 GB streamed, 0.9 GB kept), adapted from `archive/run_adsb_stage2.sh`. Runs detached under a memory-limited job runner (3 GB). Resumable; retries missing days once. | `data/adsb/points/2026-02-*.parquet`, `2026-06-*.parquet`; `logs/adsb_final_download.log` |
| 10 | `10_base.py` | Parameterised copy of `prc/final.py` main(): the v92 base model refitted on all of 2025, serving the chosen months from a chosen ranking file. Features are built one calendar month at a time; fitted members are cached so a crash resumes; outputs go to `data/final/`. `--dry-run` builds the matrices and stops. | `data/final/<tag>_base.parquet` |
| 10 | `10_base.sh` | Launcher for J2 under a memory-limited job runner: smoke, refit (`--tag J2`), the three LightGBM member fits, `archive/member_delta.py` (mode `set:lgb_routed,lgb_extratrees`, i.e. the v94 base). Runs detached with 9 to 10 GB. | `data/final/J2_base.parquet`, `J2_members.npz`, `J2_v94model.parquet`; `logs/final_10_base.log` |
| (shared) | `fcommon.py` | Paths, template and ranking readers, training-label statistics. | |

How to launch the two long jobs (each returns at once):

    bash final/05_adsb_download.sh
    bash final/10_base.sh



## ADS-B stage and re-stack (stages 20 to 22)

Each stage runs on a month set (default February and June) and writes `data/final/<tag>_*`. A shipped script that reads
`data/raw/ranking.parquet` by name runs unchanged through `_run_as_ranking.py`, which points that name at the month frame
written by `_mkrank.py` (the ranking file with the departure rows of the chosen months and all arrival rows). Jobs over 1 GB
go through `2x_run.sh` (a memory-limited job runner: job lock, 3 GB commit cap).

| step | file | what it does | output |
|---|---|---|---|
| 20 | `20_adsb_dumps.py` | Per-departure ADS-B dumps for the chosen months, from the day files: `_dump_rows.py` (off-block estimates, anchors, track shape; from `agent_restack3_01_dump.py`), `_dump_ff.py` (first ground fix; from `agent_backx_01_dump.py`), `_dump_evt.py` (push-back events; from `agent_monster2_01_events.py`), `_dump_seq.py` (take-off sequence; from `agent_smallbundle_01_seqfeat.py`), `_dump_bx.py` (back-extrapolation estimates fitted on 2025; from `agent_backx_02_estimate.py`). Prints rows per airport-month and the share of departures with a surface track. | `<tag>_rows/_ff/_evt/_seq/_bx.parquet` (default tag `J6`) |
| 21 | `21_adsb_stage.py` | Rome pre-rules on the base, ten stage realisations per stream (103, 105, 105sc) and the Rome-path run, as `merged_chain.sh` and `agent_stagecap_stageserve.sh`. | `<tag>_v<stream>_s1/_gatebag_s1.parquet` (default tag `J5`) |
| 22 | `22_restack.py` | Re-stack serves (pieces a to d), post-chain, Rome-path splice, stream splice, monster rule, airport-wise composite. | `<tag>_piece_<a-d>.parquet`, `<tag>_restack.parquet` (default tag `J7`) |
| cov | `29_adsb_coverage.py` | Ground aircraft per movement by airport and month from the day files. | prints |

    bash final/2x_run.sh "why" final/20_adsb_dumps.py --months 2,6
    python final/_mkrank.py --months 2,6 --out data/final/J5_frame.parquet      # done by 21 itself

## Rule layers and the published daily layer (steps 30 to 33, 38, 39)

Rules are functions `rule(inputs, prediction) -> prediction` with a counter of rows moved per airport and month. They are defined on input
columns, fitted on 2025, and fire on any month. Constants are the shipped ones; jitter allowance 12 s on the plan window (use `--jitter-allow 0`
to reproduce the files of the January/July run). Nothing here reads a 2026 departure label.

| step | file | what it does |
|---|---|---|
| 30 | `30_general_rules.py` | `pre`: Rome day-early bands, Rome plan gate (bagged classifier x regressor mixture; heavy, needs the feature matrix). `post`: gap clamp, plan-clock window with the hard-binding lift, per-airport floor, habitual-copy operators, hidden-plan lower bound, hidden-plan upper bound (flight string and callsign routes), hidden-plan reading. `rows`: Rome late-plan-clock rule, Paris duplicate pairs, Rome calendar condition, Istanbul no-plan bound, Madrid ceiling, Rome plan band, Rome hard-binding window. `last`: gap clamp, exact plan window and hidden-plan lower bound again. `--sample F` takes a random fraction of the departures. |
| 31 | `31_scored_row_fixes.py` | The only file that names flights: six Rome flight strings left at their value from before the daily layer (table with reason and source) and one never-uploaded single-row decision. Off unless `--with-row-fixes`. |
| 32 | `32_published_daily.py` | Day-level calibration of delayed flights from the published daily delay (EDDF EDDM EGLL LEBL LEMD LIRF, Istanbul with the snow stop-day exclusion, Paris) and the Rome record-type posterior. `--on/--off`. Alternatives: `--lam-cap 0.5 --own-bias` (month-agnostic form), `--rule-hedge`, `--update-null-flight`, `--no-edge-guard`; Zurich (`--pieces ...,lszh`) is an approximation of the shipped joint solve and off by default. |
| 33 | `33_daylayer_validation.py` | Validation (a): the month-agnostic form of the day layer, leave-one-month-out on all twelve 2025 months. Design and pass rule: `docs/GENERALISATION.md` section 10. |
| 38 | `38_rules_plumbing_test.py` | Rows moved per rule and airport-month on `data/final/F0.parquet` for February and June, next to the same months of 2025. |
| 39 | `39_rules_repro_test.py` | Each rule or layer applied to the stored file that precedes it in the January/July lineage and compared with the stored file that follows it. |

    bash final/2x_run.sh "why" final/38_rules_plumbing_test.py      # shared job lock, 3 GB commit cap
    python final/30_general_rules.py --pred P.parquet --out Q.parquet --stage post,rows,last
    python final/32_published_daily.py --pred Q.parquet --out R.parquet
    python final/31_scored_row_fixes.py --before Q.parquet --pred R.parquet --out S.parquet --with-row-fixes

## Correctors and Istanbul (stages 40 to 44)

Order: the base chain value of the served rows (re-stack, rules, Istanbul bulk and snow refits) -> `42` (small stages, writes the
value the correctors are served on) -> `40` (serve-side inputs) -> `41` (serve fits) -> assembly of the corrector terms by the
replay of the shipped build scripts (`archive/agent_arith3_07_build.py`, `agent_bundle_v143.py`, `agent_pcdx_03_build.py`).

| step | file | what it does | output |
|---|---|---|---|
| 40 | `40_serve_inputs.py` | Serve-side inputs of the residual correctors for the departures of the chosen months: 140 base columns, 52 push-back/track/map/occupancy/sequence columns, the two variant-difference columns, drift columns, hashed key cells, nine plan-clock distance columns, AOBT_3 flags, take-off times. Parameterised copy of the January/July builders (named in its header); every list, map and mode is read from the stored 2025-side files. `--profile stored` reads the stored January/July tables (reproduction test); `--profile final` reads the J6/J7 files. Rank features are built one calendar month at a time and cached. | `data/final/<tag>_serve/` |
| 41 | `41_corrector_serve.py` | `stage` links the 2025-side files next to the serve-side arrays; `arith` (round-three arm KYQSR, 845 rounds, 12 seeds), `pcd` (plan-clock corrector C4ff7V5, 775 rounds, 12 seeds) and `v5` (the earlier 12-month corrector, 676 rounds, 3 seeds for the tail) run the shipped fit code on the staged inputs; `check` compares the staged arrays with the stored January/July files; `summary` prints per airport-month means. Cloud launcher: `final_corrector_app.py` (not launched). | `data/final/<tag>_stage/out/` |
| 42 | `42_small_stages.py` | M1, M2, M3, M4 and the tail component (copy of `agent_bndl_02_serve.py` and `agent_tailc_03_serve.py`); nudge and meta stay zero. | `data/final/J8_deltas.npz`, `J8_pv.parquet`, `J8_small.parquet` |
| 43 | `43_istanbul.py` | Plan and preflight of the Istanbul path (bulk and snow refits, views members and corrector, Istanbul corrector): hours, RAM, cost, missing inputs. Nothing is fitted. | prints |
| 44 | `44_support_check.py` | `snow`: snow hours and the Istanbul snow path on the served months next to the 2025 stop days. `support`: per feature and airport-month, the share of served rows outside the 1st to 99th percentile of the 2025 fold rows of the airport, next to the January/July 2026 serve file. | prints, `data/final/support_check.csv` |
| (shared) | `4x_run.sh` | Memory-limited job runner: job lock and 3 GB cap for one job (`SLOT_GB=10 SLOT_MIN=300 SLOT_ETA=240 bash final/4x_run.sh "why" final/41_corrector_serve.py arith ...` for the real fits). | |

Reproduction against the stored January/July files (`logs/final_40_check_full.log`, `final_41_check.log`): all rows of
`agent_bigc13_serve_in.npz` (194 columns), the drift columns, all hashed keys, the plan-clock columns and the flags are reproduced
with 0 differing values.

### Outputs the corrector stage (40 to 44) reads from stages 20 to 22 (paths, schemas)

All keyed by `MVT_ID_mvt` (Float64 when cast; one row per matched departure unless noted), month set `--months 2,6`.

| path | writer | schema |
|---|---|---|
| `data/final/J6_rows.parquet` | `20` part rows | 47 off-block columns plus anchors and track shape (`ci_*`, `appear_raw`, `gp_*`, `tf_*`): `day, apt, plan, src, stand, type, hour, mvt_t, block_t(null), prod_kind, prod_t, e_*, P_*, ...` |
| `data/final/J6_trk.parquet` | `20` part trk | id + the 14 anchor + 8 track-shape columns (a view of the rows file) |
| `data/final/J6_evt.parquet` | `20` part evt | push-back event columns (`n_push, t_first, ..., max_stop, ...`, `src, day, apt`) |
| `data/final/J6_seq.parquet` | `20` part seq | 21 `sq_*` columns |
| `data/final/J6_bx.parquet` | `20` part bx | `bx_c, bx_g, tt_g` |
| `data/final/J6_osm.parquet` | `20` part osm | `osm_dc, osm_dff` |
| `data/final/J6_occ.parquet` | `20` part occ | `occ_push` |
| `data/final/J6_census.parquet` | `20` part census | one row per DEPARTURE of the month set: `apt, plan, category, day` (category as in the January/July census file) |
| `data/final/J7_restack.parquet` | `22` | template-format file (`MVT_ID_mvt, TAXITIME_SEC_mvt`) for the departures of the month set: re-stack + post-chain + splices + monster rule, airport-wise composite of pieces a to d. This is the input of the later additive steps (blend, Istanbul, rules); the corrector value `pv` is built from it by those steps. |
| `data/final/J7_piece_<a-d>.parquet`, `J7_sb2_t_<a-d>.parquet` | `22` | the four piece builds; `sb2_t_<pc>` is the same file under the name `40_serve_inputs.py` reads as the reference variant (`VARIANTS['ref']`) |

Not written (needs the 354-day refit serves of the earlier work): the other three variant streams of the 52 view columns,
`J7_bulkr_t_<a-d>.parquet` (bulk-refit re-stack: `agent_md_bxserve.py`/`agent_md_serve.py` with `--new-npz` and the
`agent_md_*_new` rows of the added days, built by `agent_bulkr_build.sh` on the serve outputs), `J7_etrs_t_<a-d>.parquet` and
`J7_etrs_t_<a-d>_avg.parquet` (ExtraTrees re-stack: `agent_etrs_bxserve.py`/`agent_etrs_serve.py`, built by
`agent_etrs_build.sh`). Both need training rows and fold arrays for the added 2025 days and unc/nov caches; without them
`d_NEW`/`d_ET` stay NaN for the pieces concerned.

### Istanbul path

| step | file | what it does |
|---|---|---|
| 43_gen_wrappers | `43_gen_wrappers.py` -> `ist_kg_ltfm_final.py`, `ist_ltreg_final.py` | The shipped Istanbul bulk-refit wrappers (`archive/agent_kg_ltfm_final.py`, `agent_ltreg_final.py`) with their patching of `prc.final` pointed at `final/10_base.py` and the served frame read from `FINAL_RANKING` / `FINAL_MONTHS`. Arms: control (`--ltfm-seq off --seeds 3`), snow k0 (`--seeds 1`), snow k12 (`--seed-offset 1 --seeds 2`). |
| 43_ist_build | `43_ist_build.py` | Istanbul rows = base + (snow - control), regime correction re-applied, plan-window box, rounding. |
| 43_gen_lv | `43_gen_lv.py` -> `ist_lv_members.py` | The views members, serve mode, for the served months. |
| 43_ist_serve_inputs | `43_ist_serve_inputs.py` | Serve halves of the keys, plan-clock, handover and rank-feature files of the Istanbul corrector. |
| 43_gen_ist_serve | `43_gen_ist_serve.py` -> `ist_corrector_serve.py` | The Istanbul corrector with the nine-airport column lists (D1), served on those inputs. |
| launchers | `43_kaggle_prepare.py` (prepare only), `modal/final_istanbul_app.py` (bulk arms and views members) | Not launched by any script. |

**Deliberate simplification: the treatment arm is not run.** The shipped chain applied two Istanbul steps, (sequence arm - control arm) in
v113 and (snow arms - sequence arm) in v137. They telescope: base + (treat - ctrl) + (snow - treat) = base + (snow - ctrl). The build
therefore takes one delta, snow minus control, and skips the sequence-feature "treat" arm (8.5 h on Kaggle). What differs from the
shipped two-step result is one rounding and one application of the plan-window box instead of two; the January/July size of that
difference is printed by `final/_chk_ist_telescope.py` (log `logs/final_43_telescope.log`).

### Notes on stage 22 and the variant streams

- `apply_eddm_january.py` is left out of the post-chain on purpose: it lowers 7,062 Munich January rows by 38 s, and the shipped chain
  removes that shift again at upload v144 (`docs/BUILD-TRACE.md` stage 17; the v144 step adds the 38 s back on exactly those rows).
  The reproduction test of piece a confirms it: the rebuilt piece differs from the stored one on exactly those 7,062 rows, by 38 s,
  and on no other row. It matches no row in February or June.
- Variant streams `J7_bulkr_t_<a-d>` and `J7_etrs_t_<a-d>(_avg)` are NOT built, so `d_NEW` and `d_ET` are NaN for February and June.
  The bulk-refit stream needs the 2026 stage served again on the 354-day calibration (ctrl and S1 streams, 10 realisations plus the
  Rome path each, 22 stage runs: about 4 h at 11 GB with three workers, 22 Modal containers in the original) plus 8 re-stack builds;
  the ExtraTrees stream needs its own 8 or more builds with uncertainty caches. Together well over 3 hours and 11 GB.
  The corrector's 2025 fold has NaN views for months 4, 5, 6 and 12, so NaN at serve is inside what the fitted model has seen.
- The ADS-B stage serving process peaks a little above 3 GB (a test under a 3 GB cap stopped at 3.01 GB), so stage 21 needs the
  larger memory limit; `2x_chain.sh` runs the pre-rules under 3 GB and the realisations after the 3 GB window.
- `2x_chain.sh` (start with `final/2x_chain_start.ps1`) waits for `J2_v94model.parquet`, then runs 21 and 22 unattended; log
  `logs/final_2x_chain.log`.

Measured on January/July with the stored arm outputs (`final/_chk_ist_telescope.py`): the build reproduces the shipped v137 Istanbul step
exactly (45,183 rows changed by the build, 0 differing from v137, max abs 0); one step (snow - ctrl) against the two shipped steps on the same
base differs by at most 1 s, on 11,686 of 47,581 Istanbul rows (rounding).

## Assembly and driver (stages 50 to 52, run_all)

`50_assemble.py` joins the stage outputs into the four-month file, one function per join, in the shipped order; each join asserts the row set
and prints rows, rows changed and mean and rms change per airport-month (a stage that does nothing on February or June rows shows as
`changed 0` with a warning). Order for the months built from stage outputs: base (J2) -> Rome pre-rules and ADS-B stage (inside J5, report
only) -> re-stack with post-chain (J7) -> neighbour blend (not ported) -> Istanbul bulk/snow rows (J3) -> small stages (J8_small) ->
correctors (arithmetic round three at EDDF EDDM EGLL EHAM LEBL LFPG LEMD, plan-clock term at LIRF LSZH, shipped doses; tail at half dose) ->
Istanbul corrector term -> record rules (`rows,last`) -> Rome mixtures (generic months only). January and July (`--carried 1,7`) enter at the
same point from the stored lineage: `agent_bundle_v152_final.parquet` (everything up to and including the round-three correctors, the Rome
mixtures and the rule layers) plus the Istanbul corrector delta (`agent_istnext_final` minus `agent_bundle_v150_final`). Then the variants:
A = plan-clock window last; B = daily layer (`--lam-cap 0.5 --own-bias`) then window; C = B with the named-row fixes (final/31) then window.

| step | file | what it does |
|---|---|---|
| 50 | `50_assemble.py` | The joins above. `--placeholder` replaces missing generic-month inputs and writes only `_ph_*` files; `--sample` uses days 10 to 12 of each month and writes `_sample_*`; `--stop-after-chain` writes `J11_chain.parquet`, the `--base` of stage 42. Full real run writes `final_A/B/C.parquet` and runs `check_file.py` and `51`. January/July only: `J11_jj_{A,B,C}.parquet`. |
| 51 | `51_roundtrip_check.py` | Per airport-month: share of days whose IFR count lies within +-2 of the published count; delay-total residual per departure as z against the 2025 true-stamp identity (report only). |
| 52 | `52_jj_verify.py` | January/July rows against `submissions/jolly-lobster_v162.parquet`, split into day-layer decisions and everything else (shipped-form reconstruction S). |
| run_all | `run_all.py`, `run_all.sh` | Stage table with inputs and outputs, skips up-to-date stages, `--dry-run` prints the plan and missing outputs. |
| launcher | `5x_run.sh` | Memory-limited job runner (job lock, 3 GB cap) for stages 50 to 52. |

    bash final/run_all.sh --dry-run
    bash final/5x_run.sh "assembly" final/50_assemble.py --months 1,2,6,7 --carried 1,7      # builds final_A, final_B, final_C once the inputs exist

Left out of the assembly: the neighbour blend (needs the serve values of about ten older lineages; ablation row C6, -0.087 s), the
silent-start nudge and the NNLS meta layer (stage 42 keeps them at zero; C6, -0.153 s together with the blend share), the Istanbul views
corrector (S4; the views members exist, the corrector fit is not ported), the separate treatment arm (telescoped, see above), the v114 splice
delta, the cell split of upload v162 (flag `--cell-split`, needs a coverage census of the served months), the Zurich day layer (an approximation
behind `--pieces lszh` in final/32), the Munich January shift (removed in v144).

Stop days of the daily layer (final/32 `--stop-days`): default `signature` = a day with a METAR snow or freezing report in the 3 h before take-off whose published
total is more than 3 sigma below the implied total (gate parameters of each airport from the 2025 held-out chain values; sigma_f 5 min, 33 at Rome), at every airport
of the layer; `metar` = the earlier blunt rule (every snow-report day at EDDF EDDM LSZH EHAM, plus the Istanbul signature); `ltfm` = Istanbul only (the January/July
file); `off`. `32 --stop-report` prints, per airport and month, the days the signature, the deficit alone and the METAR rule exclude.

## Published IFR count in the Rome record-type step (`final/36_rome_count.py`, flag `--count-equation` of `32_published_daily.py`)

The published daily count of IFR departures at Rome equals the number of rows filed under the off-block date, so for a no-plan row whose schedule is the evening before its take-off the count
separates a schedule copy (filed on the schedule's date) from a real late push. The flag multiplies each enumerated assignment's minutes weight by the likelihood of the counts (exact
forward-backward over consecutive days, boundary flips and isolated day extras as noise, month-last days and days dropped by the step's own gates carry no factor). Off by default; with the flag
off the step is unchanged (checked by `agent_romecount_gate.py`). It FAILED its pre-written 2025 gate: only a handful of rows are visible to
the count and the noise model was over-confident, so it should stay off. Validation: `agent_romecount_gate.py`, `agent_romecount_exposure.py`; run through `agent_romecount_run.sh`.

### January/July rebuilt from the final-phase input file (tags J5jj, J6jj, J7jj)

Same stages, run for months 1 and 7 on `data/raw/final_ranking.parquet` (its take-off times), base `data/final/J2_v94model.parquet`:
`J6jj_{rows,ff,evt,seq,bx,trk,osm,occ,census}.parquet` (stage 20), `J5jj_*` (stage 21: `J5jj_frame`, `J5jj_base`, `J5jj_s0ship`,
`J5jj_gatebag_s0`, `J5jj_v<103|105|105sc>_adsb_<1-10>`, `_s1`, `_gatebag_s1`), `J7jj_piece_<a-d>`, `J7jj_sb2_t_<a-d>`,
`J7jj_restack.parquet` (stage 22). Same schemas as the February/June files. Driver `2x_jj.sh` (stage 21 with one worker under a 9 GB limit, then 22 under 6 GB);
comparison with the shipped lineage `26_jj_compare.py`. Rome in July runs through the shipped Rome path (the Rome-path stage run), no special case.

### January/July through the same stages (tag jj)

Inputs `J6jj_*`, `J7jj_restack.parquet` (stages 20 to 22). Outputs, all under `data/final/`, none overwriting a February/June file:

| file | stage | content |
|---|---|---|
| `J3jj_ist.parquet` | 43_ist_build | `J7jj_restack` with the Istanbul rows replaced (base + snow - control). **The Istanbul arm outputs in `J3_kaggle/` cover February and June only** (325,949 rows); for January/July the build uses the stored arm outputs of the old frame, `results/preds_ltfmfin_ctrl_s3.parquet`, `preds_lrfin_snow_k0.parquet`, `preds_lrfin_snow_k12.parquet` (same models, inputs without the new +-12 s jitter). Rerunning the three arms on months 1,7 (Modal about 3.5 USD) would remove that. Full chain value of months 1 and 7 (this file is the `J11_chain` of those months). |
| `J8jj_deltas.npz`, `J8jj_pv.parquet`, `J8jj_small.parquet` | 42 | M1-M4, tail `ehat`, the corrector pv, the small-stage value |
| `J9jj_serve/`, `J9jjv5_serve/` | 40 | corrector serve inputs (52 view columns and d_NEW/d_ET empty, as in February/June) |
| `J10jj_stage/out/{arith,pcd,v5}/` | 41 (Modal) | 12 + 12 + 3 serve fits for the rows of months 1 and 7 |
| `jj_corrector_compare.csv` | 45 | new corrector terms against the stored shipped terms, per airport-month |

The assembly reads fixed names (`J3_ist`, `J8_*`, `J10_stage`); for months 1 and 7 point it at the jj files above. Istanbul's own corrector term and the views members were not run.

### All four months through final/ (`final_{A,B,C}_rebuilt`)

`50_assemble_rebuilt.py` joins the stage files of all four months (February and June from the files above, January and July from the jj tags)
into `data/final/final_{A,B,C}_rebuilt.parquet` (`50_assemble.py` with `--carried ""`, the same layer settings); `54_rebuilt_compare.py` compares the
January and July rows with the carried version. The rebuilt file passes every check of `check_file.py` and `51_roundtrip_check.py`. It differs from
the carried January/July rows by 37.5 s rms (January 34.1, July 40.0); the table by airport, the comparison by stage and the reason it is not the
submitted file are in `docs/REPRODUCE.md` section A.7.

### Which months come from where (the submitted file, variant B)

- **February and June**: entirely from `final/` (stages 10, 20 to 22, 40 to 44, 50).
- **January and July**: carried from the stored lineage for the expensive stages. Base predictions: the same refit. ADS-B stage, re-stack and correctors
  (with the 52 view columns, the small blend and nudge layers and Istanbul's own corrector term): the stored results of the earlier runs that served those
  months (`agent_bundle_v152_final.parquet` plus the Istanbul corrector delta, see Assembly above). Rule layers and the published-daily layer: rebuilt by
  `final/` on top. The reduced archive of the old scripts (`archive/` in the release tree) ships for that reason.
- **Istanbul's own corrector term** is in the submitted file only inside the carried January and July lineage. It is not in February or June, and not in the
  all-four-months rebuild.

