# Build trace: from raw data to the last uploaded file

This page is the January/July lineage: the shortest route a reader can follow from the raw inputs to the last uploaded file. The pipeline that ports it to all four months is `final/` (`docs/PIPELINE.md` Part I, `final/README.md`). It lists every stage in the
order the last uploaded file passes through it, which script builds it, what goes in, where it was fitted and what comes out, and it
writes down how the last versions of the file were assembled from files of earlier stages (`archive/reproduce_final.py` performs
that assembly and checks it). What each stage is for: `docs/PIPELINE.md`. How much each stage added and in which upload it first
shipped (the ablation table): `docs/PIPELINE.md` section 2. The rules that act on few rows: `docs/ROW-RULES.md`. Why some pieces
are specific to an airport or a month: `docs/GENERALISATION.md`. Commands, gaps and a reviewer's checks: `docs/REPRODUCE.md`.
Every upload with its predicted and realised result: `docs/SUBMISSION-LOG.md`.

**Which months of the submitted file this lineage is.** February and June of the submitted file come entirely from `final/`. January and July are
carried from this lineage for the expensive stages: the base predictions are those of the same refit; the ADS-B stage, the re-stack and the correctors (stages 2, 3 and 8 to 22
below, with their 52 view columns, the small blend and nudge layers of stage 7 and Istanbul's own corrector term of stage 25) are the stored results of the runs
that served those months; the rule layers and the published-daily layer are rebuilt by `final/` on top. Istanbul's own corrector term is therefore in the submitted file
only inside this carried lineage, not in February or June. `final/` also ran January and July through all of its stages (a demonstration that the pipeline runs end to
end; it differs from the carried version by 37.5 s rms and leaves out the pieces above): `docs/REPRODUCE.md` section A.7.

The best scored upload is **v162** (leaderboard RMSE 213.2898 s on 2026-10-08 UTC; the team stopped uploading that day, when the final phase was announced). Stages 1 to 25 build the file of upload v156 (213.7101 s), the last file that `archive/reproduce_final.py` assembles; stages 26 to 29 (uploads v158 to v162) are applied on top of it. **Without the steps that use published figures about the scored months (stages 23, 24, 27, 28 and 29) the best file is v152, 216.2405 s.** The file of v156 is the file v137 plus the deltas of stages 13 to 25: the corrector refit at seven airports (stage 13, upload v141), the small
pieces of upload v142 (stage 14), uploaded together in v143 the plan-clock distance inputs of the corrector (stage 15) and the
Rome with-plan copy mixture (stage 16), then the removal of the Munich January level shift together with Amsterdam on the plan-clock term (stage 17, v144), the calendar condition on
the Rome day-early band rule (stage 18, v145), six row-level bounds (stage 19, v146), the Rome hard-binding rule (stage 20, v148), the corrector with 52 and then 45 further arithmetic columns (stages 21 and 22, v150 and v152), the day-level calibration and the Rome record types from the published daily delay (stages 23 and 24, v153 to v155) and the Istanbul corrector (stage 25, v156). Stage 26 re-applies the plan-clock window (v158) and stages 27 to 29 are the later day-level calibrations and the cell split (v159 to v162). Every experiment that lost on the leaderboard (v138, v139, v140, v147, v149, v151 and v157) is not part of it as a file; pieces of v139 were uploaded again in v142 and the v134 mixture was rebuilt for v143 (`docs/ROW-RULES.md`).

How the files are named. Intermediate files live in `data/interim/` and are not committed (they are large and are derived from
the competition data, which is not redistributed). The uploaded files (for example `submissions/jolly-lobster_v156.parquet`) are data files and are not part of the
public repository; `archive/reproduce_final.py` assembles them from the stored intermediate files. "Lineage" below means the chain of such files, each built on the previous one.

Conventions. "Where fitted": LOCAL = one CPU workstation; KAGGLE = free CPU kernels; MODAL = rented CPU containers (paid).
Tier: CORE = the core model, SPECIFIC = competition-specific handling, RECONCILIATION = the step that uses published figures about the scored months (`docs/PIPELINE.md` sections 1 and 5, and Part I for the pipeline of `final/`). "Ablation row" points to the
row of the ablation table in `docs/PIPELINE.md` section 2.

## 1. The stages in order

| # | Stage | Tier | Scripts | Inputs | Fitted where | Output | Ablation row |
|---|---|---|---|---|---|---|---|
| 1 | Base model blend: gradient-boosted members routed by airport, a separate no-plan specialist | CORE | `prc/features.py`, `prc/final.py`, `prc/crossval.py`, `prc/members.py`, `archive/build_v93model.sh`, `archive/auto_v94.sh`, `archive/member_delta.py`, `archive/merge_members_v95.py` | the twelve 2025 training files, METAR | final fit LOCAL; fold fits KAGGLE and MODAL (`modal/agent_bulkr_bulk.py`) | one base prediction per departure (`p0`) | C1 |
| 2 | ADS-B stage: off-block and push-back estimated from ground tracks where the track is good, with the Rome pre-rules (`apply_rome_dayearly.py`, `apply_rome_plan_gate.py`, SPECIFIC) applied before it | CORE (pre-rules SPECIFIC) | `prc/adsb.py`, `prc/adsb_match.py`, `archive/adsb_apply.py`, `archive/agent_merged_apply.py`, `archive/merged_chain.sh`, `archive/agent_stagecap_bxbuild.sh`, `modal/stage1.py` | stage 1 output; adsb.lol day extracts (354 days of 2025, January and July 2026); OSM stand positions | calibration MODAL (leave-one-day-out on the 264-day list); serving LOCAL (about 2 h, 11 GB) | stage prediction per departure | C2, S1 |
| 3 | Pooled LightGBM re-stack on the stage output, with a per-airport splice of its output at five airports | CORE | `archive/agent_restack5_bxserve.py`, `archive/agent_stagecap_bxbuild.sh`, `archive/build_combo.sh`, `modal/agent_md_restack.py`, `modal/agent_bulkr_restack.py`, `modal/agent_sb2_app.py` | stage 2 output, ADS-B event, track and back-extrapolation columns | fits MODAL; assembly LOCAL | re-stack prediction; at EDDF, EDDM, EHAM, LEBL, LSZH the larger-stage re-stack output is spliced in | C3 |
| 4 | Post-chain rules, in this order: gap clamp, plan-clock window (LOBT box with hard-binding lift), per-airport floor, Rome habitual-copy operators and hidden-plan lower bound, hidden-plan upper bound, hidden-plan reading, Munich January shift | window and clamp CORE; the rest SPECIFIC | `archive/apply_gap_clamp.py`, `apply_lobt_box.py`, `apply_min_floor.py`, `apply_v35_rules.py`, `apply_upper_bound.py`, `apply_hidden_reading.py`, `apply_eddm_january.py` | stage 3 output, 2025 labels (constants only), the flight-plan clocks | LOCAL | prediction clipped or moved on a few rows (counts in section 1a) | C5, S2, S3 |
| 5 | Neighbour blend: per-airport non-negative blend over older lineages | CORE (generic addition) | `archive/agent_blend_run.sh`, `archive/agent_blend_03_build.py` | stage 3 and 4 output, member files of older lineages | LOCAL | blended prediction | C6 |
| 6 | Istanbul refits: a bulk refit with take-off-sequence features, then a snow-aware refit | SPECIFIC | `archive/agent_kg_ltfm_kaggle.py`, `agent_kg_ltfm_final.py`, `agent_kg_ltfm_delta.py`; `archive/agent_ltreg_buildall.sh`, `agent_ltreg_final.py` | 2025 Istanbul rows, METAR snow columns | KAGGLE (the snow serving fit takes about 8.5 h) | Istanbul rows replaced or shifted | S4 |
| 7 | Small correctors: uplift GBM at half dose (M1), arrival-lateness ridge (M2), matched-row operator offset (M3); a silent-start nudge at three airports; a small per-airport blend toward rebuilt predictions | CORE (generic addition) | `archive/agent_bndl_01_bundle.py`, `agent_bndl_02_serve.py`, `agent_nxt_02_silent_serve.py`, `agent_meta_03_serve.py` | stage 5 and 6 output, 2025 labels | LOCAL, each leave-one-month-out | additive per-row terms | C6 |
| 8 | Leave-one-month-out residual corrector with per-airport dose: LightGBM on the residual, features include out-of-month key statistics and variant views | CORE (the per-airport doses SPECIFIC) | `archive/agent_bigc_te.py`, `agent_bigc_v2_15_keys.py`, `agent_vfac_03_assemble.py`, `agent_bigc_v2_03_judge.py`, `agent_bigc_v2_36_serve_prep.py`, `agent_bigc_v2_37_serve_assemble.py`, `agent_m12_serve_assemble.py`, `agent_m12_v5_assemble.py`; `modal/agent_bigc_app.py`, `modal/agent_bigc12_app.py`, `modal/agent_vfac_app.py` | stage 7 prediction, the fold composite (2025 months), key statistics, variant-view columns, and from v143 the nine plan-clock distance columns (stage 15) | fits MODAL (about $0.04 per fit), preparation and assembly LOCAL | one additive term per row at nine airports | C4, S6 |
| 9 | Istanbul views corrector (members refitted at Istanbul only, quantile views, key statistics), plus stand-handover columns | SPECIFIC | `archive/agent_lv_02_screen.py`, `agent_lv_04_serve.py`, `agent_lv_07_handover.py`, `agent_lv_members.py`; `modal/agent_lv_app.py` | stage 6 Istanbul rows, 2025 Istanbul months | member and quantile fits MODAL (`modal/agent_lv_app.py`); corrector screening and serving fits and assembly LOCAL; none on KAGGLE | additive term on Istanbul rows | S4 |
| 10 | Tail component: a second model of the part of the residual beyond the corrector's +-900 s cut, at half dose | CORE (generic addition) | `archive/agent_tailc_01_fit.py`, `agent_tailc_03_serve.py`, `agent_tailc_04_assemble.py` | stage 8 output and corrector columns, rows with a flight plan | LOCAL | additive term on rows with a plan at nine airports | C6 |
| 11 | Rome no-plan mixture and Rome plan rules: a copy-or-real mixture on mid-gap no-plan rows (two rounds), the callsign-keyed reach of the hidden-plan reading, a hard rule for five hidden-plan rows | SPECIFIC | `archive/agent_rome2_44_build.py`, `agent_rome2_54_build.py`, `agent_fix2_01_rome_callsign.py`, `agent_fix2_03_build.py`, `agent_bundle_v133.py`, `agent_bundle_v136.py` | stage 10 output, Rome 2025 rows, destination arrival records | LOCAL | replaced values on 259 + 11 + 5 rows | S5 |
| 12 | Paris duplicate rule and hidden-plan callsign cap | SPECIFIC | `archive/agent_multipush_19_build.py`, `agent_detrules_30_build.py`, `agent_bundle_v137.py` | stage 11 output, LFPG records, destination arrival records | LOCAL | replaced values on 16 + 6 rows | S5 |
| 13 | Corrector refit at seven airports on the base the corrector is actually served on | CORE (the doses SPECIFIC) | `archive/agent_refit2_06_assemble.py`, `agent_mega_01_deltas.py`, `agent_mega_02_build.py`, `agent_mega_03_refit.py`, `agent_v14z_build.py`, `agent_v14z_chain.sh`; `modal/agent_bigc13_app.py` | stage 12 file (v137), the refit corrector fits | fits MODAL (six serving fits), assembly LOCAL | delta of the corrector term at EDDF, EDDM, EGLL, LEBL, LEMD, LIRF, LSZH | C4 |
| 14 | Small pieces of upload v142: bounds re-applied last, Rome plan-gate extra weight 0.1 (and 0.2 on July with-plan rows not seen on the ground), Istanbul corrector dose 1.0 on non-snow rows, corrector serve term from 12 seeds instead of 6 | bounds and 12 seeds CORE; gate weights and Istanbul dose SPECIFIC | `archive/agent_bundle_v142.py`, `archive/agent_mega_03_refit12.py`; the components come from `agent_mega_*` and `agent_regime_*` builds | the v141 file, the component files named in section 2 | LOCAL; the 12-seed serve fits MODAL | delta on 200,541 rows (the gate weights move 14,502 rows, the other pieces 53,030, the seeds 218,889) | S8 |
| 15 | Plan-clock distance inputs of the residual corrector: nine columns (the prediction minus the taxi time implied by each plan clock, the distance to the schedule-gap clamp, the distance to the nearer plan-window edge, the `AOBT_3` distance split by a label-free default-or-report flag), corrector refitted at the same seven airports and doses with 12 serve seeds. The idea is credited to the public repository https://github.com/EnioAguiar/prc-taxiout-2026 (GPL-3.0); no code was copied | CORE | `archive/agent_pcd_09_refit12.py`, `modal/agent_pcd_app.py`, `archive/agent_bundle_v143.py` | the v142 file, the corrector fits with the new columns | fits MODAL (month fits and 12 serve fits), assembly LOCAL | delta of the corrector term on 218,889 rows (rms 9.5 s) | C7 |
| 16 | Rome with-plan copy mixture, clean two-sided build (a copy-or-real type model times a within-type regressor), dose 0.5 | SPECIFIC | `archive/agent_bundle_v143.py` (component `agent_rome1c_final`) | the v141 file, Rome 2025 plan rows | LOCAL | delta on 11,031 Rome plan rows (rms 79 s, maximum 1,723 s) | S7 |
| 17 | Munich January level shift removed (the 38 s added back on the same 7,062 rows; the rule of stage 4 rested on a comparison with a published figure that excludes flights de-iced after off-block) and Amsterdam moved to the plan-clock term of the corrector (the 31,254 Amsterdam rows take the corrector term with the nine plan-clock columns, which stage 15 had applied at the other six airports of the refit) | SPECIFIC (Munich), CORE (Amsterdam term) | `archive/agent_apt_EDDM_01_level.py` to `agent_apt_EDDM_07_build.py`, `archive/agent_pcdx_01_judge.py`, `agent_pcdx_02_eham.py`, `agent_pcdx_03_build.py`, `archive/agent_bundle_v144_v145.py` | the v143 file | LOCAL | delta on 38,316 rows (Munich 7,062 at +38 s; Amsterdam 31,254, rms 8.2 s; disjoint) | S9, C7 |
| 18 | Calendar condition on the Rome day-early band rule: a no-plan departure with a gap of 50,000 to 100,000 s that took off on the same local day as the schedule takes the schedule gap | SPECIFIC | `archive/agent_bundle_v144_v145.py` | the v144 file, input columns only | LOCAL | one row (65,347 s to 53,520 s) | S10 |
| 19 | Six row-level bounds from the logic audit (Istanbul no-plan early bound, Madrid one-hour ceiling, Rome with-plan day-early band, bounds re-applied last, plan-window edge, hidden-plan lower bound) | bounds observed on 2025, each with its count of exceptions (`docs/ROW-RULES.md`) | `archive/agent_logicC_10_ltfm_noplan_bound.py`, `agent_logicC_12_madrid_ceiling.py`, `agent_logic_06_build.py`, `agent_logicB_20_build.py`, `archive/agent_bundle_v146.py` | the v145 file, input columns only | LOCAL | 51 rows (13 + 4 + 1 + 29 + 10 + 2, six rows moved by two pieces) | S11 |
| 20 | Rome plan rows with a hard-binding plan window served at take-off minus LOBT (14 scored rows) | SPECIFIC | `archive/agent_romenear_01_item1.py`, `agent_romenear_02_item1_eval.py`, `agent_romenear_05_build_item1.py` | the v146 file, Rome plan rows, input columns only | LOCAL | 14 rows (`agent_romenear_hardbind_final.parquet`) | S12 |
| 21 | Residual corrector with 52 arithmetic columns (round two) at six airports | CORE (the airport list and doses SPECIFIC) | `archive/agent_arith_02_fit.py`, `agent_arith_03_judge.py`, `agent_arith_14_fit_serve.py`, `agent_arith_13_build.py`, `agent_bundle_v150_v151.py` | stage 15 corrector columns and fold rows, 2025 labels | LOCAL (held-out fits and 12 serve fits, about six minutes each) | delta on 189,076 rows (rms 11.0 s, maximum 309 s) | C8 |
| 22 | Round three: 45 further arithmetic columns without the view-geometry group, seven airports | CORE (airport list and doses SPECIFIC) | `archive/agent_arith3_02_cols.py`, `agent_arith3_02_fit.py`, `agent_arith3_03_judge.py`, `agent_arith3_04_fit_temp.py`, `agent_arith3_05_servecols.py`, `agent_arith3_06_fit_serve.py`, `agent_arith3_07_build.py` (suffix `_noG`), job lists `agent_arith3_jobs_cloud*.txt`, `agent_v152_auto.sh`; `modal/agent_arith_app.py` | stage 21 columns, 2025 labels | MODAL for the held-out, temporal and serve fits (about 140 fits), LOCAL for judging and the build | file `agent_arith3_final_noG.parquet`: 209,078 rows differ from v150 | C8 |
| 23 | Day-level calibration of delayed flights from the published daily delay (six airports) | RECONCILIATION | `archive/agent_dayeq_01_identity.py`, `agent_dayeq_02_bdate.py`, `agent_dayeq_04_level.py`, `agent_bundle_dayeq.py` | the v150 file, `data/external/eurocontrol/all_pre_departure_delays_{2025,2026}.csv` | LOCAL (no fit beyond one shrinkage per airport) | 139,724 rows (rms 23.5 s, maximum 119 s) | R1 |
| 24 | Rome no-plan record types from the published daily delay | RECONCILIATION | `archive/agent_dayeq_03_resolve.py`, `agent_dayeq_05_rome_types.py`, `agent_dayeq_06_dayearly.py`, `agent_bundle_dayeq.py` | the v150 file, the same daily file | LOCAL | 168 rows (rms 1,861 s, maximum 9,486 s) | R2 |
| 25 | Istanbul corrector with the nine-airport column lists on Istanbul cells | SPECIFIC | `archive/agent_istnext_01_pool.py`, `agent_istnext_02_fit.py`, `agent_istnext_03_judge.py`, `agent_istnext_04_serve.py`, `agent_istnext_05_build.py`, `agent_bundle_v156_v157.py` | the v150 file, 2025 Istanbul rows | LOCAL (about 330 fits) | 42,538 non-snow Istanbul rows (rms 10.7 s) | S13 |
| 26 | Plan-clock window re-applied last | CORE (an observed window, 0 exceptions in 2025) | `archive/agent_bundle_v158.py` | the v156 file, plan columns of the ranking file | LOCAL | 10 rows | S14 |
| 27 | Day-level calibration at Istanbul (snow stop days excluded) and at Zurich (IFR flights) from the published daily delay | RECONCILIATION | `archive/agent_dayeq2_04_ltfm_level.py`, `agent_bundle_v159.py`, `agent_bundle_v160.py` | the v158 and v159 files, the daily file | LOCAL | 33,779 rows (rms 33 s, maximum 155 s) and about 19,900 rows (rms 14 s) | R3, R4 |
| 28 | Day-level calibration at Paris CDG from the published daily delay | RECONCILIATION | `archive/agent_nextpools_04_score.py`, `agent_nextpools_05_build.py` | the v160 file, the daily file | LOCAL | 30,899 rows (rms 16.7 s, maximum 60 s) | R5 |
| 29 | Day shift split by coverage class and lateness bin at five airports (v162; two clauses of its gate waived before upload) | RECONCILIATION | `archive/agent_nextpools_10_slope.py`, `agent_nextpools_11_buildc5.py` | the v161 file, the daily file | LOCAL | 104,752 rows (rms 15.3 s, maximum 190 s) | R6 |

Stages 5 to 29 are all applied after the re-stack. Rows that a rule of stage 4, 11 or 12 sets are protected in later steps:
the later additive terms skip them where the build script says so (the corrector assembly leaves the rule-set rows alone). The
consequence was that a few predictions lay slightly outside a bound that an earlier rule had enforced (section 1a); stage 19 moves those back.

### 1a. Facts read from the code and measured on the chain

Rows moved by each rule of stage 4, measured by running the repository's own rule scripts in the documented order on the stage
output of one piece of the shipped lineage (`data/interim/sb2serve/ta_v105_s1.parquet`, the stage output that supplies Heathrow;
the other pieces differ by a few rows). Every count uses only input columns; no 2026 departure label was read.

| Rule | Scored rows moved | Detail |
|---|---|---|
| Gap clamp | 39 | Rome 35, Heathrow 4 (29 in an early build) |
| Plan-clock window with hard-binding lift | 82, all raised | 86 rows bind hard and 40 of the moved rows are lifted by the hard-binding rule; by airport EDDF 7, EDDM 6, EGLL 6, EHAM 18, LEBL 5, LEMD 10, LFPG 16, LIRF 4, LSZH 6, LTFM 4 |
| Per-airport floor | 1,307 | median +37 s, mean +52 s, maximum +428 s |
| Habitual-copy operators (Rome) and hidden-plan lower bound | 6 and 9 (15 changed) | operators ISR, LAV, MSC, TKJ, TWB |
| Hidden-plan upper bound | 8 lowered of 30 in scope | |
| Hidden-plan reading (Rome) | 18 in scope | measured again on 2026-10-05 from input columns only (no departure label), by the flight-string key of `archive/apply_hidden_reading.py`: 18 scored rows are in scope, and on the stage output all 18 change; the callsign extension of stage 11 adds 11 rows and the callsign cap of stage 12 6 rows. An earlier note counted 3 rows by flight string; that count is superseded by the measurement and is not reproduced. In the last file at the time of that measurement the 18 predictions are within 60 s of the reading on 8 of them (5 in the v141 file), because later additive terms and rules follow the reading |
| Munich January shift (removed in stage 17, v144) | 7,062 | all by -38 s in v143; none in v144 and later |

Measured on 2026-10-05 from input columns only, on the last file of that date and its predecessors (no label read): the Rome plan gate has 25,756
scored plan rows in its scope (a plan row with a schedule gap up to 7,200 s; the build printout said about 26,300, the difference is the
definition of a plan row). The extra gate weights added in v142 move 49 of them by more than 60 s relative to v141; the corrector change
and with-plan mixture of v143 move 1,823 of them by more than 60 s relative to v142. The number moved by the gate itself, relative to
the stage output, needs a re-run of the gate and has not been re-measured; the figure of 25 rows moved by more than 60 s in the upload
log belongs to a variant (extra weight 0.1 on an earlier base), not to the shipped gate. Rome day-early and extreme-gap bands: 6 scored
rows have a schedule gap of 50,000 to 100,000 s in the v143 file (2 in each band; predictions 65,347 to 88,168 s); stage 18 sets one of them to the schedule gap and stage 19 moves the plan row to the plan's own reading.

The v143 file: label-free checks (`archive/reproduce_final.py`) found 21 predictions above the schedule gap among the
rows of the clamp's scope (excess median 71 s, maximum 246 s; Rome 20, Heathrow 1) and 6 plan rows outside the plan-clock window
beyond the 14 s slack (EDDM 2, LTFM 2, LEMD 1, LIRF 1; by 1 to 252 s). On the v146 file the same checks found 0 and 0 (stage 19). On the v156 file the gap-clamp check finds 0 and the plan-window check finds 6 plan rows beyond the 14 s slack (EDDF 1, LEBL 3, LEMD 1, LTFM 1; 15 to 25 s outside the window, so 1 to 11 s beyond the slack), moved there by the corrector terms of stages 21, 22 and 25; they are not re-clamped. For the v141 file the same checks gave 25 (Rome 22, Heathrow 3; median
85 s, maximum 369 s) and 10 (by 2 to 46 s). The additive terms of stages 7 to 16 were applied after the bounds of stage 4 and are not re-clamped. The counts fell between
the v141 and v143 files; the v142 piece "bounds re-applied last" (12 rows) is the only step that re-applies a bound, and the
v143 deltas were not re-clamped; stage 19 re-applies them again on the v146 file. Re-applying the bounds last was tested held out and went out first inside the v139 bundle, which
lost (`docs/SUBMISSION-LOG.md`).

Other facts closed by reading the code and the build scripts:

- **First upload that contained the per-airport floor and the Munich January shift (the shift was removed again in v144): v39 (242.36).** The earliest build script that
  contains the floor is `archive/build_v38.sh` (its output was never uploaded), and `archive/build_v40.sh` names the Munich shift
  "v39's rider" and lists the floor, so both entered with v39. The upload log does not itemise v39, so this is read from the build
  scripts, not from a log line.
- **Istanbul views fits.** The member and quantile fits ran on rented CPU (`modal/agent_lv_app.py`, about $2 for the arms of the
  first round); the LightGBM corrector screening and serving fits and the assembly ran on the local machine
  (`archive/agent_lv_02_screen.py`, `agent_lv_04_serve.py`); none of the views fits ran on Kaggle (Kaggle was used for the Istanbul
  bulk refits of stage 6). Which of the later individual corrector fits was local and which rented is not recorded per fit.
- **Lever B (Rome hidden-plan rows with a late plan clock), 2025 count.** 7 of 7 real taxis with the flight-string key of the
  shipped rule; 8 of 8 with the wider callsign-extended key of a later check. Both are right: they are different keys.

## 2. Exact assembly of the last versions

Several late steps do not re-run the earlier stages. They take the file of the previous upload and either replace the rows that
a component file changed, or add the difference between a component file and a reference file. For each step the three roles are:

- the BASE file: the previous upload (or the stage output the step builds on);
- the COMPONENT file: the same rows after the new component has been applied (built with the component's own script on the base
  or on an equivalent file);
- the REFERENCE file: the same rows without the new component (only in the delta form).

`archive/reproduce_final.py` performs the steps of the table below (v133, v135, v136, v137, v141, v142, v143, v144, v145, v146, v148, v150, v152, v153, v154, v155, v156) from the file of
upload v132 and compares each intermediate result with the stored file of that upload, and the files of v143 to v146, v148, v150 and v152 to v156 also with
`submissions/jolly-lobster_v<V>.parquet`: the maximum absolute difference is 0 on all rows (all ten airports) for each (22 comparisons). Uploads v147, v149, v151 and v157 are not in the chain.
The v135 step, whose application line was never saved as a script, is now one of those steps.

There are three forms. In the REPLACE form the rows where the component differs from the base take the component's value (the
component's script guarantees that its rows are disjoint from every other component's rows, and the assembly asserts it).
In the DELTA form the output is `base + (component - reference)`, rounded to whole seconds and clipped at zero. In the DELTAS form
(v142 to v144) several float deltas, each a component minus its reference, are summed on the base and the sum is rounded once and
clipped at zero; the files are read in id order, so the assembled files from v142 on are in id order. v145 and v146 are rules on input columns: the v145 step reads the schedule, take-off and plan columns of the ranking file; the v146 step takes the value of six piece files (each equal to the v145 file except where its bound applies).

| Step | Form | Base | Component | Reference | Script | Rows |
|---|---|---|---|---|---|---|
| v133 | REPLACE | `agent_bundle_v132_final.parquet` | `agent_rome2_A_final.parquet` (Rome no-plan mixture at half weight) and `agent_fix2_c_final.parquet` (callsign reach) | none | `archive/agent_bundle_v133.py` (components from `agent_rome2_44_build.py`, `agent_fix2_03_build.py`) | 259 + 11 |
| v135 | REPLACE | `agent_bundle_v133_final.parquet` | `agent_rome3_final.parquet` (round 2: new copy indicators and per-bucket weights) | none | component `archive/agent_rome2_54_build.py`; applied by `archive/reproduce_final.py` (the original line was not saved); the result equals the stored `agent_bundle_v135_final.parquet` on every row | the same 259 |
| v136 | REPLACE | `agent_bundle_v135_final.parquet` | `agent_rome3B_final.parquet` (hard rule for five hidden-plan rows, built on v133, so its rows are those where it differs from v133) | none | `archive/agent_bundle_v136.py` | 5 |
| v137 | REPLACE | `agent_bundle_v136_final.parquet` | `agent_multipush_final.parquet` (duplicate rule) and `agent_detrules_final.parquet` (callsign cap) | none | `archive/agent_bundle_v137.py` (components from `agent_multipush_19_build.py`, `agent_detrules_30_build.py`) | 16 + 6 |
| v141 | DELTA | `submissions/jolly-lobster_v137.parquet` | `agent_mega_final.parquet` | `agent_mega_no4_final.parquet` | `archive/agent_v14z_build.py 141 137 agent_mega_final agent_mega_no4_final`, called by `archive/agent_v14z_chain.sh` | about 191,700 |
| v142 | DELTAS | `agent_bundle_v141_final.parquet` | (A) `agent_mega_no4_final.parquet` minus `agent_basefix_final.parquet`: bounds re-applied last, Rome plan-gate extra weight 0.1, Istanbul corrector dose 1.0 (53,030 rows; dropped on the rows B moves); (B) `agent_regime_gatejul_final.parquet` minus the v141 file: plan-gate extra weight 0.2 on July with-plan rows not seen on the ground (14,502 rows); (C) `agent_detrules_final.parquet` minus `agent_bundle_v136_final.parquet` on rows v141 left at the v136 value: hidden-plan upper bound (no row, already in v141); (D) `agent_mega_refit12_delta.npz` minus `agent_mega_refit_delta.npz`: corrector serve term from 12 seeds instead of 6 (218,889 rows) | the references are named with each component | `archive/agent_bundle_v142.py` | 200,541 changed |
| v143 | DELTAS | `agent_bundle_v142_final.parquet` | (E) `agent_pcd_refit12_delta.npz` minus `agent_mega_refit12_delta.npz`: the corrector refitted with the nine plan-clock distance columns, 12 serve seeds, same seven airports and doses (218,889 rows); (F) `agent_rome1c_final.parquet` minus the v141 file: Rome with-plan two-sided copy mixture, dose 0.5 (11,031 rows; 11,031 rows are moved by both E and F) | the references are named with each component | `archive/agent_bundle_v143.py`, `archive/agent_pcd_09_refit12.py`, `modal/agent_pcd_app.py` | 190,957 changed |
| v144 | DELTAS | `agent_bundle_v143_final.parquet` | (H) `agent_apt_EDDM_final.parquet` minus the v143 file: the Munich January shift removed (7,062 rows, +38 s each); (I) `agent_pcdx_final.parquet` minus the v143 file: Amsterdam on the plan-clock corrector term (31,254 rows; the two row sets are disjoint, asserted) | the references are the v143 file | `archive/agent_bundle_v144_v145.py` (components from `agent_apt_EDDM_07_build.py`, `agent_pcdx_03_build.py`) | 38,316 changed |
| v145 | RULE | `agent_bundle_v144_final.parquet` | Rome no-plan departures with a gap of 50,000 to 100,000 s and take-off on the same Europe/Rome day as the schedule take the schedule gap | none | `archive/agent_bundle_v144_v145.py` | 1 |
| v146 | BOUNDS | `agent_bundle_v145_final.parquet` | six piece files: `agent_logicC_LTFM_noplan-early-bound_final`, `agent_logicC_LEMD_ceiling3606_final`, `agent_logic_planband_final`, `agent_logic_bounds_final`, `agent_logicB_window_final`, `agent_logicB_hiddenlower_final` (each equals the v145 file except where its bound applies; where two move a row the value farthest from v145 is taken) | none | `archive/agent_bundle_v146.py` (pieces from `agent_logicC_10_ltfm_noplan_bound.py`, `agent_logicC_12_madrid_ceiling.py`, `agent_logic_06_build.py`, `agent_logicB_20_build.py`) | 51 changed (13 + 4 + 1 + 29 + 10 + 2, six rows moved by two pieces) |
| v148 | PIECE | `agent_bundle_v146_final.parquet` | `agent_romenear_hardbind_final.parquet` minus the v146 file (the piece is v146 plus the rule) | the v146 file | `archive/agent_romenear_05_build_item1.py`; step in `reproduce_final.py` | 14 |
| v150 | PIECE | `agent_bundle_v148_final.parquet` | `agent_arith_final.parquet` minus the v146 file (corrector with 52 arithmetic columns at six airports; the row sets are disjoint from the base's changes, asserted) | the v146 file | `archive/agent_bundle_v150_v151.py 150 <v148 file>` | 189,076 |
| v152 | PIECE | `agent_bundle_v150_final.parquet` | `agent_arith3_final_noG.parquet` (round three without the view-geometry group; equal to v150 outside its seven airports, asserted) | none | `archive/agent_arith3_07_build.py build --suffix _noG`, chosen by `archive/agent_v152_auto.sh` (log `logs/auto_v152.log`) | 209,078 |
| v153 | DELTA | `agent_bundle_v152_final.parquet` | `agent_dayeq_level_final.parquet` minus the v150 file (day-level calibration) | the v150 file | `archive/agent_bundle_dayeq.py` | 139,724 |
| v154 | DELTA | `agent_bundle_v152_final.parquet` | `agent_dayeq_types_final.parquet` minus the v150 file (Rome record types) | the v150 file | `archive/agent_bundle_dayeq.py` | 168 |
| v155 | BOTH | `agent_bundle_v152_final.parquet` | both deltas; on rows in both, the types value (none) | the v150 file | `archive/agent_bundle_dayeq.py` | 139,892 |
| v156 | PIECE | `agent_bundle_v155_final.parquet` | `agent_istnext_final.parquet` minus the v150 file (Istanbul corrector; rows disjoint from the base's changes, asserted) | the v150 file | `archive/agent_bundle_v156_v157.py` | 42,538 |
| v158 | RULE | `agent_bundle_v156_final.parquet` | plan rows outside the plan-clock window set to its edge | none | `archive/agent_bundle_v158.py` (not assembled by `reproduce_final.py`) | 10 |
| v159 | PIECE | `agent_bundle_v158_final.parquet` | `agent_dayeq2_ltfm_level_final.parquet` (Istanbul day-level calibration) | none | `archive/agent_bundle_v159.py` (not assembled by `reproduce_final.py`) | 33,779 |
| v160 | PIECE | `agent_bundle_v159_final.parquet` | `agent_dayeq2_lszh_final.parquet` (Zurich IFR day-level calibration) | none | `archive/agent_bundle_v160.py` (not assembled by `reproduce_final.py`) | about 19,900 |
| v161 and v162 | PIECE | `agent_bundle_v160_final.parquet`, then the v161 file | Paris day-level calibration (`agent_nextpools_lfpg_final.parquet`), then the cell split of the day shift | none | `archive/agent_nextpools_05_build.py`, `agent_nextpools_11_buildc5.py` (not assembled by `reproduce_final.py`) | 30,899 and 104,752 |

How the v141 delta is defined. `agent_mega_final.parquet` and `agent_mega_no4_final.parquet` are two builds on the same base
file (`agent_basefix_final.parquet`, a file that is not part of the lineage of the submission). They differ in exactly one
ingredient, the corrector refit at seven airports (stage 13). Their difference is therefore the change of the corrector term and
nothing else: the term of the refit corrector minus the term of the shipped corrector, per row, at the seven airports, with doses
chosen leave-one-month-out (EDDF 1.25, EDDM 1.25, EGLL 1.0, LEBL 1.0, LEMD 1.0, LIRF 0.75, LSZH 1.25). Added to v137 it replaces
the old term by the new one without carrying along the other differences between the base of those two files and v137. EHAM and
LFPG keep the shipped corrector, so their delta is zero. The build script chose v137 as the base because it was the best scored
file at that point (the base choice is mechanical: the best-scoring earlier upload). The delta is nonzero on 191,726 rows and changes 191,725 predictions of v137 (mean -1.1 s, from -260 to +148 s).

How the v142 and v143 deltas are defined. In each, the component and its reference differ in exactly one ingredient, so the difference
is the change of that ingredient and nothing else: for E the corrector term with the plan-clock distance columns minus the 12-seed
corrector term without them; for F the file with the with-plan mixture minus the v141 file (the Rome plan rows only). The two v143 pieces
were uploaded together and the build does not separate them on the leaderboard. Where a row is moved by both (the 11,031 Rome plan rows of
F), the deltas are summed. Sizes (rows moved, mean, rms of the change) are printed by `archive/reproduce_final.py`: v142 mean shift
+0.096 s, rms 5.4 s; v143 mean shift +0.347 s, rms 16.4 s, maximum move 1,521 s.

Steps before v130 follow the same forms and are listed here for completeness, each with the script that does it:

| Step | Form | Roles and script |
|---|---|---|
| E9S splice (v114) | DELTA | base = the current best file; component = the build with the larger stage and its re-stack output at five airports; reference = the build before the splice; the formula is `max(base + (component - reference), 0)` (builds: `archive/agent_stagecap_bxbuild.sh`; the line that applies the delta is not saved as a script and is not part of the final assembly) |
| neighbour blend (v112) | DELTA | base = the serving build; component = the serving value of each older lineage member; the per-airport weights come from a non-negative fit (`archive/agent_blend_03_build.py`) |
| Istanbul rows (v113, v117) | REPLACE | the Istanbul rows of the base are replaced by the refit rows (`archive/agent_kg_ltfm_final.py`, `agent_ltreg_final.py`); other rows untouched |
| small correctors (v122, v123, v124) | DELTA | additive terms M1, M2, M3, nudge, blend added to the base (`archive/agent_bndl_02_serve.py`, `agent_nxt_02_silent_serve.py`, `agent_meta_03_serve.py`) |
| corrector term replacements (v126 to v129) | DELTA | the old corrector term inside the base is subtracted and the new term added, so the term itself is swapped: `archive/agent_bigc_v2_37_serve_assemble.py` (base file, new term from the serving fits, old term stored as `dp` in an array file) |
| 12-month corrector (v130) and its views (V5) | DELTA | `archive/agent_m12_serve_assemble.py` (new 12-month term minus the 8-month term inside the base), `archive/agent_m12_v5_assemble.py` (V5 term replaces V4) |
| v130 Istanbul rows | REPLACE | the Istanbul rows carry the views corrector (`archive/agent_lv_04_serve.py`); the other rows carry the 12-month corrector |
| v132 | REPLACE on disjoint sets | base = the v130 file; the Istanbul rows come from the file with the stand-handover columns (`agent_lv4r_ltfm_final.parquet`), the nine other airports from the file with the half-dose tail term (`agent_tailc_board.parquet`, `archive/agent_tailc_04_assemble.py`); the line that merges the two is not saved as a script; the result is verified equal to the two components on every row, and the file it produced (`agent_bundle_v132_final.parquet`) is the starting point of `archive/reproduce_final.py` |

Every one of these assembly files was checked against its predecessor before upload with `archive/agent_smallbundle_06_check.py`
(same ids and row count, correlation above 0.99, mean shift under 5 s, no negative values), and the changed rows of each step
were checked to be exactly the rows the step's component moves.

## 3. What cannot be re-run exactly, and why

1. **Seeded fits on rented machines.** The large fits (re-stack, residual corrector, Istanbul views) ran in rented containers.
   Seeds are set in the scripts, but the served terms are means over 3 to 9 seeds whose individual files are not committed; two
   builds of the same corrector that differ only in where they ran differ by about 7 s rms per row (seed noise), and the 9-seed
   mean lowers that to about 4 s. At the level of the score, the seed noise of the served corrector term is about 0.03 s of RMSE at
   infinitely many seeds, and under 0.1 s for the whole chain. A re-run reproduces the result to that precision, not bit for bit. January and July in the submitted file are stored results; a from-raw rebuild differs by 37.5 s rms per row (`docs/REPRODUCE.md` section A.7). The container images pin
   the library versions listed in `docs/REPRODUCE.md`.
2. **Thread count.** A CatBoost fit moved a paired comparison by about 0.9 s when the thread count changed between two arms; the
   thread counts are set in the scripts and have to be kept.
3. **Archive days fetched from replica releases.** The ADS-B inputs are one archive per day, fetched from the public release
   pages of the open ADS-B history and cut down to the points inside the ten airport boxes, on the ground or below 3,000 ft.
   The day lists are in `docs/lists/` (the 264-day list and the 116-day list overlap on 26 days; their union is exactly the 354
   days of 2025 that were extracted, checked on 2026-10-04). The archives are third-party files that can move or be
   re-published, an alternate replica carries 99.99% of the same points, and reception is a property of the month: a fetch made
   later returns the same archive only if the publisher has not changed it.
4. **Rome July 2026 is a feed gap at the source** (very few ground points for the whole month); a re-fetch cannot recover it.
5. **Third-party snapshots.** The OSM stand and way extracts are a snapshot of 2026-09-17 and 2026-09-20 from a public query
   service and can differ on a later fetch.
6. **Intermediate files are not committed.** The lineage files, the fold composites and the term arrays are large and derived
   from data that is not redistributed; a rebuild regenerates them, stage by stage, in the order of section 1. Two assembly
   lines (the v132 merge and the E9S delta) were run as commands and are not saved as scripts; the effect of each is
   verified in section 2 by exact equality with its components. The v135 line is now part of `archive/reproduce_final.py`.
7. **Order of fits.** The final corrector terms were fitted on predictions of earlier stages of the same lineage; a rebuild
   must follow the order of section 1, because a corrector fitted on an older base and served on a newer one is not the same
   model (the stage 13 refit exists because of exactly this).

Gaps that needed only a sentence and are now closed (the items that remain unestablished are listed after this sentence): the day lists (item 3) are complete; the uploaded files are data files, not part of the repository;
the bases of v141 to v156 and their deltas and rules are defined by the table in section 2 and are re-run by `archive/reproduce_final.py`.

Still unestablished: (a) the exact command of the stage 1 fit of the January/July lineage (the final pipeline refits it with `final/10_base.sh`); (b) whether the first upload with the floor and the (since removed) Munich shift was v39, which is read from build scripts and not from a log line (section 1a); (c) which individual later Istanbul corrector fits ran locally and which on rented CPU; (d) the two assembly lines (the v132 merge and the E9S delta) that were run as commands and never saved as scripts. Settled: the corrector's OSM columns use the stand positions only; the hidden-plan reading count is 18 rows in scope, measured from input columns.
