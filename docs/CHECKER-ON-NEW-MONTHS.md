# The stamp checker on February and June 2026

`tools/check_offblock_stamps.py` and the counts of `docs/DATA-DEFECTS.md` were written on 2025 and on January and July 2026.
This page runs the checker on two months it was never built on: February and June 2026, from the final phase's input file
(`final_ranking.parquet`: February 144,158 departures, June 181,791, month = UTC month of take-off). It is a demonstration that the
catalogue's definitions carry to new months, and a check that the model's inputs for those months contain no defect family at a
scale the 2025 months do not.

## What was run, and what can be seen

The checker's functions (`normalise` and `add_flags`) were applied to 16 month tables: the twelve months of 2025 and January,
February, June and July 2026, at the ten airports. The 2026 departure rows carry no label and no off-block stamp, so only the flags
that are defined on input columns can fire: `no_plan`, `copy_impossible_plan`, `nm_default_offblock`, `duplicate_movement`,
`heli_or_runway_h`, `vfr`, the plan-window population and the populations in which the label-based defects live (a no-plan row with
a large schedule gap, a no-plan row with a gap over 21,600 s whose take-off is on a later local day). `nm_default_offblock` is learned
per month table, as the checker does. The label-based flags (`schedule_copy`, `day_shifted`, `near_zero_label`, `hold_like`,
`label_outside_window`) cannot be evaluated without labels and are not claimed here. No 2026 departure label was read.
The counts come from the scripts `agent_recon_01_extract.py` to `agent_recon_10_ranges.py`.

Each cell was compared with the range of the same quantity over the twelve 2025 months and January and July 2026. About 1,000
airport-quantity cells were tested; 37 for February or June 2026 fall outside that range, none by a margin that suggests a new
regime. The largest, by excess over the range:

| cell | value | range of 2025 and January/July 2026 | reading |
|---|---|---|---|
| Rome stand null, February | 3 rows (0.03%) | 0 to 0.01% | 3 rows, noise |
| Rome runway count, February | 4 runways | 5 to 7 | 94.8% on runway 25: weather or configuration, not a data change |
| Istanbul stand count, June | 357 | 329 to 346 | about 20 more distinct stands; watch stand-keyed features |
| Barcelona take-off minus the network manager's off-block, median, June | 776 s | 758 to 770 | +6 s, the stand-default taxi drifts |
| Amsterdam stand-default share, June | 2.91% (595 rows) | 0.27 to 2.49% | a real-stamp airport; the flag is weak there |
| Paris duplicate records, June | 30 (0.14%) | 14 to 24 records | 15 pairs against a 2025 maximum of 12: the one family count above range |
| Rome take-off before schedule, February | 1.74% (179) | 0.34 to 1.50% | mild, January 2026 was 1.50% |

## Counts per defect family and airport-month

Columns: February 2025, February 2026, June 2025, June 2026, and for scale January 2026 and July 2026. Counts of rows (records for
duplicates). Departures per airport-month are in the first table.

### Departures

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 15099 | 14654 | 20706 | 19700 | 15667 | 20650 |
| EDDM | 10381 | 10534 | 15661 | 14372 | 10953 | 15013 |
| EGLL | 17956 | 17719 | 20063 | 19662 | 19283 | 20557 |
| EHAM | 17852 | 16802 | 21257 | 20881 | 16232 | 21950 |
| LEBL | 12169 | 12167 | 16315 | 17067 | 12201 | 17879 |
| LEMD | 15390 | 16191 | 17992 | 19335 | 16802 | 20152 |
| LFPG | 16806 | 16934 | 20887 | 20714 | 18129 | 21743 |
| LIRF | 10703 | 10264 | 14696 | 14983 | 11061 | 15838 |
| LSZH | 9048 | 9374 | 12083 | 12384 | 9995 | 13155 |
| LTFM | 18327 | 19519 | 23482 | 22693 | 22396 | 25185 |

### D1, schedule copy without a flight plan: the population (no plan, schedule gap of 5,400 s or more)

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 33 | 111 | 192 | 95 | 140 | 125 |
| EDDM | 11 | 69 | 88 | 74 | 93 | 74 |
| EGLL | 28 | 33 | 103 | 156 | 75 | 113 |
| EHAM | 66 | 73 | 125 | 183 | 604 | 152 |
| LEBL | 33 | 50 | 124 | 175 | 52 | 283 |
| LEMD | 14 | 65 | 77 | 82 | 46 | 119 |
| LFPG | 63 | 81 | 295 | 253 | 237 | 252 |
| LIRF | 42 | 40 | 131 | 116 | 75 | 212 |
| LSZH | 8 | 23 | 86 | 110 | 165 | 136 |
| LTFM | 152 | 25 | 24 | 20 | 57 | 22 |

At Rome, February 2026 has 40 such rows on 20 days (11 days with two or more, none above 50,000 s); June 2026 has 116 rows on 30
days (27 days with two or more, 12 rows of 14,400 s or more, 3 rows of 50,000 s or more). June is the Rome month with real mass,
February is thin.

### D4, stamp dated a day early: populations

No plan, gap of 21,600 s or more, take-off on a later local day than the schedule:

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 0 | 1 | 5 | 2 | 3 | 1 |
| EDDM | 0 | 0 | 0 | 0 | 0 | 1 |
| EHAM | 0 | 3 | 1 | 2 | 9 | 2 |
| LEBL | 0 | 0 | 1 | 0 | 1 | 4 |
| LFPG | 0 | 0 | 2 | 0 | 1 | 0 |
| LIRF | 1 | 1 | 3 | 5 | 6 | 10 |
| LTFM | 2 | 0 | 0 | 0 | 0 | 1 |

(EGLL, LEMD and LSZH: 0 in all six months.) Plan rows with a schedule gap of 70,000 to 100,000 s:

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 10 | 10 | 6 | 12 | 9 | 7 |
| EDDM | 0 | 0 | 1 | 0 | 0 | 2 |
| EHAM | 1 | 7 | 6 | 6 | 11 | 7 |
| LFPG | 0 | 0 | 0 | 0 | 3 | 1 |
| LIRF | 0 | 0 | 0 | 0 | 0 | 1 |
| LTFM | 6 | 0 | 0 | 0 | 0 | 0 |

(EGLL, LEBL, LEMD and LSZH: 0 in all six months.) Frankfurt's 10 and 12 are inside its 2025 range (0.02 to 0.09% of its rows).

### D2 and D6, schedule copy with a flight plan

Plan rows with a schedule gap of 1,800 s or more where a copy is possible (the schedule is within 3,606 s of the plan's off-block time):

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 3387 | 3752 | 7664 | 7244 | 5889 | 6890 |
| EDDM | 1730 | 2093 | 3954 | 3409 | 3448 | 3949 |
| EGLL | 4877 | 5185 | 8159 | 8463 | 6733 | 8560 |
| EHAM | 4852 | 3797 | 6381 | 6220 | 4966 | 7016 |
| LEBL | 2263 | 2884 | 5515 | 5842 | 2830 | 6544 |
| LEMD | 3322 | 5603 | 6513 | 7118 | 5779 | 7662 |
| LFPG | 5263 | 4791 | 7926 | 8210 | 6793 | 8475 |
| LIRF | 2425 | 2247 | 6378 | 5791 | 2573 | 7095 |
| LSZH | 1842 | 2205 | 4293 | 3953 | 3455 | 4772 |
| LTFM | 4609 | 4690 | 5071 | 4202 | 8179 | 5781 |

Plan rows where a copy is impossible (the window or the gap rules it out):

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 276 | 446 | 656 | 509 | 680 | 466 |
| EDDM | 128 | 201 | 404 | 370 | 329 | 365 |
| EGLL | 300 | 317 | 463 | 624 | 620 | 575 |
| EHAM | 552 | 562 | 698 | 768 | 1196 | 705 |
| LEBL | 223 | 403 | 615 | 730 | 464 | 958 |
| LEMD | 303 | 681 | 746 | 753 | 715 | 1012 |
| LFPG | 517 | 589 | 1000 | 921 | 1380 | 958 |
| LIRF | 225 | 297 | 669 | 639 | 471 | 1079 |
| LSZH | 141 | 167 | 310 | 355 | 383 | 337 |
| LTFM | 1083 | 727 | 649 | 718 | 1298 | 948 |

At Rome the copy-possible plan rows are 22.1% (February 2026) and 39.1% (June 2026) of plan rows (2025 range 22.7 to 49.5%), and the
copy-impossible rows 2.9% and 4.3%; no cell is outside its 2025 range.

### D7, duplicate movement records (records, both rows of a pair)

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| LFPG | 16 | 18 | 14 | 30 | 12 | 22 |
| LSZH | 0 | 0 | 0 | 0 | 6 | 0 |

(All other airports: 0 in all six months. Across the 2025 months Paris has 14 to 24 records.) June 2026 is the highest of the 16 months at
Paris, 15 pairs against a 2025 maximum of 12; the Paris duplicate rule will have about 15 pairs to act on.

### D11, helicopter and runway-H movements

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EHAM | 84 | 77 | 114 | 85 | 70 | 85 |
| LSZH | 47 | 39 | 43 | 16 | 52 | 40 |
| EDDM | 6 | 0 | 0 | 4 | 0 | 3 |
| EDDF | 0 | 0 | 0 | 1 | 0 | 1 |
| LFPG | 0 | 0 | 0 | 0 | 0 | 1 |

(Other airports: 0.)

### D12, plan rows whose network-manager off-block equals a stand default

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 2043 | 1767 | 3794 | 3642 | 1644 | 4342 |
| EDDM | 3497 | 3622 | 6076 | 6583 | 2271 | 7794 |
| EGLL | 0 | 0 | 0 | 0 | 0 | 0 |
| EHAM | 124 | 106 | 452 | 595 | 102 | 59 |
| LEBL | 7270 | 7490 | 10280 | 10519 | 6751 | 10741 |
| LEMD | 4849 | 4794 | 5642 | 6000 | 4996 | 5932 |
| LFPG | 769 | 1430 | 789 | 1136 | 912 | 798 |
| LIRF | 817 | 1347 | 2124 | 2358 | 1171 | 2079 |
| LSZH | 3678 | 3750 | 5053 | 5326 | 3516 | 6233 |
| LTFM | 5504 | 6216 | 12933 | 11347 | 8914 | 13509 |

As a share of plan rows the shares are Rome 13.2% (February 2026) and 15.9% (June 2026) against a 2025 range of 7.5 to 16.9%, Munich
34.7% and 46.2% (26.0 to 47.4%), Barcelona 62.0% and 62.5%, Madrid 29.8% and 31.2%, Istanbul 32.3% and 50.7%, Zurich 40.6% and 43.9%,
Frankfurt 12.2% and 18.7%, Paris 8.5% and 5.6% (2025 range 1.03 to 9.82%), Amsterdam 0.6% and 2.9%, Heathrow 0.

### No-plan rows with a recoverable plan (the hidden-plan reading)

| airport | 2025-02 | 2026-02 | 2025-06 | 2026-06 | 2026-01 | 2026-07 |
|---|---:|---:|---:|---:|---:|---:|
| EDDF | 4 | 2 | 6 | 1 | 10 | 0 |
| EDDM | 0 | 12 | 13 | 19 | 14 | 13 |
| EGLL | 0 | 1 | 8 | 18 | 5 | 13 |
| EHAM | 1 | 2 | 2 | 0 | 12 | 1 |
| LEBL | 2 | 6 | 11 | 14 | 11 | 23 |
| LEMD | 1 | 10 | 9 | 16 | 11 | 16 |
| LFPG | 2 | 4 | 17 | 28 | 15 | 22 |
| LIRF | 1 | 6 | 11 | 13 | 18 | 22 |
| LSZH | 0 | 1 | 3 | 4 | 2 | 3 |
| LTFM | 12 | 3 | 6 | 3 | 13 | 3 |

At Rome, 6 rows in February 2026 (2 in the scope of the hidden-plan rules) and 13 in June (11 in scope); the 2025 range is 1 to 25.

## What the demonstration shows

- **No new regime.** The no-plan share at Rome is 0.73% in February 2026 and 1.07% in June 2026 (2025: 0.54% and 1.26%), inside the
  2025 range. The January 2026 excess at Amsterdam (4.96% no plan), Zurich (3.00%) and Frankfurt (1.35%) does not recur in February
  (1.39%, 1.34%).
- **Dates and clocks.** The schedule equals the plan's off-block time (within 60 s) on 79.1% of Rome plan rows in February and 71.9%
  in June (2025: 64.6 to 80.0%). No new stand, runway, operator or aircraft-type key with 20 rows or more appears in either month.
- **Changes that exist in both 2026 months of a season and not in 2025**, so they are not new for February and June: the
  network manager's off-block is written on whole minutes at Amsterdam (70.3% in February 2026 against 86.9% in February 2025),
  Barcelona (96.0% against 99.8%) and Paris (97.8% against 99.9%), as in January 2026; and in June the schedule equals the plan's
  off-block time at Istanbul on 81.0% of rows (59.5% in June 2025) and the second plan update equals it at Paris on 60.7% (73.6%)
  and Heathrow on 47.7% (54.5%), as in July 2026.
- **Covariate shifts** are configuration and weather, not coding: Madrid's runways 14L and 14R carry 39.4% of February departures
  against 5.3% in February 2025; Istanbul's February delay level is lower than February 2025, which was the snow month.
- **Disruption days.** February 2026 has 8 airport-days with at least a quarter of departures an hour or more late, a very low
  volume or a high published delay per departure (Frankfurt 3, 12, 14 and 19 February, Munich 19, Barcelona 12, Madrid 5 and 13) against
  35 in January 2026; June has 7 (Heathrow 2 and 27 June, Zurich 21 and 28, Munich 21 and 29, Frankfurt 28).

## The two published identities on the new months

Both daily series are present for every day of February (28) and June (30) at all ten airports. A stamp can be dated a day before its
take-off, so the last day of a month needs the next month's departures: 28 February and 31 July are not usable, 30 June and 31 January are.

- **Count identity** (`FLT_DEP_IFR_2` equals the count of IFR rows filed under the stamp date, with an input-only range): inside the
  range for 27 of 27 usable February days at all ten airports, and for 30 of 30 June days at eight airports (29 at Amsterdam and at Istanbul, where 30 June is the known Istanbul month-end misfit: 778 against 696 to 708; 28 at Paris).
  Paris misses 4 to 7 days a month in 2025 as well.
- **Delay identity** (`DLY_ALL_PRE_2` against the expected delay with the airport-month's median taxi time, as residual minutes per
  departure z-scored on the airport's 2025 days): days with |z| above 6: February 2026 11, June 2026 0 (January 2026 52, July 2026 1;
  February 2025 14, June 2025 0). All 11 in February are negative (published delay well below the typical-taxi expectation, which is
  longer taxis, de-icing and holds): Munich 19 February (z -20), Frankfurt 19 February (-16), Zurich 18 February (-15) and others, none
  at Rome. The published totals are in the same direction and size class as the 2025 snow days.

Nothing in the published series points to a regime the 2025 months do not show. The days that will move the day layer are the 11
snow and de-icing days of February; the stop-day rule leaves none of them unchanged (`docs/PIPELINE.md` I.3).

## Reproducing it

`python tools/check_offblock_stamps.py <month table> --out flags.parquet --summary summary.csv` produces the flags and the per-airport-month
summary for a table with the challenge's columns (one month at a time for the 2026 input file; the label-based flags stay empty without
labels). The counts above were produced with the checker's functions in `agent_recon_02_census.py` (the census table) and
`agent_recon_03_report.py` (the range test), under a 3 GB cap.
