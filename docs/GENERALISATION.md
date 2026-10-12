# Why some pieces are specific to an airport or a month, and what carries over

The entry has three layers (README, `docs/PIPELINE.md` section I.2): a **general model**, a **reconciliation layer** that uses a
published daily figure of the served days, and **fixes that name specific January and July flights**. The general model contains
record-quality handling that is fitted to the recording habits of the ten airports; this page argues that it is needed for this
target, says what it rests on, says which parts are a method that transfers and which are only constants, and lists what someone
applying the model at another airport or in another year should do first. Sections 8 to 10 give the results on months never used
for selection, what was weighted toward January and July and what the month-agnostic alternative costs, and the half-strength day
layer; section 11 says which months of the submitted file are carried from the earlier lineage and what the corrector's view columns are worth. Every number is ours, from 2025 hold-out evidence or from the label-free checks named in `docs/ROW-RULES.md`, unless it is
marked as a leaderboard result.


## 1. The label is a record, not a physical event

The target is `MVT_TIME - BLOCK_TIME`, and the take-off time is an input. The task is therefore to locate the off-block time that
the airport recorded. That time is the airport operator's stamp, and the stamp differs from the pushback by habit, differently at
every airport:

- At London Heathrow (2025, the 9,524 departures of eight months whose track is seen leaving the stand, all operators) the first movement of the track comes a median 484 s after the recorded off-block (5th to 95th percentile 347 to
  742 s; for the largest operator alone, to the end of the stop at the aircraft's own stand, the median is 396 s, a different definition and a narrower population); at Paris CDG the first fix near the stand comes a median 401 s after it.
- The flight plan carries its own off-block field (`AOBT_3`). On a large share of rows it is not an observation but the take-off
  time minus a stand-specific default taxi time rounded to the minute: Barcelona 91% of rows, Zurich 79%, Madrid 75%, Munich 72%,
  Istanbul 67%, Frankfurt 35%, Rome 19%, Paris 9%, Amsterdam 1%, Heathrow 0%; the share falls by 2 to 9 points in 2026 at Frankfurt, Munich, Madrid, Zurich and Paris.
- At Rome the recorded off-block is, on many departures, a copy of the scheduled time. The copy habit belongs to the reporting
  airline (the intraclass correlation of the copy flag is 0.31 by operator and 0.004 by day). About 8% of plan departures taking
  off at least 30 minutes late are copies, and roughly half of the departures without a plan.
- A few hundred departures per year have a stamp that is not a pushback at all (a copy of the schedule, a stamp a calendar day
  early, a duplicate record), and the label is then hours long. On the 2025 hold-out view of the final model, rows missed by more
  than 1,800 s are 0.16% of the rows and 38% of the squared error; rows missed by more than 600 s are 1.65% of the rows and 64%.
  Seventy-six rows with a label over four hours or below zero carry 20% of it.

The organisers stated that block time is what the airport provided, that no substitution is made in the export, and that dealing
with noisy data is part of the challenge. A model that ignores these rows is largely measuring luck on them, and a squared-error
metric multiplies the effect. So any honest approach to this target needs explicit handling of the recording process. What
transfers to another airport is the **method**; the **fitted constants** are airport-specific by nature.

The method, as applied here (details and counts in `docs/ROW-RULES.md`):

1. Define the default-stamp population from **input columns that exist at prediction time** (schedule gap, flight-plan clocks, the
   plan recovered from the destination's arrival record, operator, stand), never from the label.
2. Estimate the probability that a row of that population is a default stamp from **out-of-period habit rates** (operator and
   stand rates computed on the other months), and predict the mixture mean, not one of the two outcomes.
3. Apply a **hard rule only where the database implies it**. The plan-join window is the example: a movement is attached to its
   flight plan only when the recorded off-block lies within an hour of the plan's last off-block time, so a plan row's label lies
   in a window of known width (zero exceptions on 2,062,577 plan departures in 2025). Hidden-plan bounds are derived the same way.
4. **Count the exceptions on training data, airport by airport, and use a rule only where it holds.** The hidden-plan upper bound
   holds on 1,277 of 1,278 rows over seven airports and is used there; at Amsterdam it holds on 23 of 27 and is left out; the
   lower bound holds at Rome and Istanbul and fails at Barcelona (2 of 13), where "no plan" has another cause.

## 2. Reception is a feed state, so ground-track components are judged regime-matched

The ADS-B stage needs aircraft to be heard on the ground. Whether they are heard is a property of the open receiver network in
a given month, not of the airport, and it changes:

| Airport and month | Share of plan departures seen on the ground, 2025 | same month, 2026 |
|---|---|---|
| Rome, January | 0% (January to April 2025 all 0%) | 65% |
| Rome, July | 37% (May to December 2025: 18 to 92%) | 0.1% |
| Heathrow, July | 23% | 97% |
| Madrid, January | 0% | 96% |
| Munich, July | 37% | 97% |
| Frankfurt, January | 93% | 73% |

A calibration or a gain measured in one regime and served in another keeps only part of its size: the ADS-B stage kept 76 to 81%
of its held-out gain on the leaderboard, bulk-model changes about 38%. The general procedure that follows is to **take a census of
coverage by airport and month for training and for the target period, and judge any ground-track component on months in the same
regime as the target month**. For Rome that means months 5 to 12 of 2025 for January 2026 and months 1 to 4 for July 2026. We
applied this late, and found that earlier Rome verdicts had had the months the wrong way round. We also found its limits: used as a
predictor of leaderboard outcomes, the regime-matched frame failed its own check (it predicted a gain for two uploads that lost), so
we use it to size and to inspect, not to accept a change. Two attempts to train around the problem did not work (hiding the ground track
in lit months so that the model learns to live without it: +0.9 s worse through the chain; weighting the re-stack by coverage:
+0.32 s worse, every airport worse).

## 3. A year-over-year level change cannot be validated inside one training year

Every fold that trains on 2025 and tests on 2025 is blind to a change of level between 2025 and 2026. Earlier versions of the
solution carried one rule that used 2026 information in setting its size: a shift of -38 s on the 7,062 Munich January 2026 rows
with a schedule gap under 1,800 s (`archive/apply_eddm_january.py`). It was an exception to "fitted and validated on 2025 only", and
it is in no file from v144 on. What went wrong is the lesson of this section.

- The size came from three instruments, none using a 2026 departure label: the published monthly Munich taxi-out figure (the
  organisers answered on 2026-09-27 that the published monthly figures for the scored months may be used as model input; the shift was first uploaded about 2026-09-20, before that answer), after
  fitting the portal's coverage effect on 2025 (an excess of 76 s); our own 2026 ADS-B estimates, bias-calibrated on 2025 (16 s, a floor);
  and the flight plan's own taxi proxy (an excess of 76 s). The shift applied was half the largest.
- The comparison was not like for like. The published January mean leaves out flights de-iced after off-block (3,426 of 11,033 January
  2026 Munich departures, 31%; 20% in January 2025), which take about 550 s longer, so a comparison with all flights reads a chain that
  is right as too high, and the apparent excess grows with the de-iced share (all flights, the gap between our predictions and the
  published mean was +254 s in January 2026 and +164 s in January 2025). The same applies to a later label-free check on the chain that
  showed about +61 s (4.65 spread units over 17 placebo airport-months): it was the same all-flight comparison, and it is superseded.
- Like for like (the published reference population rebuilt with de-iced flights removed and a term for the de-iced flights the
  detector misses, fitted on 12 months of 2025; and our ADS-B estimates on seen, unflagged flights, calibrated on 2025 months) the bias
  of the chain before the shift is about +2 s, interval -10 to +15 s. A shift of c against a true bias b costs c^2 - 2bc per row, so
  c = 38 s costs about 0.06 s at b = 2 s. The shift was removed (v144). The plan-clock proxy was too noisy to vote.

No component of the last uploaded file or of the submitted file uses a value derived from the published monthly figures. They remain useful, as here, as a check on a
level, and then only after the population behind the published figure has been rebuilt.

What the general rule would be for a new year: published level statistics (a monthly mean, a count above a threshold) are
legitimate **checks** (does the chain's implied monthly figure match the published one?) and, where an airport-month is clearly off
by every independent instrument, a **level anchor** applied as a shift on a defined population. They should never be used per row.
The chain passes the same kind of check where it can be run: the number of Rome labels over 7,200 s implied by the file of that time is
43.6 for January and 82.1 for July 2026 against the 44 and 75 published, and the 2025 months agree to a mean absolute error of
2.7 rows. A second example of a year-specific effect inside training is Istanbul in July 2025, a dated one-off (about 5 to 26
July, +100 to +200 s per day); the served July 2026 Istanbul file carries no such inflation (-1.4 s against normal-month cells,
where the window was +91 s).

## 4. What someone at another airport, or in another year, would do

| Step | What | Runs unchanged? |
|---|---|---|
| 1 | **Census the recording habits.** Compare the label with every physical or planned timestamp available; measure the share of rows whose off-block is a default, a copy or a duplicate, by operator and by month. | No: needs the airport's own data. This is the airport-specific input. |
| 2 | **Census reception** by month, for the training period and the target period (share of departures with a ground track). | Procedure unchanged; numbers are the airport's. |
| 3 | **Fit the core** on the airport's own history: base gradient-boosted blend on flight and plan features (features and members unchanged; retrain). | Yes. |
| 4 | **ADS-B stage**, where reception exists in both periods: stand map, matcher, calibration on days with tracks. | Code unchanged; needs the airport's stand positions and track days. |
| 5 | **Pooled re-stack** over the stage output, so the model learns when to trust the track. | Yes. |
| 6 | **Leave-one-month-out residual corrector** with out-of-month key statistics (operator, stand, destination, and similar keys). | Yes; the keys are generic columns. |
| 7 | **Database constraints**, if the data come from a database with a matching window like the plan join: measure the window on training data and apply it as a bound (zero exceptions required). | Procedure unchanged; the width is measured. |
| 8 | **Default-stamp handling** (section 1 method), only if step 1 found such a population. | Method unchanged; constants refitted. |
| 9 | **Judge** with month-held-out and airline-held-out folds, the scored months first, minus-top-50 and a day bootstrap; regime-matched where a component uses tracks. | Yes. |

Order of work: steps 1 and 2 before any modelling. A new airport that has no default-stamp population gets the core model
alone; the ablation table in `docs/PIPELINE.md` gives the size of each core stage in this competition.

## 5. What does not generalise, and why

- **Per-airport constants.** The corrector's dose is chosen per airport leave-one-month-out among 0.5, 0.75, 1.0 and 1.25 (in
  the final refit: Frankfurt, Munich and Zurich 1.25, Heathrow, Barcelona and Madrid 1.0, Rome 0.75). Choosing it per airport
  added -0.09 to -0.11 s in sample on the fold; a user at another airport would choose their own. The Rome mixture weights, the
  Paris offset (750 s) and the Munich shift of earlier versions (removed in v144) are likewise constants of one airport and one pair of years.
- **Rules that rest on a count are not logic.** The calendar condition on the Rome day-early bands (v145) says that a no-plan
  departure that took off on the same local day as its schedule does not carry a day-early stamp. What we know is a count: 0 of about
  159,600 same-day Rome departures in 2025, and all 13 Rome day-early records (12 without a flight plan, one with) have a take-off on a later local day. It was never observed at
  Rome on a same-day row; it is not established as impossible, and at Zurich one same-day flight in 2025 carries a day-early stamp. The
  same holds for the other row-level bounds of v146 (Istanbul no-plan early bound, 0 exceptions in 3,585 rows; Madrid one-hour
  ceiling, maximum label 3,598 s in 212,242 rows): each is a count with its population stated, applied where its count holds.
- **One group of rows stays hedged by choice.** Five scored Rome no-plan departures with a schedule gap of 14 h or more that took off on a
  later local day are served between the two record types that 2025 shows for such rows (a copy of the schedule, or a stamp made a day
  early): 2025 next-day no-plan rows are 14 to 17 h 5 copies and 5 day-early, 17 to 24 h 2 and 5, over 24 h 1 and 1. No input column
  separates the two types, and a move to either would lose a large squared error if the other were true, so the rows are neither moved
  nor re-dosed; a change needs a database fact or a recovered plan record for the flight (`docs/ROW-RULES.md`).
- **Airline habit rates across years.** This is a negative result of ours, and it is a limit of the Rome components. A Rome
  with-plan copy mixture gained +0.93 s on the fold that preceded the upload of its first, one-sided version, which then lost
  +0.16 s on the leaderboard (v134, not kept). A clean two-sided rebuild gained +0.79 s on a month-held-out fold (January +0.09,
  July +0.85); judged with whole airline groups held out it gained +0.02 s in January and +0.06 s in July, and restricted to the
  airlines whose habit is stable between years it gained +0.20 s. The clean version is in every file from v143 on, at dose 0.5,
  uploaded together with a corrector change whose expected gain was larger; an independent review put its expected value
  at about +0.2 s (range -0.4 to +0.9), and its own leaderboard effect is not known. The reading is that a month-held-out gain of a
  copy mixture is same-year airline-habit information, and does not transfer across years, so that its place in the chain is
  a bet and not a result. The copy classifier's AUC inside the airlines whose habit is uncertain is 0.831 month-held-out and 0.665 with
  airlines held out and the habit inputs excluded. A different learner on the same inputs adds nothing (an in-context tabular
  classifier changed the AUC by -0.0002 to +0.003). The shipped Rome no-plan mixture, which also uses airline habit, keeps about
  two thirds of its held-out gain (+1.55 of +2.31) when judged forward in time instead of across months, all of it on rows it
  moves down toward the real taxi level; its best weight depends on the frame (0.78 month-held-out, 0.66 forward in time, 0.86
  on year-stable airlines, 0.58 on unstable ones) and no alternative beats the shipped weight with probability 0.9. We ship it
  as a concentrated bet and say so (`docs/ROW-RULES.md`).
- **Airline-keyed key statistics in the corrector.** 51 of its 116 key statistics are airline-keyed. A proxy judged forward in
  time says they add nothing there, and that a transfer-honest dose would be worth up to about 0.04 s at Frankfurt. The shipped
  settings were not changed on that evidence, which is weak.
- **Gains that depend on a feed state** (section 2) and **hedges on rows whose regime has no 2025 analogue**: two of those (long-gap
  Rome rows, an Amsterdam snow period) lost on the leaderboard and are not in the January/July chain (`docs/SUBMISSION-LOG.md`).

What we did not find: any airport where the body of the prediction error is above the information floor in the inputs (the body
RMSE is 40 to 75% of the label's standard deviation within a stand, runway, hour and month cell at every airport), and any open
per-flight source of off-block events. The remaining error is mostly the record, which is the point of section 1.

## 6. The three layers

Scores: v152, 216.2405 s, contains no published figure about the scored months; v162, 213.2898 s, contains the reconciliation layer at full strength (the step is worth 2.4780 s on the board in its first form, v155 against v152; the Istanbul corrector of v156 a further 0.0524 s).

1. **The general model** (what one would run at another airport or in another year, retrained on that airport's history): the base blend, the ADS-B stage where reception exists, the pooled re-stack, the leave-one-month-out residual corrector with out-of-month key statistics, the plan-clock distance inputs and the arithmetic columns of its rounds two and three (the airport lists and doses are fitted constants), the plan-clock window and the schedule-gap clamp, and the record rules of the competition-specific handling: the Rome schedule-copy and day-early treatment (mixtures, plan-gate weights, the hard-binding rule), the Paris duplicate rule, the Istanbul models and corrector, the row-level bounds and the per-airport doses. The method of each transfers; the constants do not. They rest on 2025 counts and are fitted and judged on 2025 only (`docs/ROW-RULES.md`). Their measured contributions are in the ablation table of `docs/PIPELINE.md` (rows C1 to C8, S1 to S14) and, by month, in section 8.
2. **The reconciliation layer** (a day-level calibration of delayed flights and posterior Rome record types). It uses a published figure about the served months, the daily all-cause pre-departure delay of EUROCONTROL's performance portal, which on 2025 is a function of the recorded off-block stamps (`docs/PIPELINE.md` section I.3). It transfers as a method wherever such a daily aggregate is published at the same grain as the records. The organisers' answers are quoted in `docs/DATA-SOURCES.md`. Nothing in layer 1 reads those figures. The submitted form is the half-strength form of section 10.
3. **Fixes that name specific January and July flights**: six Rome flight strings and one single-row decision (`final/31_scored_row_fixes.py`). They cannot be re-judged on 2025 and are off in the submitted file.

For someone using the general model at another airport or in another year, layer 1 is the model and layer 2 is an audit of the records when the figures exist.

## 7. Definitions and intended use

- **Intended use.** The organisers describe the model as a tool for post-operations analysis: identifying constrained operating periods and measuring excess fuel and CO2 against the unconstrained state. The label's recording behaviour (section 1) decides what such a figure means per airport, stand or airline.
- **Their indicator against our label.** The published additional taxi-out time (EUROCONTROL, indicator document, edition 01.00, 16 March 2023, read in full; page numbers are its printed pages) groups flights by departure runway and stand (section 3.2, pp. 10 to 11), takes the 10th percentile of the taxi-out times of the rolling year (month i-11 to month i, by local take-off time) as the reference (section 3.4, step 2, p. 13; section 3.5, p. 14), valid with at least 10 flights at or below it (p. 15), and filters out flights with some missing data (no runway, no stand or no off-block time), a taxi-out over 120 minutes, helicopters and flights de-iced after off-block (section 3.4, step 1, p. 13); flights without a valid reference get neither a reference nor an additional time and are not considered (section 3.4, step 5; section 6.3, p. 16). Our label keeps all of those rows. Our rebuild of the rule reproduces the published no-reference counts for December 2025 at five airports (ours / published: EHAM 1,146 / 1,145, LEBL 587 / 590, LEMD 1,550 / 1,544, EDDF 385 / 380, LSZH 671 / 708); the other airports are inflated by exclusions (de-icing, helicopters) that we cannot see. The document calls the precision of the off-block time "key" and says it should reflect when the aircraft starts moving on the apron (section 4, p. 15).
- **Origin of the network off-block constants.** The ATFCM Users Manual (EUROCONTROL, edition MAINT-2, valid 18 June 2024, section 8.1.10.1, p. 41) says the default taxi time is specified for each runway of an aerodrome in the Network Manager's Central Airspace and Capacity Database; the taxi time that the aircraft operator gives in the flight plan overrides it; a modification of the aerodrome parameter for a period, requested by the flow management position or the tower, takes precedence over the flight-plan taxi time for flights with an estimated off-block time in the period; and at CDM airports taxi times in departure planning information messages take precedence over the flight-plan taxi time and over tactical updates to the aerodrome parameter. The manual does not list the CDM airports (chapter 2.5 refers to an Information Notice); in IN/26-013 (20 May 2026) Munich, Amsterdam and Zurich are CDM airports, Frankfurt, Heathrow, Barcelona, Madrid, Paris Charles de Gaulle and Rome are ANI airports (a step beyond CDM), and Istanbul is not listed. The manual gives per-runway defaults only; what we measure (a per-runway constant at Istanbul, per-stand constants at most listed airports, section 1, `AOBT_3`) is consistent with it but is not stated in it.
- **The indicator document names the precision of the off-block stamp as the key error source**, which is the premise of section 1.


## 8. Results on months never used for selection

The constants of the January/July lineage were fitted on 2025 labels and judged first on held-out January and July 2025; which
candidates stayed was in some cases decided by a 2026 leaderboard score (`docs/SUBMISSION-LOG.md`). This section
and the next two report what holds on the other months. Everything is from held-out chains on 2025 (every month predicted by
models and constants that never saw it), in **board seconds** (`SSE / 1.5e8`, positive = the stage helps; a gain on one month's
rows is about the same number of seconds on a two-month score and half that on the four-month score). The 2025 level is about 18%
above the leaderboard level (the January/July analogue is 250.7 where the board reads 213.3), so compare ratios. Month 6 is a
different base, noted where it appears. Source: `scripts/agent_genaudit_0[1-8]_*.py`.

### 8.1 Stage by stage

Nine airports; Istanbul is fitted separately and is not in the tables of this subsection (it appears by airport in section 8.3).
RMSE in seconds after each stage:

| after stage | Jan | Feb | Mar | Jul | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|---|---|---|
| bulk base | 283.9 | 192.9 | 194.7 | 292.0 | 237.9 | 249.1 | 211.2 | 210.1 |
| + stage, re-stack, blend | 259.7 | 166.4 | 164.3 | 263.1 | 215.7 | 223.2 | 180.5 | 175.9 |
| + six rules | 258.2 | 165.9 | 161.6 | 257.8 | 213.7 | 204.2 | 178.2 | 175.6 |
| + three small correctors, silent-start nudge, meta layer | 257.9 | 165.7 | 161.8 | 257.4 | 213.6 | 204.0 | 177.9 | 175.8 |
| + corrector | 256.7 | 164.4 | 160.0 | 255.3 | 211.5 | 202.0 | 176.1 | 173.5 |
| + tail, + Rome (served) | 253.7 | 164.6 | 161.2 | 248.8 | 209.9 | 201.5 | 174.9 | 173.1 |

Gain of each stage, board seconds:

| stage | Jan | Feb | Mar | Jul | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|---|---|---|
| stage + re-stack + blend | 9.84 | 7.95 | 10.44 | 17.72 | 11.10 | 13.06 | 12.93 | 12.33 |
| six rules | 0.59 | 0.14 | 0.85 | 3.07 | 0.98 | 8.64 (one row) | 0.93 | 0.12 |
| M1 to M3 | 0.08 | 0.03 | -0.11 | 0.13 | -0.03 | 0.00 | 0.06 | -0.13 |
| silent + meta | 0.04 | 0.03 | 0.03 | 0.11 | 0.09 | 0.06 | 0.05 | 0.04 |
| corrector | 0.47 | 0.36 | 0.56 | 1.20 | 0.97 | 0.89 | 0.65 | 0.75 |
| tail | 0.01 | 0.04 | 0.03 | 0.14 | 0.03 | 0.01 | 0.10 | 0.00 |
| Rome mixtures and rules | 1.13 | -0.08 | -0.41 | 3.43 | 0.71 | 0.19 | 0.39 | 0.14 |
| neighbour blend (removal test) | 0.06 | 0.04 | 0.11 | 0.04 | 0.07 | 0.07 | 0.10 | 0.10 |

The stage, re-stack, corrector, tail, blend and silent/meta pieces are positive in all eight months, and February's gain is about
80% of January's (81% for the stage, re-stack and blend, 77% for the corrector). The rules gain less in February because they fire on copy and
window cases that cluster in summer and autumn. The Rome block loses in February and March: the mixtures put mass on copy rows that
winter months barely have. M1 to M3 are a coin flip outside January and July (a loss in March, August and November).

### 8.2 Months 4, 5, 6 and 12

These months have no stage output in the eight-month frame (the 12-month extension produced them later), so only part of the chain exists.
For June 2025 (out of month: stage and re-stack fitted on 238 days that exclude June): bulk base 209.8 s, after stage and re-stack
178.8 s (12.39 board-s, inside the January to November range of 7.9 to 17.7), plus the 12-month corrector 177.2 s (0.55 board-s).
The stage generalises to a summer month it never saw. The same table gives April 134.4 to 133.0, May 277.7 to 277.0 and
December 147.4 to 145.9 (stage gains 8.3, 9.7 and 13.5). Missing for these months: the rules, blend, silent/meta, tail, Rome rows
and every published-figure piece.

Corrector gain per month, 12-month leave-one-month-out fit (board seconds; this is a later fit on all twelve months, not the
eight-month frame's corrector in the table of section 8.1, which is why January reads 0.53 here and 0.47 there): January 0.53, February 0.37, March 0.59, April 0.33,
May 0.31, June 0.55, July 1.45, August 1.08, September 1.01, October 0.75, November 0.81, December 0.34. Positive in all twelve.

### 8.3 Where February and June are decided

Held-out RMSE by airport on the 2025 analogue (June is the pre-rules chain and has no Istanbul, so read June as an upper bound;
LIRF June could fall by 0 to about 40 s with Rome handling).

| airport | Jan | Feb | Jun | Jul | squared error, board-s, Jan / Feb / Jun / Jul |
|---|---|---|---|---|---|
| LTFM | 209.0 | 542.8 | n/a | 256.9 | 5.2 / 36.0 / n/a / 11.1 |
| LIRF | 397.2 | 343.5 | 374.4 | 581.1 | 10.1 / 8.4 / 13.4 / 34.2 |
| LFPG | 528.5 | 196.9 | 228.4 | 232.2 | 28.3 (one label: 21.9) / 4.3 / 6.1 / 7.8 |
| EGLL | 170.4 | 152.9 | 208.1 | 267.8 | 3.2 / 2.8 / 5.7 / 9.9 |
| LEBL | 82.1 | 83.6 | 129.1 | 225.9 | 0.5 / 0.6 / 1.8 / 5.7 |
| LEMD | 151.2 | 143.7 | 115.2 | 160.5 | 2.1 / 2.1 / 1.6 / 3.2 |
| LSZH | 137.6 | 199.1 | 100.7 | 137.7 | 1.0 / 2.4 / 0.8 / 1.6 |
| EDDF | 113.9 | 93.3 | 95.0 | 114.6 | 1.1 / 0.9 / 1.2 / 1.9 |
| EDDM | 57.6 | 70.5 | 90.8 | 139.5 | 0.2 / 0.3 / 0.9 / 2.1 |
| EHAM | 124.0 | 80.9 | 79.5 | 112.5 | 1.6 / 0.8 / 0.9 / 1.9 |

February is decided by Istanbul snow: Istanbul is 63% of February's squared error (36.0 of 57.0 board-s, RMSE 542) and the other
nine airports are at RMSE 164. June has no such tail: Rome 41%, Paris 21%, Heathrow 17%. February 2026 is a milder Istanbul month
(35 snow-coded hours against 131 in February 2025) and a harsher Central-European de-icing month (Munich 39 against 15, Frankfurt 28
against 6, Zurich 29 against 7), so February 2025 is not a fair preview in either direction. In January 2025 the top 20 rows hold 60%
of the squared error, in February 22%, June 9% and July 17%: February and June are the best months for judging the body of the model.

Reception differs by month. February 2025 has no ground tracks at Madrid, Rome and Paris (rows by tracked / estimated / untracked:
72,394 / 36,530 / 16,480, RMSE 87 / 219 / 258); July 2025 has 97,100 / 63,416 / 5,026 (198 / 229 / 788). Rows without tracks score
RMSE 258 to 788 against 87 to 198 with tracks, which is why the ADS-B download for February and June is the stage with the largest
value (8 to 13 board-s a month).

### 8.4 Row rules by month

Rows differing under removal of each rule on the eight frame months (counts per month 1, 2, 3, 7, 8, 9, 10, 11) and the gain:

| rule | rows per month | gain by month | verdict |
|---|---|---|---|
| plan-clock window | 11, 11, 9, 72, 33, 37, 20, 14 | +0.32, +0.04, +0.77, +1.32, +0.04, +0.43, +0.33, -0.01 | general (hard bound, 0 exceptions on 2,062,577 rows); positive in 7 of 8 |
| gap clamp | 6, 18, 6, 17, 18, 16, 22, 9 | +0.01, +0.04, +0.02, +0.02, +0.35, +1.24, +0.06, 0.00 | general; August and September dominated by a few rows |
| per-airport floor | 598, 665, 816, 644, 645, 683, 645, 596 | -0.00, -0.01, -0.01, +0.07, 0.00, 0.00, 0.00, -0.01 | general; fires on 600 to 800 rows a month and is worth about zero |
| habitual-copy operators and hidden lower bound | 4, 4, 6, 18, 14, 15, 7, 10 | +0.01, +0.02, 0.00, +0.07, +0.03, +0.02, +0.01, +0.01 | general in form, Rome-specific in effect |
| Rome row rules (R1, R2, R3a, R3b) | 12, 2, 3, 36, 21, 14, 17, 8 | inside the Rome block of 8.1 | fitted to Rome 2025 copy behaviour |

The six row-level bounds of v146 (51 rows in the January/July run) are general and fire on February and June rows by the same condition.

## 9. What was weighted toward January and July, and the month-agnostic alternative

| piece | how it was chosen | evidence on other months | month-agnostic alternative |
|---|---|---|---|
| Corrector per-airport doses (1.25 EDDF/EDDM/EHAM/LSZH, 1.0 EGLL/LEBL/LEMD, 0.75 LFPG/LIRF) | nested leave-one-month-out among 0.5, 0.75, 1.0, 1.25 | scaling the shipped term on the eight-month frame, the best scale is 1.0 at every airport; January/July alone prefers other doses at three airports, worth 0.006, 0.016 and 0.006 board-s | keep: the dose is not a January/July artefact |
| Corrector column lists (arithmetic round three at four airports) | nested all-eight-month per-airport selection, then a January/July gate | candidate beat control in 8 of 8 months; the temporal frame kept 0.62 of the gain | not specific to January/July; the refused airports are those with a tail |
| 12-month corrector (kept) against the 8-month fit (Istanbul keeps 8) | the rule of 2026-10-02: judge on held months 1 and 7 | the 12-month fit is positive in all twelve months; Istanbul's 12-month fit looked worse on months 1 and 7 (-0.322 against -0.377) | Istanbul's 8-month choice is a January/July choice and ignores February snow |
| Rome no-plan mixture (weights 0.5 and 0.85), with-plan mixture (dose 0.5), plan-gate extra weights | months 1 and 7 (+2.71) | removal costs +0.60 January, +3.17 July, +0.69 August, +0.44 October but -0.22 February and -0.49 March | scale the weight by the day's copy mass; tested in section 10: neither form passes, so the flat form stays and is off in February |
| Tail component (k 0.5) | months 1 and 7 (-0.199) | positive in all eight months (+0.00 to +0.14) | keep |
| Istanbul views corrector (v156) | eight-month leave-one-month-out; July carries 94% in one cell split | by month +0.39, +2.32 (February), +0.07, +0.15, +0.64, +0.43, +0.62, +0.04 | eight of eight positive; February is its largest month; the term is not in February of the submitted file |
| Day layer: airports, shrink factor, bias | nested shrink factor 0.8 to 1.0 per airport; airports where the identity holds to 1 to 2 minutes a day | section 10 | `--lam-cap 0.5 --own-bias` (the submitted form) |
| Munich January shift | removed in v144 | n/a | already month-agnostic |
| Six Rome flight strings and one never-uploaded single-row decision | named 2026 rows | no 2025 analogue | layer 3, off in the submitted file |
| Rome hard-binding rule (14 scored rows), calendar condition (1 scored row) | condition on input columns, counted on 2025 (38 rows; 63 rows) | few rows, so the 2025 evidence is a count | layer 1; loss if wrong 0.07 to 0.5 s |

Two pieces have no value outside January and July: the per-airport floor (about zero on 600 to 800 rows a month) and M1 to M3
(negative in 3 of 8 months, pooled removal gain -0.03). Both stay in the pipeline because removing them is not better.

## 10. The half-strength day layer, and why

**The finding.** The day layer's constants (shrink factors 0.8 to 1.0 and a fixed bias per airport) were fitted on eight 2025 frame
months. On the four 2025 months the frame never saw, the same procedure lost: April -0.08, May -0.13, June -0.15, December -0.29
board seconds. With each month's own median as the bias term those months sit at +0.01, -0.06, 0.00 and -0.09, so the loss is partly
a bias that does not carry to a different base, and the gain also disappears. February and June 2026 are such months: the constants
have not been shown to carry to them.

**The form.** The submitted layer caps the shrink factor at 0.5 and takes each month's bias from that month's own published series
(the median of the day shifts; label-free). The grid of shrink factors was cut at 0.5, and 0.5 was the choice in every cell.

**The test**, written before it was run and run once (`final/33_daylayer_validation.py`; pass rule: no month below -0.05 board-s,
and months 4, 5, 6 and 12 not negative in sum), seven airports (EDDF, EDDM, EGLL, LEBL, LEMD, LFPG, LIRF):

| month | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| capped, own bias | +0.108 | +0.088 | +0.172 | +0.033 | +0.002 | +0.116 | +0.338 | +0.138 | +0.138 | +0.117 | +0.133 | -0.004 |
| January/July form | +0.143 | +0.110 | +0.266 | -0.079 | -0.131 | -0.151 | +0.537 | +0.196 | +0.198 | +0.179 | +0.187 | -0.289 |

It passes: worst month -0.004, months 4, 5, 6 and 12 sum +0.147, twelve-month total +1.379. January plus July +0.446 (the
January/July form is worth about +0.68 there, so the cap costs about 0.23 on that pair); February plus June +0.204 (January/July
form -0.04). Months 4, 5, 6 and 12 are measured on the weaker base (no rules, Rome handling, blend, tail or published-figure
pieces), so the result shows that the layer does not hurt there, not what it gains on the submitted file. Istanbul and Zurich are not
covered by this test: Istanbul's layer has its own seven-month validation and the snow stop-day rule; Zurich is worth 0.005 s and is off.
Istanbul's shift was tested on seven 2025 months at the uploaded strength, not with the cap; the Rome part on eight
(`docs/PIPELINE.md` I.3).

**Other pieces of the day layer, tested the same way.**

- Snow and de-icing days at Frankfurt, Munich, Zurich and Amsterdam (February 2026 has 28, 39 and 29 snow-coded hours at the first
  three against 6, 15 and 7 in February 2025). A rule excluding a day with a snow report whose discrepancy is more than 3 sigma below
  the airport's non-snow days qualified three days in twelve 2025 months (all Zurich, January; excluding them changes the score by
  -0.0014), too few to judge. The fallback written before that test, to leave every snow-report day unshifted at Frankfurt, Munich,
  Zurich and Amsterdam, was replaced on 2026-10-08 by the signature rule of `docs/PIPELINE.md` I.2 at every airport of the layer. In
  the submitted file it leaves Frankfurt 10 and 29 January unshifted and no day of February or June; the fallback would have left 20
  February airport-days unshifted. Neither rule could be judged on 2025.
- Rome no-plan mixture weight scaled by the day's copy mass, in months 1, 2, 3, 7, 8, 9, 10 and 11 (footprints of 30, 44, 46, 250,
  150, 132, 84 and 37 rows). Flat: +0.562, -0.075, -0.386, +4.247, +0.791, +4.650, +0.533, +0.482. Scaled: +0.415, +0.152, -0.334,
  +3.395, +0.841, +4.468, +0.578, +0.370. Both forms failed the rule written before the run, in March (flat -0.386, scaled
  -0.334). Eight months could be tested: no base predictions exist for those rows in April, May, June and December. The written rule
  gave February off and June the scaled form. We kept the flat form for June, a decision taken after the result: both forms are
  positive in July to September (+9.69 flat, +8.71 scaled) and the flat form was already built.
- Published IFR count in the Rome step (`final/36_rome_count.py`): failed its gate (-0.348 board-s leave-one-month-out, nine crossing
  rows in eight months, ceiling +0.047) and is off.

Worth on the four-month score of all published-figure steps in the January/July form: about 2.4 s on January plus July, about 0.4 to
1.1 s on February plus June, so about 1.4 to 1.8 s on all four months. The capped form gives up part of that for a result that is not
negative on months it was not fitted on.

## 11. The submitted January and July, and the view columns of the correctors

**Which months come from where in the submitted file.** February and June come entirely from `final/`. January and July are carried
from the stored lineage for the expensive stages: the base predictions are those of the same refit; the ADS-B stage, the re-stack and
the correctors (with their 52 view columns, the small blend and nudge layers and Istanbul's own corrector term) are the stored results
of the earlier runs that served those months; the rule layers and the published-daily layer are rebuilt by `final/` on top. Istanbul's
own corrector term is in the submitted file only inside that carried lineage; it is not in February or June.

**What the view columns are worth, from 2025.** The 52 view columns (and two variant columns) are empty for February and June and for the
all-four-months rebuild (`docs/REPRODUCE.md` section A.7). The 12-month leave-one-month-out corrector's held-out gain (section 8.2) is, in
board seconds, 0.53, 0.37, 0.59, 1.45, 1.08, 1.01, 0.75 and 0.81 in the eight months whose fold has the columns (January, February, March,
July to November), mean 0.82, and 0.33, 0.31, 0.55 and 0.34 in the four months without them (April, May, June, December), mean 0.38. It is
positive in all twelve, so a corrector with the columns empty still helps, at roughly half the size. The two groups differ in more than the
columns (the months without them have a weaker base chain with no rules, blend or Rome handling, and fall in other seasons; July's 1.45
is a July effect), so the gap of 0.44 is an upper bound on what the columns are worth, not a measurement.
