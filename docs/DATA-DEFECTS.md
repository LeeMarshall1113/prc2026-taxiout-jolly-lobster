# Data defects: where and how the recorded off-block stamp goes wrong

The label of the PRC Data Challenge 2026 is `MVT_TIME - BLOCK_TIME`: the recorded take-off minus the recorded off-block stamp. The
take-off is an input, so the whole task is to find the stamp that the airport's system wrote down. This page catalogues the ways
that stamp is not a pushback: what each defect is, where it occurs in the 2025 training files, how to detect it, what this entry did
about it, and what would carry over to another airport or year. It is the first thing to read about this project; the model
(`docs/PIPELINE.md`) comes after it, because most of what moved the score was the identification of these records and not the
model.

Provenance. Every count in this page that is not marked "(record)" was re-run from the 12 training files of 2025
(2,085,047 departures at the ten airports) with `tools/defect_counts_2025.py`, or with the stamp checker
`tools/check_offblock_stamps.py` (the worked example is at the end). Numbers marked "(record)" are taken from the working record of
the project (listed in `docs/LAB-NOTES.md`) and were not re-run for this page, usually because they need the ADS-B ground tracks or
the held-out prediction tables. No label of the scored months (January and July 2026) was read for this page. "Squared deviation"
below always means the squared deviation of the label from the median label of its airport and month: a model-free measure of how
much of the spread of the stamp sits in the rows concerned, before any prediction. Dates and times are UTC unless a local zone is named.

Final phase. The entry has three layers (a general model, a reconciliation layer that uses the published daily delay, and fixes that name specific January and July flights; `docs/PIPELINE.md` section I.2). The counts are 2025; D2, D10, D12, D13 and D14 also use 2026 inputs. The checker was then run label-free on February and June 2026, months it was not built on, where only the input-defined flags and populations can be evaluated: 37 airport cells fall outside the range of the reference months, and the only defect-family count among them is Paris duplicate records in June ([docs/CHECKER-ON-NEW-MONTHS.md](CHECKER-ON-NEW-MONTHS.md)).

## Summary

The headline. In 2025 the ten airports' training files hold 2,085,047 departures. 584 of them (0.028%) have a label above 7,200 s
and carry 46.3% of the squared deviation of the label from its airport-month median; 4,126 (0.20%) have a label above 3,600 s and
carry 54.7%. The largest 1,447 rows carry half of it. Fifteen day-shifted stamps (a stamp dated a day early) carry 20.0% by
themselves, and Rome's 693 no-plan schedule copies carry 24.8% (48.1% of Rome's own). At Rome the 25 largest rows carry half of the
airport's squared deviation, and the schedule copies, day-shifted stamps and the plan-clock family together (5,675 of 160,704 rows)
carry 89.6% of it. The same concentration appears in the model's held-out error: rows missed by more than 1,800 s are 0.16% of rows
and 38% of squared error (`docs/ROW-RULES.md`). One corrected record moved the leaderboard score by 0.93 s (upload v145: a single
Rome departure whose long label was settled by a calendar condition on the stamp, defined on input columns and counted on 2025;
see section D4). By comparison, 30 level-distance and within-cell columns offered to the Istanbul corrector gained nothing (they
lost 0.30 s on Istanbul rows, about 0.04 s on the total; record). The columns that did help were worth 0.40 s and 0.43 s in total for
52 and 45 arithmetic columns (v150, v152), so the point is not that features are useless but that a record is worth more than a column.

| # | Defect or recording convention | Airports | Rows in 2025 | Share of those airports' rows | Share of squared deviation: of those airports / of all | Detectable from inputs |
|---|---|---|---:|---:|---|---|
| D1 | Schedule copy, no flight plan | LIRF | 693 | 0.43% | 48.1% / 24.8% | partly: the population is defined on inputs, the copy is not |
| D2 | Schedule copy, with flight plan | LIRF | 4,958 | 3.09% | 10.2% / 5.3% | partly: impossible on 5,518 Rome plan rows, a probability elsewhere |
| D3 | Schedule copy, rare but real | LEBL | 609 flagged (about 400 above background) | 0.34% | 5.5% / 0.2% | no: 1-7% habit, never confident |
| D4 | Stamp dated a day early | LIRF 13, LFPG 1, LSZH 1 | 15 | 0.003% | 31.1% / 20.0% | no: the date is the label; the population is |
| D5 | Stamp at the plan clock, Rome flights that left long after it | LIRF | 38 | 0.024% | 0.6% / 0.3% | yes |
| D6 | Plan-clock window (observed window) | all ten | 2,062,577 plan rows, 0 exceptions | 98.9% of rows | not applicable: a bound | yes |
| D7 | Duplicate movement records | LFPG (EDDM, EHAM, LIRF, LSZH: 12 records in all) | 230 records in 115 pairs | 0.096% | 6.0% / 0.5% | yes |
| D8 | Late stamp (label under 60 s) | LSZH 305, LFPG 136, EHAM 105, LEBL 65 | 611 | 0.076% | 0.3% / 0.1% | no |
| D9 | Hold record (stamp near the schedule, then a wait of hours) | EGLL 562, LTFM 522, LFPG 256, LIRF 166 | 1,506 | 0.165% | 5.1% / 3.9% | partly |
| D10 | Two populations in "no plan" | LTFM | 874 (non-Ercan) of 3,645 | 0.32% | 8.9% / 0.8% | yes for the population, no for the stamp |
| D11 | Helicopter and general-aviation movements recorded as departures | LSZH (runway H), EHAM (HG stands) | 387 + 1,572 | 0.51% | 1.8% / 0.1% | yes |
| D12 | Network manager's off-block equal to a stand default | eight airports | 904,795 plan rows (43%) | 43% | not an error | yes |
| D13 | De-icing sits inside or outside the stamp | EDDM, LSZH, EHAM, LEMD, EGLL inside; EDDF outside | see D13 | | not measured here (record) | partly |
| D14 | No ground positions (coverage) | all, by airport-month | 0% to 99% of plan departures seen | | not applicable | yes: a count |
| D15 | Portal daily delay against the stamps | all ten | 3,650 airport-days | | to 1 to 2 minutes a day at seven airports | not applicable |
| | for reference: any label above 7,200 s | EGLL, EHAM, LFPG, LIRF, LSZH, LTFM | 584 | 0.028% | 54.2% / 46.3% | |

Reading the table. "Share of squared deviation" is measured on 2025 labels and before any model; the rows of different lines overlap
(a Rome schedule copy is also a label above 7,200 s), so the shares do not add up. "Detectable from inputs" means from columns
present before the label is known; a "partly" says the population can be named but the outcome cannot.

---

## D1. Schedule copy at Rome, no flight plan

### What it is
The airport's off-block stamp is the scheduled off-block time, although the aircraft left hours later. The label is then the schedule gap
(take-off minus schedule), not a taxi time. Example, 2025-02-07, `LALL1980` to OMDW: scheduled 12:00:00, recorded off-block 12:00:05,
take-off 16:58:55. Label 17,930 s; schedule gap 17,935 s; no flight plan attached.

### Where and how often
Rome only at this strength. Of 61,016 Rome departures that took off at least 1,800 s after their schedule, 5,651 (9.3%) have a stamp within
90 s of the schedule. Of the 1,412 such departures without a flight plan, 693 (49.1%); of the 59,604 with a plan, 4,958 (8.3%) (D2). By
month the no-plan copy share runs from 0.33 (July: 107 of 328) to 0.81 (March: 44 of 54); the counts per month are 25/55, 41/52, 44/54,
41/61, 46/93, 85/174, 107/328, 88/193, 72/165, 53/108, 32/51, 59/78 (January to December). Copies belong to the reporting airline: the
intraclass correlation of the copy flag is 0.31 by operator and 0.004 by day (record). The 693 rows carry 48.1% of Rome's squared
deviation. The other nine airports have 0 to 31 no-plan copies a year (table in `tools/defect_counts_2025.py --section copies`) and no
spike at the schedule (see D3).

### How to detect it
On inputs: the population is "no flight plan, take-off at least 1,800 s after the schedule" (1,412 rows); a copy is not visible there. On
labels: `|label - schedule gap| <= 90 s` and gap >= 1,800 s; the checker flag is `schedule_copy`. False positives: an ordinary stamp
that lies within 90 s of the schedule by chance. The checker prints the evidence per airport: rows within 30 s of the schedule against the
mean of the ten neighbouring minutes (Rome 13.7 on 2025, so about 7% of the flagged plan rows in the 1,800-3,700 s band are background).
Copy status among the 1,412 rows is predictable only as a probability, from the operator's habit, the schedule gap, hour, destination
and the inbound turn (record).

### What we do about it in this entry
A mixture, not a rule: for no-plan Rome rows with a gap of 3,600-14,400 s the served value is the model value moved toward
`P(copy) x gap + (1 - P(copy)) x 1,025`, with `P(copy)` from a classifier fitted leave-one-month-out on 2025 (round 1 weight 0.5; round
2 adds destination and garbled-flight-string indicators and weights per gap bucket). Held out on 773 mid-gap rows over eight months:
+2.58 s over all months (round 1; +2.71 s in January and July, 91% of it from ten rows); round 2 over round 1 +1.17 s (January +1.14,
July +1.18, day-bootstrap probability of a gain 0.98; February and March lose). We call it a concentrated bet. Operators with at least
97% copies over at least 8 rows are set to the gap (97.9% right out of fold; 6 scored rows). Since upload v154 the daily published delay
adds a posterior for rows with a gap of at least 5,400 s (section D15). Worth: the Rome and Paris rules of v133 to v137 together -2.58 s on
the leaderboard; the posterior (v154) -2.12 s against v152. Details: `docs/ROW-RULES.md`.

### How it generalises
What transfers is the method: name the default-stamp population on input columns, estimate its probability from out-of-period habit,
serve the mixture mean and count exceptions on training data. What does not: the constants (49% copies is a Rome number), and the airline
habit, which drifts between years (a with-plan copy mixture gained +0.79 s month-held-out but +0.02 to +0.06 s with whole airline groups
held out). To apply the detector elsewhere one needs the schedule, take-off and stamp columns and a year of labels to see whether a spike
at the schedule exists (the checker's evidence line). We could not determine why Rome records a copy for about half of its no-plan
departures: whether it is a reporting default of the airlines' interface or of the airport's.

---

## D2. Schedule copy at Rome, with a flight plan

### What it is
The same copy, on a departure whose movement is attached to a flight plan. Example, 2025-07-05, `NOS523` to LIPX: schedule 09:35:00, plan
off-block (LOBT) 10:00:00, network manager's off-block 10:13:00, recorded off-block 09:34:58, take-off 10:31:02. Label 3,364 s; schedule
gap 3,362 s.

### Where and how often
4,958 of 59,604 Rome plan departures with a gap of at least 1,800 s (8.3%), flat by month at 7 to 10% (record: 2,597 of 28,043 at gaps of
1.8-3.6 ks, 649 of 8,831 at 3.6-7.2 ks, 25 of 1,631 at 7.2-14.4 ks, 1 of 275 above). They carry 10.2% of Rome's squared deviation. Elsewhere
plan copies are the coincidence rate of a smooth distribution (D3), except at Barcelona.

### How to detect it
Inputs: the plan window excludes a copy whenever the schedule is more than 3,606 s from the plan's last off-block. That holds on 66,665 plan
rows over the ten airports (5,518 at Rome) and none of them is a copy (0 of 66,665); the checker flag is `copy_impossible_plan`. On the
remaining Rome rows a copy is possible and not detectable. Labels: as D1.

### What we do about it in this entry
The Rome plan gate: for plan rows with a gap up to 7,200 s the value is half the model's and half a calibrated mixture
`P(copy) x gap + (1 - P(copy)) x taxi` (-0.53, -1.19, -1.15 s on three held-out month pairs, -0.35 s on a confirmation fold; with the cap
removed it fails). A two-sided copy mixture at dose 0.5 was added in v143 together with a corrector change (-0.26 s in all, so its own
effect is not known); it gained +0.79 s month-held-out but +0.02 to +0.06 s with airline groups held out, and an independent review put
the expected value at about +0.2 s. We list it as a bet. A one-sided earlier version lost on the leaderboard (+0.163) and was not kept.
For the 66,665 impossible rows the copy hedge is removed by the window (D6), for example `ITY340` on 2026-07-16 (a Rome plan row with a
gap of 94,560 s): the window is [83,214, 90,426] s, a copy at 94,560 s cannot have been joined, so the served value is inside it.

### How it generalises
The window rule needs only the plan clock and applies wherever movements are attached to plans by the same join (D6). The probability
part needs a year of labels and drifts with airline habit between years, which is why it is judged leave-one-airline-out.

---

## D3. Schedule copy at Barcelona (real but rare), and nowhere else

### What it is
At Barcelona a small share of stamps equals the schedule to the minute, almost always on departures whose plan clock was never moved.
Example, 2025-07-09, `ICE597` to BIKF: schedule 13:45:00 and plan clock 13:45:00, recorded off-block 13:45:05, network manager's off-block
14:10:00, take-off 14:30:49. Label 2,744 s; gap 2,749 s.

### Where and how often
Among Barcelona plan rows with a gap of 1,800-3,700 s (42,874) 482 have a stamp within 30 s of the schedule, against a mean of 83 in each of the
ten neighbouring minutes (5.8 times), so about 400 above background. The flag counts 609 rows over all gaps (0.34%); they carry 5.5% of
Barcelona's squared deviation. The copy rate is 2.5% where the plan clock equals the schedule and 0.26% where it does not, and runs by month from
0.36% (February) to 3.2% (July). At the other eight airports the same test reads 0.6 to 1.1 times background (Heathrow 1.1, Madrid 1.1, Paris
1.1, Istanbul 1.0), so their flagged rows are coincidences, not a copy habit. Rome reads 13.7.

### How to detect it
Inputs: none that separates them: a boosted classifier on raw fields reaches AUC 0.82 on Barcelona copies and is never confident (49 of
27,578 rows above probability 0.2; record). Labels: as D1, together with the minute test of the checker (`copy_spike`).

### What we do about it in this entry
Nothing is served for it. A type mixture built for it was screened and failed (held-out gain -0.001 s, probability of a gain 0.43;
record); knowing the type for every row would be worth 1.3 s of RMSE, which is not reachable from inputs. Barcelona's labels reach 3,604 s once in 179,705 rows; a one-hour ceiling is
served at Madrid only (`docs/ROW-RULES.md`).

### How it generalises
Run the minute test on any airport before trusting a schedule-copy flag: a ratio near 1 means the rows are background. What is specific is
the habit (airline-independent here, summer-heavy) and its size. We could not determine its cause.

---

## D4. Stamp dated a day early

### What it is
The stamp carries the actual clock time of the pushback written on the schedule's calendar date, so it is 24 hours too early and the
label is about 86,400 s plus an ordinary taxi. Example (Rome local dates, the form in which the rule was found): `ITYTY680`, scheduled
Monday 3 November 22:40, recorded off-block 3 November 14:04, take-off 4 November 14:22 (UTC: 21:40, 13:04, 13:22). Label 87,480 s; gap 56,520 s.

### Where and how often
Labels of 82,800-95,000 s in 2025: 15 rows in all. Rome 13: 12 without a flight plan (one at a mid gap of 11,694 s, `ITY332`; eleven at gaps of
50,581-93,535 s) and one with a plan (`AEZ146R`, label 87,002 s). Paris 1 (`EJU983W`, gap 1,740 s, label 84,240 s) and Zurich 1 (`NJE389D`,
gap 939 s, label 87,341 s). All 13 Rome records have the stamp on the schedule's local date and a take-off on a later local day. The 15
carry 31.1% of the squared deviation of the three airports, 20.0% of all; the single Paris row is 42.7% of Paris's January.

The calendar condition, stated as a count. Rome has 160,704 departures; 1,087 take off on a later local day than their schedule and 159,617 on the
same day. A day-early stamp was never observed on a same-day Rome row (0 of 159,617); all 13 are later-day rows. Among Rome no-plan
departures with a gap of 6 h or more (63 rows, all with the stamp on the schedule's local date): later-day 44 (33 copies, 11 day-early), same-day 19
(18 copies, 1 real, 0 day-early). This is not Rome's dating rule: a midnight crossing creates no day-length label in the ordinary case
(1,024 later-day Rome rows with a gap under 6 h and no day-length label, against `ITY332`); the stamp is dated by the actual clock. It is a
defect of a dozen records. It is also not universal: Paris and Zurich each have a same-day row stamped the day before.

### How to detect it
Labels: 82,800 <= label <= 95,000 (checker flag `day_shifted`; the flag selects exactly these 15 rows in 2025). Inputs: only the population
(no plan, long gap, take-off on a later local day); among the 33+11 Rome later-day rows of 6 h or more, copy and day-early share every raw field
we could test (the offset of the would-be stamp from the schedule overlaps, runway differs: 0 of 12 day-early on 16R against 10 of 33 copies;
record), and the copy share by gap runs from 25 of 25 at 6-12.6 h to 5 of 10 at 14-17 h and 2 of 7 at 17-24 h (record).

### What we do about it in this entry
Bands by schedule gap (no-plan 50-60 ks, 60-70 ks, 70-100 ks; plan 70-100 ks) serve `P(day-early) x (86,400 + offset) + P(copy) x gap`, with
single-digit counts per band; plus the calendar condition of v145: a band applies only to rows that take off on a later local day than their
schedule. One scored row changed (65,347 s to 53,520 s, a Rome departure with a gap of 14.9 h that took off the same local day). The squared-error
arithmetic was 1.4e8 s^2, about 0.93 s of RMSE if the row is a copy, and the measured step was -0.93 s; if it had been day-early the cost would
have been about 4.3 s (break-even probability 18%). The rule rests on the count above, not on a known property of the airport's system, and is stated that way.
The five other long-gap rows are served between the two record types by choice (their 2025 analogues split evenly, and no input separates them);
a re-centring of four of them from a 45-record estimate scored 0.474 s worse (v157) and was not kept. The leaderboard use around these
one-row uploads is described in `docs/ROW-RULES.md`.

### How it generalises
To apply the detector: take the label band and tabulate, for every airport, whether the stamped date equals the schedule date and whether
take-off is on the same local day. The rule "needs a later-day take-off" holds at Rome on 2025 and fails as a general statement. Specific
to this dataset: the 86,400 s offset (a date field filled from the schedule), the airline list. We could not determine how the airport system
fills the date field, which is what would turn the count into a rule.

---

## D5. Stamp at the plan clock on Rome flights that left long after it

### What it is
For a small family of Rome plan departures the stamp sits at the plan's last off-block time (LOBT), and the plan window forces it: the lower
edge of the window is far above the network manager's own reading of the taxi, so an ordinary taxi is excluded and only the stamp at the plan
clock (which is usually also the schedule) is left. Example, 2025-04-14, `ISR348` to LLBG: schedule 10:10:00 = LOBT 10:10:00, network manager's
off-block 11:56:00, recorded off-block 10:10:02, take-off 12:17:58. Label 7,676 s = take-off minus LOBT (7,678 s) minus 2 s.

### Where and how often
Condition on inputs: plan row, lower window edge `L = (take-off - LOBT) - 3,606` above 2,800 s and more than 1,800 s above the network manager's
reading (take-off minus AOBT_3). Rome 2025: 38 rows; the label is within 60 s of take-off minus LOBT on 21 and within 900 s on 31; the mean position of the label in
the window (0 = lower edge, 1 = LOBT) is 1.00 and none sits at the lower edge; the plan clock equals the schedule on 29 of the 38. The same
condition elsewhere: Heathrow 72 rows (5 within 60 s of LOBT; mean position 0.68), Paris 48 (1; 0.66), Istanbul 6 (0.46), Frankfurt 7 (0.49),
Munich 4 (0.32), Zurich 3, Madrid 1, Amsterdam and Barcelona 0: no stamp-at-clock family. The 38 rows carry 0.6% of Rome's squared deviation.
A softer neighbour (EOBT_1 minus LOBT at least 3,600 s, lower edge 0-2,800 s): 30 rows, 18 within 60 s of LOBT, mean position 0.97.

### How to detect it
Inputs, all of them: LOBT, take-off, AOBT_3. Labels: the position of the label in the window. Expected false positives at other airports:
most, since the condition selects long waits at stand and only at Rome does the stamp land on the plan clock.

### What we do about it in this entry
Serve take-off minus LOBT on those rows (v148: 14 scored rows, January 3 and July 11). Expected 0.2 s, measured -0.21 s; the held-out gain is
concentrated in 37 rows, and the worst case, if all rows sat at the window edge, is +0.58 s. The softer neighbour (v151, three scored rows) scored 0.067 s worse and was
not kept. `docs/ROW-RULES.md`.

### How it generalises
The check is cheap anywhere with a plan clock and a network manager clock: tabulate where in the window the label sits for rows whose lower
edge is high. If the histogram has a mass at the plan clock, the family exists. It exists at Rome only on 2025; its cause (a stamp not updated when
the aircraft is held after a re-filed plan) is our reading, not a documented process.

---

## D6. The plan-clock window: an observed window

### What it is
In 2025 every plan departure has its recorded off-block within 3,606 s of the plan's last off-block time (2,062,577 rows, 0 exceptions). We read this as the
tolerance of the join that attaches a movement to its plan, so a plan row's label satisfies `take-off - LOBT - 3,606 <= label <= take-off - LOBT + 3,606`
(one hour plus 6 s; 3,606 s is the largest value observed). On that reading it is a property of the join in the database and not of the aircraft. Example, 2025-01-05, `NOZ1719` at Madrid: LOBT 19:30:00, recorded off-block
20:30:05, take-off 20:46:51: label 1,006 s, take-off minus LOBT 4,611 s, so the label sits 3,605 s inside the upper edge.

### Where and how often
2,062,577 plan departures (98.9% of all), 0 exceptions; maximum `|label - (take-off - LOBT)|` is 3,606 s (EDDM, EGLL, LEBL, LEMD, LFPG,
LIRF, LTFM) and 3,602-3,605 s at the others; every month has a maximum of 3,604-3,606 s. In 66,665 plan rows the schedule is more than 3,606 s from the
plan clock, so a schedule copy is impossible there (0 copies). The no-plan rows (22,470; 1.1%) are the movements for which this join failed or
no plan exists (D10).

### How to detect it
Inputs give the bound on the label; labels test it (checker flag `label_outside_window`, zero expected). A hit means a join error in the table,
not a flight.

### What we do about it in this entry
A hard bound on every prediction (plan rows are moved to the nearer edge), plus a fitted lift for the 359 rows whose lower edge is more than 1,800
s above the network manager's reading. Held out on three month-pair folds, predictions outside the box were 94, 46 and 61 rows with RMSE
changes of -1.20, -0.15 and -0.98 s; removing the rule from the January/July chain costs +0.77 s. 82 scored predictions were raised
(`docs/ROW-RULES.md`). Later additive terms pushed ten plan predictions out again, and v146 moved them back.

### What breaks if it is ignored
A model that does not know the window predicts a copy or an ordinary taxi where the join excludes it (D2: a copy hedge on 66,665 rows where a
copy cannot be), predicts outside [lower, upper] on the long-wait rows (the held-out changes above), and misreads the no-plan population as
ordinary flights (D10): the no-plan rows are exactly those whose stamp or plan failed the join, which is what gives the hidden-plan bounds
(`docs/ROW-RULES.md`).

### How it generalises
Any dataset that attaches movements to plans with a time-window join has such a bound; the width has to be re-counted (here 3,600 s of
tolerance, 6 s of margin). Specific to this dataset: the width, the field names, and that no exception occurs in 2.06 million rows. What we could not
determine is whether the join rule is the same in the export for other years; the checker's `label_outside_window` is the first test to run.

---

## D7. Duplicate movement records at Paris

### What it is
Two records of one flight within minutes: one carries a flight plan and an obsolete early stamp near the schedule, the other has no plan and an
ordinary stamp. Example, 2025-01-17, `AFR1192` to LSSB: schedule 07:30:00. Record 1 (plan, LOBT 08:20, network manager's off-block 09:41): stamp
07:51:57, take-off 09:54:53, label 7,376 s. Record 2 (no plan): stamp 09:23:01, take-off 09:54:59, label 1,918 s.

### Where and how often
Paris 2025: 115 pairs (230 records), none in a larger group; 107 have one plan record and one without, 8 have a plan on both. In 101 of the 107
the plan record's label exceeds its partner's by more than 300 s, and in 85 of 85 pairs whose plan record has a gap of 3,000-50,000 s. Median
take-off spread within a pair 22 s (1 to 1,739 s); median label of the plan record 4,453 s, of the no-plan record 1,082 s. The 107 one-plan pairs by month: 11, 8, 9, 7, 11, 6, 9, 9,
11, 7, 10, 9. Other airports: EDDM 4 records, LIRF 4, EHAM 2, LSZH 2. The Paris records carry 6.0% of Paris's squared deviation.

### How to detect it
Inputs only: same airport and flight string, take-off within 1,800 s of another record (checker flag `duplicate_movement`). Expected false
positives: a flight number reused for two flights within half an hour; none seen in 2025.

### What we do about it in this entry
For a plan record whose partner has no plan and with `take-off - schedule - 780 <= 14,400 s`: value `0.75 x clip(take-off - schedule - 750, 60, 14,400) + 0.25 x
prediction` (750 s is the median offset over the other months' pairs). Leave-one-month-out over 98 rows: January +0.41 s, July +0.46 s,
months 1 and 7 together +0.87 s, all 12 months better; the weight 0.75 is the best of 0.5, 0.75 and 1.0. 16 scored pairs (5 in January, 11 in
July). The six exceptions in 2025 are five pairs with a plan-record gap under 3,000 s and one with a day-shifted schedule. Part of the Rome and Paris
bundle, -2.58 s on the leaderboard (v133 to v137).

### How it generalises
Transfers as a check (count close pairs by airport). The rule needs the same pairing of a plan record with a plan-less one; the offset and weight are
Paris numbers. We could not determine why the Paris system emits a second record.

---

## D8. Late stamps

### What it is
The stamp is written close to take-off, so the label is near zero on an ordinary departure. Example, 2025-01-03, `AFR1510` at Paris: schedule 14:30:00, network
manager's off-block 14:49:00, recorded off-block 14:59:56, take-off 15:00:04. Label 8 s.

### Where and how often
Labels under 60 s on rows that are not helicopter or runway-H movements: Zurich 305 (0.23% of rows), Paris 136 (0.057%), Amsterdam 105 (0.042%), Barcelona 65 (0.036%),
Rome 19, others at most 7. By month the rate is 0.12-0.62% at Zurich, 0.02-0.13% at Paris, 0.01-0.07% at Amsterdam, 0.00-0.13% at Barcelona: no
season. A wider view at Amsterdam: 6,004 non-helicopter plan rows (2.46%) have a label under 300 s, and the network manager's off-block precedes their stamp
by a median 7.8 min against 1.7 min on the others. On the wider measure (plan row, label under 300 s, network manager's off-block 5 minutes or more before the stamp) no hour of the day exceeds 3.6% at Amsterdam, and at Paris the excess sits at 00-01 UTC (4.8%, against 0.8% or less in every other hour except 23-24 UTC at 1.2%). The rows carry 0.3% of
the squared deviation of their airports because the label is close to the median.

### How to detect it
Labels only (checker flag `near_zero_label`). Inputs: none (a boosted classifier reaches AUC 0.90 for Amsterdam and Barcelona, but the chain already serves those rows low; record).

### What we do about it in this entry
Nothing specific. A type mixture failed its screen (held-out gain -0.108 s, probability of a gain 0.00; record): the corrector prices what can be
priced. Knowing the type for every row is worth 0.6 s of RMSE (record).

### How it generalises
Count by airport and month; look at the network manager's field against the stamp as the signature. The cause (a stamp made at departure) is our reading. Not determined
whether it is a handling procedure or an interface behaviour.

---

## D9. Hold records

### What it is
The stamp is within 10 minutes before to 40 minutes after the schedule, and the aircraft then waits one and a half to three hours before taking off. The label is the
schedule gap minus a few minutes. Example, 2025-01-21, `BAW46EH` at Heathrow to EHAM: schedule 13:50:00 = plan clock 13:50:00, recorded off-block 13:51:59,
network manager's off-block 15:41:00, take-off 15:51:00. Label 7,141 s; the network manager's reading says 600 s.

### Where and how often
Label of at least 3,600 s, stamp 10 min before to 40 min after the schedule, not a copy, not day-shifted (checker flag `hold_like`): Heathrow 562, Istanbul 522,
Paris 256, Rome 166, Amsterdam 52, Frankfurt 47, Munich 35, Zurich 19, Barcelona 1, Madrid 0; 1,506 rows at the first four, 5.1% of their squared deviation. Days:
Istanbul 62% on five February days (19, 23, 11, 6 and 24 February, 129 rows on the 19th); Heathrow 562 rows on 211 days, 16% on the five busiest (31 December 32, 27 July 17, 1 June 15, 3 July
15, 6 July 11), 27 days with at least five rows; Paris 22 on 17 January, 17 on 21 January, 15 on 22 November; Rome 21 on 13 July; Amsterdam 17 on 5 January.

### How to detect it
Labels. Inputs: partly. For plan rows with a gap of 5,400-50,000 s, the rate of hold-like rows when the plan clock is within 15 minutes of the schedule is 59% at Heathrow (162 of
275), 47% at Paris (82 of 173), 57% at Istanbul (37 of 65) and 24% at Rome (26 of 107); when the plan clock is 15-45 minutes after the schedule 3% (Heathrow), 5% (Paris), 13% (Istanbul),
5% (Rome); beyond 45 minutes 0.2-0.7%. Those first cells hold at least 29% (Heathrow), 32% (Paris), 7% (Istanbul) and 16% (Rome) of all the hold-like rows, so most hold records sit where
the rate is low, and where it is high it is a coin flip.

### What we do about it in this entry
Nothing served. A boosted type model reaches AUC 0.965 held out by month, but the mixture gains +0.56 s [-0.08, +1.45] pooled on January and July, with January -0.107,
airline-held-out negative and -0.25 after removing the 20 best rows; one event is two thirds of the gain (record). With the type known the pool would be worth 4.9 s (no-plan
long labels) and 2.4-3.2 s (plan rows with a stale stamp) of RMSE (record). The window and the network manager's reading bound the long rows without a model of the hold.

### How it generalises
Count by airport and day; a hold record needs a disruption (snow at Istanbul in February, stop days at Heathrow and Paris) and so depends on the year's weather and operations, which a
training year cannot predict. Specific: the 10- and 40-minute edges and the day lists. Not determined: whether the stamp reflects the departure being scheduled and then held
by the airport, or the stamp being made at the schedule by default.

---

## D10. "No plan" is not one population: Istanbul

### What it is
Istanbul's flights without a flight plan are two groups with different physics: flights to Ercan (LCEN), for which a plan is never matched, and the rest, whose plan or stamp failed the join.
Example, a normal Ercan row, 2025-01-01, `THY6KN`: schedule 11:30:00, recorded off-block 11:22:03, take-off 11:39:56, label 1,073 s. A displaced stamp to the rest, 2025-01-01, `THY92` to OEJN:
schedule 06:00:00, recorded off-block 06:10:03, take-off 08:12:57, label 7,374 s.

### Where and how often
3,645 Istanbul no-plan departures: 2,771 to LCEN (mean label 1,046 s; 4 above 3,600 s; 94.5% with a gap under 3,000 s) and 874 to other destinations (mean 2,111 s; 217 above
3,600 s). Only one of the 2,772 LCEN departures has a plan. In 2026, 594 of 721 no-plan rows are to LCEN (record). The 874 carry 8.9% of Istanbul's squared deviation.

### How to detect it
Inputs: destination, and the plan flag. Labels: not needed. A bound on inputs: for no-plan rows that took off at or after the schedule the label cannot exceed the gap by more than 1,200 s:
0 exceptions in 3,585 rows (maximum excess 903 s, all 12 months); outside Istanbul and Rome the same bound has 7 exceptions in 16,672 no-plan rows, so it is built for Istanbul only.

### What we do about it in this entry
The February snow refit had learned its large no-plan values from the 874 and applied them to short-gap Ercan flights; v146 caps Istanbul no-plan predictions at gap + 1,200 s (13 scored rows
lowered, up to 3,093 s; worth 0.16 to 0.42 s depending on the truth) (`docs/ROW-RULES.md`).

### How it generalises
Split every "no plan" population by destination and by the join's failure mode before pooling. Specific: LCEN and the Istanbul join. Not determined: why a plan is never matched for LCEN.

---

## D11. Helicopter and general-aviation movements recorded as departures

### What it is
Movements that are not airline departures sit in the file with a stamp that is a take-off, or a position, not a pushback. Example, Zurich, 2025-01-13, `HBXAT` on runway H, stand HPW: schedule 13:31,
recorded off-block 13:30:12, take-off 13:30:00: label -12 s. At Amsterdam, `NCG01A` on stand HG01 on 2025-01-06: recorded off-block 05:35:04, take-off 08:38:56: label 11,032 s.

### Where and how often
Zurich runway code H: 387 rows (368 under 120 s; mean 36 s; 384 without a plan); the types are helicopters (AS50 135, A109 83, B429 43, EC20 38, EC30 25, AS32 13, A139 10) and a few light aircraft. Runway H exists at Zurich only. Amsterdam
stands HG*: 1,572 rows (1,423 without a plan; median label 187 s), 992 of type A139 and 459 of a fixed-wing type (DH8A) at helicopter stands; the one row over 7,200 s is `NCG01A` (11,032 s), which is 8.8% of Amsterdam's held-out squared error (record). Flight rule V: Amsterdam
1,619 rows (23% under 120 s), Zurich 1,125 (41%), Barcelona 136. Together the helicopter and runway-H rows carry 1.8% of the squared deviation of the two airports.

### How to detect it
Inputs: runway code and aircraft type (checker flag `heli_or_runway_h`) and flight rule (flag `vfr`). Expected false positives: fixed-wing aircraft on helicopter stands.

### What we do about it in this entry
No row rule. The chain serves them low (no-plan helicopter cell served mean 202 s against 192 s in 2025). About 12 Zurich rescue-helicopter rows were served 400-1,100 s against a 2025 median of 0 s: a hedge worth at
most 0.04 s was identified and not built (record).

### How it generalises
Flag these movements in any airport table before modelling. Which stands and types qualify is airport-specific (the checker's helicopter type list is editable). Not determined: how the single Amsterdam 11,032 s row arose.

---

## D12. The network manager's off-block field is often a stand default

### What it is
The flight plan carries the network manager's own off-block (`AOBT_3`). On many rows it is not an observation but take-off minus a stand-specific default taxi time, rounded to the minute. Example, Barcelona,
stand 264, runway 24L (2,254 rows in 2025; 87% have `AOBT_3` = take-off minus 585-655 s, mode about 630 s): `TAP10X`, 2025-01-03: `AOBT_3` 17:25:00, take-off 17:35:47 (647 s), recorded off-block 17:18:01, label 1,066 s.

### Where and how often
Share of plan rows that look defaulted by the checker's label-free flag (stand-and-runway mode of take-off minus `AOBT_3` on a 10 s grid, box +-35 s, at least 100 rows and 30% of them in the box), whole of 2025: Barcelona 87%, Madrid 80%, Munich 79%, Zurich 75%,
Istanbul 51%, Frankfurt 41%, Paris 31%, Rome 22%, Amsterdam 4%, Heathrow 0.06%. `docs/GENERALISATION.md` quotes 91, 79, 75, 72, 67, 35, 19, 9, 1 and 0% (Barcelona, Zurich, Madrid, Munich, Istanbul, Frankfurt, Rome, Paris, Amsterdam, Heathrow) from a stricter earlier flag. The two agree on the extremes (high at Barcelona, Madrid, Munich and Zurich, near zero at Amsterdam and Heathrow); the middle of the order and the sizes depend on the definition (Paris 9% there, 31% here).
Drift: the share falls by 2-9 points between 2025 and 2026 at Frankfurt, Munich, Madrid, Zurich and Paris (record).

### How to detect it
Inputs only (checker flag `nm_default_offblock`; learned on the table given, so a single month finds fewer keys than a year). False positives: a stand with a genuinely peaked real taxi time (dense stand peaks at large hubs).

### What we do about it in this entry
The plan-clock distance columns of the corrector are split by a default-or-report flag. As a stand-alone corrector input the flag gained -0.057 s pooled on January and July but -0.043 against a shuffled flag, and the 20 most-improved rows carry 155% of
it (record): no gain. This is a recording convention, not an error: it says the "network manager taxi" feature is a stand constant on those rows.

### How it generalises
Any feature built from this field (taxi proxies, neighbours' proxies) means "stand default" at the airports above and "observed" at Heathrow and Amsterdam; check the share per airport and year before using it.

---

## D13. De-icing: where the stamp is taken, and which airports carry it inside taxi-out

### What it is
The stamp is taken when the aircraft leaves its stand. De-icing at a remote pad after that is inside the label; de-icing on the stand before push-back is not. So the same published count of de-iced flights means different things at different airports.

### Where and how often
(Record; needs ground tracks, not re-run here.) A label-free detector from the ADS-B ground tracks finds the pads as dwells of at least 180 s, at least 200 m from the stand, that are enriched on cold observations (80 cells in 21 pads at six airports); a flight is de-iced when a dwell lies in a pad. Munich 11 pads
(runway heads 2.1-3.7 km from the stand), Zurich one main pad, Amsterdam one platform (1.5 km), Madrid two, Heathrow remote stands, Frankfurt three small cells; Paris, Rome, Barcelona none found; Istanbul has no tracks. The excess taxi-out of a flagged flight over its unflagged
airport-day-hour-runway neighbours is +551 s at Munich, +478 Zurich, +922 Amsterdam, +557 Madrid, +887 Heathrow, +635 Frankfurt. First movement minus the stamp is the same on cold and warm days at every airport, so no treatment appears after the stamp on the stand. Agreement with the portal's published monthly
count of excluded flights (`TF - VALID_FL`) at airports where that exclusion is de-icing: Munich 0.83 to 1.08 over five months of 2025 and 0.95 in January 2026 (3,239 of 3,426); Zurich 0.43 in January 2025, 1.04 and 0.95 in November and December, 0.79 in January 2026; Madrid 0.90 to 0.96. Frankfurt: 0.02 to 0.36: the published
1,993 January 2026 de-icings mostly sit before the stamp, not in the label. At Amsterdam the published exclusions are helicopters and local movements, not de-icing; at Heathrow the portal does not exclude de-icing; Paris and Istanbul are blind.

### How to detect it
Inputs: the ground track (pad dwell), or a weather-only propensity (AUC 0.94 in January, calibrated per month, ranking within an airport-month; its level needs the published count). Labels are not needed.

### What we do about it in this entry
The chain already prices seen de-iced flights: the mean held-out residual on flagged flights is +1 to +5 s at Munich, Zurich and Amsterdam, -9 s at Madrid, +26 s at Heathrow. The unseen-row problem does not occur in the scored January at the airports where de-icing can be seen (97% of Munich rows are seen). No de-icing model was built; the
detector is an instrument.

### How it generalises
For any airport, find where the stamp is taken (stand leave, off-block, push) before comparing it with a de-icing count. The detector needs surface tracks and a temperature; the pad cells are airport-specific. Not determined: the on-stand treatment at Frankfurt (treatment before the stamp) cannot be seen in the label at all.

---

## D14. Ground-track coverage is a data condition

### What it is
The open ADS-B network hears an aircraft on the ground only where a receiver covers the apron, and that changes by airport, month and year. A ground-track stage built on 2025 can find itself in a month with no ground positions.

### Where and how often
Share of plan departures with a matched ground track and a seen lift-off (saved matched-departure tables, 2025; definitions differ somewhat from the table in `docs/GENERALISATION.md`, which has its own definition of "seen"): Rome 0% in January to April 2025, 9% May, 68% June, 88% July, 82% August, 18% September, 37% October,
37% November, 57% December; Paris 0% January to May 2025, never above 19%; Madrid 0% January and February 2025, 12-13% July and August; Munich 36% July 2025; Heathrow 4-9% in April to June and 43-48% in January to February; Istanbul 0% all year. Scored months, 2026 (same definition): Rome 65% in January and 0.1% in July (18 of 15,562); Paris 6%;
Madrid 96% and 53%; Frankfurt 55% and 44%; Heathrow 72% and 97%; Istanbul 0%. In July 2026 the ground positions around Rome fell to 2,184 points against 403,000 in January, with zero on 27 of 31 days. The data carries no receiver identifier, so whether one receiver or several went quiet cannot be told (record). A third of the January and July departures (about 111,000) are first heard on a taxiway more than 100 m from their stand, so
there is no stand-leave to measure (record).

### How to detect it
Count matched tracks by airport and month, for the training year and the target year. Label-free.

### What we do about it in this entry
The ADS-B stage is judged on months of the same reception regime as the target month, and the re-stack learns when to trust the track. The stage kept 76 to 81% of its held-out gain on the leaderboard; bulk-model changes about 38% (`docs/GENERALISATION.md`). A physical back-projection from a taxiway-first fix gave nothing usable (record).

### How it generalises
Always take the census first. Specific: the receivers around these airports and their history. Not determined: the cause of the Rome July 2026 gap.

---

## D15. The portal's daily delay against the recorded stamps

### What it is
EUROCONTROL's performance portal publishes a daily all-cause pre-departure delay per airport (`DLY_ALL_PRE_2`). On 2025 it is a function of the recorded stamps: the sum over the day's departures of max(off-block minus schedule, 0), each flight filed under the UTC date of its stamp.

### Where and how often
Over the 365 days of 2025, median and 90th percentile of the absolute daily difference (published minus ours) in minutes, and the same with VFR departures dropped (flight rule V is an input):

| airport | all rows: median / p90 | mean difference | without VFR: median / p90 | reading |
|---|---|---:|---|---|
| LIRF | 0.8 / 2.0 | -0.7 | 0.8 / 2.0 | matches |
| EGLL | 1.0 / 2.7 | +0.9 | 1.0 / 2.7 | matches |
| EDDM | 1.0 / 2.4 | +2.2 | 0.9 / 2.4 | matches |
| LTFM | 1.3 / 3.5 | +16.3 | 1.3 / 3.5 | matches |
| LEMD | 1.5 / 3.4 | +1.5 | 1.4 / 3.3 | matches |
| EDDF | 2.1 / 4.2 | -0.5 | 2.1 / 4.0 | matches |
| LEBL | 1.3 / 19.4 | -5.2 | 0.9 / 2.5 | matches on most days; matches without VFR |
| LSZH | 6.0 / 49.1 | -18.1 | 1.2 / 2.9 | different flight set: matches without VFR |
| EHAM | 39.8 / 172.1 | -66.5 | 2.5 / 4.8 | different flight set: matches without VFR |
| LFPG | 34.1 / 156.4 | +68.7 | 34.1 / 156.4 | different flight set: the portal counts about 69 min a day more than our rows; not resolved |

The slope is 1.000 to 1.005 at every airport. The portal follows the stamp: at Rome the 429 long-gap schedule copies (label within 90 s of the gap, gap over 7,200 s) fall on 224 days
of 2025. Published minus the stamp-based sum is -0.7 min a day on those days, and it would be -458 min a day if the copies had counted as real late departures. A copy counts as an on-time
departure, and so does a stamp written a day early.

### How to detect it
Inputs plus the published file; the test is the table above.

### What we do about it in this entry
The identity is used as a method, in the reconciliation layer (layer 2; `docs/PIPELINE.md` section I.3 has the full statement). The organisers' answers are quoted in `docs/DATA-SOURCES.md`. The layer is kept apart from the model (`docs/GENERALISATION.md` section 6).

**The score both ways.** Without any published figure about the scored months: v152, 216.2405 s. With them: v155, 213.7625 s, and v162, 213.2898 s (v162 also contains the Istanbul corrector, 0.0524 s, which uses no published figure). The first form of the step is worth 2.4780 s on the board (v155 minus v152); the day-level calibration alone was worth 0.3540 s (v153) and the record types alone 2.1205 s (v154), each against v152. The final pipeline uses the half-strength form of the day shift; the Rome record-type part is not capped (`docs/GENERALISATION.md` section 10).

### How it generalises
As a method it needs the daily file and the stamps of the same flight set; seven airports of ten reproduce it, and two of the other three once VFR departures are dropped. It cannot be applied before the figures exist. Not determined: the flight set of the portal at Paris.

### A finding for the data's owners
The portal's daily delay counts a copied stamp as an on-time departure. At Rome, on 224 days of 2025, 429 flights carry an off-block stamp equal to the schedule although they left on average several hours late; on those days the published delay is lower by about 458 minutes a day than if those flights were counted from schedule to take-off (their taxi included). A stamp written a day early is counted the same way. Where such records exist the published pre-departure delay understates real delay, and the identity above is a way to measure how many.

Counted on the UTC date of the stamp: 429 copies on 224 days, mean schedule gap of the copies 239 minutes, published minus the stamp-based daily sum -0.7 minutes, and -458 minutes if the copies were counted as real delay. The stamp checker finds the records. At Paris the published series is about 69 minutes a day higher than the stamps give, and we cannot attribute that.

---

## A worked example: the stamp checker on two 2025 months

`tools/check_offblock_stamps.py` flags the record defects (`schedule_copy`, `day_shifted`, `duplicate_movement`, `near_zero_label`, `hold_like`, `heli_or_runway_h`, `label_outside_window`)
and the input-only conditions (`copy_impossible_plan`, `nm_default_offblock`, `no_plan`, `vfr`), writes one row per flight, and prints a per-airport-month summary with the share of squared deviation from the airport-month median that the flagged rows carry.

```
python tools/check_offblock_stamps.py data/raw/training_2025-01-01_2025-02-01.parquet data/raw/training_2025-07-01_2025-08-01.parquet --out flags.parquet
```

Both files together: 344,419 departures, one second of run time, 1.55 GB peak memory (measured under a 2 GB cap). Extract of the printed summary (the columns give flagged rows, rate in % of the airport's rows, and the share of squared deviation in %):

```
LIRF   (rows: 2025-01 11317, 2025-07 15211)
  flag                       2025-01  rate % sq-dev %     2025-07  rate % sq-dev %
  schedule_copy                  234    2.07     49.7         791    5.20     51.9
  day_shifted                      1    0.01     42.6           5    0.03     51.3
  duplicate_movement               0    0.00      0.0           4    0.03     20.5
  near_zero_label                  0    0.00      0.0           1    0.01      0.0
  hold_like                        7    0.06      0.6          61    0.40      1.1
  heli_or_runway_h                 0    0.00      0.0           0    0.00      0.0
  label_outside_window             0    0.00      0.0           0    0.00      0.0
  copy_impossible_plan           286    2.53      0.4        1069    7.03      0.9
  nm_default_offblock           1088    9.61      1.2        1989   13.08      1.1
  no_plan                         60    0.53     88.4         337    2.22     78.0
  vfr                              0    0.00      0.0           0    0.00      0.0
  any_record_defect              242    2.14     92.9         858    5.64     94.1

EGLL   (rows: 2025-01 19420, 2025-07 20790)
  flag                       2025-01  rate % sq-dev %     2025-07  rate % sq-dev %
  schedule_copy                  235    1.21      6.7         271    1.30      7.0
  day_shifted                      0    0.00      0.0           0    0.00      0.0
  duplicate_movement               0    0.00      0.0           0    0.00      0.0
  near_zero_label                  3    0.02      0.1           1    0.00      0.0
  hold_like                       51    0.26     21.6         131    0.63     30.8
  label_outside_window             0    0.00      0.0           0    0.00      0.0
  copy_impossible_plan           534    2.75      3.3         802    3.86      4.9
  nm_default_offblock              0    0.00      0.0           0    0.00      0.0
  no_plan                        128    0.66      2.0         338    1.63     10.5
  any_record_defect              289    1.49     28.4         403    1.94     37.8

LFPG   (rows: 2025-01 17900, 2025-07 21712)
  schedule_copy                  127    0.71      1.9          49    0.23      3.1
  day_shifted                      1    0.01     42.7           0    0.00      0.0
  duplicate_movement              24    0.13      1.1          24    0.11     15.0
  near_zero_label                 24    0.13      0.1          11    0.05      0.3
  hold_like                       66    0.37      4.9          22    0.10     18.0
  any_record_defect              235    1.31     49.7          97    0.45     21.7

LSZH   (rows: 2025-01 9735, 2025-07 12611)
  near_zero_label                 26    0.27      0.7          33    0.26      1.5
  heli_or_runway_h                66    0.68      4.5          37    0.29      1.6
  vfr                            108    1.11      4.8         118    0.94      9.9
  any_record_defect              114    1.17     14.8          81    0.64     15.0

schedule-copy evidence (plan rows, gap 1,800-3,700 s): rows within +-30 s of the schedule / mean of the 10 flanking minutes (ratio)
  EDDF: 19 / 22 = 0.9    EDDM: 11 / 17 = 0.7    EGLL: 162 / 166 = 1.0    EHAM: 14 / 18 = 0.8    LEBL: 151 / 12 = 12.4
  LEMD: 22 / 25 = 0.9    LFPG: 63 / 56 = 1.1    LIRF: 614 / 50 = 12.4    LSZH: 8 / 11 = 0.7     LTFM: 108 / 104 = 1.0
```

How to read it. At Rome the flagged record defects (`any_record_defect`) are 2% to 6% of rows and 93% to 94% of the squared deviation of the stamp in these months; in January a single day-shifted record is 42.6% of Rome's January. At Heathrow the schedule-copy flag
fires on 1.2% to 1.3% of rows but the evidence line reads 1.0: those are coincidences, and the real record defect at Heathrow is the hold record (22% and 31% of the squared deviation on 0.26% and 0.63% of rows). At Paris 24 duplicate records a month carry 15% of July. `nm_default_offblock` depends on the table size: on the two
months together it reads lower at some airports than on the whole year (D12). `any_record_defect` includes the coincidental schedule copies at airports where the evidence line is near 1.
