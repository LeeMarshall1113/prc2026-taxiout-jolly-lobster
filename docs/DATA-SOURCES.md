# Data sources

Every external dataset the entry uses or tested, for the PRC Data Challenge 2026 documentation requirement ("external datasets openly
accessible and documented"). Items marked **UNVERIFIED** were not checked against the primary source. Licence statements in sections 1 and 2 are what our scripts and records said when they were written; section 1b records the licence text read from each official page on 2026-10-02, and where the two differ, section 1b is the later reading.

No external data is committed to this repository (`data/` is git-ignored). Every external file below is re-fetchable
with the script named in its row.

## The rule on external data, and organiser statements

The written rule is on the challenge's Eligibility page (https://prc-data-challenge-2026.netlify.app/eligibility.html, read
2026-10-08). For a prize, it requires that "All used external datasets are openly accessible/usable and documented", that
additional datasets are available under an open licence, that the source code is public on GitHub under GPLv3, and that the
documentation is sufficient to understand and reproduce the results. The Ranking page adds that submissions are monitored for
attempts to learn from or exploit the ranking process. Every external source this release uses is listed below with where it
comes from, its licence line and the script that fetches it.

The organisers stated to participants (2026-09-08 and 2026-10-01) that METAR weather data may be used, that open-source data
may be used if it is declared in the documentation of the solution, and that external data must be openly licensed (data
without an open licence is not open). They also answered on 2026-09-17 that EUROCONTROL's performance-portal files can be
considered open data for the challenge, and on 2026-09-27 that the published monthly taxi-out figures for the scored months
(January and July 2026) may be used as model input. On 2026-10-08, after the final phase was announced, they answered a
participant who asked whether shifting predictions toward a published EUROCONTROL aggregate of the scored months was still
acceptable that nothing had changed and it was. The daily pre-departure delay file used here comes from the same portal and is
the same kind of published aggregate; none of these answers names it specifically. Dates against uses: the Munich shift was first uploaded about 2026-09-20 and v101 on 2026-09-24, both before the 2026-09-27 answer about the monthly figures.

What in the last uploaded file uses published EUROCONTROL figures. **The published daily delay files** (section 2.5) are used by the reconciliation pieces of the best scored file: a day-level calibration and posterior Rome record types (uploads v153 to v155), the day-level calibration extended to Istanbul, Zurich and Paris (v159 to v161) and a split of the day shift by coverage class and lateness bin (v162; two clauses of its gate waived before upload). They are described in `docs/PIPELINE.md` section I.3 and section 2.5 below; in the final pipeline only the half-strength day layer (layer 2) reads them. The organisers answered on 2026-09-17 that the performance portal's files can be considered open data for the challenge. Two pieces that were built and not uploaded (v163, v164) read the same files; v164 also read the daily departure count at Rome by off-block date. **The published monthly figures**: no feature, no fitted component and no post-chain rule of the last uploaded file is built from them (checked by reading the scripts, 2026-10-05, for the v146 chain; the later pieces read only the daily files). They were used in two places earlier: (a) one early upload, v101, aligned
airport-month means to them; it scored worse and is not in the later lineage; (b) one post-chain level-shift rule, Munich January
2026 (`archive/apply_eddm_january.py`, 7,062 rows, -38 s), whose size was set from three instruments, one of which was the published
January 2026 figure for Munich. That comparison was made on all flights while the published figure excludes flights de-iced after
off-block; the rule was removed in v144 (`docs/ROW-RULES.md`, `docs/GENERALISATION.md` section 3). The like-for-like check that led to
the removal (`archive/agent_apt_EDDM_01_level.py`) reads the published January mean as an analysis input; its output is not a component
of the file. What remains of EUROCONTROL work in the repository: the reference taxi time `ref_taxi_s` (`prc/reference.py`), computed
from our own training data by the recipe of the indicator document (a published method, no EUROCONTROL data file is read), and the
analysis scripts that read `data/external/eurocontrol/` (the scripts whose names begin `eurocontrol_`, and several other studies), which are tests and are not part of the chain.
A further use of the published monthly figures was tested on 2026-10-04 and dropped: the published monthly counts of Rome taxi-outs over 7,200 s and per-runway means were used as constraints and as checks on the served file (the file already implies the published
counts: January 43.6 against 44, July 82.1 against 75), and no component reads them. The daily delay series is a different use and is in the uploaded files from v153 (section 2.5).

## 1. Summary table

| Source | Used by the CURRENT best pipeline? | Licence as documented | Open licence documented? |
|---|---|---|---|
| Competition data (OpenSky bucket `prc-2026-datasets`) | YES, the only labelled data | Organiser terms; not redistributed here | n/a (organiser-supplied, not an external dataset) |
| adsb.lol historical ADS-B (`globe_history_2025`, `_2026`) | YES (ADS-B stage, stand map, all ADS-B features) | ODbL 1.0 (README and the adsb.lol site); a CC0 file is also shipped (section 1d, R1) | YES: ODbL 1.0 |
| IEM ASOS METAR (Iowa Environmental Mesonet) | YES (weather columns in the bulk model; LTFM snow refit v117) | no formal licence; disclaimer page states public domain (section 1b) | ACCEPTED BY ORGANISER STATEMENT (METAR allowed); public-domain statement only (section 1d, R2) |
| OpenStreetMap stand data via Overpass | YES (OSM stand fill in `prc/adsb_match.py`) | ODbL 1.0, attribution (c) OpenStreetMap contributors | YES |
| OpenStreetMap taxiway/runway ways via Overpass | NO in the final pipeline: only the stand positions are read (see 2.4) | ODbL 1.0 | YES |
| wind direction/speed re-fetch (IEM ASOS, `data/raw/weather_wind`) | NO, tested and unused | same as METAR | see METAR |
| EUROCONTROL portal daily airport file `all_pre_departure_delays_<year>.csv` (daily all-cause pre-departure delay and departure count; January and July 2026 included) | YES, from v153: the published daily delay is an identity on the recorded off-block stamps; used for a day-level calibration at six airports and for Rome no-plan record types (v153 to v155), for the same calibration at Istanbul, Zurich and Paris (v159 to v161) and for a cell split of the day shift (v162) (section 2.5; `docs/PIPELINE.md` section I.3). Without it the best file scores 216.2405 (v152), with it 213.2898 (v162) | the portal's terms: free to copy with EUROCONTROL cited, non-commercial use; the organisers answered on 2026-09-17 that the portal's daily airport files can be considered open data for the challenge | ACCEPTED BY ORGANISER STATEMENT (no formal open licence; risk R3); layers 1 and 3 do not read it, so the entry without it is variant A |
| EUROCONTROL ansperformance aggregates (published monthly taxi-out) | NOT as a model input, and no component of the last uploaded file reads them. One early upload (v101) did and lost; one rule of earlier versions (Munich January 2026 level shift) was sized with the published January figure as one of three instruments and was removed in v144 | attribution, non-commercial; the organisers answered on 2026-09-17 that the portal files can be considered open data for the challenge and on 2026-09-27 that the published monthly figures for the scored months may be used as model input | ACCEPTED BY ORGANISER STATEMENT; no licence text recorded (risk R3) |
| OPDI flight events (opdi.aero) | NO, probe only | no formal licence named; "non-commercial" framing | NO (risk R4) |
| ADSBiq feeder parquet (one day, 2026-07-15) | NO, tested, zero coverage gain | ODbL 1.0 | YES (unused) |
| adsb.lol ACARS probe (2026-07-15) | NO, tested, dead | CC0 | YES (unused) |
| Public holidays JSON (Nager.Date) | NO, screen only | UNVERIFIED | UNVERIFIED (unused) |
| French SIA AIP supplements (PDF) | NO, notes only | French state publication; UNVERIFIED | UNVERIFIED (unused) |
| X-Plane Scenery Gateway apt.dat | NO, tested 2026-10-04 and unused (no gain) | GNU GPL: the format specification page states "The GNU GPL (general public License) under which this data is released" (https://developer.x-plane.com/article/airport-data-apt-dat-file-format-specification/); the per-airport Gateway exports carry no copyright line | YES (unused, files not kept in repo) |
| ERA5 hourly snow depth and snowfall (Copernicus Climate Change Service / ECMWF, read through the public ARCO-ERA5 archive) | NO, tested, no estimator passed, no shipped column | Copernicus licence for ERA5 (open, attribution required); text not re-read for this release because the data is not used | not used, so no obligation arises |

## 1b. Licence terms, read from the official pages on 2026-10-02

This section was written by reading each source's own page (web pages only, no dataset downloads) on 2026-10-02. Where it
quotes, the quote is the page's own wording. Where a page states no licence we say so and list it as an open risk; we did
not infer a licence. The licence summaries are ours and are not legal advice; the full texts are at the URLs given.

| source (used by the shipped pipeline) | licence as stated | attribution wording | URL where stated | obligations that follow for us |
|---|---|---|---|---|
| Challenge data (organisers' OpenSky bucket `prc-2026-datasets`) | none stated: "no licence stated at https://prc-data-challenge-2026.netlify.app/data.html on 2026-10-02"; none stated at https://prc-data-challenge-2026.netlify.app/eligibility.html either | none prescribed | https://prc-data-challenge-2026.netlify.app/data.html, https://prc-data-challenge-2026.netlify.app/eligibility.html | We do not redistribute the data (it is git-ignored). The eligibility page requires that "All produced source code are made openly available on GitHub under the GNU GPLv3 license", that "All used external datasets are openly accessible/usable and documented", that "All additional datasets used are openly available under an open source license" and that "Sufficient documentation is provided to understand and reproduce the results". |
| adsb.lol historical ADS-B, `adsblol/globe_history_2025` and `adsblol/globe_history_2026` | Open Database License 1.0: "This database is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/". The repository front pages also list CC0-1.0 and ship `LICENSE-ODbL.txt` and `LICENSE-cc0.txt`; which of the two covers which part is not stated on the pages we read | The pages ask for no specific string. The 2026 page says the data is from "ADSB.lol feeders and friends, FlyItalyADSB and TheAirTraffic.com"; we credit "adsb.lol" and those contributors | https://github.com/adsblol/globe_history_2025 and https://github.com/adsblol/globe_history_2026 | Attribute adsb.lol and its feeders; ODbL attribution and share-alike apply to any derived database we publicly use or distribute. We publish code only, no point extracts. R1 is resolved in section 1d (ODbL 1.0; the CC0 file is not described on the pages) |
| OpenStreetMap (stand positions and taxiway/runway ways, fetched through the Overpass API) | Open Data Commons Open Database License (ODbL) | The page says to "credit OpenStreetMap and its contributors" and to make clear the data is available under the ODbL. We use "(c) OpenStreetMap contributors, ODbL 1.0" | https://www.openstreetmap.org/copyright | Share-alike: "If you alter or build upon our data, you may distribute the result only under the same license." We publish no OSM extract; if `data/external/osm/*.parquet` is ever published it must carry the notice and be ODbL |
| Overpass API (the query service) | server software GNU AGPL v3; it serves OSM data and states no separate data licence | none | https://wiki.openstreetmap.org/wiki/Overpass_API | Public-instance fair use: under 10,000 queries and 1 GB per day for ordinary use, a User-Agent or Referer header, no parallel scripts, and "Commercial use should use self-hosted or paid Overpass servers". Our fetch scripts set a User-Agent |
| Weather: Iowa Environmental Mesonet (IEM) ASOS/METAR archive | no formal licence. The disclaimer page says: "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose." The ASOS download page itself states no licence ("no licence stated at https://mesonet.agron.iastate.edu/request/download.phtml on 2026-10-02"; it carries an Iowa State University copyright notice) | requested, not required: "Attributing the Iowa Environmental Mesonet of Iowa State University would be appreciated." | https://mesonet.agron.iastate.edu/disclaimer.php | We credit the Iowa Environmental Mesonet of Iowa State University. The origin of the underlying METAR reports (national weather services) is not stated on those pages, which is open risk R2 |

Evaluated but not used by the current chain (kept for completeness): EUROCONTROL ansperformance aggregates were used by one
early upload (v101, which lost) and are not in the later lineage; the data page https://ansperformance.eu/data/ states no licence itself; the terms are on the portal's disclaimer page (section 2.5). OPDI, ADSBiq and adsb.lol ACARS are as in section 3.

## 1d. Licence verdicts (official pages read 2026-10-03; reading only, nothing downloaded)

**R1, adsb.lol historical ADS-B: OPEN (ODbL 1.0).**
- `https://github.com/adsblol/globe_history_2025` README: "This database is made available under the Open Database License: http://opendatacommons.org/licenses/odbl/1.0/". `globe_history_2026` README: the same sentence; it also credits "ADSB.lol feeders and friends, FlyItalyADSB and TheAirTraffic.com".
- Both repositories list two files, `LICENSE-ODbL.txt` and `LICENSE-cc0.txt`. The READMEs state only ODbL for the database; no page says what the CC0 file covers (plausibly the repository's own scripts or text). The adsb.lol site's historical-data page (`https://www.adsb.lol/docs/open-data/historical/`) states "License: ODbL 1.0" and nothing else.
- Reading: the data is ODbL 1.0; CC0 is more permissive, so using the data under ODbL satisfies both. Neither page asks for a specific attribution string. Attribution used here: "Contains information from adsb.lol (ADSB.lol feeders and friends, FlyItalyADSB and TheAirTraffic.com), made available under the ODbL 1.0."
- Share-alike applies only to a publicly used derived database. This release contains code only and no adsb.lol extract or point file.

**R2, METAR via the Iowa Environmental Mesonet: ACCEPTED BY ORGANISER STATEMENT.**
- `https://mesonet.agron.iastate.edu/disclaimer.php`: "The materials found on this website are in the public domain and may be used freely by anyone for any lawful purpose." and "Attributing the Iowa Environmental Mesonet of Iowa State University would be appreciated."
- The disclaimer page does not name the primary source of the METAR reports. They originate with national meteorological services and airport operators via the NOAA/WMO exchange; that is general knowledge and is NOT stated on the IEM page we read.
- There is no formal open-licence name, only the public-domain statement; the organisers stated METAR data may be used. Attribution used here: "Weather observations: Iowa Environmental Mesonet, Iowa State University (ASOS/METAR archive)." If it must be closed formally, the question for the organisers is: "Is the IEM public-domain statement for METAR sufficient as the open licence for the external-data rule?"

**R5, OpenStreetMap: OPEN (ODbL 1.0).**
- `https://www.openstreetmap.org/copyright`: "You are free to copy, distribute, transmit and adapt our data, as long as you credit OpenStreetMap and its contributors." and "If you alter or build upon our data, you may distribute the result only under the same license."
- Attribution line: "(c) OpenStreetMap contributors, ODbL 1.0" (https://www.openstreetmap.org/copyright).
- Derived database: this release publishes none. The fetch scripts (`archive/fetch_osm_stands.py`, `archive/fetch_osm_ways.py`) are code and write `data/external/osm/*.parquet` locally; no OSM-derived file (parquet, CSV or lists) is in the export, whose only parquet is the final submission. The submission file is a set of taxi-out predictions, not an OSM extract; the OSM stand fill is one of many inputs to the model.

## 1c. Downloaded but NOT used by the shipped solution

These sources were downloaded or probed during the work and evaluated. **No component of the final chain reads them**: EUROCONTROL's published monthly taxi-out figure for Munich, January 2026, was one of three instruments behind a Munich January level shift of earlier versions, which was removed in v144 (docs/ROW-RULES.md). They are not redistributed; they are listed so that the documentation is complete.

| source | what was done | verdict |
|---|---|---|
| ERA5 snow reanalysis (snow depth, snowfall at the ten airport grid points; Copernicus / ECMWF via the ARCO-ERA5 public archive) | fetched for a snow-severity test | does not describe the taxi-out excess better than the METAR snow flag; nothing built from it |
| OPDI flight lists / flight events (opdi.aero) | one January 2026 file, probes | no off-block milestone coverage; not used |
| EUROCONTROL aggregate performance files | calibration and correlation screens; one early upload (v101) aligned airport-month means to them | the upload lost and is not in the later lineage; the monthly taxi-out figure was used once more, for a Munich January level shift that was removed in v144 (docs/ROW-RULES.md) |
| ADSBiq feeder parquet (one day) | schema and coverage comparison | zero ground-coverage gain; not used |
| IEM wind observations (re-fetch of drct, sknt, gust) | runway-relative wind screen | adds nothing; no shipped column |
| adsb.lol ACARS release (one day) | coverage probe | no messages near the target airports; not used |
| public-holiday JSON, French AIP supplements, X-Plane apt.dat | screens and inspection | not used |

## 2. Sources used by the current best pipeline

### 2.1 Competition data
- Files: `training_2025-MM-01_...parquet` (12), `ranking.parquet`, `submitting.parquet`, from the organisers'
  OpenSky S3 bucket `prc-2026-datasets` (read access issued 2026-09-04). Fetched with `prc/data.py`
  (`python -m prc.data list|pull`), credentials from outside the tree (`docs/DATA-ACCESS.md`).
- Feeds: everything (labels `TAXITIME_SEC_mvt`, Network Manager clocks, flight-plan fields).
- Not redistributed. Rebuilders need their own bucket access from the organisers.

### 2.2 adsb.lol historical ADS-B (surface tracks)
- Source: `https://github.com/adsblol/globe_history_2025` and `.../globe_history_2026`, one GitHub release per day,
  tag `v{YYYY.MM.DD}-planes-readsb-prod-0`, tar parts `.tar.aa`, `.tar.ab`. Attribution: adsb.lol.
- Licence: ODbL 1.0 (section 1d, R1); `prc/adsb.py` records that the project also ships a CC0 file. Acquired in stages from 2026-09-16 (a probe, 11 days of 2025, then the rest).
- Days on disk (`data/adsb/points/`): 354 days of 2025 (Jan-Dec, 29-31 per month) plus all 31 days of January 2026
  and all 31 days of July 2026 (417 day files in all). The 354 days are exactly the union of `docs/lists/calib_days_264_octnov.txt`
  (264 days) and `docs/lists/agent_md_newdays.txt` (116 days); the two lists overlap on 26 days (checked 2026-10-04). Extraction dates approx. 2026-09-16 to 2026-09-26 (from file times, UNVERIFIED).
- Fetch: `python -m prc.adsb probe|extract --dates ... --out data/adsb/points`; day lists in `archive/extract_60_days.sh`,
  `extract_92_days.sh`, `extract_122_days.sh`, `agent_md_dumps.sh`, `agent_moredata_extract.py`. Only points inside the
  ten airport boxes, on the ground or below 3,000 ft, are kept; the day's tar is streamed, never stored.
- Feeds: the ADS-B stage (off-block / take-off anchoring), the stand map, `prc/adsb_match.py`, `archive/adsb_apply.py`,
  and every `bx_*`, `evt_*`, `trk_*`, `occ_*`, `m_*` ADS-B-derived column in the re-stack and the residual corrector.
- Also fetched, unused in the best file: `data/adsb/hidden/` and `data/adsb/points_alt/` (extra readsb fields and an
  alternative extraction; `archive/agent_hidden_extract.py`, `agent_alt_02_extract.py`), tests only.

### 2.3 IEM ASOS METAR
- Source: `https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py` (Iowa Environmental Mesonet), ten ICAO stations
  (LTFM LFPG EGLL EHAM LEMD EDDF LEBL LIRF EDDM LSZH), 2025-01-01 to 2026-08-01, report types 3 and 4, UTC.
  Fetched 2026-09-06 (file time) with `python -m prc.weather fetch` into `data/raw/weather/<ICAO>.csv`.
- Licence: `prc/weather.py` calls it "a free, openly accessible archive". We recorded NO licence text. The organisers stated on
  2026-09-08 that METAR may be used and that open data must be declared in the documentation.
- Feeds: `prc/weather.py` columns joined in `prc/features.py` (`weather.attach`), so the base bulk model; the LTFM
  snow features of v117 (`archive/agent_ltreg_feat.py`, `agent_ltreg_final.py`); weather columns in the correctors.
- Tested and unused: `data/raw/weather_wind/` (drct, sknt, gust), fetched 2026-09-30 by
  `archive/agent_wx2_fetch_wind.py` from the same endpoint; used only by `archive/agent_wx2_03_wind.py`
  (dead: runway-relative wind adds nothing). Not in any shipped column.

### 2.4 OpenStreetMap (Overpass API)
- Stands: `https://overpass-api.de/api/interpreter` (fallback overpass.kumi.systems), `aeroway=parking_position`
  and `aeroway=gate` inside the ten airport boxes. Fetched 2026-09-17 with `archive/fetch_osm_stands.py` into
  `data/external/osm/stands.parquet` (apt, kind, ref, lat, lon; ~97 kB). Licence ODbL 1.0, attribution
  "(c) OpenStreetMap contributors" (stated in the script header).
- Feeds: `prc/adsb_match.py` (`OSM_STANDS`, stand centroid fill for stands no tracked arrival ever reached).
- Ways (taxiways, runways, aprons): `archive/fetch_osm_ways.py`, `archive/agent_osmgeo_fetch_ways.py`, fetched
  2026-09-20 into `data/external/osm/ways.parquet`. Read by `scripts/agent_osmgeo_*` and `agent_eddf_01_classify.py`.
  The "osm" columns of the residual corrector in `final/` (`final/_dump_osm_occ.py`) are built from the stand positions only; the ways file is not read by the final pipeline.
- OSM is an ODbL database: a derived database or "produced work" carries share-alike and attribution duties. We publish
  no extract of OSM and no database derived from it; a published copy of the parquet needs the ODbL notice.

### 2.5 EUROCONTROL portal daily airport file (published daily delay)
- Source: `https://www.eurocontrol.int/performance/data/download/csv/all_pre_departure_delays_2025.csv` (6.2 MB) and `..._2026.csv` (4.1 MB), downloaded 2026-09-22 (the same directory holds the other daily and annual files tested in section 3). Stored in `data/external/eurocontrol/` (git-ignored; re-fetch from that address). The columns used: `FLT_DATE`, `APT_ICAO`, `FLT_DEP_1` (departures), `DLY_ALL_PRE_2` (total all-cause pre-departure delay, minutes), for the ten airports; the 2026 file carries January to August 2026, so the four served months.
- Licence: the portal's own terms, that the data may be copied in whole or in part with EUROCONTROL cited as the source, for non-commercial use (the disclaimer page `https://ansperformance.eu/about/disclaimer/`, linked from `https://ansperformance.eu/data/`; checked on 2026-10-10). Attribution used: "EUROCONTROL Performance Review Unit, ansperformance.eu".
- Organisers' statement of 2026-09-17: the portal's daily airport files can be considered open data for the challenge.
- Finding and use (all in `docs/PIPELINE.md` section I.3; scripts `archive/agent_dayeq_01_identity.py` to `agent_dayeq_06_dayearly.py`, `agent_bundle_dayeq.py`): on 2025 the daily delay equals the sum over the day's departures of max(off-block minus schedule, 0), exact to 1 to 2 minutes a day at seven airports; the step that uses it is the reconciliation layer (`docs/PIPELINE.md` I.3). The same directory's ATC-delay, slot-adherence and arrival-delay files were tested and are not identities.
- Feeds: `data/interim/agent_dayeq_level_final.parquet` (139,724 rows) and `agent_dayeq_types_final.parquet` (168 rows) and so uploads v153 to v155; the files of the later pieces (Istanbul, Zurich, Paris and the cell split; `scripts/agent_dayeq2_*.py`, `agent_nextpools_*.py`) feed uploads v159 to v162. The daily departure count (`FLT_DEP_1`) is used in these pieces as a day-level gate (a day is calibrated only where the published count is within 15 of ours) and, in the piece for v164, which was not uploaded, to compare the published count at Rome with ours by off-block date (equal on 358 of 365 days of 2025).

## 3. Sources tested and NOT used by the current best pipeline

| Source | Where | Result |
|---|---|---|
| EUROCONTROL ansperformance CSVs (`taxi_out_additional_time`, `apt_dly`, pre-departure delay, ATFM slot adherence), downloaded 2026-09-19 and 2026-09-22 from `eurocontrol.int/performance/data/download/csv/` | `data/external/eurocontrol/`; `scripts/eurocontrol_*.py`, `agent_aggr_01_screen.py` | Perfect-correction ceiling +0.75 s; daily correlation -0.016. One early submission (v101, `archive/apply_ec_align.py`, aligned airport-month means to EUROCONTROL's published level) scored +0.115 worse and is not in the lineage of the later builds (the ablation chain in docs/PIPELINE.md Part II has no such step; not checked numerically). The published January 2026 figure for Munich was one of the three instruments that set the size of the Munich January level shift (`archive/apply_eddm_january.py`), which was removed in v144 and is in no later file; no shipped component reads them. |
| OPDI `flight_events_20260110_20260120.parquet` (270 MB), downloaded 2026-09-17 | `data/external/opdi/`; `archive/opdi_probe.py` | Probe only; no script reads the 2026-01 events file. v0.3.0 AOBT milestone unreleased. |
| ADSBiq `aircraft_diffs_2026-07-15.parquet` (227 MB), downloaded 2026-09-23, `github.com/Sky-Power-Services/adsbiq-data` | `data/external/adsbiq/SOURCE.json`; `scripts/agent_adsbiq_*` | Zero ground coverage gain at LTFM/LFPG/LIRF. |
| adsb.lol ACARS, release `adsblol-adl-2026-07-15` of `adsblol/aircraft-data-links-2026`, 72 files, 32 MB, downloaded 2026-09-30 | `data/external/acars/20260715/` (gitignored) | 400,955 messages from 79 stations, none near LFPG/LIRF/LTFM, 120 gate-out messages worldwide: dead. |
| Nager.Date public holidays JSON, fetched 2026-09-22 | `data/external/holidays/`; `archive/agent_aggr_01_screen.py` | Screen only. |
| French SIA AIP supplements (PDF), fetched 2026-09-21 | `data/external/sia_sup/`; `archive/fetch_sia_sup.sh`, `agent_lfpgworks_01_parse.py` | Works-in-progress notes only. |
| X-Plane Gateway apt.dat (EDDF pack inspected once) | inspected once, no file kept | Licence not clearly open at that date; acquisition stopped. Superseded: the format page states the GNU GPL (section 1), all ten airports were tested on 2026-10-04 and nothing was built. |

## 4. Licence risks

- **R1, adsb.lol (used, load-bearing). RESOLVED 2026-10-03 (section 1d): OPEN, ODbL 1.0.** Exact wording, URLs and the attribution line are in section 1d. The organisers stated that OSN state vectors are discouraged; adsb.lol is a different source.
- **R2, IEM METAR (used, load-bearing).** Updated 2026-10-02 (section 1b): the IEM disclaimer page states the materials
  are in the public domain, and the ASOS download page states no licence. The organisers stated that data without an
  open licence is not open (2026-10-01); a public-domain statement on the website that serves the data is not a formal
  open licence, and the origin of the underlying METAR reports is not stated on those pages. Verdict 2026-10-03
  (section 1d): ACCEPTED BY ORGANISER STATEMENT (METAR is allowed); the question to put to the organisers if it must be closed formally is given there.
- **R3, EUROCONTROL (the daily delay file is used in the uploaded files from v153, section 2.5; the monthly figures are not; two earlier uses of the monthly figures are stated below).** "Free to copy with source cited;
  non-commercial use only" is the portal's own licence line; the organisers answered on 2026-09-17 that the portal files can be
  considered open data for the challenge and on 2026-09-27 that the published monthly figures may be used as model input. The
  files are not committed; the last uploaded file uses the daily delay file as described in section 2.5 and no model input is built from the monthly figures. Two earlier uses came before the 2026-09-27 answer: the Munich January level shift, sized partly from the published
  January figure, was first carried by an upload on about 2026-09-20 (v39, read from the build scripts; the upload log does not itemise it) and removed in v144; and v101 (uploaded 2026-09-24, aligned airport-month means to the published January and July 2026 figures) lost and is not in
  the lineage (section 2 intro; `archive/apply_ec_align.py` and the v101 build script are in the repo).
- **R4, OPDI (tested, not used).** No formal licence is named anywhere on opdi.aero ("freely used, reused and
  distributed provided attributed", "non-commercial purposes"). Declare as evaluated, not used.
- **R5, OpenStreetMap (used). OPEN (section 1d, R5).** Attribution is required and any published derivative database
  carries share-alike. No OSM-derived file is in the release; do not commit `stands.parquet`/`ways.parquet` without the notice.
- **R6, unused but downloaded (holidays, SIA, apt.dat).** Licence unclear; none is in the repo or the pipeline.
- **R7, the competition data itself** is not ours to redistribute; `.gitignore` excludes it. Reproducers need the
  organisers' bucket.
- **R8, the challenge data (new, 2026-10-02).** No licence is stated at
  https://prc-data-challenge-2026.netlify.app/data.html or https://prc-data-challenge-2026.netlify.app/eligibility.html.
  We do not redistribute it. Open risk only in the sense that no terms are written down on those pages.
- **R9, Overpass public instance.** Fair-use limits and a no-commercial-use note apply to the public servers
  (section 1b); our fetches were small one-off pulls.
