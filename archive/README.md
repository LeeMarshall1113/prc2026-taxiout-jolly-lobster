# archive/: historical record of how the January/July predictions were built, upload by upload

The maintained pipeline is in `final/`. The scripts here are only those in the lineage of the files that were actually
uploaded to the leaderboard (January and July 2026 rows): the scripts that produced each uploaded file and the scripts
they import or call. Experiments that never shipped are not included.

How to read it:
- `docs/BUILD-TRACE.md` names the script of every stage in the order the final file passed through it, and
  `docs/SUBMISSION-LOG.md` lists every upload with its prediction and its score.
- `reproduce_final.py` assembles the last uploaded files from the stored intermediate files (those are not part of this
  repository).
- File names `agent_<name>_NN_*` are a historical naming convention: one family per experiment, numbered in the order the
  steps were written. Many files are copies of an earlier script with one change; that is how the work was done.
- Some scripts hard-code the January/July ranking file (`data/raw/ranking.parquet`) and their stored intermediate
  files; they are kept as the record, and `final/` ports them to a parameterised form.
- `run_job.sh` and `job_slots.py` are pass-through stubs: the original wrappers called a local job scheduler, and the
  stubs accept its options and ignore them. Use `capped_run.py --cap-gb N` for a memory cap.
- Files with the extension `.tmpl` are templates with `@PLACEHOLDER@` tokens that a launcher fills in; they are not
  runnable as they stand.
- Modal apps of this lineage are in `modal/`.

Six named flight strings appear in this folder and in `final/31_scored_row_fixes.py`: they are disclosed row-level fixes
(`docs/ROW-RULES.md`).

Scripts in this folder: 416 (one line of purpose per script is kept with the release tooling).
