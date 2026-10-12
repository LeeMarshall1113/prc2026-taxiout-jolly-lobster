"""Generate final/ist_kg_ltfm_final.py and final/ist_ltreg_final.py from the shipped Istanbul wrappers.

Chosen route: keep the shipped wrappers' patching of prc.final and point the patches at final/10_base.py (the parameterised copy
of prc.final.main that builds the served frame one calendar month at a time) instead of rewriting the arms. 10_base keeps the
module-level names the wrappers patch (build, load_training_features, resolve_transform, DROP, FEATURES), so the only edits are:
  1. `import prc.final as fin` -> the 10_base module under the same name `fin`.
  2. the two reads of RAW_DIR/ranking.parquet -> the file named by env FINAL_RANKING; the Istanbul frame of the sequence
     features keeps only the months in env FINAL_MONTHS (level columns are demeaned per calendar month, so this changes nothing
     for the served months).
  3. the checkpoint read: <out-dir>/<tag>_prepost.parquet (10_base writes it) instead of data/interim/ranking_predictions_prepost.parquet.
  4. a --dry-run flag of 10_base stops the wrapper after main().
Everything else (flags, seeds, feature lists, snow features, the shifted CatBoost seed) is the shipped text.
    python final/43_gen_wrappers.py
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "archive"

IMPORT_NEW = '''import importlib.util as _iu  # noqa: E402
_here = os.path.dirname(os.path.abspath(__file__))
_cands = [os.path.join(_here, "..", "final", "10_base.py"), os.path.join(_here, "final_10_base.py"), os.path.join(_here, "10_base.py")]
_p10 = next(p for p in _cands if os.path.exists(p))
_sp = _iu.spec_from_file_location("base10", _p10)
fin = _iu.module_from_spec(_sp)
sys.modules["base10"] = fin
_sp.loader.exec_module(fin)   # final/10_base.py: same module-level names as prc.final that the patches below replace
from pathlib import Path  # noqa: E402
_RANKING = Path(os.environ["FINAL_RANKING"])
_MONTHS = [int(t) for t in os.environ.get("FINAL_MONTHS", "2,6").split(",")]
_OD = Path(argv[argv.index("--out-dir") + 1]) if "--out-dir" in argv else Path("data/final")
_TAG = argv[argv.index("--tag") + 1] if "--tag" in argv else "J3"
'''


def gen(src_name, dst_name, extra_note):
    t = (SRC / src_name).read_text(encoding="utf-8").replace("\r\n", "\n")
    n0 = t.count("import prc.final as fin")
    assert n0 == 1
    t = t.replace("import prc.final as fin  # noqa: E402  (import-time registry check runs on the ORIGINAL lists)", IMPORT_NEW)
    assert "base10" in t
    NL = chr(10)
    PATH_OLD = "sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))"
    t = t.replace(PATH_OLD, PATH_OLD + NL + "sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, 'archive'))")
    F0_OLD = "frame0 = fin.load_training_features()"
    F0_NEW = ("frame0 = (pl.scan_parquet(os.path.join(os.environ.get('PRC_DATA_DIR', 'data'), 'interim', 'train_features.parquet')).head(200_000).collect()" + NL
              + "          if '--dry-run' in sys.argv else fin.load_training_features())")
    assert F0_OLD in t and PATH_OLD in t
    t = t.replace(F0_OLD, F0_NEW)
    DRY = ("if '--dry-run' in sys.argv:" + NL
           + "    _lt = (pl.col('ADEP_mvt') == 'LTFM') | (pl.col('ADES_mvt') == 'LTFM')" + NL
           + "    _raw = pl.scan_parquet(_RANKING).filter(_lt).collect()" + NL
           + "    _raw = pl.concat([_raw.filter(pl.col('MVT_TIME_UTC_mvt').dt.month() == _m).head(3000) for _m in _MONTHS])" + NL
           + "    _r = fin.build_by_month(_raw, _MONTHS)" + NL
           + "    _f = [c for c in F.FEATURES if c not in set(BASEDROP)]" + NL
           + "    _x = _r.select(_f)" + NL
           + "    print('[final wrapper] dry run: served frame', _r.shape, 'features', len(_f), 'sq_win_n non-null', int(_r['sq_win_n'].is_not_null().sum()) if 'sq_win_n' in _r.columns else 'n/a')" + NL
           + "    if 'train_fr' in globals():" + NL
           + "        _y = train_fr.select(_f)" + NL
           + "        print('[final wrapper] dry run: training frame', _y.shape)" + NL
           + "    sys.exit(0)" + NL)
    t = t.replace(NL + "fin.main()" + NL, NL + DRY + "fin.main()" + NL)
    assert "dry run: served frame" in t
    # argv for 10_base: out-dir and tag are positionals there; the wrapper's own flags are already popped
    t = t.replace('r26 = pl.scan_parquet(RAW_DIR / "ranking.parquet").select(COLS).filter(ltfm).collect()',
                  'r26 = pl.scan_parquet(_RANKING).select(COLS).filter(ltfm).collect().filter(pl.col("MVT_TIME_UTC_mvt").dt.month().is_in(_MONTHS))')
    t = t.replace('pre = pl.read_parquet(fin.INTERIM_DIR / "ranking_predictions_prepost.parquet")',
                  'if "--dry-run" in sys.argv:\n    print("[final wrapper] dry run: stopping before any output"); sys.exit(0)\npre = pl.read_parquet(_OD / f"{_TAG}_prepost.parquet")')
    t = t.replace('apt = pl.read_parquet(RAW_DIR / "ranking.parquet", columns=', 'apt = pl.read_parquet(_RANKING, columns=')
    assert 'RAW_DIR / "ranking.parquet"' not in t and "INTERIM_DIR" not in t, "unreplaced path"
    hdr = (f'"""GENERATED by final/43_gen_wrappers.py from archive/{src_name} (the shipped wrapper); see that file for the changes.\n{extra_note}\n'
           'Arguments: the shipped wrapper flags (--ltfm-seq, --out, --basedrop, [--seed-offset]) followed by the final/10_base.py flags\n'
           '(--ranking --template --months, ... then the positionals OUT_DIR TAG). Env FINAL_RANKING (path of final_ranking.parquet) and\n'
           'FINAL_MONTHS (default 2,6) must be set. Writes --out like the shipped wrapper.\n"""\n')
    body = t.split('"""', 2)[2]   # drop the shipped docstring
    (HERE / dst_name).write_text(hdr + body, encoding="utf-8")
    print("wrote", dst_name)


gen("agent_kg_ltfm_final.py", "ist_kg_ltfm_final.py", "Control / treatment arms of the Istanbul bulk refit.")
gen("agent_ltreg_final.py", "ist_ltreg_final.py", "Snow arms of the Istanbul bulk refit (k0 = seed 1113, k12 = seeds 2090 and 3067 via --seed-offset 1).")
