# experiment sb2: COPY of archive/agent_stagecap_build.sh (docs/LAB-NOTES.md): serve step = archive/agent_sb2_serve.py
# (new tokens mw/lw/unc/osm/occ), logs under logs/agent_sb2_*. Original header follows.
#!/usr/bin/env bash
# experiment stagecap: COPY of agent_restack5_build.sh (docs/LAB-NOTES.md) adding ONLY three env knobs:
#   SC_TRAIN  re-stack training npz (default data/interim/modal/TX_S6.npz = the original)
#   SC_P105   stage-output prefix of the v105 path (default data/interim/v105 = the original; the new stage = data/interim/v105sc)
#   SC_PFX    intermediate tag prefix (REQUIRED; never the original's, so no other experiment's file is overwritten)
# and logs under logs/agent_stagecap_*. With SC_TRAIN/SC_P105 unset it rebuilds the original's outputs. Original header follows.
# experiment restack5: COPY of agent_restack4_build.sh (docs/LAB-NOTES.md). Differences: serve step =
# archive/agent_sb2_serve.py; R5_CB=1 adds --blend-cb; R5_SEEDS=0,1,2 adds --seeds (seed bag); intermediate tag prefix
# "r5" (never overwrites other experiments' files); nbr 2026 columns -> data/interim/agent_restack5_nbr26_*. Original header follows.
# experiment restack4: COPY of archive/build_combo.sh (docs/LAB-NOTES.md). Differences: the serve step is
# archive/agent_restack4_serve.py (tokens seq/seqm/nbr/arr/big; R4_CB=1 adds --blend-cb); nbr recipes get each path's
# stage input (--s0 = v10x_s0ship / v10x_gatebag_s0, as agent_neighbour_build.sh); intermediate tag prefix "r4"
# (never overwrites combo/sb/nb files); every python under capped_run --cap-gb 6. Original header follows.
# ADAPTIVE COMBO BUILDER (2026-09-24, the team: "make sure each one helped then add and subtract"). Copy of
# agent_restack3_build.sh with a T switch: T=1 = taxi-phase world (second path v105, splice set
# EDDF,EGLL,LEMD,LSZH,LFPG,LIRF = v105); T=0 = pre-v105 world (second path v104, splice set LFPG = v104).
#   bash agent_restack4_build.sh T FEATS RECIPE OUT        FEATS none|est_v2 ; RECIPE "" | cap | cap,evt,trk ...
cd "${PRC_ROOT:-$PWD}"; PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8 POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR=load_as_storage PYTHONPATH="$(pwd)" OMP_NUM_THREADS=4 POLARS_MAX_THREADS=4
T="${1:-1}"; FEATS="${2:-est_v2}"; RECIPE="${3:-}"; OUT="${4:-combo_chk}"; SKIP=""
if [ "$T" = "1" ]; then V2=105; SPL=EDDF,EGLL,LEMD,LSZH,LFPG,LIRF; else V2=104; SPL=LFPG; fi
RT=$(echo "$RECIPE" | tr ',' '_'); TAG="${SC_PFX:?set SC_PFX}${FEATS}${RT:+_$RT}${R5_CB:+_cb}${R5_SEEDS:+_bag}"
CBF=""; [ -n "$R5_CB" ] && CBF="--blend-cb"; [ -n "$R5_SEEDS" ] && CBF="$CBF --seeds $R5_SEEDS"
CAP="$PY archive/capped_run.py --cap-gb 6 --"
say () { echo "[$(date -u +%H:%M:%SZ)] $*"; }
chain () {
  $CAP $PY archive/apply_gap_clamp.py --in "$1" --out "$2_s2.parquet" >/dev/null 2>&1 || { echo "chain fail gap_clamp $1"; exit 1; }
  $CAP $PY archive/apply_lobt_box.py --hard-binding-blend 0.5 --in "$2_s2.parquet" --out "$2_s3.parquet" >/dev/null 2>&1 || { echo "chain fail lobt_box"; exit 1; }
  $CAP $PY -u archive/apply_min_floor.py --in "$2_s3.parquet" --out "$2_s35.parquet" >/dev/null 2>&1 || { echo "chain fail min_floor"; exit 1; }
  $CAP $PY -u archive/apply_v35_rules.py --in "$2_s35.parquet" --out "$2_s4.parquet" >/dev/null 2>&1 || { echo "chain fail v35"; exit 1; }
  $CAP $PY -u archive/apply_upper_bound.py --in "$2_s4.parquet" --out "$2_s5.parquet" >/dev/null 2>&1 || { echo "chain fail upper"; exit 1; }
  $CAP $PY -u archive/apply_hidden_reading.py --in "$2_s5.parquet" --out "$2_s6.parquet" >/dev/null 2>&1 || { echo "chain fail hidden"; exit 1; }
  $CAP $PY -u archive/apply_eddm_january.py --in "$2_s6.parquet" --out "$2_final.parquet" >/dev/null 2>&1 || { echo "chain fail eddm"; exit 1; }
}
lirf_splice () {  # nl everywhere, li at LIRF (merged_chain.sh's splice)
  $CAP $PY archive/splice_apt_parquet.py "$1" "$2" LIRF "$3" | grep -v capped_run | tail -1
}
for V in 103 $V2; do
  P=data/interim/v${V}; [ "$V" = "105" ] && P=${SC_P105:-data/interim/v105}
  for S in s1 gatebag_s1; do
    if [ "$S" = "s1" ]; then S0=${P}_s0ship.parquet; else S0=${P}_gatebag_s0.parquet; fi
    NBA=""; case ",$RECIPE," in *,nbr,*) NBA="--s0 $S0 --nbr-2026-out data/interim/agent_restack5_nbr26_${TAG}_v${V}_$S.parquet";; esac
    if [ "$FEATS" = "none" ]; then cp ${P}_$S.parquet ${P}_${S}${TAG}.parquet
    else
      $CAP $PY -u archive/agent_sb2_serve.py $CBF $NBA \
        --train-npz ${SC_TRAIN:-data/interim/modal/TX_S6.npz} --train-rows data/interim/agent_offblock_rows_c238.parquet \
        --rows-2026 data/interim/agent_offblock_rows_2026.parquet --feats $FEATS ${RECIPE:+--recipe $RECIPE} ${SKIP:+--skip $SKIP} \
        --in ${P}_$S.parquet --out ${P}_${S}${TAG}.parquet > logs/agent_sb2_serve_${TAG}_v${V}_$S.log 2>&1 || { echo "serve failed"; tail -5 logs/agent_sb2_serve_${TAG}_v${V}_$S.log; exit 1; }
      grep -E "footprint|moved|recipe" logs/agent_sb2_serve_${TAG}_v${V}_$S.log | tail -3
    fi
  done
  chain ${P}_s1${TAG}.parquet ${P}${TAG}_nl
  chain ${P}_gatebag_s1${TAG}.parquet ${P}${TAG}_li
  lirf_splice ${P}${TAG}_nl_final.parquet ${P}${TAG}_li_final.parquet data/interim/v${V}${TAG}_raw.parquet || exit 1
  say "v$V $TAG raw built"
done
$CAP $PY archive/splice_apt_parquet.py data/interim/v103${TAG}_raw.parquet data/interim/v${V2}${TAG}_raw.parquet $SPL data/interim/${OUT}_spliced.parquet | grep -v capped_run | tail -1
$CAP $PY -W ignore archive/agent_monster_apply.py --in data/interim/${OUT}_spliced.parquet --out data/interim/${OUT}_final.parquet | grep -v capped_run | tail -1
REF="${REF:-}"
if [ -n "$REF" ]; then
  $CAP $PY -c "
import polars as pl
a=pl.read_parquet('data/interim/${OUT}_final.parquet').sort('MVT_ID_mvt'); b=pl.read_parquet('$REF').sort('MVT_ID_mvt')
assert a['MVT_ID_mvt'].equals(b['MVT_ID_mvt']) and a.columns==b.columns
d=(a['TAXITIME_SEC_mvt'].cast(float)-b['TAXITIME_SEC_mvt'].cast(float)).abs()
print('RECONSTRUCTION vs $REF: rows differing', int((d>0).sum()), 'max |diff|', float(d.max()), '; frames equal:', a.equals(b))" | grep -v capped_run
fi
say "BUILD $OUT ($FEATS, recipe '$RECIPE'${R5_CB:+ + CatBoost blend}${R5_SEEDS:+ + seed bag $R5_SEEDS}) DONE -> data/interim/${OUT}_final.parquet"
