"""tailc assemble: out = rint(max(v130 + 0.5 x ehat_B, 0)) on the nine airports (LTFM untouched) -> agent_tailc_board.parquet; footprint vs the fold; smallbundle check."""
import subprocess
import sys
import numpy as np
import polars as pl
I = "data/interim/"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
K = 0.5
base_f = "agent_lv3a_ltfm_final.parquet"
sv = np.load(I + "agent_tailc_serve_ehat.npz", allow_pickle=True)
ids, apt, eh = sv["ids"], sv["apt"].astype(str), sv["ehat"]
bf = pl.read_parquet(I + base_f).sort(ID)
assert (bf[ID].to_numpy() == ids).all()
cur = bf[T].cast(pl.Float64).to_numpy()
o = np.rint(np.maximum(cur + K * eh, 0))
mv = o != cur
bf.with_columns(pl.Series(T, o).cast(bf.schema[T])).write_parquet(I + "agent_tailc_board.parquet")
use = sv["use"]
print(f"rows moved {int(mv.sum()):,}; mean shift {float((o - cur).mean()):+.3f}; max up {np.max(o-cur):.0f} down {np.min(o-cur):.0f}; sum dp^2/row (all rows) {float(((o - cur) ** 2).mean()):.3f}")
z = np.load(I + "agent_bigc12_fold.npz", allow_pickle=True)
mon = z["mon"]
res = [np.load(I + f"agent_tailc_s{s}.npz") for s in (0, 1, 2)]
o8 = np.isin(mon, [1, 2, 3, 7, 8, 9, 10, 11])
ef = np.mean([r["B"] for r in res], 0)[o8] * K
fold = float((ef ** 2).mean())
# 2026 months are Jan and Jul; also compare with the fold's Jan+Jul months
jj = np.isin(mon[o8], [1, 7])
srv = float(((K * eh[use]) ** 2).sum() / len(eh))   # per row over all 9-airport + other rows
srv9 = float(((K * eh[use]) ** 2).sum() / int(np.isin(apt, ["EDDF","EDDM","EGLL","EHAM","LEBL","LEMD","LFPG","LIRF","LSZH"]).sum()))
fold9 = fold
print(f"sum dp^2/row: fold (8 months, nine airports) {fold:.2f}; fold Jan+Jul {float((ef[jj]**2).mean()):.2f}; 2026 nine-airport rows {srv9:.2f}; ratio {srv9/fold:.2f} (vs Jan+Jul {srv9/float((ef[jj]**2).mean()):.2f}) ({'PASS' if srv9/fold<=2.0 else 'FAIL'} at <= 2.0)")
print("LTFM rows changed:", int((o[apt == 'LTFM'] != cur[apt == 'LTFM']).sum()))
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", I + "agent_tailc_board.parquet", I + base_f], capture_output=True, text=True, encoding="utf-8", errors="replace",
                   env=dict(__import__("os").environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage"))
print(r.stdout[-1500:], r.stderr[-500:])
