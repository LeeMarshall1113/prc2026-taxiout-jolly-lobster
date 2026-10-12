"""arith3 step 7 (experiment arith3-serve, 2026-10-06): build data/interim/agent_arith3_final<suffix>.parquet = v150 with the residual-corrector term replaced by the round-three term
(mean of the 12 serve fits <serve-dir>/<ARM>_serve_s0..s11.npz) at the listed airports, applied as archive/agent_arith_13_build.py applied the round-two term:
per-airport doses as served (EDDF 1.25, EDDM 1.25, EGLL 1.0, EHAM 1.25, LEBL 1.0, LEMD 1.0, LSZH 1.25, LIRF 0.75, LFPG 0.75), rows nr = is9 & ~RR (agent_bigc13_serve_base) minus every row that
v145, v146, v147 or v148 moved against its parent; tail and Rome rules are later pieces and are not touched. Rome July rows are never touched.
The swap is made from v146 (the last file that carries the plan-clock term on these rows): new value = rint(max(v146 + dose x round-three term - term in v146, 0)), term in v146 = plan-clock term
(dose x mean of agent_pcd_modal/serve/C4ff7V5_s*) or the old LFPG term (kF_old). This is one rule for both cases: an airport v150 serves with the round-two term (the six of agent_arith_airports.json)
and an airport still on the plan-clock term (LEMD, LIRF, LSZH): no second rounding on top of v150. Before anything is written the script proves it knows what v150 carries on those rows:
v150 == rint(max(v146 + round-two term - term in v146, 0)) at the six airports and v150 == v146 at the other three (exact, row by row).
usage: python agent_arith3_07_build.py selftest                      (round-two serve fits put in place of round three, list = the six: must reproduce v150 exactly)
       python agent_arith3_07_build.py build [--airports EDDF,EDDM,EGLL,EHAM,LEBL,LFPG,LEMD] [--suffix _four] [--serve-dir data/interim/agent_arith3_serve/] [--arm KYQGSR-pw-s0-r0]
Guards (any failure: no file): 12 serve files with ids in file order and equal rounds, corr(new term, term served now) >= 0.7 per changed airport, |mean shift| <= 5 s, |row move| <= 600 s,
airports outside the list equal v150 exactly, Rome July untouched. Then archive/agent_smallbundle_06_check.py CAND v150."""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import polars as pl

I = "data/interim/"
ID, T = "MVT_ID_mvt", "TAXITIME_SEC_mvt"
DOSE = {"EDDF": 1.25, "EDDM": 1.25, "EGLL": 1.0, "LEBL": 1.0, "LEMD": 1.0, "LIRF": 0.75, "LSZH": 1.25, "EHAM": 1.25, "LFPG": 0.75}
DEFAULT = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LFPG", "LEMD"]
args = sys.argv[1:]
cmd = args[0] if args else "build"


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


AIRPORTS = [a for a in opt("--airports", ",".join(DEFAULT)).split(",") if a]
SUFFIX = opt("--suffix", "")
SDIR = opt("--serve-dir", I + "agent_arith3_serve/")
ARM = opt("--arm", "KYQGSR-pw-s0-r0")
assert cmd in ("selftest", "build") and all(a in DOSE for a in AIRPORTS), (cmd, AIRPORTS)
sb = np.load(I + "agent_bigc13_serve_base.npz", allow_pickle=True)
si = np.load(I + "agent_bigc13_serve_in.npz", allow_pickle=True)
rd = lambda n: pl.read_parquet(I + n + ".parquet").sort(ID)   # noqa: E731
val = lambda n: rd("agent_bundle_%s_final" % n)[T].cast(pl.Float64).to_numpy()   # noqa: E731
v150f = rd("agent_bundle_v150_final")
ids = v150f[ID].to_numpy()
v150 = v150f[T].cast(pl.Float64).to_numpy()
v146 = val("v146")
assert (sb["ids"] == ids).all() and (si["ids"] == ids).all() and (rd("agent_bundle_v146_final")[ID].to_numpy() == ids).all()
apt = si["apt"].astype(str)
mon_s = si["mon_s"]
is9, RR, kold = sb["is9"], sb["RR"], sb["kF_old"]
nr = is9 & ~RR
moved_later = np.zeros(len(ids), bool)
for a, b in (("v144", "v145"), ("v145", "v146"), ("v146", "v147"), ("v146", "v148")):
    moved_later |= val(b) != val(a)
rome_july = (apt == "LIRF") & (mon_s == 7)
print("rows moved by v145/v146/v147/v148 vs parent (excluded):", int(moved_later.sum()), "| Rome July rows (never touched):", int(rome_july.sum()))
six = [a for a, on in json.load(open(I + "agent_arith_airports.json")).items() if on]
print("airports v150 serves with the round-two term:", sorted(six), "| list for the round-three term:", AIRPORTS, "| suffix '%s'" % SUFFIX)


def mean_pred(pattern, need_ids):
    fs = sorted(f for f in glob.glob(pattern) if "_wire" not in f and ".tmp" not in f)
    assert len(fs) == 12, (pattern, len(fs))
    zs = [np.load(f, allow_pickle=True) for f in fs]
    if need_ids:
        assert all(np.array_equal(z["ids"].astype(float), ids.astype(float)) for z in zs), "serve file ids are not in file order"
        assert len({int(z["rounds"]) for z in zs}) == 1 and sorted(int(z["seed"]) for z in zs) == list(range(12)), "seeds / rounds of the 12 serve files"
        print("round-three serve fits: 12 seeds, rounds", int(zs[0]["rounds"]), "columns", len(zs[0]["names"]))
    Pm = np.mean([z["pred"] for z in zs], 0)
    assert Pm.shape == ids.shape and np.isfinite(Pm).all()
    return Pm


Pp = mean_pred(I + "agent_pcd_modal/serve/C4ff7V5_s*.npz", False)
Pa = mean_pred(I + "agent_arith_serve/ABCDE-pw-s0-r0_s*.npz", False)
P3 = Pa if cmd == "selftest" else mean_pred(SDIR + ARM + "_serve_s*.npz", True)
if cmd == "selftest":
    AIRPORTS = six
kap = np.array([DOSE.get(a, 0.0) for a in apt])
t146 = np.where(apt == "LFPG", kold, kap * Pp)          # corrector term inside v146
tA = kap * Pa                                            # round-two term (LFPG dose 0.75 is in DOSE)
t3 = kap * P3
el = nr & ~moved_later                                   # rows on which the corrector term may be swapped
swap = lambda t: np.rint(np.maximum(v146 + (t - t146), 0))   # noqa: E731
# what v150 carries (must be exact, otherwise the swap below would not be a replacement)
for a in sorted(DOSE):
    m = el & (apt == a)
    exp = np.where((tA - t146) != 0, swap(tA), v146) if a in six else v146
    bad = int((v150[m] != exp[m]).sum())
    print("v150 at %s (%d rows): %s term, rows not matching the reconstruction: %d" % (a, int(m.sum()), "round-two" if a in six else "plan-clock (old)", bad))
    assert bad == 0, "v150 is not what the build assumes at " + a
out = v150.copy()
ok = True
print("airport | rows | term served now | corr(new, now) | mean move s | rms move s | max |move| s")
for a in AIRPORTS:
    m = el & (apt == a) & ~rome_july
    now = tA if a in six else t146
    out[m] = np.where((t3 - t146)[m] != 0, swap(t3)[m], v146[m])
    d = out[m] - v150[m]
    c = float(np.corrcoef(t3[m], now[m])[0, 1])
    bad = (c < 0.7 or abs(d.mean()) > 5 or np.abs(d).max() > 600) and cmd != "selftest"
    ok &= not bad
    print("%s | %d | %s | %.3f | %+.3f | %.2f | %.0f%s" % (a, int(m.sum()), "round-two" if a in six else "plan-clock (old)", c, d.mean(), np.sqrt((d ** 2).mean()), np.abs(d).max(), "  <-- GUARD FAIL" if bad else ""))
mv = out != v150
assert not (mv & ~np.isin(apt, AIRPORTS)).any(), "rows outside the airport list moved"
for a in ("LIRF", "LSZH"):
    if a not in AIRPORTS:
        assert (out[apt == a] == v150[apt == a]).all(), a + " differs from v150"
assert not (mv & rome_july).any() and not (mv & (RR | ~is9 | moved_later)).any() and (out >= 0).all()
if cmd == "selftest":
    assert not mv.any(), "selftest: %d rows differ from v150" % int(mv.sum())
    print("SELFTEST OK: round-two fits in place of round three reproduce v150 exactly (0 rows differ)")
    sys.exit(0)
if not ok:
    print("GUARD FAILED: no file written")
    sys.exit(1)
name = I + "agent_arith3_final%s.parquet" % SUFFIX
v150f.with_columns(pl.Series(T, out).cast(v150f.schema[T])).write_parquet(name)
dl = out - v150
print("wrote %s: rows moved %d; mean shift %+.3f; rms (moved) %.2f; max %.0f; airports moved %s" % (name, int(mv.sum()), dl.mean(), np.sqrt((dl[mv] ** 2).mean()), np.abs(dl).max(), sorted(set(apt[mv].tolist()))))
print("LIRF rows equal v150: %s; LSZH rows equal v150: %s; Rome July rows moved: %d" % ((out[apt == "LIRF"] == v150[apt == "LIRF"]).all(), (out[apt == "LSZH"] == v150[apt == "LSZH"]).all(), int((mv & rome_july).sum())))
r = subprocess.run([sys.executable, "archive/agent_smallbundle_06_check.py", name, I + "agent_bundle_v150_final.parquet"], capture_output=True, text=True, encoding="utf-8", errors="replace",
                   env=dict(os.environ, PYTHONIOENCODING="utf-8", POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR="load_as_storage"))
print(r.stdout[-2500:])
print(r.stderr[-500:])
sys.exit(r.returncode)
