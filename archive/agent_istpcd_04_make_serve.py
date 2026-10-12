"""istpcd 04 (generator): writes archive/agent_istpcd_04_serve.py = agent_istjul_03_serve.py (read only, unchanged) with
(a) a --cols switch appending the nine plan-clock distance columns (fold D, serve Ds from agent_istpcd_cols.npz) after the view columns, and
(b) the tail replaced by: save the served term (k x clipped mean of the seeds, snow rows 0) in the serve-row order, compared with the shipped lv3a term.
Run once (no fitting)."""
src = open("archive/agent_istjul_03_serve.py", encoding="utf-8").read()
head = src.split("d = a.k * d; d[snow_s] = 0")[0]
head = head.replace('ap.add_argument("--arm", default="AQC");', 'ap.add_argument("--cols", default="none", help="none | pcd | shuf | rand: extra nine columns"); ap.add_argument("--arm", default="AQC");', 1)
assert "--cols" in head
needle = "Ftr2 = np.hstack(parts_tr).astype(np.float32); Fs2 = np.hstack(parts_sv).astype(np.float32)"
assert needle in head
add = '''ZC = np.load(I + "agent_istpcd_cols.npz", allow_pickle=True)
assert (ZC["fid"] == fid).all() and (ZC["sid"] == rl[ID].to_numpy()).all()
if a.cols == "pcd":
    parts_tr.append(ZC["D"]); parts_sv.append(ZC["Ds"])
elif a.cols == "shuf":
    parts_tr.append(ZC["Dshuf"][0]); parts_sv.append(ZC["Ds"])   # control: shuffled prediction in the fold columns, served columns as built
elif a.cols == "rand":
    parts_tr.append(ZC["Dnoise"][0]); parts_sv.append(np.random.default_rng(3000).normal(0, 1000, ZC["Ds"].shape).astype(np.float32))
print("extra columns", a.cols, flush=True)
'''
head = head.replace(needle, add + needle, 1)
tail = '''d = a.k * d; d[snow_s] = 0
old = np.load(I + "agent_lv3a_ltfm_final_delta.npz"); v128 = pl.read_parquet(I + "agent_bigc_v4_board_final.parquet").sort(ID)
assert (old["ids"] == v128[ID].to_numpy()).all(); idpos = {x: i for i, x in enumerate(v128[ID].to_numpy())}
new = np.zeros(len(v128)); new[[idpos[x] for x in rl[ID].to_numpy()]] = d
lt = old["lt"] if "lt" in old.files else None
print("served term: mean %.3f sd %.3f; shipped lv3a term on the same rows: mean %.3f sd %.3f" % (float(d.mean()), float(d.std()), float(old["new"][new != 0].mean()), float(old["new"][new != 0].std())))
dd = new - old["new"]
print("this fit's term minus the shipped lv3a term (all LTFM rows): mean %.4f rms %.4f max abs %.4f (0 = reproduces)" % (float(dd[np.isin(np.arange(len(dd)), [idpos[x] for x in rl[ID].to_numpy()])].mean()), float(np.sqrt((dd ** 2).mean() * len(dd) / len(rl))), float(np.abs(dd).max())))
np.savez(I + a.out + "_delta.npz", ids=v128[ID].to_numpy(), new=new, snow_serve=snow_s, sid=rl[ID].to_numpy(), d=d)
print("wrote", a.out)
'''
open("archive/agent_istpcd_04_serve.py", "w", encoding="utf-8").write(head + tail)
print("written archive/agent_istpcd_04_serve.py", len(head + tail))
