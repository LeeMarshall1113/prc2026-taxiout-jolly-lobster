"""download-only (no numpy): prc-stage1:/bigc/out/*.npz -> data/interim/bigc_out/ (skips existing). run with the python that has modal."""
import os, modal
vol=modal.Volume.from_name("prc-stage1"); os.makedirs("data/interim/bigc_out",exist_ok=True); n=0
for e in vol.listdir("bigc/out"):
    b=os.path.basename(e.path)
    if not b.endswith(".npz") or ".tmp." in b or os.path.exists("data/interim/bigc_out/"+b): continue
    with open("data/interim/bigc_out/"+b,"wb") as fh:
        for c in vol.read_file(e.path): fh.write(c)
    n+=1
print("downloaded",n)

# serve-fit outputs (modal/agent_bigc_app.py::serve_one)
os.makedirs("data/interim/bigc_serve",exist_ok=True); m=0
try:
    for e in vol.listdir("bigc/serve"):
        b=os.path.basename(e.path)
        if not b.endswith(".npz") or ".tmp." in b or os.path.exists("data/interim/bigc_serve/"+b): continue
        with open("data/interim/bigc_serve/"+b,"wb") as fh:
            for c in vol.read_file(e.path): fh.write(c)
        m+=1
except Exception as ex: print("no serve dir yet",ex)
print("downloaded serve",m)
