"""List every adsb.lol globe_history release (metadata only) + PREFERRED_RELEASES.txt.
Writes data/interim/agent_alt_releases.csv: date, tag, replica, n_parts, bytes, preferred."""
import csv, json, re, sys, time, urllib.request
UA = {"User-Agent": "prc-taxiout-2026 research (adsb.lol open data)", "Accept": "application/vnd.github+json"}
rows = []
for year in (2025, 2026):
    pref = urllib.request.urlopen(urllib.request.Request(
        f"https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt", headers=UA), timeout=60).read().decode()
    preferred = set(re.findall(r"/download/(v[^/]+)/", pref))
    page = 1
    while True:
        url = f"https://api.github.com/repos/adsblol/globe_history_{year}/releases?per_page=100&page={page}"
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            rels = json.load(r)
            left = r.headers.get("X-RateLimit-Remaining")
        if not rels:
            break
        for rel in rels:
            tag = rel["tag_name"]
            m = re.match(r"v(\d{4})\.(\d{2})\.(\d{2})-planes-readsb-(.+)$", tag)
            if not m:
                continue
            parts = [a for a in rel["assets"] if a["name"].startswith(tag + ".tar.")]
            rows.append({"date": f"{m[1]}-{m[2]}-{m[3]}", "tag": tag, "replica": m[4], "n_parts": len(parts),
                         "bytes": sum(int(a["size"]) for a in parts), "preferred": int(tag in preferred)})
        print(year, "page", page, len(rels), "rate left", left, flush=True)
        if len(rels) < 100:
            break
        page += 1
rows.sort(key=lambda r: (r["date"], r["replica"]))
with open("data/interim/agent_alt_releases.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), "releases")
