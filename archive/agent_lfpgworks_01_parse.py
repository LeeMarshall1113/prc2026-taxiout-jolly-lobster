"""LFPG works closures from the French AIP supplements (SUP AIP, metropolitan series), 2025 + 2026.

SOURCE AND LICENCE. PDFs in data/external/sia_sup/ (archive/fetch_sia_sup.sh, fetched
2026-09-21), published by the Service de l'Information Aeronautique (SIA) under the Licence
Ouverte / Open Licence: reuse allowed, attribution required ("SIA" + date of last update), no
alteration of meaning. Every row written here carries the supplement's own number and
publication date, which is that attribution; nothing in the text is paraphrased into a
different meaning -- dates, taxiways and stands are copied, and each stand row keeps the
sentence it came from.

WHAT IT DOES
  1. pdftotext -enc UTF-8 -layout every PDF into data/interim/agent_lfpgworks/txt/ (cached).
  2. keeps the supplements whose text concerns LFPG (Charles de Gaulle), and classifies each:
     AD_WORKS (an aerodrome restriction at LFPG), OTHER_LFPG (LFPG only mentioned: airspace,
     another aerodrome, chart representation).
  3. parses, per supplement: number, AIRAC flag, publication date, object, 'En vigueur' dates,
     the DATES ET HEURES start/end with HHMM where given, "annule et remplace" references,
     runways, taxiways/exits, and NAMED STANDS with a role:
        closed    -- the text says the stand is closed ("Postes E24, E26 et E30 fermes")
        modified  -- the stand keeps operating under a changed departure procedure or access
                     rule (long pushback, restricted exit, span limit)
        range     -- inside a named range ("entre les postes E24 et E30 inclus"), no role given
        named     -- named, role not stated
  4. writes data/interim/agent_lfpgworks/sup_lfpg.tsv (one row per supplement) and
     sup_lfpg_stands.tsv (one row per supplement x stand), and prints the stand match rate
     against STAND_mvt (2025 LFPG departures, and 2026 ranking rows) and against OSM refs.

The automatic stand roles are checked by hand against every AD_WORKS text; the corrections
live in MANUAL below, each with the sentence that justifies it, so the parse is auditable.

    PYTHONPATH=. .venv/Scripts/python.exe archive/agent_lfpgworks_01_parse.py
"""
from __future__ import annotations

import re
import subprocess
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, ".")

import polars as pl

PDF_DIR = Path("data/external/sia_sup")
OUT = Path("data/interim/agent_lfpgworks")
TXT = OUT / "txt"
PDFTOTEXT = "C:/Program Files/Git/mingw64/bin/pdftotext.exe"

MONTHS = {"janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6, "juillet": 7,
          "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11, "decembre": 12}
MON_ABBR = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
            "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def fold(s: str) -> str:
    """Strip accents and normalise apostrophes/dashes, for matching only (never for output)."""
    s = s.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def extract(pdf: Path) -> str:
    TXT.mkdir(parents=True, exist_ok=True)
    out = TXT / (pdf.stem + ".txt")
    if not out.exists():
        exe = PDFTOTEXT if Path(PDFTOTEXT).exists() else "pdftotext"
        subprocess.run([exe, "-enc", "UTF-8", "-layout", str(pdf), str(out)], check=False)
    return out.read_text(encoding="utf-8", errors="replace") if out.exists() else ""


# --------------------------------------------------------------------------- dates
DATE_RE = re.compile(
    r"(\d{1,2})(?:er)?\s*(janvier|fevrier|mars|avril|mai|juin|juillet|aout|septembre|octobre|"
    r"novembre|decembre)?\s*(\d{4})?(?:\s*a\s*(\d{4}))?", re.I)


def _dates_in(s: str):
    """All '<day> [<month>] [<year>] [a HHMM]' tokens in a folded string, in order."""
    out = []
    for m in DATE_RE.finditer(s):
        d, mon, yr, hhmm = m.groups()
        if mon is None and yr is None and hhmm is None:
            continue
        out.append((int(d), MONTHS.get(mon.lower()) if mon else None, int(yr) if yr else None, hhmm))
    return out


def parse_span(s: str, default_year: int | None):
    """'du 30 octobre 2025 a 0300 au 2 fevrier 2026 a 0400' -> (start, end) UTC datetimes.

    Missing month/year on the start borrow the end's ("Du 09 au 14 juillet 2025"); a missing
    year everywhere borrows the supplement's own year ("Du 05 mai a 0300 au 07 juin a 0300").
    Times are UTC, the French AIP convention (GEN 2.1), and 045/25 states it explicitly.
    A missing time is 00:00 at the start and 23:59 at the end (whole-day reading).
    """
    s = fold(s)
    s = re.sub(r"\b(lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche)\s+", "", s, flags=re.I)
    m = re.search(r"(?:du|de)?\s*(\d{1,2}(?:er)?\s*[a-z]*\s*\d{0,4}(?:\s*a\s*\d{4})?)\s+au\s+"
                  r"(\d{1,2}(?:er)?\s*[a-z]+\s*\d{0,4}(?:\s*a\s*\d{4})?)", s, re.I)
    if not m:
        return None, None
    a, b = _dates_in(m.group(1)), _dates_in(m.group(2))
    if not a:
        dd = re.match(r"\s*(\d{1,2})", m.group(1))
        a = [(int(dd.group(1)), None, None, None)] if dd else []
    if not a or not b:
        return None, None
    (d1, m1, y1, t1), (d2, m2, y2, t2) = a[0], b[0]
    y2 = y2 or default_year
    m1 = m1 or m2
    y1 = y1 or (y2 if (m1 or 0) <= (m2 or 0) else (y2 - 1 if y2 else None))
    if None in (m1, m2, y1, y2):
        return None, None

    def mk(y, mo, d, t, end):
        hh, mm = (int(t[:2]), int(t[2:])) if t else ((23, 59) if end else (0, 0))
        if hh == 24:
            hh, mm = 23, 59
        return datetime(y, mo, d, hh, mm, tzinfo=timezone.utc)

    return mk(y1, m1, d1, t1, False), mk(y2, m2, d2, t2, True)


# --------------------------------------------------------------------------- sections
def header(text: str, stem: str) -> dict:
    f = fold(text)
    m = re.search(r"lf_sup_(a_)?(\d{4})_(\d{3})", stem)
    airac, year, num = bool(m.group(1)), int(m.group(2)), int(m.group(3))
    pub = re.search(r"Date de publication\s*:\s*(\d{1,2})\s+([A-Z]{3})\s+(\d{4})", f)
    pub_date = (datetime(int(pub.group(3)), MON_ABBR.get(pub.group(2).upper(), 1), int(pub.group(1))).date()
                if pub and pub.group(2).upper() in MON_ABBR else None)
    # object: from "Objet" to "Lieu", with any interleaved 'En vigueur' fragment removed
    ob = re.search(r"Objet\s*:?(.*?)\n\s*Lieu", f, re.S)
    objet = ""
    if ob:
        objet = re.sub(r"En vigueur\s*:?[^\n]*", " ", ob.group(1))
        objet = re.sub(r"^\s*:\s*du.*$", " ", objet, flags=re.M | re.I)
        objet = re.sub(r"\s+", " ", objet).strip(" :")
    ev = re.search(r"En vigueur(.*?)\n\s*Lieu", f, re.S)
    ev_txt = re.sub(r"\s+", " ", ev.group(1)) if ev else ""
    # the whole "Du <date> au <date> <year>" span, wherever the layout put it in that block
    ev_span = re.search(r"\b((?:du|de)\s+(?:[a-z]+\s+)?\d{1,2}.*?\bau\b.*?\d{4})", ev_txt, re.I)
    ev_txt = ev_span.group(1) if ev_span else ev_txt
    lieu = re.search(r"Lieu\s*:?([^\n]*)", f)
    dh = re.search(r"DATES ET HEURES D.ACTIVITE\s*\n+\s*([^\n]+)", f)
    repl = sorted(set(re.findall(r"(?:annule|remplace)[^.\n]{0,40}?SUP\s+AIP\s+(?:AIRAC\s+)?(?:N.?\s*)?"
                                 r"(\d{1,3})\s*/\s*(\d{2,4})", f, re.I)))
    return dict(sup_id=f"{year}/{num:03d}", year=year, num=num, airac=airac, pub_date=pub_date,
                objet=objet, en_vigueur=ev_txt.strip(), lieu=(lieu.group(1).strip() if lieu else ""),
                dates_heures=(dh.group(1).strip() if dh else ""),
                replaces=";".join(f"{int(a)}/{b[-2:]}" for a, b in repl))


LFPG_RE = re.compile(r"LFPG|Charles\s+de\s+Gaulle", re.I)


def classify(h: dict, text: str) -> str:
    """AD_WORKS = a supplement ABOUT the aerodrome LFPG (works, closures, ground procedures);
    OTHER_LFPG = LFPG only mentioned (airspace zones, another aerodrome, frequencies)."""
    f = fold(text)
    if not LFPG_RE.search(f):
        return ""
    lieu, objet = fold(h["lieu"]), fold(h["objet"])
    airspace = re.search(r"zones?\s+(interdite|reglementee|dangereuse)|\bZIT\b|\bZRT\b|\bZDT\b", objet, re.I)
    other_ad = re.search(r"\bLF(?!PG)[A-Z]{2}\b", objet) and not LFPG_RE.search(objet)
    if airspace or other_ad:
        return "OTHER_LFPG"
    lieu_is_lfpg = LFPG_RE.search(lieu) and re.search(r"\bAD\b", lieu) and \
        len(re.findall(r"\bLF[A-Z]{2}\b", lieu.replace("LFFF", ""))) <= 1
    if lieu_is_lfpg or (LFPG_RE.search(objet) and re.search(r"Restrictions? d.utilisation", objet, re.I)):
        return "AD_WORKS"
    return "OTHER_LFPG"


# --------------------------------------------------------------------------- stands, taxiways
STAND_TOK = r"[A-Z]\d{2}[A-Z]?"          # CDG stands are letter + two digits (E08, F32, N41)


def stands(text: str, universe: set[str]) -> list[dict]:
    """Stands named in the text, with an automatic role and the sentence they came from."""
    f = fold(text)
    rows = []
    # sentence units: a full stop/semicolon, a dash-list item, or a line that starts with a
    # stand token ("E14 : repoussage ...", "E08 : Poste ferme") -- a colon does NOT end a unit,
    # or "E14 :" would be cut away from its own instruction
    sentences = re.split(r"(?<=[.;])\s+|\n\s*[-•⚬]\s+|\n(?=\s*[A-Z]\d{2}\b)", f)
    for sent in sentences:
        s1 = re.sub(r"\s+", " ", sent).strip()
        toks = re.findall(rf"\b{STAND_TOK}\b", s1)
        # "H24" is the round-the-clock marker ("fermeture H24"), not stand H24, unless a
        # stand list names it
        if "H24" in toks and not re.search(rf"postes?\s+(?:avions?\s+)?(?:{STAND_TOK}\s*(?:,|et)\s*)*H24", s1):
            toks = [x for x in toks if x != "H24"]
        if not toks:
            continue
        has_poste = re.search(r"\bpostes?\b|\btiroir\b", s1, re.I) is not None
        lead = re.match(rf"^-?\s*((?:{STAND_TOK})(?:\s*(?:,|et)\s*{STAND_TOK})*)\s*:(.*)$", s1)
        leading = lead is not None
        if not (has_poste or leading):
            continue
        # ranges: "entre les postes (avions) X et Y", "X a Y"
        for a, b in re.findall(rf"entre\s+les\s+postes?\s+(?:avions?\s+|de\s+stationnement\s+)?"
                               rf"({STAND_TOK})\s+(?:et|a)\s+({STAND_TOK})", s1, re.I) + \
                re.findall(rf"postes?\s+(?:avions?\s+)?({STAND_TOK})\s+a\s+({STAND_TOK})", s1, re.I):
            if a[0] == b[0]:
                lo, hi = sorted((int(a[1:3]), int(b[1:3])))
                for u in universe:
                    if u[0] == a[0] and u[1:3].isdigit() and lo <= int(u[1:3]) <= hi and len(u) == 3:
                        rows.append(dict(stand=u, role="range", sentence=s1[:300]))
        body = lead.group(2) if leading else s1
        if re.search(r"\bfermee?s?\b|fermeture\s+(?:H24\s+)?(?:des|du)\s+(?:voies\s+et\s+des\s+)?postes?", body, re.I):
            role = "closed"
        elif re.search(r"\bgarage\b", body, re.I):
            role = "garage"
        elif leading or re.search(r"repoussage|sortie|limite|accessible|interdit|envergure|code", s1, re.I):
            role = "modified"
        else:
            role = "named"
        for t in sorted(set(toks)):
            rows.append(dict(stand=t, role=role, sentence=s1[:300]))
    return rows


TWY_TOK = r"[A-Z]{1,3}\d{0,2}[A-Z]?"
NOT_TWY = {"AD", "LFPG", "SUP", "AIP", "AIRAC", "FIR", "UTC", "NOTA", "ATIS", "ATC", "SIA", "CDG",
           "SIAE", "API", "ILS", "RNP", "LOC", "DME", "SID", "MVL", "NM", "VH", "O", "S", "H", "ET",
           "PAI", "DGV", "RETIL", "TWR", "APP", "GMC", "PCA", "REL", "THL", "RWSL", "FR", "LFFF", "N",
           "AF", "B", "BENE", "AGEN", "OUEST", "EST", "WORKS", "A380", "B748", "H24"}


def runways_taxiways(text: str) -> tuple[str, str]:
    """Runways and taxiways/exits a supplement closes or restricts, recorded for the non-stand
    supplements. Descriptive only: nothing downstream tests on it."""
    f = fold(text)
    rwy = sorted(set(f"{a}/{b}" for a, b in re.findall(r"pistes?\s+(\d{2}[LRC]?)\s*[/-]\s*(\d{2}[LRC]?)", f)))
    twy = set()
    for m in re.finditer(rf"\bvoies?\s+(?:de\s+circulation\s+)?(?:d.aire\s+de\s+trafic\s+)?({TWY_TOK})\b", f):
        twy.add(m.group(1))
    # "taxiways RT5, RT7 , WT et TANGO", "TWY P2, P3, P4", "degagements ... W4 et W5", "sortie ... TE1"
    for m in re.finditer(rf"\b(?:TWY|taxiways?|degagements?\s+a\s+grande\s+vitesse(?:\s*\(DGV\))?|"
                         rf"sorties?(?:\s+rapides?)?(?:\s+au\s+moteur)?\s+par)\s+((?:TANGO|MIKE|{TWY_TOK})\b"
                         rf"(?:\s*(?:,|et)\s*(?:TANGO|MIKE|{TWY_TOK})\b)*)", f, re.I):
        twy.update(t.upper() for t in re.findall(rf"\b(TANGO|MIKE|{TWY_TOK})\b", m.group(1)))
    # dash-list items under any 'Fermeture' / 'Voies' line: "- T entre les voies RT8 et RT11", "- W6"
    for blk in re.finditer(r"(?:Fermeture|Voies)[^\n]*\n((?:\s*-\s*[^\n]+\n)+)", f, re.I):
        for line in blk.group(1).splitlines():
            if re.search(r"\bpostes?\b|\bpiste\b", line, re.I):
                continue
            twy.update(re.findall(rf"\b({TWY_TOK})\b", line))
    twy = {t for t in twy if t not in NOT_TWY and not re.fullmatch(r"\d+", t)}
    return ";".join(rwy), ";".join(sorted(twy))


# --------------------------------------------------------------------------- manual audit
# Corrections after reading each AD_WORKS text in full (2026-09-21). Key: (sup_id, stand) ->
# (role, reason). An entry overrides the automatic role and is written to the table as such.
# Only supplements whose window touches 2025 out-of-fold months or the scored months (January,
# July 2026) were audited stand by stand; the rest keep the automatic roles, flagged below.
_170 = "SUP 170/25: 'Poste E08 ferme servant de tiroir de retournement'"
MANUAL: dict[tuple[str, str], tuple[str, str]] = {
    # 005/25 -- one list under 'Fermeture H24 des voies et des postes de stationnement avions
    # suivants', split by the sentence cutter from its heading
    **{("2025/005", s): ("closed", "listed under 'Fermeture H24 des voies et des postes de stationnement'")
       for s in ("N01", "N02", "N03", "N04", "N31", "N33", "N41", "N43")},
    # 170/25 -- the E08 sentence names E18/E20/E22 only as stands that push back INTO E08
    ("2025/170", "E08"): ("closed", _170),
    ("2025/170", "E18"): ("modified", "'repoussage long face a l'Est et retournement face au Sud sur E08'"),
    ("2025/170", "E20"): ("modified", "'repoussage long face a l'Est et retournement face au Sud sur E08'"),
    ("2025/170", "E22"): ("modified", "'repoussage long face a l'Est et retournement face au Sud sur E08'"),
    ("2025/170", "E24"): ("closed", "'Postes E24, E26 et E30 fermes'"),
    ("2025/170", "E26"): ("closed", "'Postes E24, E26 et E30 fermes'"),
    ("2025/170", "E30"): ("closed", "'Postes E24, E26 et E30 fermes'"),
    ("2025/170", "E14"): ("modified", "'repoussage face a l'Ouest et sortie des avions aux moteurs par TE1'"),
    ("2025/170", "E16"): ("modified", "'repoussage face a l'Ouest et sortie des avions aux moteurs par TE1'"),
    ("2025/170", "E32"): ("modified", "'Postes E32 et E34 limites aux avions de code C'; specific pushback"),
    ("2025/170", "E34"): ("modified", "'Postes E32 et E34 limites aux avions de code C'; specific pushback"),
    # the odd E stands are not on P2: OSM puts E25/E27/E29 ~450 m east of E24-E30, on the
    # other face (lon 2.584 vs 2.578-2.580), and the supplement's own closed list is E24, E26, E30
    ("2025/170", "E25"): ("not_affected", "range-expansion artefact; not on P2 (OSM), not in the closed list"),
    ("2025/170", "E27"): ("not_affected", "range-expansion artefact; not on P2 (OSM), not in the closed list"),
    ("2025/170", "E29"): ("not_affected", "range-expansion artefact; not on P2 (OSM), not in the closed list"),
    # 209/25 (3 Feb - 30 Jun 2026, outside every scored month) -- recorded, not used
    ("2025/209", "E08"): ("closed", "'Configuration du poste avion en tiroir de retournement'"),
    ("2025/209", "E32"): ("closed", "'Voie P2 fermee entre les postes avion E32 a E34 inclus'"),
    ("2025/209", "E34"): ("closed", "'Voie P2 fermee entre les postes avion E32 a E34 inclus'"),
    # 096/26 phase 1 (26 Jun - 4 Aug 2026, the July phase): I30/I32/I34 closed, I31/I33 garage
    ("2026/096", "I30"): ("closed", "phase 1: 'Fermeture des postes I30, I32 et I34'"),
    ("2026/096", "I31"): ("garage", "phase 1: 'Postes I31 et I33 utilises en garage pour Air France' (closed in phase 2, from 4 Aug)"),
    ("2026/096", "I32"): ("closed", "phase 1: 'Fermeture des postes I30, I32 et I34'"),
    ("2026/096", "I33"): ("garage", "phase 1: 'Postes I31 et I33 utilises en garage pour Air France'"),
    ("2026/096", "I34"): ("closed", "phase 1: 'Fermeture des postes I30, I32 et I34'"),
}
# Stands the automatic pass misses entirely: sup_id -> [(stand, role, reason)].
MANUAL_ADD: dict[str, list[tuple[str, str, str]]] = {}
# Validity overrides where the automatic date parse is wrong: sup_id -> (start, end, reason).
MANUAL_SPAN: dict[str, tuple[str, str, str]] = {}
# Supplements whose window never touches 2025 out-of-fold months nor January/July 2026: their
# automatic stand roles are recorded but were not audited one by one.
NOT_AUDITED = {"2026/004", "2026/019", "2026/151", "2026/152", "2026/169", "2026/187", "2026/188"}


OOF_2025 = [(datetime(2025, m, 1, tzinfo=timezone.utc), datetime(2025, m + 1, 1, tzinfo=timezone.utc))
            for m in (1, 2, 3, 4, 5, 7, 8, 9, 10, 11)]
SCORED_2026 = [(datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc)),
               (datetime(2026, 7, 1, tzinfo=timezone.utc), datetime(2026, 8, 1, tzinfo=timezone.utc))]


def overlaps(a, b, spans) -> bool:
    return a is not None and b is not None and any(a < y and b > x for x, y in spans)


def stand_universe() -> tuple[set[str], set[str], set[str]]:
    C = pl.col
    tr = set(pl.scan_parquet("data/raw/training_*.parquet")
             .filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LFPG"))
             .select("STAND_mvt").unique().collect()["STAND_mvt"].drop_nulls().to_list())
    rk = set(pl.scan_parquet("data/raw/ranking.parquet")
             .filter((C("PHASE_mvt") == "DEP") & (C("ADEP_mvt") == "LFPG"))
             .select("STAND_mvt").unique().collect()["STAND_mvt"].drop_nulls().to_list())
    osm = set(pl.read_parquet("data/external/osm/stands.parquet")
              .filter(C("apt") == "LFPG")["ref"].drop_nulls().to_list())
    return tr, rk, osm


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tr, rk, osm = stand_universe()
    universe = tr | rk
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    print(f"{len(pdfs)} PDFs in {PDF_DIR}")
    sups, srows = [], []
    for pdf in pdfs:
        text = extract(pdf)
        if not text:
            print("  EMPTY TEXT:", pdf.name)
            continue
        h = header(text, pdf.stem)
        cls = classify(h, text)
        if not cls:
            continue
        # dates: the 'En vigueur' span; times from DATES ET HEURES when it names the same days
        ev_start, ev_end = parse_span(h["en_vigueur"], h["year"])
        dh_start, dh_end = (parse_span(h["dates_heures"], h["year"])
                            if re.search(r"\bau\b", fold(h["dates_heures"]), re.I) else (None, None))
        start, end = ev_start, ev_end
        if start is None:
            start, end = dh_start, dh_end
        else:
            if dh_start is not None and dh_start.date() == start.date():
                start = dh_start
            if dh_end is not None and dh_end.date() == end.date():
                end = dh_end
        if start is None:
            m = re.search(r"\b(?:du|de)\s+(?:[a-z]+\s+)?\d{1,2}[^\n]{0,60}?\bau\b[^\n]{0,60}?\d{4}", fold(text), re.I)
            if m:
                start, end = parse_span(m.group(0), h["year"])
        if h["sup_id"] in MANUAL_SPAN:
            a, b, why = MANUAL_SPAN[h["sup_id"]]
            start = datetime.fromisoformat(a).replace(tzinfo=timezone.utc)
            end = datetime.fromisoformat(b).replace(tzinfo=timezone.utc)
        rwy, twy = runways_taxiways(text)
        st = stands(text, universe) if cls == "AD_WORKS" else []
        # one row per stand, strongest role wins: closed > modified > named > range
        rank = {"closed": 4, "garage": 3, "modified": 2, "named": 1, "range": 0}
        best: dict[str, dict] = {}
        for r in st:
            if r["stand"] not in best or rank[r["role"]] > rank[best[r["stand"]]["role"]]:
                best[r["stand"]] = r
        for stn, role, why in MANUAL_ADD.get(h["sup_id"], []):
            best[stn] = dict(stand=stn, role=role, sentence="MANUAL_ADD: " + why)
        for stn, r in best.items():
            role, how = r["role"], "auto"
            if (h["sup_id"], stn) in MANUAL:
                role, why = MANUAL[(h["sup_id"], stn)]
                how = "manual: " + why
            srows.append(dict(sup_id=h["sup_id"], stand=stn, role=role, how=how,
                              audited=h["sup_id"] not in NOT_AUDITED,
                              in_train_2025=stn in tr, in_ranking_2026=stn in rk, in_osm=stn in osm,
                              sentence=r["sentence"]))
        sups.append(dict(**h, cls=cls, file=pdf.name, start_utc=start, end_utc=end,
                         ev_start=ev_start, ev_end=ev_end, runways=rwy, taxiways=twy,
                         stands=";".join(sorted(best)), n_stands=len(best),
                         touches_2025_oof=overlaps(start, end, OOF_2025),
                         touches_scored_2026=overlaps(start, end, SCORED_2026)))

    S = pl.DataFrame(sups).sort("year", "num")
    # attribution travels with every row (Licence Ouverte: source + date of last update)
    S = S.with_columns(pl.format("SIA, SUP AIP {}, published {}, Licence Ouverte / Open Licence",
                                 pl.col("sup_id"), pl.col("pub_date").cast(pl.Utf8)).alias("source"))
    src = dict(zip(S["sup_id"].to_list(), S["source"].to_list()))
    srows = [dict(r, source=src.get(r["sup_id"], "SIA")) for r in srows]
    ST = (pl.DataFrame(srows) if srows else
          pl.DataFrame(schema={"sup_id": pl.Utf8, "stand": pl.Utf8, "role": pl.Utf8, "how": pl.Utf8,
                               "audited": pl.Boolean, "in_train_2025": pl.Boolean, "in_ranking_2026": pl.Boolean,
                               "in_osm": pl.Boolean, "sentence": pl.Utf8, "source": pl.Utf8}))
    ST = ST.sort("sup_id", "stand")
    S.with_columns(pl.col("pub_date").cast(pl.Utf8), pl.col("start_utc").cast(pl.Utf8),
                   pl.col("end_utc").cast(pl.Utf8), pl.col("ev_start").cast(pl.Utf8),
                   pl.col("ev_end").cast(pl.Utf8)).write_csv(OUT / "sup_lfpg.tsv", separator="\t")
    ST.write_csv(OUT / "sup_lfpg_stands.tsv", separator="\t")
    S.write_parquet(OUT / "sup_lfpg.parquet")
    ST.write_parquet(OUT / "sup_lfpg_stands.parquet")

    print(f"\nLFPG supplements: {S.height}  (AD_WORKS {S.filter(pl.col('cls') == 'AD_WORKS').height}, "
          f"OTHER_LFPG {S.filter(pl.col('cls') == 'OTHER_LFPG').height})")
    for r in S.iter_rows(named=True):
        print(f"\n[{r['sup_id']}] {r['cls']}{' AIRAC' if r['airac'] else ''}  pub {r['pub_date']}  "
              f"{r['start_utc']} -> {r['end_utc']}"
              f"{'  replaces ' + r['replaces'] if r['replaces'] else ''}"
              f"{'  [2025 OUT-OF-FOLD]' if r['touches_2025_oof'] else ''}"
              f"{'  [SCORED 2026 MONTH]' if r['touches_scored_2026'] else ''}")
        print(f"   objet: {r['objet'][:220]}")
        print(f"   en vigueur: {r['en_vigueur'][:120]} | dates/heures: {r['dates_heures'][:120]}")
        if r["cls"] == "AD_WORKS":
            print(f"   runways: {r['runways'] or '-'}   taxiways: {r['taxiways'] or '-'}")
            sub = ST.filter(pl.col("sup_id") == r["sup_id"])
            for role in ("closed", "garage", "modified", "named", "range", "not_affected"):
                x = sub.filter(pl.col("role") == role)["stand"].to_list()
                if x:
                    print(f"   stands {role:<8}: {' '.join(x)}")

    if ST.height:
        print("\nSTAND NAME MATCH (stands the supplements name):")
        used = S.filter(pl.col("touches_2025_oof") | pl.col("touches_scored_2026"))["sup_id"].to_list()
        for scope, sub in (("all AD_WORKS", ST),
                           ("supplements touching 2025 out-of-fold or scored 2026 months",
                            ST.filter(pl.col("sup_id").is_in(used) & (pl.col("role") != "not_affected")))):
            n = sub.select("stand").unique().height
            print(f"   {scope}: {n} distinct stands; in 2025 STAND_mvt {sub.filter('in_train_2025').select('stand').unique().height}"
                  f"/{n}; in 2026 ranking {sub.filter('in_ranking_2026').select('stand').unique().height}/{n};"
                  f" in OSM refs {sub.filter('in_osm').select('stand').unique().height}/{n}")
        miss = ST.filter(~pl.col("in_train_2025") & ~pl.col("in_ranking_2026"))
        if miss.height:
            print("   named but in NEITHER competition file:", miss.select("sup_id", "stand").rows())


if __name__ == "__main__":
    main()
