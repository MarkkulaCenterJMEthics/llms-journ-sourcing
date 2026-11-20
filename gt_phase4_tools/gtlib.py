"""Shared helpers for GT Phase 4 step1/step2 (rebuilt 2026-11-18 after reboot).

Text: extracted_articles_boilerplate/<n>-*.txt -- metadata header between two
'=====' lines, then the body (one paragraph per line).
GT: benchmarking/GT data/GT-II/<n>-*.csv, columns
No, Sourced Statements, Type of Source, Name of Source, Title of Source,
Source Descriptors, Source Justification.
"""
import csv, glob, io, os, re

# Repo root: $GT_REPO, else the parent folder when this copy lives inside the repo, else ~/Documents/GitHub/...
_here = os.path.dirname(os.path.abspath(__file__))
REPO = (os.environ.get("GT_REPO")
        or (os.path.dirname(_here) if os.path.isdir(os.path.join(os.path.dirname(_here), "extracted_articles_boilerplate")) else None)
        or os.path.expanduser("~/Documents/GitHub/llms-journ-sourcing"))
TEXT_DIR = os.path.join(REPO, "extracted_articles_boilerplate")
GT_DIRS = {"GT-II": os.path.join(REPO, "benchmarking/GT data/GT-II"),
           "GT-I": os.path.join(REPO, "benchmarking/GT data/GT-2026")}

# Original GT-II batch (take-stock 2026-11-15) and SZ-9.
ORIG_GTII = [51, 52, 55, 56, 61, 63, 64, 65, 66, 70, 74, 75, 76, 77, 78, 87, 89, 94,
             95, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113, 116,
             121, 122, 124, 125, 135, 152, 154, 155, 156, 159, 160]
SZ9 = [163, 164, 177, 178, 179, 180, 182, 183, 184]
HELD = set()   # 52/55 resolved 2026-11-19: dual-coded pairs (see ANNOTATOR_BY_FILE)

COLS = ["No", "Sourced Statements", "Type of Source", "Name of Source",
        "Title of Source", "Source Descriptors", "Source Justification"]
TYPES = {"Named Person", "Named Organization", "Document", "Anonymous Source",
         "Unnamed Person", "Unnamed Group of People"}

QMAP = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'",
                      " ": " ", "‑": "-", "ʼ": "'"})
OPEN_Q = "\"'“‘"
CLOSE_Q = "\"'”’"


def story_files(nums, corpus="GT-II", include_held=False):
    d = GT_DIRS[corpus]
    out = []
    for n in nums:
        if n in HELD and not include_held:
            continue
        out += sorted(glob.glob(os.path.join(d, f"{n}-*.csv")))
    return out


def story_num(path):
    return int(os.path.basename(path).split("-")[0])


def load_body(n):
    fs = glob.glob(os.path.join(TEXT_DIR, f"{n}-*.txt"))
    if len(fs) != 1:
        raise FileNotFoundError(f"text for {n}: {fs}")
    raw = open(fs[0], encoding="utf-8").read()
    parts = re.split(r"^={10,}\s*$", raw, flags=re.M)
    body = parts[2] if len(parts) >= 3 else raw
    return [p for p in body.split("\n") if p.strip()]


def read_csv(path):
    raw = open(path, "rb").read()
    bom = raw.startswith(b"\xef\xbb\xbf")
    txt = raw.decode("utf-8-sig")
    nl = "\r\n" if "\r\n" in txt else "\n"
    rows = list(csv.DictReader(io.StringIO(txt, newline="")))
    return rows, {"bom": bom, "nl": nl}


def write_csv(path, rows, fmt):
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=COLS, lineterminator=fmt["nl"])
    w.writeheader()
    for r in rows:
        w.writerow({c: r.get(c, "") for c in COLS})
    data = buf.getvalue().encode("utf-8")
    if fmt["bom"]:
        data = b"\xef\xbb\xbf" + data
    open(path, "wb").write(data)


# ---------- sentences ----------
ABBR = r"(?:Mr|Mrs|Ms|Dr|Gov|Sen|Rep|Reps|Sens|Lt|Gen|Col|Sgt|Capt|Cmdr|Adm|Prof|St|Jr|Sr|No|Inc|Co|Corp|Ltd|vs|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec|Ave|Blvd|Dept|Mt|Ft|U\.S|U\.N|U\.K|D\.C|a\.m|p\.m|e\.g|i\.e)"


def sentence_spans(p):
    """(start, end) spans of sentences in paragraph p."""
    spans, start = [], 0
    for m in re.finditer(r"[.!?]+[" + re.escape(CLOSE_Q) + r")\]]*(?=\s+[\"'“‘(\[]{0,2}[A-Z0-9])", p):
        before = p[:m.start() + 1]
        if re.search(r"\b" + ABBR + r"\.$", before) or re.search(r"\b[A-Z]\.$", before):
            continue
        end = m.end()
        spans.append((start, end))
        start = end
        while start < len(p) and p[start].isspace():
            start += 1
    if start < len(p):
        spans.append((start, len(p.rstrip())))
    return spans


def norm_map(s):
    """normalized string (quotes straightened, whitespace collapsed) + index map to s."""
    out, idx, prev_space = [], [], False
    for i, ch in enumerate(s.translate(QMAP)):
        if ch.isspace():
            if prev_space:
                continue
            ch, prev_space = " ", True
        else:
            prev_space = False
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def locate(ss, paras):
    """Find ss in the body. Returns list of (para_idx, start, end) pieces (exact
    original offsets) or None. Tries exact, then normalized, then with outer quote
    marks stripped, then across paragraph breaks."""
    s = ss.strip()
    for pi, p in enumerate(paras):
        k = p.find(s)
        if k >= 0:
            return [(pi, k, k + len(s))], "exact"
    variants = [("norm", s), ("strip-quotes", s.strip(OPEN_Q + CLOSE_Q + " "))]
    for how, v in variants:
        nv, _ = norm_map(v)
        nv = nv.strip()
        if not nv:
            continue
        for pi, p in enumerate(paras):
            np_, idx = norm_map(p)
            k = np_.find(nv)
            if k >= 0:
                return [(pi, idx[k], idx[k + len(nv) - 1] + 1)], how
    # spans paragraphs: split on line breaks / big gaps
    pieces = [x for x in re.split(r"\s*\n\s*", s) if x.strip()]
    if len(pieces) > 1:
        res = []
        for piece in pieces:
            r = locate(piece, paras)
            if not r:
                return None
            res += r[0]
        return res, "multi-para"
    # try sentence-by-sentence (paragraph-spanning without line break)
    sents = [s[a:b] for a, b in sentence_spans(s)]
    if len(sents) > 1:
        res = []
        for piece in sents:
            r = locate(piece, paras)
            if not r:
                return None
            res += r[0]
        if len({pi for pi, _, _ in res}) > 1:
            return res, "multi-para"
    return None


def expand_to_sentences(p, a, b):
    spans = sentence_spans(p)
    sa = next((x for x, y in spans if x <= a < y), a)
    sb = next((y for x, y in spans if x < b <= y), b)
    # include closing quote right after
    while sb < len(p) and p[sb] in CLOSE_Q:
        sb += 1
    # a lead-in label / bullet / dateline ending in ':' or a dash is not part of the sentence
    pre = p[sa:a]
    # ...but a speech lead-in ("Barber continued:") is part of the sentence (user 2026-11-19: SS = full sentence)
    speech = re.search(r"\b(said|says|say|told|tells|added|adds|continued|continues|wrote|writes|explained|noted|asked|stated|declared|replied|recalled|argued|warned)\b", pre, re.I)
    if sa < a and len(pre) <= 60 and not speech and re.search(r"(^[-•*]\s*$)|([:—–]\s*[\"“]?$)", pre) \
            and not re.search(r"[\"“]$", pre.rstrip()) or (sa < a and re.fullmatch(r"[-•*]\s*", pre)):
        sa = a
    # only punctuation after a closing quote (e.g. '”.') -> leave as is
    if sb > b and re.fullmatch(r"[.,;]+", p[b:sb]) and p[b - 1] in CLOSE_Q:
        sb = b
    return sa, sb


def best_fuzzy(ss, paras):
    try:
        from rapidfuzz import fuzz
    except ImportError:
        from fuzzywuzzy import fuzz
    best = max(range(len(paras)), key=lambda i: fuzz.partial_ratio(ss, paras[i]))
    return best, fuzz.partial_ratio(ss, paras[best])


def key_source(r):
    return (r["Type of Source"].strip().lower(), re.sub(r"\W+", " ", r["Name of Source"]).strip().lower())


# Annotator per GT-II file: Expansion List (Sep29) -- Aashka list = AV, Suha list = SZ;
# dual-coded pairs: AV_/SZ_ prefix decides, the unprefixed partner is the other annotator.
# Checked 2026-11-18 against every file already logged in gt-ii-student-housekeeping.md (all agree).
SHEET_AV = {56, 63, 64, 75, 77, 87, 89, 113, 116, 121}
SHEET_SZ = {51, 65, 66, 70, 78, 95, 101, 102, 103, 104, 105, 107, 110, 111, 124, 135, 152, 154,
            155, 156, 159, 160}
DUAL = {61, 74, 76, 94, 108, 109, 112, 122, 125, 52, 55}


# Dual pairs where neither file carries an AV_/SZ_ prefix: AV's deliveries are the "#NN-..." xlsx files
# in ~/Documents/GT-II-finishedfiles-XLSx (Sep11), SZ's the "NN - ..." ones; housekeeping agrees (55).
ANNOTATOR_BY_FILE = {"52-Swalwell_Allegations.csv": "AV", "52-East_Bay_Voters.csv": "SZ",
                     "55-Pentagon_Lab.csv": "AV", "55-Chicago_Pentagon.csv": "SZ",
                     "67-California_Ships_Whales.csv": "AV"}   # 67 (AV) and 70 (SZ) are the same article


def annotator(path):
    b = os.path.basename(path)
    if b in ANNOTATOR_BY_FILE: return ANNOTATOR_BY_FILE[b]
    n = story_num(path)
    if "-AV_" in b: return "AV"
    if "-SZ_" in b: return "SZ"
    if n in DUAL:
        sib = [os.path.basename(x) for x in glob.glob(os.path.join(os.path.dirname(path), f"{n}-*.csv")) if x != path]
        if any("-SZ_" in s for s in sib): return "AV"
        if any("-AV_" in s for s in sib): return "SZ"
        return "?"
    return "AV" if n in SHEET_AV else "SZ" if n in SHEET_SZ else "?"
