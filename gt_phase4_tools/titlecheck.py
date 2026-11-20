"""Title-vs-text wording check ("case/wording as written", Phase 2.6 Pass 1 standard of 2026-11-13).
Built 2026-11-18; calibrated on commit 5647714 (SZ batch 191-211 Pass 1) with calib_title.py.
Used by step2.py. Capitalization-only differences are Step 1's job (TCAP) and ignored here.

For each named source, collect how the reporter describes them:
  pre-name    words right before a mention:   "Jackson police Det. Samuel Dukes"
  appositive  words right after a mention:    "Suzanne Ford, executive director of San Francisco Pride,"
  is-a        "Ford is the executive director of ..."
  state       "Senator Ed Markey of Massachusetts", "(D-Mass.)"
and compare with the GT Title.
"""
import re

TITLE_WORDS = re.compile(
    r"\b(director|president|professor|chief|executive|officer|ceo|cfo|coo|cto|economist|analyst|"
    r"senator|sen\.|rep\.|representative|gov\.|governor|mayor|spokes\w+|attorney|lawyer|judge|"
    r"manager|founder|co-?founder|chair\w*|secretary|minister|scientist|researcher|engineer|"
    r"doctor|dr\.|physician|nurse|teacher|coordinator|organizer|fellow|dean|superintendent|"
    r"commissioner|councilmember|council member|supervisor|sheriff|det\.|detective|editor|"
    r"reporter|journalist|author|co-author|consultant|advis[eo]r|leader|head|owner|partner|"
    r"member|principal|curator|investor|strategist|pastor|rabbi|lieutenant|captain|sgt\.|"
    r"agent|trustee|ambassador|envoy|general|admiral|commander|deputy|vice|associate|"
    r"assistant|lead|senior|instructor|lecturer|psychologist|psychiatrist|pollster|historian|"
    r"expert|specialist|planner|architect|developer|broker|administrator|treasurer|controller|"
    r"comptroller|clerk|marshal|prosecutor|public defender|counsel|activist|advocate|scholar|"
    r"democrat|republican|td|mp|mep|lawmaker|legislator|congressman|congresswoman|assemblymember)(?![\w])",
    re.I)

STOP_BEFORE = {"said", "says", "told", "tells", "according", "to", "by", "and", "with", "that", "when",
               "as", "but", "from", "of", "for", "asked", "wrote", "added", "in", "on", "at", "about",
               "like", "than", "while", "after", "before", "because", "if", "where", "which", "who",
               "whom", "including", "between", "among", "against", "via", "per", "or", "nor", "so",
               "then", "also", "even", "both", "recalled", "explained", "noted", "joined", "prompted",
               "called", "met", "praised", "welcomed", "pressed", "urged", "named", "hired", "fired",
               "replaced", "succeeded", "thanked", "criticized", "attacked", "spoke", "speaks", "is", "was",
               "are", "were", "be", "been", "has", "have", "had", "will", "would", "could", "can", "may",
               "might", "did", "does", "do", "say", "tell", "let", "pick", "similarly"}
ABBREV = re.compile(r"^(?:[A-Z][a-z]{0,4}\.|(?:[A-Z]\.)+)$")       # Gov.  Det.  Dr.  U.S.
STATES = ["New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
          "Rhode Island", "South Carolina", "South Dakota", "West Virginia"] + (
    "Alabama Alaska Arizona Arkansas California Colorado Connecticut Delaware Florida Georgia Hawaii "
    "Idaho Illinois Indiana Iowa Kansas Kentucky Louisiana Maine Maryland Massachusetts Michigan "
    "Minnesota Mississippi Missouri Montana Nebraska Nevada Ohio Oklahoma Oregon Pennsylvania "
    "Tennessee Texas Utah Vermont Virginia Washington Wisconsin Wyoming").split()
ELECTED = re.compile(r"(senator|sen\.|rep\.|representative|congress\w+|governor|gov\.)", re.I)


def norm(s):
    s = s.lower().replace("’", "'").replace("u.s.", "us")
    s = re.sub(r"[^\w\s']", " ", s)
    s = re.sub(r"\b(the|a|an)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mentions(name, body, others=()):
    name = name.strip()
    shared = {o.strip().split()[-1] for o in others if o.strip() and o.strip() != name and len(o.split()) >= 1}
    out = [(m.start(), m.end(), True) for m in re.finditer(re.escape(name), body)]
    parts = name.split()
    if len(parts) >= 2 and parts[-1] not in shared:
        for m in re.finditer(r"(?<![\w-])" + re.escape(parts[-1]) + r"(?![\w-])", body):
            if not any(a <= m.start() < b for a, b, _ in out):
                out.append((m.start(), m.end(), False))
    return out


def phrase_after(right):
    """Appositive after a name, up to the clause end (abbreviation-aware)."""
    m = re.match(r"(?:’s|'s)?,\s+(?:who\s+(?:is|was|serves as|served as|has been)\s+)?", right)
    if not m:
        return None
    toks, out = re.findall(r"\S+", right[m.end():m.end() + 220]), []
    for t in toks:
        if t.lower() in ("who", "which", "that", "said", "says", "told", "said,", "wrote") or t in ("—", "–", "-"):
            break
        core = t.rstrip(",;:)”\"")
        if t.endswith((",", ";", ":", ")", "”", '"')):
            out.append(core); break
        if t.endswith(".") and not ABBREV.match(t):
            out.append(t[:-1]); break
        out.append(t)
        if len(out) > 18:
            return None
    ph = " ".join(out).strip()
    return ph or None


def phrases_for(name, paras, others=()):
    body = "\n".join(paras)
    first = name.strip().split()[0] if name.strip() else ""
    found = []
    for a, b, full in mentions(name, body, others):
        if not full and first and body[max(0, a - len(first) - 1):a].strip() == first:
            a -= len(first) + 1
        # ---- pre-name ----
        left = body[max(0, a - 160):a].split("\n")[-1]
        pre = []
        for t in reversed(re.findall(r"\S+", left)):
            if re.search(r"[—–]", t):                       # dateline WASHINGTON—President
                tail = re.split(r"[—–]", t)[-1]
                if tail: pre.insert(0, tail)
                break
            if t in ("-",) or t.endswith((",", ";", ":", ")", "”", '"')):
                break
            if t.endswith((".", "!", "?")) and not ABBREV.match(t):
                break
            lt = t.lower().strip("“\"(")
            if lt in STOP_BEFORE or (t[:1].islower() and lt.endswith("ly")):
                break
            pre.insert(0, t.strip("“\"("))
            if len(pre) >= 9:
                break
        while pre and pre[0].lower() in ("the", "a", "an"):
            pre.pop(0)
        ph = " ".join(pre)
        if ph and TITLE_WORDS.search(ph):
            found.append(("pre-name", ph))
        # ---- after the name ----
        right = body[b:b + 240].split("\n")[0]
        st = re.match(r"\s+of\s+(" + "|".join(map(re.escape, STATES)) + r")\b", right) or \
            re.match(r"\s*,?\s*\(?([DRI]-[A-Z][A-Za-z.]{1,8}\.?)\)?", right)
        if st:
            found.append(("state", st.group(1)))
            right = right[st.end():]
        ap = phrase_after(right)
        if ap and not (ap[:1].islower() or re.match(r"(a|an|the|his|her|its|their)\s", ap, re.I)
                       or (TITLE_WORDS.match(ap) and len(ap.split()) <= 6 and not re.search(r"\s[a-z]+s?$", ap)
                           and not re.search(r"\b[A-Z][a-z]+ [a-z]+$", ap))):
            ap = None
        if ap and TITLE_WORDS.search(ap):
            found.append(("appositive", ap))
        m2 = re.match(r"\s+(?:is|was|serves as|served as|has been)\s+(?:the|a|an)\s+", right)
        if m2:
            ap2 = phrase_after(", " + right[m2.end():])
            if ap2 and TITLE_WORDS.search(ap2):
                found.append(("is-a", ap2))
    seen, out = set(), []
    for k, ph in found:
        key = (k == "state", norm(ph))
        if key not in seen:
            seen.add(key); out.append((k, ph))
    return out


PARTY = re.compile(r"^(democrat|republican|democratic|gop|labour|conservative|independent)$", re.I)


def check_source(name, title, paras, others=()):
    """None if the Title matches the text's wording; else {"kind": ..., "text": [(kind, phrase), ...]}."""
    phs = phrases_for(name, paras, others)
    titles = [(k, p) for k, p in phs if k != "state"]
    states = [p for k, p in phs if k == "state"]
    if not phs:
        return None
    nt = norm(title)
    if not nt:
        real = [(k, p) for k, p in titles
                if any(not PARTY.match(m.group(0)) for m in TITLE_WORDS.finditer(p))]
        return {"kind": "Title blank; text gives a title", "text": phs} if real else None
    body = norm(" ".join(paras))
    nps = [norm(p) for _, p in titles]
    out = []
    if ELECTED.search(title) and states:
        if not any(re.sub(r"[^a-z]", "", s.lower().replace("of ", "")) in re.sub(r"[^a-z]", "", title.lower())
                   for s in states):
            out.append("elected: state/party in text not in Title")
    parts = [x for x in title.split(";") if norm(x)]

    def found(p):
        p = norm(p)
        return any(p == q or re.search(r"(^|\s)" + re.escape(p) + r"(\s|$)", q) for q in nps) or \
            (len(p.split()) >= 2 and re.search(r"(^|\s)" + re.escape(p) + r"(\s|$)", body))

    def ok(part):
        if found(part):
            return True
        # "US Senator, Louisiana" / "Dr., preventive medicine physician": the extra comma pieces
        # may only be a state or an abbreviation; everything else must be found as a whole
        subs = [x.strip() for x in part.split(",") if norm(x)]
        if len(subs) < 2:
            return False
        extra_ok = lambda x: norm(x) in [norm(s) for s in STATES] or ABBREV.match(x)
        rest = [x for x in subs if not extra_ok(x)]
        if any(extra_ok(x) for x in subs) and (not rest or found(", ".join(rest))):
            return True
        # pieces each taken whole from a *different* phrase of the text (Raskin: pre-name + appositive)
        homes = []
        for x in rest:
            hs = [i for i, q in enumerate(nps) if re.search(r"(^|\s)" + re.escape(norm(x)) + r"(\s|$)", q)]
            if not hs:
                return False
            homes.append(set(hs))
        return len(set.union(*homes)) > 1 and all(len(h - set.union(*(o for o in homes if o is not h))) > 0
                                                 or len(h) > 1 for h in homes)

    if not all(ok(p) for p in parts):
        out.append("wording differs from text")
    else:
        tl = title.lower()
        for _, ph in titles:
            if norm(ph) == nt or not any(norm(x) in norm(ph) for x in parts):
                continue
            extra = [w for w in re.findall(r"[A-Z][\w'’.-]*", ph) if w.lower() not in tl
                     and w.lower().rstrip(".") not in ("the", "a", "an", "his", "her", "its", "their", "our",
                                                       "my", "this", "that", "these", "those")]
            if extra:
                out.append("Title shorter than the text's phrase"); break
    # the text's appositive adds credentials the Title leaves out (Raskin: "a Maryland Democrat and
    # constitutional scholar" while the Title is only "representative")
    tw = set(norm(title).split())
    def covered(ph):
        ws = [w for w in norm(ph).split() if w not in ("of", "and", "at", "for", "in", "on", "s")]
        return ws and sum(w in tw for w in ws) >= max(1, len(ws) // 2 + (len(ws) % 2))
    if not out and not any(covered(p) for k, p in titles if k in ("appositive", "is-a")):
        for k, ph in titles:
            if k not in ("appositive", "is-a"):
                continue
            words = [w for w in norm(ph).split() if w not in ("of", "and", "at", "for", "in", "on", "s", "'s")]
            missing = [w for w in words if w not in tw]
            if words and len(missing) >= max(2, len(words) // 2):
                out.append("text's appositive not in Title"); break
    return {"kind": "; ".join(out), "text": phs} if out else None
