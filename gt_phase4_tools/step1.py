"""Phase 4 Step 1: settled mechanical fixes, as one batch table.

Detect:   python3 step1.py [--corpus GT-II] [--set orig|sz9|N,N,..] [--include-held]
          -> step1_proposals.json + step1_report.md (in this folder)
Apply:    python3 step1.py --apply all [--except ID,ID,...]   or  --apply ID,ID,...

Categories (settled rules, GT-I Phase 4 / dev doc item 76):
  WS     edge whitespace / inner double spaces or line breaks, any field
  FID    SS differs from the text only in quotes/apostrophes/spaces -> exact text
  QUOTE  outer quote mark the text doesn't have -> removed
  FRAG   SS is part of a sentence (incl. missing final period) -> full sentence(s)
  MERGE  same source, same paragraph, separate rows -> one row (continuous span)
  SPLIT  SS spans two paragraphs -> one row per paragraph
  TCAP   Title starts with a capital the text doesn't use -> lowercased
  TART   Title starts with an article (the/a/an) -> dropped
  NPER   Name of Source ends with a stray period -> dropped
  NUM    row numbers duplicated/out of order -> renumbered
  ---- flags only (no auto fix, go to Step 2) ----
  NOTFOUND  SS not in text;  TYPE  Type outside the six;  TDIFF  Title differs within a source
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlib import *

HERE = os.path.dirname(os.path.abspath(__file__))
PROP = os.path.join(HERE, "step1_proposals.json")
REPORT = os.path.join(HERE, "step1_report.md")
# elected / head-of-state / cabinet titles: kept as written, flagged (GT-I: "Gov." kept)
AS_WRITTEN = {"gov", "sen", "rep", "mayor", "president", "governor", "senator", "representative",
              "pope", "king", "queen", "prime", "minister", "secretary", "supervisor",
              "councilmember", "councilman", "councilwoman", "assemblymember", "congressman",
              "congresswoman", "sheriff", "rabbi", "rev", "reverend", "bishop", "imam", "father", "pastor"}
_DICT = None


def in_dict_lower(w):
    global _DICT
    if _DICT is None:
        try:
            _DICT = {x.strip() for x in open("/usr/share/dict/words") if x[:1].islower()}
        except OSError:
            _DICT = set()
    return w.lower() in _DICT


AS_WRITTEN_PHRASES = {"city attorney", "district attorney", "attorney general", "county executive",
                      "vice mayor", "council member", "city council", "board of supervisors",
                      "county supervisor", "state attorney", "county commissioner", "comptroller",
                      "controller", "treasurer", "lieutenant governor", "lt. gov", "speaker"}
ORG_HEAD = {"shop", "center", "centre", "institute", "foundation", "office", "agency", "company",
            "group", "association", "society", "university", "college", "school", "department",
            "coalition", "council", "committee", "club", "church", "bank", "network", "project",
            "alliance", "union", "lab", "laboratory", "inc", "llc", "corp", "museum", "hospital",
            "district", "commission", "board", "caucus", "party", "fund", "trust", "services", "salon",
            "restaurant", "cafe", "store", "market", "studio", "academy", "program", "initiative"}
CONNECT = {"of", "and", "the", "for", "on", "de", "del", "la", "du", "&", "der", "von", "van"}


def title_case_fix(t, name, paras):
    """Lowercase common words in a Title. A capitalized word is kept when it is:
    an acronym; a word not in the dictionary and never lowercase in the text; a word the
    reporter capitalizes mid-sentence (not right before the source's name) and never
    writes lowercase; part of a phrase (capitalized words, small connectors allowed) the
    reporter writes mid-sentence; or in the same unbroken capitalized run as a kept word.
    Elected/cabinet titles: whole run kept and flagged. Returns (new_title, flagged)."""
    body = " ".join(paras)
    name_toks = set(re.findall(r"[A-Z][\w’'-]+", name or ""))
    toks = [m for m in re.finditer(r"[\w][\w’'.-]*|&", t)]
    W = [m.group(0) for m in toks]
    bare = [re.sub(r"(’s|'s)$", "", w).rstrip(".") for w in W]
    cap = [w[0].isupper() or w[0].isdigit() for w in W]   # digits ride with names (360Value)

    def mid(phrase):
        for m in re.finditer(r"(?<![\w’'])" + re.escape(phrase) + r"(?![\w])", body):
            pre = body[:m.start()].rstrip()
            if not pre or pre[-1] in ".!?:“\"—–-":
                continue
            # (2026-11-18, user) capitals the reporter uses are on purpose -- stylebook or the
            # source's own preference -- including before a name ("Executive Director Jane Doe").
            return True
        return False

    def lower_in_body(w):
        return re.search(r"(?<![\w’'])" + re.escape(w.lower()) + r"(?![\w])", body) is not None

    keep = [False] * len(W)
    for i, w in enumerate(W):
        if not cap[i]:
            continue
        b = bare[i]
        if b[0].isdigit() or (b.isupper() and len(b) > 1) or re.search(r"[a-z][A-Z]|\d[A-Z]", b) or W[i].endswith("."):
            keep[i] = True
        elif not in_dict_lower(b.split("-")[0]) and not lower_in_body(b):
            keep[i] = True
        elif mid(W[i]):
            keep[i] = True       # reporter capitalizes this word mid-sentence
        elif re.search(r"[“\"][^”\"]*$", t[:toks[i].start()]):
            keep[i] = True       # (2026-11-24) inside quotation marks in the Title (a book title)
        elif name_toks and re.search(r"(?<![\w’'])" + re.escape(W[i]) + r"(?:\s+[A-Z][\w.’'-]*)*\s+(?:" +
                                     "|".join(map(re.escape, name_toks)) + r")\b", body):
            keep[i] = True       # (2026-11-24) capitalized title before the name, as written ("Judge Gregory Carro")
    # phrases: capitalized words with connectors between, tested on the source text
    def adjacent(i, j):   # tokens i and j separated only by whitespace in t
        return t[toks[i].end():toks[j].start()].strip() == ""
    n = len(W)
    for i in range(n):
        if not cap[i]:
            continue
        for j in range(i + 1, n):
            if not adjacent(j - 1, j) or not (cap[j] or W[j].lower() in CONNECT):
                break
            if cap[j] and mid(t[toks[i].start():toks[j].end()]):
                for k in range(i, j + 1):
                    if cap[k]:
                        keep[k] = True
    # unbroken capitalized runs share proper-noun status; as-written titles flag the run
    flagged, i = [], 0
    while i < n:
        if not cap[i]:
            i += 1; continue
        j = i
        while j + 1 < n and cap[j + 1] and adjacent(j, j + 1):
            j += 1
        run = range(i, j + 1)
        runtxt = " ".join(W[i:j + 1]).lower()
        # extend over connectors to see the whole capitalized phrase (for org-name test)
        pj = j
        while pj + 2 < n and W[pj + 1].lower() in CONNECT and cap[pj + 2] and adjacent(pj, pj + 1) and adjacent(pj + 1, pj + 2):
            pj += 2
            while pj + 1 < n and cap[pj + 1] and adjacent(pj, pj + 1):
                pj += 1
        phrase_words = [bare[k].lower() for k in range(i, pj + 1)]
        orgish = (phrase_words[-1] in ORG_HEAD or "&" in W[i:pj + 1]) and (pj > i)
        whole = t.strip().rstrip(".").lower()
        aw_hit = [bare[k].lower() for k in run if bare[k].lower() in AS_WRITTEN]
        if aw_hit == ["president"] and whole not in ("president", "former president", "vice president", "then-president"):
            aw_hit = []          # president of an organization: ordinary title
        if orgish and not any(keep[k] for k in run) and not lower_in_body(" ".join(W[i:pj + 1])):
            # looks like an organization/business name the text can't confirm either way
            flagged.append(" ".join(W[i:pj + 1]) + " [possible name; kept]")
            for k in range(i, pj + 1): keep[k] = keep[k] or cap[k]
            i = pj + 1
            continue
        if orgish:
            aw_hit = []          # "Sheriff's Office" etc.
        if aw_hit or any(p in runtxt for p in AS_WRITTEN_PHRASES):
            # extend over "of/for ..." capitalized continuation (Secretary of Education)
            while j + 2 < n and W[j + 1].lower() in CONNECT and cap[j + 2] and adjacent(j, j + 1) and adjacent(j + 1, j + 2):
                j += 2
                while j + 1 < n and cap[j + 1] and adjacent(j, j + 1):
                    j += 1
            run = range(i, j + 1)
            flagged.append(" ".join(W[i:j + 1]))
            for k in run: keep[k] = True
        elif any(keep[k] and not (bare[k].isupper() and len(bare[k]) > 1) for k in run):
            for k in run: keep[k] = True   # an acronym alone doesn't carry the run (YVA Education...)
        i = j + 1
    out = t
    for k in reversed(range(n)):
        if cap[k] and not keep[k]:
            nw = "-".join(p[:1].lower() + p[1:] for p in W[k].split("-"))
            out = out[:toks[k].start()] + nw + out[toks[k].end():]
    return out, flagged


def clean_ws(v):
    return re.sub(r"[ \t]*\n[ \t]*|[ \t]{2,}", " ", v).strip()


def detect(files):
    props, flags = [], []
    pid = 0

    def add(f, cat, row, field, old, new, note=""):
        nonlocal pid
        pid += 1
        props.append(dict(id=pid, file=os.path.basename(f), path=f, cat=cat, row=row,
                          field=field, old=old, new=new, note=note))

    for f in files:
        n = story_num(f)
        rows, _ = read_csv(f)
        try:
            paras = load_body(n)
        except FileNotFoundError as e:
            flags.append((os.path.basename(f), "-", "NOTEXT", str(e)))
            continue
        # numbering
        nos = [r["No"].strip() for r in rows]
        if nos != [str(i + 1) for i in range(len(rows))]:
            add(f, "NUM", "-", "No", ",".join(nos), "1-%d" % len(rows))
        locs = {}
        for r in rows:
            no = r["No"].strip()
            # whitespace, all text fields except SS (SS handled with fidelity below)
            for c in COLS[2:]:
                v = r.get(c) or ""
                if v != clean_ws(v):
                    add(f, "WS", no, c, v, clean_ws(v))
            if r["Type of Source"].strip() and r["Type of Source"].strip() not in TYPES:
                flags.append((os.path.basename(f), no, "TYPE", r["Type of Source"]))
            # Name trailing period
            nm = clean_ws(r["Name of Source"])
            if nm.endswith(".") and not re.search(r"\b(Inc|Jr|Sr|Co|Corp|Ltd|St|[A-Z])\.$", nm):
                add(f, "NPER", no, "Name of Source", nm, nm[:-1])
            # Title
            t = clean_ws(r["Title of Source"])
            m = re.match(r"(?i)(the|a|an)\s+", t)
            if m:
                add(f, "TART", no, "Title of Source", t, t[m.end():])
                t = t[m.end():]
            if t:
                nt, fl = title_case_fix(t, r["Name of Source"], paras)
                if nt != t:
                    add(f, "TCAP", no, "Title of Source", t, nt)
                for w in fl:
                    flags.append((os.path.basename(f), no, "TCAP?", f"'{w}' in Title '{t}' -- elected/cabinet title, as written?"))
            # SS location / fidelity / fragment
            ss = r["Sourced Statements"]
            loc = locate(ss, paras)
            if not loc:
                bi, sc = best_fuzzy(ss.strip(), paras)
                al = None
                if sc >= 94 and len(ss.strip()) <= len(paras[bi]) + 5:
                    from rapidfuzz import fuzz
                    al = fuzz.partial_ratio_alignment(ss.strip(), paras[bi])
                if al:
                    sa, sb = expand_to_sentences(paras[bi], al.dest_start, al.dest_end)
                    full = paras[bi][sa:sb]
                    add(f, "FIDX", no, "Sourced Statements", ss, full,
                        f"fuzzy {sc:.0f} -- text differs beyond quote style; check")
                    locs[no] = (bi, sa, sb, r)
                else:
                    flags.append((os.path.basename(f), no, "NOTFOUND", f"best para {bi + 1} score {sc:.0f}: {ss.strip()[:90]}"))
                continue
            pieces, how = loc
            if how == "multi-para" or len({p for p, _, _ in pieces}) > 1:
                parts = []
                for pi in sorted({p for p, _, _ in pieces}):
                    a = min(x for p, x, _ in pieces if p == pi)
                    b = max(y for p, _, y in pieces if p == pi)
                    sa, sb = expand_to_sentences(paras[pi], a, b)
                    parts.append(paras[pi][sa:sb])
                add(f, "SPLIT", no, "Sourced Statements", ss, " || ".join(parts),
                    "one row per paragraph (other fields copied)")
                continue
            pi, a, b = pieces[0]
            locs[no] = (pi, a, b, r)
            exact = paras[pi][a:b]
            sa, sb = expand_to_sentences(paras[pi], a, b)
            full = paras[pi][sa:sb]
            if full != exact:
                cat = "FRAG"
                add(f, cat, no, "Sourced Statements", ss, full)
                locs[no] = (pi, sa, sb, r)
            elif ss != exact:
                if ss.strip() == exact:
                    cat = "WS"
                elif ss.strip().strip(OPEN_Q + CLOSE_Q) != ss.strip() and how == "strip-quotes":
                    cat = "QUOTE"
                else:
                    cat = "FID"
                add(f, cat, no, "Sourced Statements", ss, exact)
        # merges: same source, same paragraph
        groups = {}
        for no, (pi, a, b, r) in locs.items():
            groups.setdefault((key_source(r), pi), []).append((a, b, no))
        for (src, pi), lst in groups.items():
            if len(lst) < 2:
                continue
            if not src[1]:
                flags.append((os.path.basename(f), "+".join(x[2] for x in sorted(lst)), "MERGE?",
                              f"blank Name, same type ({src[0]}), same paragraph {pi + 1} -- same source?"))
                continue
            lst.sort()
            a, b = lst[0][0], max(x[1] for x in lst)
            span = paras[pi][a:b]
            gap = []
            for (a1, b1, _), (a2, _, _) in zip(lst, lst[1:]):
                if a2 > b1 and paras[pi][b1:a2].strip():
                    gap.append(paras[pi][b1:a2].strip())
            add(f, "MERGE", "+".join(x[2] for x in lst), "Sourced Statements",
                " | ".join(paras[pi][x[0]:x[1]] for x in lst), span,
                (("in-between text: " + " / ".join(gap)) if gap else "") +
                ("" if src[1] else " [blank Name -- check same source]"))
        # Title differs within a source
        bysrc = {}
        for r in rows:
            if r["Name of Source"].strip():
                bysrc.setdefault(key_source(r), set()).add(clean_ws(r["Title of Source"]))
        for s, ts in bysrc.items():
            if len(ts) > 1:
                flags.append((os.path.basename(f), "-", "TDIFF", f"{s[1]}: " + " / ".join(sorted(ts))))
    return props, flags


def report(props, flags):
    L = ["# Step 1 proposals\n"]
    from collections import Counter
    c = Counter(p["cat"] for p in props)
    L.append("Counts: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())) + "\n")
    L.append("| ID | File | Row | Cat | Field | Old | New | Note |\n|---|---|---|---|---|---|---|---|")
    for p in props:
        e = lambda s: str(s).replace("|", "\\|").replace("\n", "\\n")
        L.append(f"| {p['id']} | {p['file']} | {p['row']} | {p['cat']} | {p['field']} | {e(p['old'])} | {e(p['new'])} | {e(p['note'])} |")
    L.append("\n# Flags (Step 2)\n\n| File | Row | Flag | Detail |\n|---|---|---|---|")
    for fl in flags:
        L.append("| " + " | ".join(str(x).replace("|", "\\|") for x in fl) + " |")
    open(REPORT, "w").write("\n".join(L) + "\n")


def apply(sel, exc):
    props = json.load(open(PROP))["props"]
    chosen = [p for p in props if (sel == "all" or p["id"] in sel) and p["id"] not in exc]
    byfile = {}
    for p in chosen:
        byfile.setdefault(p["path"], []).append(p)
    for path, ps in byfile.items():
        rows, fmt = read_csv(path)
        idx = {r["No"].strip(): r for r in rows}
        drop, inserts = set(), {}
        for p in ps:
            if p["cat"] in ("NUM",):
                continue
            if p["cat"] == "MERGE":
                nos = p["row"].split("+")
                keep = idx[nos[0]]
                keep["Sourced Statements"] = p["new"]
                for o in nos[1:]:
                    other = idx[o]
                    if (other["Source Justification"].strip() and
                            other["Source Justification"].strip() != keep["Source Justification"].strip()):
                        print(f"  NOTE {os.path.basename(path)} merge {p['row']}: differing SJ kept from r{nos[0]}; r{o} SJ was: {other['Source Justification']}")
                    drop.add(id(other))
                continue
            r = idx[p["row"]]
            if p["cat"] == "SPLIT":
                parts = p["new"].split(" || ")
                r["Sourced Statements"] = parts[0]
                inserts[id(r)] = [dict(r, **{"Sourced Statements": x}) for x in parts[1:]]
                continue
            cur = r[p["field"]]
            if cur.strip() != p["old"].strip() and cur != p["old"]:
                print(f"  SKIP {p['id']}: field changed since detection")
                continue
            r[p["field"]] = p["new"]
        out = []
        for r in rows:
            if id(r) in drop:
                continue
            out.append(r)
            out += inserts.get(id(r), [])
        for i, r in enumerate(out):
            r["No"] = str(i + 1)
        write_csv(path, out, fmt)
        print(f"applied {len(ps)} to {os.path.basename(path)} ({len(rows)} -> {len(out)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="GT-II")
    ap.add_argument("--set", default="orig")
    ap.add_argument("--include-held", action="store_true")
    ap.add_argument("--apply")
    ap.add_argument("--except", dest="exc", default="")
    a = ap.parse_args()
    if a.apply:
        sel = "all" if a.apply == "all" else {int(x) for x in a.apply.split(",")}
        exc = {int(x) for x in a.exc.split(",") if x}
        apply(sel, exc)
        sys.exit()
    nums = ORIG_GTII if a.set == "orig" else SZ9 if a.set == "sz9" else [int(x) for x in a.set.split(",")]
    files = story_files(nums, a.corpus, a.include_held)
    props, flags = detect(files)
    json.dump({"props": props, "flags": flags}, open(PROP, "w"), indent=1, ensure_ascii=False)
    report(props, flags)
    from collections import Counter
    print(len(files), "files;", len(props), "proposals", dict(Counter(p["cat"] for p in props)))
    print(len(flags), "flags", dict(Counter(f[2] for f in flags)))
