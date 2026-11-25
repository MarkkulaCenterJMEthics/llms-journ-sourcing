"""Phase 4 Step 2: judgment-phase candidate lists (4a recall, 4b precision, SJ).
Rebuilt 2026-11-18 after the reboot. Candidates only -- never edits a CSV.

    python3 step2.py [--corpus GT-II] [--set orig|sz9|N,N,..] [--include-held]
        -> step2_candidates.json + step2_report.md (stories sorted by row count)

4a  text sentences with an attribution signal not covered by any row (all types compared,
    methodology item 37); nearest existing row shown so paraphrase coverage is visible.
    Settled skips counted, not listed: non-responses, headline, captions/boilerplate.
4b  speaker check (speakercheck.py, 2026-11-19): the sentence names a speaker not in the row's Name/Title/SD;
    rows whose sentence has no attribution signal (implied-continuation context shown);
    reported feelings/beliefs; first-person without quotes; source name not near the row.
Title  (titlecheck.py) Title wording vs how the reporter describes the source -- the "as written"
    standard the original GT-II batch and SZ-9 missed (their Pass 1/2 ran in Sept, before it existed).
SJ  attribution/channel-only; not in text; = Title or name+title; piece of the row's own SS;
    differs across a source's rows; names another source. Engagement with a public body
    (meeting, hearing, forum, testimony, earnings call...) is valid SJ (v62 item 19).
"""
import argparse, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlib import *
from speakercheck import check_row as speaker_check
from rapidfuzz import fuzz
import titlecheck

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "step2_candidates.json")
OUT_MD = os.path.join(HERE, "step2_report.md")

VERBS = (r"said|says|say|told|tells|tell|according to|reported|reports|announced|announces|argued|argues|argue|"
         r"warned|warns|warn|wrote|writes|added|adds|noted|notes|claimed|claims|claim|stated|states|explained|"
         r"explains|recalled|recalls|asked|asks|estimated|estimates|found|finds|showed|shows|show|insisted|"
         r"insists|acknowledged|acknowledges|testified|alleged|alleges|denied|denies|confirmed|confirms|"
         r"disputed|disputes|contend|contends|contended|urged|urges|pointed to|points to|described|describes|"
         r"predicted|predicts|projected|projects|suggested|suggests|believes|maintained|maintains|"
         r"called for|calls for|called on|calls on|called it|calls it|criticized|criticizes|praised|praises|"
         r"emphasized|stressed|cautioned|countered|responded|replied|declared|announcing|saying|telling|"
         r"according|agreed|agrees|admitted|admits|concluded|concludes|indicated|indicates|revealed|reveals|"
         r"posted|tweeted|wrote in|in a statement|in an interview|in an email|in a post|in a letter")
DOCS = r"\b(report|study|survey|poll|data|filing|lawsuit|complaint|ruling|order|analysis|records?|documents?|statement|letter|memo|audit|census|estimates?)\b"
NONRESP = re.compile(r"(did not|didn[’']t|has not|hasn[’']t|have not|declined|could not be|couldn[’']t be|was not immediately|were not immediately)\s+"
                     r"(immediately\s+)?(respond|return|reply|comment|reached|available|be reached)", re.I)
FEEL = re.compile(r"\b(feels?|felt|hopes?|hoped|hoping|worr(y|ies|ied)|fears?|feared|wants?|wanted|wish(es|ed)?|"
                  r"believes?|believed|thinks?|thought|knows?|knew|loves?|loved|dreams?|dreamed|unsure|wonder\w*|"
                  r"afraid|angry|frustrated|proud|relieved|grateful|scared|anxious|excited)\b", re.I)
SIGNAL = re.compile(r"\b(" + VERBS + r")\b", re.I)
QUOTE = re.compile(r"[“”\"]")
CHANNEL = re.compile(r"^(he|she|they|it|\w+)?\s*(said|says|told|wrote|added|noted|explained)?\s*"
                     r"(by|in an?|via|over the|on the|through)\s+(e-?mail|phone|telephone|text|statement|interview|"
                     r"call|message|written statement|emailed statement|phone interview|video call|zoom)\b.*$", re.I)
ENGAGE = re.compile(r"\b(meeting|hearing|forum|conference|testimony|testif\w+|earnings call|town hall|rally|debate|"
                    r"council|commission|board|committee|session|court|panel|summit|briefing|vigil|protest|press)\b", re.I)


def sents(paras):
    out = []
    for pi, p in enumerate(paras):
        for a, b in sentence_spans(p):
            out.append((pi, a, b, p[a:b]))
    return out


def coverage(rows, paras):
    cov = {}
    for r in rows:
        loc = locate(r["Sourced Statements"], paras)
        if not loc:
            bi, sc = best_fuzzy(r["Sourced Statements"].strip(), paras)
            if sc >= 90:
                al = fuzz.partial_ratio_alignment(r["Sourced Statements"].strip(), paras[bi])
                pieces = [(bi, al.dest_start, al.dest_end)]
            else:
                pieces = []
        else:
            pieces = loc[0]
        cov[r["No"].strip()] = pieces
    return cov


def covered(pi, a, b, cov):
    for no, pieces in cov.items():
        for qi, x, y in pieces:
            if qi == pi and x < b and y > a and (min(b, y) - max(a, x)) >= 0.5 * (b - a):
                return no
    return None


def name_tokens(name):
    return [t for t in re.findall(r"[A-Z][\w’'-]{2,}", name or "")]


def analyze(path):
    n = story_num(path)
    rows, _ = read_csv(path)
    paras = load_body(n)
    cov = coverage(rows, paras)
    S = sents(paras)
    res = {"file": os.path.basename(path), "rows": len(rows), "4a": [], "4b": [], "sj": [], "title": [], "skips": {}}
    skip = lambda k: res["skips"].__setitem__(k, res["skips"].get(k, 0) + 1)
    headline = paras[0] if paras else ""
    names = {r["Name of Source"].strip() for r in rows if r["Name of Source"].strip()}

    # ---- 4a ----
    for pi, a, b, s in S:
        if pi == 0 and s.strip() in headline:
            continue
        if covered(pi, a, b, cov):
            continue
        sig = SIGNAL.search(s)
        q = QUOTE.search(s)
        doc = re.search(DOCS, s, re.I)
        if not (sig or q):
            continue
        if NONRESP.search(s):
            skip("non-response"); continue
        if re.match(r"^\s*(Photo|Caption|Credit|Image|Video|Read more|Related|Sign up|Subscribe|Advertisement)\b", s, re.I):
            skip("caption/boilerplate"); continue
        best = max(rows, key=lambda r: fuzz.partial_ratio(s, r["Sourced Statements"]), default=None)
        bsc = fuzz.partial_ratio(s, best["Sourced Statements"]) if best else 0
        who = [nm for nm in names if any(t in s for t in name_tokens(nm))]
        res["4a"].append(dict(para=pi + 1, text=s, signal=(sig.group(0) if sig else "quote marks"),
                              doc=bool(doc), near_row=best["No"].strip() if best else "",
                              near_score=round(bsc), near_src=(best["Name of Source"] or best["Type of Source"]) if best else "",
                              known_sources_named=who))

    # ---- 4b ----
    for r in rows:
        no = r["No"].strip()
        pieces = cov.get(no) or []
        ss = r["Sourced Statements"].strip()
        if not pieces:
            res["4b"].append(dict(row=no, kind="not located in text", ss=ss, src=r["Name of Source"])); continue
        pi = pieces[0][0]
        para = paras[pi]
        # the full sentences the row covers
        sent_txt = " ".join(s for qi, a, b, s in S if qi == pi and any(a < y and b > x for _, x, y in pieces if _ == pi))
        sig = SIGNAL.search(sent_txt) or QUOTE.search(sent_txt)
        toks = name_tokens(r["Name of Source"])
        near = " ".join(paras[max(0, pi - 2):pi + 1])
        kinds = []
        if not sig:
            # implied continuation context: same source attributed in this or previous paragraph
            same = [o["No"].strip() for o in rows if o is not r and key_source(o) == key_source(r)
                    and any(abs(q[0] - pi) <= 1 for q in (cov.get(o["No"].strip()) or []))]
            kinds.append("no attribution signal" + (f" (same source nearby: r{','.join(same)})" if same else ""))
        f = FEEL.search(sent_txt)
        if f and not QUOTE.search(sent_txt) and not re.search(r"\b(said|says|told|according)\b", sent_txt, re.I):
            kinds.append(f"reported feeling/belief ('{f.group(0)}')")
        if re.search(r"\b(I|I[’']m|my|we|our)\b", ss) and not QUOTE.search(sent_txt):
            kinds.append("first person without quote marks")
        if toks and not any(t in near for t in toks) and r["Type of Source"].strip() in ("Named Person", "Named Organization"):
            kinds.append("source name not in this or the 2 previous paragraphs")
        spk = speaker_check(r, [o for o in rows if o is not r])
        if spk:
            kinds.append("speaker check: sentence names " + " / ".join(f"'{x}'" for x in spk["speakers"][:2])
                         + (f" (matches r{','.join(spk['other_rows'])})" if spk["other_rows"] else ""))
        if kinds:
            res["4b"].append(dict(row=no, kind="; ".join(kinds), ss=ss, src=r["Name of Source"] or r["Type of Source"], para=pi + 1))

    # ---- SJ ----
    body = " ".join(paras)
    bysrc = {}
    for r in rows:
        bysrc.setdefault(key_source(r), []).append(r)
    for r in rows:
        no, sj = r["No"].strip(), r["Source Justification"].strip()
        if not sj:
            continue
        kinds = []
        if CHANNEL.match(sj) and not ENGAGE.search(sj):
            kinds.append("channel/attribution only")
        elif len(sj.split()) <= 4 and re.search(r"\b(said|told|says|wrote)\b", sj, re.I) and not ENGAGE.search(sj):
            kinds.append("attribution only")
        nsj, _ = norm_map(sj)
        if nsj.strip().rstrip(".") and nsj.strip().rstrip(".") not in norm_map(body)[0]:
            if fuzz.partial_ratio(sj, body) < 90:
                kinds.append("not in text")
        t = r["Title of Source"].strip().lower()
        if t and (sj.lower().rstrip(".") == t or sj.lower().rstrip(".") == (r["Name of Source"].strip() + ", " + t).lower()):
            skip("SJ repeats Title (item 58 sweep)")
        if len(sj) > 25 and sj.rstrip(".") in r["Sourced Statements"] and not kinds:
            skip("SJ is part of own SS (item 58 sweep)")
        others = [nm for nm in names if nm != r["Name of Source"].strip() and len(nm) > 3 and nm in sj
                  and not any(t in sj for t in name_tokens(r["Name of Source"]))]
        if others:
            kinds.append(f"names another source ({', '.join(others)})")
        # partial sentence: found in text but not starting/ending at sentence boundaries
        if "not in text" not in "; ".join(kinds):
            loc = locate(sj, paras)
            if loc and len(loc[0]) == 1:
                qi, x, y = loc[0][0]
                sa, sb = expand_to_sentences(paras[qi], x, y)
                before = paras[qi][:x].rstrip()[-1:] if x > sa else ""
                after = paras[qi][y:].lstrip()[:1]
                appositive = (x == sa or before in ",(\u2014\u2013:") and (y >= sb or after in ",.;)\u2014\u2013")
                if (sa, sb) != (x, y) and paras[qi][sa:sb].strip() != sj and not appositive:
                    kinds.append("partial sentence (text: " + paras[qi][sa:sb][:120] + ")")
        # about something else (Note 22b): a full sentence with no tie to this source
        refs = name_tokens(r["Name of Source"]) + [w for w in re.findall(r"[A-Za-z]{4,}", r["Title of Source"])]
        if (len(sj.split()) >= 6 and sj[:1].isupper() and sj.rstrip()[-1:] in ".!?\u201d\"" and SIGNAL.search(sj) is None
                and not any(t.lower() in sj.lower() for t in refs)
                and not re.search(r"\b(he|she|they|his|her|their|its|him|them|who|whose|i|my|we|our)\b", sj, re.I)):
            kinds.append("no tie to this source -- about something else? (Note 22b)")
        if kinds:
            res["sj"].append(dict(row=no, kind="; ".join(kinds), sj=sj, src=r["Name of Source"] or r["Type of Source"]))
    for k, rs in bysrc.items():
        vals = {x["Source Justification"].strip() for x in rs if x["Source Justification"].strip()}
        blanks = [x["No"].strip() for x in rs if not x["Source Justification"].strip()]
        if k[1] and (len(vals) > 1 or (vals and blanks)):
            res["sj"].append(dict(row=",".join(x["No"].strip() for x in rs), kind="differs across the source's rows"
                                  + (f" (blank on r{','.join(blanks)})" if blanks else ""),
                                  sj=" / ".join(sorted(vals)), src=rs[0]["Name of Source"]))
    # ---- Title as written (titlecheck.py; Phase 2.6 Pass 1 standard of 2026-11-13) ----
    allnames = {r["Name of Source"] for r in rows}
    bytitle = {}
    for r in rows:
        if r["Type of Source"].strip() != "Named Person" or not r["Name of Source"].strip():
            continue
        bytitle.setdefault((r["Name of Source"].strip(), r["Title of Source"].strip()), []).append(r["No"].strip())
    for (nm, t), nos in bytitle.items():
        c = titlecheck.check_source(nm, t, paras, allnames)
        if c:
            res["title"].append(dict(row=",".join(nos), src=nm, title=t, kind=c["kind"],
                                     text=[f"{k}: {p}" for k, p in c["text"]]))
    # (2026-11-24) Title of an unnamed source, and SD of any source: words the text never uses
    # (caught late in Step 3: 155 "legal team" for "His attorneys argued")
    body_l = " ".join(paras).lower().replace("’", "'").replace("‘", "'")
    STOPW = {"the", "and", "for", "with", "from", "who", "that", "this", "its", "their", "his", "her"}
    def missing(v):
        out = []
        for w in re.findall(r"[A-Za-z][A-Za-z’'-]+", v.replace("’", "'")):
            lw = w.lower().replace("’", "'").strip("'")
            if len(lw) < 3 or lw in STOPW:
                continue
            forms = {lw, lw.rstrip("s"), lw + "s", re.sub(r"ies$", "y", lw), re.sub(r"y$", "ies", lw)}
            stem = lw.rstrip("s")
            if not any(re.search(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", body_l) for f in forms if f) \
                    and not (len(stem) >= 4 and re.search(r"(?<![a-z])" + re.escape(stem) + r"(?:ed|ing|er|ers|s)?(?![a-z])", body_l)):
                out.append(w)
        return out
    seen = {}
    for r in rows:
        typ = r["Type of Source"].strip()
        for fld in (["Title of Source"] if typ != "Named Person" else []) + ["Source Descriptors"]:
            v = r[fld].strip()
            if not v:
                continue
            miss = missing(v)
            if miss:
                key = (r["Name of Source"].strip(), fld, v)
                seen.setdefault(key, []).append(r["No"].strip())
                seen[key + ("m",)] = miss
    for key, nos in list(seen.items()):
        if len(key) == 4:
            continue
        nm, fld, v = key
        res["title"].append(dict(row=",".join(nos), src=nm or "(unnamed)", title=v,
                                 kind=f"{'SD' if fld.startswith('Source') else 'Title'} word(s) not in the text: " + ", ".join(seen[key + ("m",)]),
                                 text=[]))
    return res


def report(results):
    L = ["# Step 2 candidates (stories sorted by row count)\n"]
    for R in results:
        L.append(f"\n## {R['file']} ({R['rows']} rows)  skips: {R['skips'] or '-'}\n")
        if R["4a"]:
            L.append("**4a**")
            for c in R["4a"]:
                L.append(f"- p{c['para']} [{c['signal']}{', doc' if c['doc'] else ''}] {c['text']}  "
                         f"_(nearest r{c['near_row']} {c['near_src']} {c['near_score']})_")
        if R["4b"]:
            L.append("**4b**")
            for c in R["4b"]:
                L.append(f"- r{c['row']} {c['src']}: {c['kind']} -- {c['ss'][:220]}")
        if R["title"]:
            L.append("**Title**")
            for c in R["title"]:
                L.append(f"- r{c['row']} {c['src']}: \"{c['title']}\" -- {c['kind']} -- text: " + " | ".join(c["text"]))
        if R["sj"]:
            L.append("**SJ**")
            for c in R["sj"]:
                L.append(f"- r{c['row']} {c['src']}: {c['kind']} -- {c['sj'][:200]}")
    open(OUT_MD, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="GT-II")
    ap.add_argument("--set", default="orig")
    ap.add_argument("--include-held", action="store_true")
    a = ap.parse_args()
    nums = ORIG_GTII if a.set == "orig" else SZ9 if a.set == "sz9" else [int(x) for x in a.set.split(",")]
    results = []
    for f in story_files(nums, a.corpus, a.include_held):
        try:
            results.append(analyze(f))
        except FileNotFoundError as e:
            print("NO TEXT", f, e)
    results.sort(key=lambda R: R["rows"])
    json.dump(results, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    report(results)
    tot = lambda k: sum(len(R[k]) for R in results)
    print(f"{len(results)} files; 4a {tot('4a')}, 4b {tot('4b')}, SJ {tot('sj')}, Title {tot('title')}")
