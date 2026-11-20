"""4b speaker check (built 2026-11-19 after GT-II 116 r5, a heckler's words typed as "attendees").

For each GT row, find who the Sourced Statement itself says is speaking -- the subject of an
attribution verb ("X said", "said X", "according to X", "X accused") -- and flag the row when
none of those speakers matches the row's Name / Title / Source Descriptors (or, for a Document,
a document word). Pronoun-only speakers ("he said") are skipped: they can't be checked without
coreference. Read-only; used by step2.py and runnable on its own:

    python3 speakercheck.py [--corpus GT-II|GT-I] [--set orig|sz9|N,N,..]
"""
import re

VERBS = (r"said|says|say|told|tells|tell|wrote|writes|write|added|adds|noted|notes|explained|explains|"
         r"argued|argues|warned|warns|claimed|claims|announced|announces|proclaimed|proclaims|accused|"
         r"accuse|accuses|asked|asks|stated|states|reported|reports|testified|testifies|recalled|recalls|"
         r"insisted|insists|acknowledged|acknowledges|admitted|admits|confirmed|confirms|denied|denies|"
         r"declared|declares|replied|replies|responded|responds|estimated|estimates|predicted|predicts|"
         r"believe|believes|believed|found|finds|showed|shows|show|cited|cites|described|describes|"
         r"called|calls|urged|urges|criticized|criticizes|praised|praises|posted|posts|tweeted|"
         r"commented|comments|suggested|suggests|agreed|agrees|disagreed|objected|complained|complains|"
         r"chanted|chant|shouted|yelled|told|indicated|indicates|acknowledged|conceded|concedes|"
         r"emphasized|stressed|stresses|pointed out|points out|spoke|speaks|expressed|voiced")
PRON = {"he", "she", "they", "it", "we", "i", "you", "his", "her", "their", "its", "our", "my",
        "him", "them", "us", "me", "who", "which", "that", "this", "these", "those", "one", "both",
        "some", "many", "others", "other", "each", "all", "most", "several"}
STOP = PRON | {"the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with", "from",
               "as", "also", "has", "have", "had", "was", "were", "is", "are", "be", "been", "not",
               "but", "then", "later", "earlier", "previously", "recently", "once", "still", "had",
               "would", "will", "could", "can", "may", "might", "did", "does", "do", "just", "even",
               "who", "whom", "whose", "after", "before", "when", "while", "what", "where", "how",
               "statement", "interview", "email", "press", "release", "conference", "news", "told",
               "last", "week", "year", "month", "monday", "tuesday", "wednesday", "thursday", "friday",
               "saturday", "sunday", "today", "yesterday", "mr", "mrs", "ms", "dr", "said", "says",
               "very", "already", "repeatedly", "publicly", "privately", "first", "new", "own"}
DOCWORDS = {"report", "reports", "study", "studies", "data", "records", "record", "memo", "document",
            "documents", "letter", "analysis", "poll", "survey", "filing", "filings", "lawsuit",
            "complaint", "order", "ruling", "affidavit", "petition", "statement", "release", "research",
            "review", "audit", "figures", "numbers", "estimates", "transcript", "video", "post",
            "court", "documents", "papers", "notice", "policy", "bill", "law", "report's", "editorial"}
import unicodedata
_deacc = lambda w: "".join(c for c in unicodedata.normalize("NFKD", w) if not unicodedata.combining(c))
STEM = lambda w: re.sub(r"(’s|'s|ies|es|s)$", "", _deacc(w).lower())
WORD = re.compile(r"[^\W\d_][\w’'.-]*")


def _words(s):
    return [w.strip(".’'").lower() for w in WORD.findall(s or "")]


def speakers(ss):
    """Candidate speaker phrases named in the sentence(s)."""
    out = []
    s = ss.replace("“", '"').replace("”", '"')
    # X said / X, a ..., said  (take up to 8 words before the verb, back to a clause break)
    for m in re.finditer(r"\b(" + VERBS + r")\b", s):
        left = s[max(0, m.start() - 120):m.start()]
        left = re.split(r'[.;:!?"(]|\b(?:and|but)\b\s+(?=[a-z])', left)[-1]
        ws = WORD.findall(left)[-8:]
        if ws:
            out.append(("before", " ".join(ws)))
        # "..., said X" / "said X," inversion after a quote or comma
        before = s[max(0, m.start() - 3):m.start()]
        if re.search(r'[,"]\s*$', before) or re.search(r'"\s*$', s[:m.start()].rstrip()[-2:]):
            right = re.split(r"[,.;:!?\"]", s[m.end():m.end() + 120])[0]
            ws = WORD.findall(right)[:8]
            if ws:
                out.append(("after", " ".join(ws)))
    for m in re.finditer(r"\baccording to\s+([^,.;:!?\"]{2,120})", s, re.I):
        out.append(("according", " ".join(WORD.findall(m.group(1))[:8])))
    return out


def content(phrase):
    return [w for w in _words(phrase) if w not in STOP and len(w) > 2 and not w.isdigit()]


def identifiers(r):
    ids = set()
    for f in ("Name of Source", "Title of Source", "Source Descriptors"):
        for w in _words(r.get(f, "")):
            if w not in STOP and len(w) > 1:
                ids.add(STEM(w))
                ids.add(STEM(w.rstrip(".")))
    if r.get("Type of Source", "").strip().lower() == "document":
        ids |= {STEM(w) for w in DOCWORDS}
    return ids


PERSON_NOUNS = set("""people person persons official officials spokesperson spokesman spokeswoman spokespeople
representative representatives source sources resident residents expert experts analyst analysts
scientist scientists researcher researchers attendee attendees protester protesters protestor
protestors demonstrator demonstrators heckler hecklers critic critics supporter supporters opponent
opponents advocate advocates activist activists authority authorities police officer officers
investigator investigators prosecutor prosecutors lawyer lawyers attorney attorneys worker workers
employee employees staff staffer staffers student students parent parents teacher teachers member
members leader leaders organizer organizers organisers woman women man men commenter commenters
witness witnesses neighbor neighbors neighbour customer customers voter voters director directors
ceo chief president executive executives manager managers founder owner owners candidate
candidates lawmaker lawmakers legislator legislators senator senators mayor governor judge judges
doctor doctors nurse nurses economist economists historian professor professors official's
deputies deputy agent agents soldier soldiers troops survivor survivors victim victims family
families relative relatives friend friends colleague colleagues spokesperson's officials'
administrator administrators commissioner commissioners chair chairman chairwoman secretary
minister ministers diplomat diplomats observer observers insider insiders aide aides adviser
advisers advisor advisors consultant consultants farmer farmers miner miners driver drivers
passenger passengers patient patients tenant tenants landlord landlords homeowner homeowners
performer performers speaker speakers host hosts viewer viewers user users fan fans players player
coach coaches climbers climber guide guides crew captain commander narrator""".split())
ORG_NOUNS = set("""company companies agency department office organization organisation group firm
university school campaign administration government ministry council board court commission
committee institute center centre foundation association union party police coalition network
outlet newspaper station team club church hospital bank fund military army navy""".split())


ORG_AGENTS = {"spokesperson", "spokesman", "spokeswoman", "representative", "representatives", "official",
              "officials", "executives", "executive", "leaders", "leader", "lawyer", "lawyers", "attorney",
              "attorneys", "staff", "district", "poll", "pollsters", "data", "analysis", "survey"}


def _initials(name):
    ws = [w for w in WORD.findall(name) if w.lower() not in ("of", "the", "and", "for", "on", "in")]
    return "".join(w[0] for w in ws).lower() if len(ws) > 1 else ""


def key_words(phrase):
    """Names (capitalized, not sentence-function words) and person/group nouns in a speaker phrase."""
    ws = WORD.findall(phrase)
    if ws and ws[-1].lower() in PRON:          # "... she said" -> the speaker is a pronoun
        return []
    keys = []
    for i, w in enumerate(ws):
        lw = w.strip(".’'").lower()
        if lw in STOP:
            continue
        if len(lw) < 2:
            continue
        if lw in PERSON_NOUNS or lw in ORG_NOUNS or w[0].isupper():
            keys.append(lw)
    return keys


def check_row(r, others=()):
    """None, or {"kind":..., "speakers":[...], "other_rows":[...]}"""
    typ = r.get("Type of Source", "").strip().lower()
    sp = [(k, p, key_words(p)) for k, p in speakers(r["Sourced Statements"])]
    named = [(k, p, ks) for k, p, ks in sp if ks]
    if not named:
        return None
    ids = identifiers(r)
    if typ == "document" and any(STEM(w) in {STEM(d) for d in DOCWORDS} for w in _words(r["Sourced Statements"])):
        return None                            # the document is named in the sentence
    unnamed = typ in ("unnamed person", "unnamed group of people", "anonymous source")
    for k, p, ks in named:
        for w in ks:
            if STEM(w) in ids:
                return None
            if unnamed and w in PERSON_NOUNS and not ids - {""}:
                return None                    # nothing to compare against
            if typ == "named organization" and (w in ORG_NOUNS | ORG_AGENTS or STEM(w) in {STEM(d) for d in DOCWORDS}):
                return None
            if len(w) >= 2 and _initials(r.get("Name of Source", "")) and _initials(r.get("Name of Source", "")).endswith(w):
                return None
            if unnamed and w in {"people", "person", "sources", "source", "official", "officials"}:
                return None
    hits = []
    for o in others:
        oid = identifiers(o) - {STEM(w) for w in DOCWORDS}
        for k, p, ks in named:
            if any(STEM(w) in oid for w in ks):
                hits.append(o["No"])
    return {"kind": "speaker in the sentence not in this row's Name/Title/SD",
            "speakers": [p for _, p, _ in named], "other_rows": sorted(set(hits), key=lambda x: int(x))}


if __name__ == "__main__":
    import argparse, os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gtlib import story_files, read_csv, ORIG_GTII, SZ9
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="GT-II")
    ap.add_argument("--set", default="orig")
    a = ap.parse_args()
    nums = (ORIG_GTII if a.set == "orig" else SZ9 if a.set == "sz9" else
            list(range(1, 44)) if a.set == "all" else [int(x) for x in a.set.split(",")])
    n = 0
    for f in story_files(nums, a.corpus, include_held=True):
        rows, _ = read_csv(f)
        for r in rows:
            res = check_row(r, [o for o in rows if o is not r])
            if res:
                n += 1
                print(f"{os.path.basename(f)} r{r['No']} [{r['Type of Source']} | {r['Name of Source']} | "
                      f"{r['Title of Source']} | {r['Source Descriptors']}] speakers: {res['speakers']}"
                      f"{' ~ rows ' + ','.join(res['other_rows']) if res['other_rows'] else ''}")
    print(n, "flags")
