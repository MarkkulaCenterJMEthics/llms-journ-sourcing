"""Calibrate step2.py: run on GT-I pre-Step-2 (0903058) and check what Step 2 changed by 541e8e2."""
import sys, os, glob, gtlib, step2
from rapidfuzz import fuzz
pre, post = sys.argv[1], sys.argv[2]
hit = {"4a": [0, 0], "4b": [0, 0], "sj": [0, 0]}
miss = []
ncand = {"4a": 0, "4b": 0, "sj": 0}
for f in sorted(glob.glob(pre + "/*.csv"), key=gtlib.story_num):
    n = gtlib.story_num(f)
    if n < 13: continue
    g = os.path.join(post, os.path.basename(f))
    A, _ = gtlib.read_csv(f); B, _ = gtlib.read_csv(g)
    gtlib.GT_DIRS["GT-I"] = pre
    R = step2.analyze(f)
    for k in ncand: ncand[k] += len(R[k])
    sa = {r["Sourced Statements"].strip() for r in A}; sb = {r["Sourced Statements"].strip() for r in B}
    for s in sb - sa:   # added or changed rows
        if any(fuzz.ratio(s, x) > 85 for x in sa): continue   # an edit, not an addition
        hit["4a"][1] += 1
        if any(fuzz.partial_ratio(c["text"], s) >= 85 for c in R["4a"]): hit["4a"][0] += 1
        else: miss.append(("4a", n, s[:140]))
    for s in sa - sb:   # removed rows
        if any(fuzz.ratio(s, x) > 85 for x in sb): continue
        rr = [r["No"].strip() for r in A if r["Sourced Statements"].strip() == s]
        hit["4b"][1] += 1
        if any(c["row"] in rr for c in R["4b"]): hit["4b"][0] += 1
        else: miss.append(("4b", n, s[:140]))
    sjA = {(r["Sourced Statements"].strip(), r["Name of Source"].strip()): r["Source Justification"].strip() for r in A}
    for r in B:
        k = (r["Sourced Statements"].strip(), r["Name of Source"].strip())
        if k in sjA and sjA[k] != r["Source Justification"].strip():
            no = [x["No"].strip() for x in A if (x["Sourced Statements"].strip(), x["Name of Source"].strip()) == k]
            hit["sj"][1] += 1
            if any(set(c["row"].split(",")) & set(no) for c in R["sj"]): hit["sj"][0] += 1
            else: miss.append(("sj", n, f"{sjA[k][:60]!r} -> {r['Source Justification'][:60]!r}"))
print("recall (found/changed):", hit, " candidates listed:", ncand)
for m in miss: print("MISS", *m)
