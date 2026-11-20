"""Print Step 1 proposals as compact grouped markdown (Part A = mechanical, Part B = content)."""
import json, sys, os
from collections import OrderedDict, Counter
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "step1_proposals.json")))
part = sys.argv[1] if len(sys.argv) > 1 else "A"
P = d["props"]
if part == "A":
    fid = Counter(p["file"].replace(".csv", "") for p in P if p["cat"] == "FID")
    print("**FID (straight quotes/apostrophes -> the text's curly ones), %d rows:** " % sum(fid.values())
          + "; ".join(f"{f} {n}" for f, n in sorted(fid.items(), key=lambda x: int(x[0].split('-')[0]))))
    print("\n**NUM (renumber 1..n):** " + "; ".join(f"{p['file'].replace('.csv','')} ({p['old'][:24]}…)" for p in P if p["cat"] == "NUM"))
    print("\n| # | File | Rows | Cat | Old | New |\n|---|---|---|---|---|---|")
    labels = {"A0-FID": [p["id"] for p in P if p["cat"] == "FID"], "A0-NUM": [p["id"] for p in P if p["cat"] == "NUM"]}
    g = OrderedDict()
    for p in P:
        if p["cat"] in ("TCAP", "TART", "QUOTE", "NPER"):
            g.setdefault((p["file"].replace(".csv", ""), p["cat"], p["old"], p["new"]), []).append(p)
    for gi, ((f, c, o, n), ps) in enumerate(g.items(), 1):
        e = lambda s: s.replace("|", "\\|")
        ids = f"A{gi}"
        labels[ids] = [p["id"] for p in ps]
        print(f"| {ids} | {f} | {','.join(p['row'].replace('.0','') for p in ps)} | {c} | {e(o)} | {e(n)} |")
    json.dump(labels, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "step1_labels_A.json"), "w"), indent=0)
