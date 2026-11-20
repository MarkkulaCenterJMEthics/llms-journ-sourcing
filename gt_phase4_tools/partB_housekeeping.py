"""Write the housekeeping text for Step 1 Part B (applied 2026-11-19)."""
import json, os, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlib import read_csv, annotator, DUAL, story_num
P = {p["id"]: p for p in json.load(open("step1_proposals_partB_applied_from.json"))["props"]}
ids = [int(x) for x in open("partB_ids.txt").read().split(",")]
byfile = {}
for i in ids:
    byfile.setdefault(P[i]["path"], []).append(P[i])
def fname(path):
    n = story_num(path); b = os.path.basename(path)
    return f"`{b}`" if n in DUAL else f"`{n}`"
out = {"SZ": {"FRAG": [], "SPLIT": [], "FIDX": [], "RENUM": []}, "AV": {"FRAG": [], "SPLIT": [], "FIDX": [], "RENUM": []}}
for path, ps in sorted(byfile.items(), key=lambda x: (story_num(x[0]), x[0])):
    a = annotator(path); f = fname(path)
    old, _ = read_csv(os.path.join("backup_GT-II_before_partB", os.path.basename(path)))
    splits = {p["row"] for p in ps if p["cat"] == "SPLIT"}
    m, k = {}, 0
    for r in old:
        k += 1; m[r["No"].strip()] = [k]
        if r["No"].strip() in splits:
            k += 1; m[r["No"].strip()].append(k)
    for c in ("FRAG", "SPLIT", "FIDX"):
        rows = sorted((p["row"] for p in ps if p["cat"] == c), key=int)
        if rows:
            if c == "SPLIT":
                out[a][c].append(f"{f} " + ", ".join(f"row {r} -> rows {m[r][0]}, {m[r][1]}" for r in rows))
            else:
                out[a][c].append(f"{f} row{'s' if len(rows)>1 else ''} " + ", ".join(rows))
    if splits:
        out[a]["RENUM"].append(f"{f} ({len(old)} -> {k} rows)")
for a in ("SZ", "AV"):
    o = out[a]
    t = ("- **Phase 4 Step 1 Part B (text fidelity), original GT-II batch, 2026-11-19.** "
         "Approved item by item with full before-rows. Sourced Statements only; row numbers are before the splits unless shown with an arrow. "
         "Fragment -> the full sentence(s) from the article, lead-in included "
         "(v61 Note 12a): " + "; ".join(o["FRAG"]) + ". "
         "A row crossing a paragraph break -> one row per paragraph, other fields copied "
         "(Note 12a; breaks checked against the live pages): " + "; ".join(o["SPLIT"]) + ". "
         "Wording made to match the article text (typos, dropped/changed words, dashes, ellipses, "
         "quote marks): " + "; ".join(o["FIDX"]) + ". "
         "Rows renumbered after the splits: " + "; ".join(o["RENUM"]) + ". "
         "Same-source merges deferred until after 4b (Step 3).")
    t = t.replace("Fragment -> the full sentence(s) from the article, lead-in included (v61 Note 12a): . ", "")
    print(f"=== {a}\n" + textwrap.fill(t, 76, subsequent_indent="  ", break_on_hyphens=False) + "\n")
