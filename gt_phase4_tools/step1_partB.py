"""Print Step 1 Part B (FRAG, MERGE, SPLIT, FIDX) with full before-rows, labels B1..Bn.
Writes step1_labels_B.json and step1_partB_<CAT>.md. Usage: python3 step1_partB.py [CAT ...]"""
import json, os, sys, difflib, re
from gtlib import read_csv, COLS
D = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(os.path.join(D, "step1_proposals.json")))["props"]
ORDER = ["FRAG", "MERGE", "SPLIT", "FIDX"]
cats = sys.argv[1:] or ORDER
KEYS = json.load(open(os.path.join(D, "step1_labelkeys_B.json"))) if os.path.exists(os.path.join(D, "step1_labelkeys_B.json")) else {}
labels, n = {}, max([int(v[1:]) for v in KEYS.values()] or [0])
e = lambda s: str(s).replace("|", "\\|").replace("\n", " ")

def wdiff(a, b):
    """word diff: [-old-] {+new+}"""
    A, B = re.findall(r"\S+", a), re.findall(r"\S+", b)
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        if op == "equal":
            seg = A[i1:i2]
            out.append(" ".join(seg) if len(seg) <= 8 else " ".join(seg[:3]) + " … " + " ".join(seg[-3:]))
        else:
            if i2 > i1: out.append("[-" + " ".join(A[i1:i2]) + "-]")
            if j2 > j1: out.append("{+" + " ".join(B[j1:j2]) + "+}")
    return " ".join(out)

for cat in ORDER:
    ps = sorted([p for p in P if p["cat"] == cat], key=lambda p: (int(p["file"].split("-")[0]), p["file"], int(p["row"].split("+")[0])))
    ps.sort(key=lambda p: int(KEYS.get("|".join((p["file"], p["row"], p["cat"])), "B999")[1:]))
    L = [f"## {cat} ({len(ps)})\n"]
    for p in ps:
        k = "|".join((p["file"], p["row"], p["cat"]))
        if k not in KEYS:
            n += 1; KEYS[k] = f"B{n}"
        lab = KEYS[k]
        labels[lab] = p["id"]
        rows, _ = read_csv(p["path"])
        idx = {r["No"].strip(): r for r in rows}
        L.append(f"### {lab} · {p['file'].replace('.csv','')} · row {p['row']} · {cat} · Field: Sourced Statements")
        if p["note"]: L.append(f"*{p['note']}*\n")
        L.append("Before (full row" + ("s" if "+" in p["row"] else "") + "):\n")
        L.append("| " + " | ".join(COLS) + " |\n|" + "---|" * len(COLS))
        for no in p["row"].split("+"):
            r = idx.get(no)
            L.append("| " + " | ".join(e(r[c]) for c in COLS) + " |" if r else f"| {no} | (row missing) |")
        if cat == "SPLIT":
            L.append("\nAfter — Sourced Statements, one row per paragraph (other fields copied):\n")
            for i, part in enumerate(p["new"].split(" || "), 1):
                L.append(f"{i}. {part}")
        elif cat == "MERGE":
            L.append(f"\nAfter — one row (row {p['row'].split('+')[0]} kept), Sourced Statements:\n\n> {p['new']}")
        else:
            L.append(f"\nAfter — Sourced Statements:\n\n> {p['new']}\n\nChange: {wdiff(p['old'], p['new'])}")
        L.append("")
    if cat in cats:
        open(os.path.join(D, f"step1_partB_{cat}.md"), "w").write("\n".join(L) + "\n")
json.dump(labels, open(os.path.join(D, "step1_labels_B.json"), "w"), indent=0)
json.dump(KEYS, open(os.path.join(D, "step1_labelkeys_B.json"), "w"), indent=0)
print(n, "labels")
