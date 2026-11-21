"""Apply approved Step 2 edits given as a list of ops (by content match, not row number).
op: {"file":..., "kind":"set", "match":<SS prefix>, "field":..., "old":..., "new":...}
    {"file":..., "kind":"add", "row":{...}}            -- placed in text order, rows renumbered
    {"file":..., "kind":"remove", "match":<SS prefix>}
Prints before/after and a row-number map."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gtlib import read_csv, write_csv, GT_DIRS, load_body, locate, story_num, COLS

def para_pos(ss, paras):
    loc = locate(ss, paras)
    if not loc:  # a row whose sentences were joined across a left-out sentence: use its first sentence
        from gtlib import sentence_spans
        sp = sentence_spans(ss.strip())
        loc = locate(ss.strip()[sp[0][0]:sp[0][1]], paras) if sp else None
    if loc: return (loc[0][0][0], loc[0][0][1])
    return (10**6, 0)

def run(ops):
    byf = {}
    for o in ops: byf.setdefault(o["file"], []).append(o)
    log = {}
    for fn, os_ in byf.items():
        path = os.path.join(GT_DIRS["GT-II"], fn)
        rows, fmt = read_csv(path); paras = load_body(story_num(path))
        for r in rows: r["_old"] = r["No"].strip()
        for o in os_:
            if o["kind"] in ("set", "remove"):
                hit = [r for r in rows if r["Sourced Statements"].strip().startswith(o["match"])]
                hit = [r for r in hit if "rows" not in o or r["_old"] in o["rows"]]
                assert hit, (fn, o)
                for r in hit:
                    if o["kind"] == "remove": rows.remove(r); continue
                    assert r[o["field"]].strip() == o["old"].strip(), (fn, r["_old"], o["field"], r[o["field"]])
                    r[o["field"]] = o["new"]
            elif o["kind"] == "add":
                nr = {c: "" for c in COLS}; nr.update(o["row"]); nr["_old"] = "new"
                assert locate(nr["Sourced Statements"], paras), ("not in text", fn, nr["Sourced Statements"][:60])
                p = para_pos(nr["Sourced Statements"], paras)
                i = 0
                while i < len(rows) and para_pos(rows[i]["Sourced Statements"], paras) <= p: i += 1
                rows.insert(i, nr)
        m = []
        for k, r in enumerate(rows, 1):
            r["No"] = str(k); m.append((r.pop("_old"), str(k)))
        write_csv(path, rows, fmt)
        log[fn] = m
        print(fn, "->", len(rows), "rows; map:", ", ".join(f"{a}->{b}" for a, b in m if a != b))
    return log

if __name__ == "__main__":
    print(json.dumps(run(json.load(open(sys.argv[1]))), indent=0)[:0])
