"""Calibrate titlecheck on commit 5647714 (SZ batch 191-211, Phase 2.6 Pass 1, 'case/wording as written').
recall: rows whose Title wording changed in the commit -> flagged at the pre state?
precision: flags remaining on the post state and on today's HEAD (all decided)."""
import sys, glob, os, re
import gtlib, titlecheck
S = sys.argv[1]
def run(gtdir, textdir, nums):
    gtlib.TEXT_DIR = textdir
    res = {}
    for f in sorted(glob.glob(gtdir + "/*.csv")):
        b = os.path.basename(f)
        if not re.match(r"\d+-", b) or gtlib.story_num(f) not in nums: continue
        rows, _ = gtlib.read_csv(f); paras = gtlib.load_body(gtlib.story_num(f))
        for r in rows:
            if r["Type of Source"].strip() != "Named Person" or not r["Name of Source"].strip(): continue
            c = titlecheck.check_source(r["Name of Source"], r["Title of Source"], paras, {x["Name of Source"] for x in rows})
            if c: res[(b, r["No"].strip())] = (r["Name of Source"], r["Title of Source"], c)
    return res
nums = set(range(191, 212))
pre = run(S + "/pre/benchmarking/GT data/GT-II", S + "/pre/extracted_articles_boilerplate", nums)
post = run(S + "/post/benchmarking/GT data/GT-II", S + "/pre/extracted_articles_boilerplate", nums)
# changed rows (wording, not case-only, not SD-only)
changed = []
for f in glob.glob(S + "/pre/benchmarking/GT data/GT-II/*.csv"):
    b = os.path.basename(f)
    if not re.match(r"\d+-", b) or gtlib.story_num(f) not in nums: continue
    A, _ = gtlib.read_csv(f); B, _ = gtlib.read_csv(f.replace("/pre/", "/post/"))
    for a, c in zip(A, B):
        if a["Title of Source"] != c["Title of Source"] and a["Title of Source"].lower() != c["Title of Source"].lower() \
                and a["Type of Source"].strip() == "Named Person":
            changed.append((b, a["No"].strip(), a["Title of Source"], c["Title of Source"]))
hit = [x for x in changed if (x[0], x[1]) in pre]
print(f"RECALL {len(hit)}/{len(changed)}")
for x in changed:
    print(("  HIT " if (x[0], x[1]) in pre else "  MISS"), x[0][:24], "r" + x[1], repr(x[2][:50]), "->", repr(x[3][:60]),
          "| text:", [p for _, p in pre.get((x[0], x[1]), (0, 0, {"text": []}))[2]["text"]][:3])
print(f"\nPRE flags {len(pre)}, POST flags {len(post)} (post = after Pass 1; remaining = noise or later decisions)")
for k, (n, t, c) in list(post.items()):
    print("  POST", k[0][:24], "r" + k[1], n[:18], "|", repr(t[:50]), "|", c["kind"], "|", [p for _, p in c["text"]][:3])
