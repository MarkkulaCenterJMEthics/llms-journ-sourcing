#!/usr/bin/env python3
"""
Re-flow a hard-wrapped story text in extracted_articles_boilerplate/ to
one paragraph per line, using the story's source PDF to tell paragraph
breaks from line wraps. Only whitespace changes: every non-space
character of the body is kept, in order (checked before writing).

Why: v10-extract-multiple-LLMs.py sends each text body to the models as
is, so its line breaks are the paragraph structure a model sees. Texts
converted from PDFs were hard-wrapped at the PDF's line width, so about 1
in 5 paragraph breaks was invisible and PDF page breaks showed up as
false blank lines mid-sentence. GT Sourced Statements are split one row
per paragraph, so the text needs to mark paragraphs the same way.
See extracted_articles_boilerplate/REFLOW_2026-11.md for the run report.

Method:
1. Match each text line to its PDF line, in order. PDF lines with no
   counterpart in the text (browser headers/footers, site navigation,
   page numbers, removed ads) are skipped. Doubled "faux-bold" glyphs are
   removed first with dedupe_chars(), as in storypdf_to_text.py.
2. Take the thresholds from the matched lines only: the body font size is
   the most common size, and the within-paragraph gap is the most common
   gap between consecutive body-size lines on a page.
3. Start a new paragraph when
   - the gap to the previous line is more than 1.5 x that gap + 1.5pt,
   - the font size changes (title, byline, captions, labels), or
   - at a page break, the last line on the previous page was short.
4. Never split a sentence: a line still continues the paragraph if it
   starts with a lowercase letter, or the line before ends in a comma or
   hyphen, or ran nearly full width without closing punctuation. This
   guard doesn't override a font-size change (so captions stay separate)
   unless the break is plainly mid-sentence (lowercase word before and
   lowercase start after, or the text so far ends on "the", "of", etc.),
   and it doesn't apply to a short label such as "Image".
5. A text line with no PDF match, and the line after it, is placed by
   punctuation: a new paragraph only if the text so far ends a sentence
   and the next line starts one.
Lines inside a paragraph are joined with a space, or with nothing after
a hyphen at the end of a line. Blank lines (PDF page breaks) are dropped.

Usage:
    python reflow_wrapped_text.py STORY.txt SOURCE.pdf [--write]
Without --write it prints the re-flowed body; with --write it replaces
STORY.txt after the whitespace-only check passes. Run it on the original
wrapped text, not on a file that has already been re-flowed.
"""
import re,sys,statistics,collections,pdfplumber
RULE='='*50
norm=lambda s:re.sub(r'[^a-z0-9]','',s.lower())
def pdf_lines(path):
    out=[]
    with pdfplumber.open(path) as pdf:
        for pi,page in enumerate(pdf.pages):
            page=page.dedupe_chars()   # faux-bold doubled glyphs (e.g. 197), as in storypdf_to_text.py
            for l in page.extract_text_lines():
                sz=round(statistics.median(c['size'] for c in l['chars']),1) if l['chars'] else 0
                out.append(dict(p=pi,top=l['top'],bot=l['bottom'],x1=l['x1'],sz=sz,k=norm(l['text'])))
    return out
def same(pk,k):
    return bool(pk and k and (pk[:30]==k[:30] or (len(pk)>=8 and k.startswith(pk[:30])) or (len(k)>=8 and pk.startswith(k[:30]))))
def reflow(text,path,trace=False):
    if text.startswith(RULE) and text.count(RULE)>=2:
        _,hdr,body=text.split(RULE,2); head=RULE+hdr+RULE
    else:
        head='';body=text
    P=pdf_lines(path)
    lines=[l.strip() for l in body.split('\n') if l.strip()]
    match=[];j=0
    for t in lines:
        k=norm(t);m=None
        for jj in range(j,len(P)):
            if same(P[jj]['k'],k):
                m=P[jj];j=jj+1;break
        match.append(m)
    M=[m for m in match if m]
    bodysz=collections.Counter(m['sz'] for m in M).most_common(1)[0][0]
    g=[round((b['top']-a['bot'])*2)/2 for a,b in zip(M,M[1:]) if a['p']==b['p'] and a['sz']==b['sz']==bodysz and 0<b['top']-a['bot']<40]
    intra=collections.Counter(g).most_common(1)[0][0] if g else 4.5
    thr=intra*1.5+1.5
    xs=[m['x1'] for m in M if m['sz']==bodysz]
    right=statistics.quantiles(xs,n=10)[8] if len(xs)>10 else max(xs)
    fulllen=statistics.quantiles([len(l) for l in lines],n=10)[7] if len(lines)>10 else max(len(l) for l in lines)
    lastlen=0;lastmatched=True
    prev=None;paras=[];cur='';log=[]
    for t,m in zip(lines,match):
        if not cur:
            newp=True;why='first'
        elif m is None or prev is None or not lastmatched:
            newp=bool(re.search(r'[.!?”"’)]$',cur)) and bool(re.match(r'[A-Z“"‘(]',t));why='nomatch'
        elif m['sz']!=prev['sz']:
            newp=True;why='size'
        elif m['p']==prev['p']:
            newp=(m['top']-prev['bot'])>thr;why='gap'
        else:
            newp=prev['x1']<right*0.85;why='page'
        # guard: a line that starts lowercase, or follows a line ending in a comma or hyphen, continues the paragraph
        if newp and cur and why=='size' and ((re.search(r'[a-z,]$',cur) and re.match(r'[a-z]',t)) or re.search(r'\b(the|a|an|of|to|in|on|and|or|for|with|by|at|from|its|his|her|their|that|as)$',cur)):
            newp=False;why+='+midsentence'
        if newp and cur and len(cur)>=25 and why in ('gap','page','nomatch') and (re.match(r'[a-z]',t) or re.search(r'[,\-–]$',cur) or (lastlen>=0.9*fulllen and not re.search(r'[.!?:;"”’)\]]$',cur))):
            newp=False;why+='+guard'
        if newp:
            if cur: paras.append(cur)
            cur=t
        else:
            cur=cur+t if (cur.endswith('-') and not cur.endswith(' -')) else cur+' '+t
        if trace: log.append((why,newp,t[:50]))
        if m is not None: prev=m
        lastmatched=m is not None
        lastlen=len(t)
    if cur: paras.append(cur)
    out=(head+'\n\n' if head else '')+'\n'.join(paras)+'\n'
    info=dict(lines=len(lines),paras=len(paras),thr=thr,intra=intra,bodysz=bodysz,unmatched=sum(1 for m in match if m is None))
    return out,info,log

if __name__=='__main__':
    if len(sys.argv)<3:
        sys.exit(__doc__)
    path,pdf=sys.argv[1],sys.argv[2]
    t=open(path,encoding='utf-8').read()
    out,info,_=reflow(t,pdf)
    if re.sub(r'\s+','',t)!=re.sub(r'\s+','',out):
        sys.exit('refusing: re-flow changed more than whitespace')
    if '--write' in sys.argv[3:]:
        open(path,'w',encoding='utf-8').write(out)
        print(path,info)
    else:
        print(out)
