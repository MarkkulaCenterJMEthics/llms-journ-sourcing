# reflow_wrapped_text_17.py -- variant used for GT-I story 17 (2026-11-18): PDF with doubled "faux-bold" glyphs and line wraps that differ from the text. Paragraph starts come from the PDF (gap > 12pt or font-size change; page break after a line ending a sentence); a text line starts a paragraph when its first 25 non-space characters match a PDF paragraph start; text lines not in the PDF stay their own line; never splits mid-sentence. Usage: python3 -I reflow_wrapped_text_17.py <text> <pdf> "<first body words>" [--write]
import pdfplumber,sys,re
txtp,pdfp=sys.argv[1],sys.argv[2]
N=lambda s:re.sub(r'\s','',s).replace('ʼ','’').replace("'",'’')
def dedup(s):
    w=s.replace(' ','')
    if len(w)>=4 and all(w[i]==w[i+1] for i in range(0,len(w)-1,2)):
        out=[];i=0
        while i<len(s):
            out.append(s[i]); i+= 2 if i+1<len(s) and s[i+1]==s[i] else 1
        return ''.join(out)
    return s
pl=[]
with pdfplumber.open(pdfp) as pdf:
    for pg in pdf.pages:
        for l in pg.extract_text_lines():
            pl.append((pg.page_number,l['top'],l['bottom'],round(l['chars'][0]['size'],1),dedup(l['text'])))
pl=[l for l in pl if not l[4].startswith(('11/18/26','https://'))]
starts=set(); full=N(''.join(l[4] for l in pl))
for i,l in enumerate(pl):
    if i==0: starts.add(N(l[4])[:25]); continue
    p=pl[i-1]
    if l[0]!=p[0]: new = bool(re.search(r'[.?!”"]$',p[4].strip())) or p[3]!=l[3]
    else: new = (l[1]-p[2])>12 or p[3]!=l[3]
    if new: starts.add(N(l[4])[:25])
lines=open(txtp,encoding='utf-8').read().split('\n')
HDR=re.compile(r'^﻿?(Headline|Subtitle|Date|Publisher|Summary|Caption|URL|Author|Title|Source PDF):')
head=[]; k=0
while k<len(lines) and HDR.match(lines[k]): head.append(lines[k]); k+=1
rest=lines[k:]
fb=[i for i,l in enumerate(rest) if l.strip().startswith(sys.argv[3])][0] if len(sys.argv)>3 and not sys.argv[3].startswith('--') else 0
pre=[]; blk=[]
for l in rest[:fb]:
    if l.strip().startswith(('By ','UPDATED')) and blk: pre.append(' '.join(blk)); blk=[]
    if l.strip(): blk.append(l.strip())
    elif blk: pre.append(' '.join(blk)); blk=[]
if blk: pre.append(' '.join(blk))
body=[l.strip() for l in rest[fb:] if l.strip()]
out=[]
for s in body:
    ns=N(s)
    if not out or ns[:25] in starts or (len(ns)<25 and ns in starts) or ns[:20] not in full:
        out.append(s)
    else:
        prevunmatched = N(out[-1])[:20] not in full
        if prevunmatched: out.append(s)
        else: out[-1]=out[-1]+' '+s
# guard: never split mid-sentence
g=[]
for s in out:
    if g and (re.search(r'[,;:—–-]["”’]?$',g[-1]) or re.match(r'^[a-z]',s) or not re.search(r'[.?!…]["”’)]*$',g[-1])) and N(g[-1])[:20] in full and N(s)[:20] in full:
        g[-1]=g[-1]+' '+s
    else: g.append(s)
new='\n'.join(head)+'\n\n'+'\n'.join(pre+g)+'\n'
assert N('\n'.join(lines[k:]))==N('\n'.join(pre+g)),'nonspace changed'
if '--write' in sys.argv: open(txtp,'w',encoding='utf-8').write(new)
else: print(new)
