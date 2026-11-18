# reflow2: paragraph starts from PDF gaps/font, matched to text LINES by their opening chars (for PDFs whose line wraps differ from the text)
# usage: python3 -I reflow2.py text.txt file.pdf [--write]
import pdfplumber,sys,re
txtp,pdfp=sys.argv[1],sys.argv[2]
N=lambda s:re.sub(r'\s','',s).replace('ʼ','’').replace("'",'’')
pl=[]
with pdfplumber.open(pdfp) as pdf:
    for pg in pdf.pages:
        for l in pg.extract_text_lines():
            pl.append((pg.page_number,l['top'],l['bottom'],round(l['chars'][0]['size'],1),l['text']))
starts=set()
for i,l in enumerate(pl):
    if i==0: starts.add(N(l[4])[:25]); continue
    p=pl[i-1]
    if l[0]!=p[0]: new = bool(re.search(r'[.?!”"]$',p[4].strip())) and len(p[4])<70 or p[3]!=l[3]
    else: new = (l[1]-p[2])>12 or p[3]!=l[3]
    if new: starts.add(N(l[4])[:25])
raw=open(txtp,encoding='utf-8').read()
sep='='*50; k=raw.rfind(sep)
head,body=(raw[:k+len(sep)],raw[k+len(sep):]) if k>=0 else ('',raw)
out=[]; 
for line in body.split('\n'):
    s=line.strip()
    if not s: continue
    if not out or N(s)[:25] in starts or len(N(s))<25 and N(s) in starts: out.append(s)
    else: out[-1]=out[-1]+(' ' if not out[-1].endswith('-') or True else '')+s
newbody='\n\n'+'\n'.join(out)+'\n'
assert N(newbody)==N(body), 'non-space changed'
lead=body[:len(body)-len(body.lstrip('\n'))]
new=head+'\n'+lead[1:] if False else head+newbody
if '--write' in sys.argv: open(txtp,'w',encoding='utf-8').write(new)
else: print(new)
