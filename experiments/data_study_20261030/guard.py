"""Transitive source/text/exact-image/pHash components, including image-only rows."""
import hashlib
from collections import defaultdict


def normalized(value):
    return ' '.join(value.casefold().split())


def tokens(row):
    out=set()
    def add(prefix,value):
        if value is not None and value != '':out.add(prefix+str(value))
    add('id:',row.get('id'))
    for p in row.get('parent_ids',[]):add('parent:',p)
    for p in row.get('article_ids',[]):add('article:',p)
    for f in ('text_pl','text_en','original_en','source_caption','accepted_text'):
        if row.get(f):add('text:',normalized(row[f]))
    for f in ('sha256','pixels_sha256'):add('hash:',row.get(f))
    add('source:',row.get('source_image')); add('did:',row.get('image_did'))
    source=row.get('source_record') or {}
    if source:
        add('id:',source.get('id')); add('parent:commons-file:',source.get('source_file_id'))
        add('page:',source.get('source_page_url'))
        for q in {source.get('wikidata_qid')} | {a.get('wikidata_qid') for a in source.get('associations',[])}:
            if q:add('parent:wikidata:',q)
    identity=row.get('source_sha1_hex') or source.get('source_sha1')
    if identity:
        add('sha1:',f'{int(identity,16 if len(identity)==40 else 36):040x}')
    return out


class Components:
    def __init__(self):
        self.parent=[];self.size=[];self.owner={};self.labels=[];self.buckets=defaultdict(list);self.hashes={}

    def find(self,i):
        while self.parent[i]!=i:
            self.parent[i]=self.parent[self.parent[i]];i=self.parent[i]
        return i

    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.size[a]<self.size[b]:a,b=b,a
        self.parent[b]=a;self.size[a]+=self.size[b];self.labels[a].update(self.labels[b])

    @staticmethod
    def bands(h):
        shift=0
        for i,width in enumerate((10,9,9,9,9,9,9)):
            yield i,(h>>shift)&((1<<width)-1)
            shift+=width

    def add(self,row,labels=()):
        i=len(self.parent);self.parent.append(i);self.size.append(1);self.labels.append(set(labels))
        for token in tokens(row):
            # Compact fixed-size keys avoid retaining all candidate captions in RAM.
            key=hashlib.sha256(token.encode()).digest()
            if key in self.owner:self.join(i,self.owner[key])
            else:self.owner[key]=i
        if row.get('phash') is not None:
            self.add_phash(i,row['phash'])
        return i

    def add_phash(self,i,value):
        h=int(value,16)
        matches={k for band in self.bands(h) for k in self.buckets[band]}
        for k in matches:
            if (self.hashes[k]^h).bit_count()<=6:self.join(i,k)
        self.hashes[i]=h
        for band in self.bands(h):self.buckets[band].append(i)

    def reasons(self,i):return sorted(self.labels[self.find(i)])


def filter_verified(rows,protected):
    graph=Components()
    for r in protected:graph.add(r,['protected'])
    indices=[graph.add(r) for r in rows]
    kept=[];rejected=[]
    for i,r in zip(indices,rows):
        if not r.get('sha256') or not r.get('pixels_sha256') or not r.get('phash') or not r.get('lineage_verified'):
            rejected.append((r,'unverified_image'))
        elif graph.reasons(i):rejected.append((r,'protected_component'))
        else:kept.append(r)
    return kept,rejected
