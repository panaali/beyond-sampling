"""Reproducible local-only exact-zero branch transform; source snapshots immutable."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent
for r in json.loads((P/'manifest.json').read_text()):assert hashlib.sha256((P/r['copy']).read_bytes()).hexdigest()==r['sha256']
s=(P/'_source/dd_geometry.cpp').read_text().replace('#include "../dd_clip.cpp"','#include "_source/dd_clip.cpp"')
(P/'original.cpp').write_text(s)
s=s.replace('counts[D+8]={}','counts[D+10]={}').replace('default:*size=D+8;','default:*size=D+10;')
anchor='   for(auto& p:pieces){Result r=clip(p.c,t+1,a,cid);'
assert s.count(anchor)==1
insert='''   bool exactzero=true;for(double v:a){++counts[D+9];if(v!=0.)exactzero=false;}
   if(exactzero){counts[D+8]+=pieces.size();for(auto& p:pieces){p.gates[k]=false;
     for(size_t i=0;i<p.c.z.size()/2;++i)p.c.z[2*i+cid/64]|=uint64_t(1)<<(cid%64);}
     continue;}
'''
s=s.replace(anchor,insert+anchor)
(P/'fixed.cpp').write_text(s)
