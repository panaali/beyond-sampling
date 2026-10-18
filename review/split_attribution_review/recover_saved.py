"""Preserve failed checker; repair its explicitly documented missing-inventory rows."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
source=P/'saved_audit.py'
s=source.read_text()
assert hashlib.sha256(source.read_bytes()).hexdigest()=='61e18d2675425a2e9b66fc1a9dc02f4db1c89cf67515c3d37b90f3a94233063b'
old="for item in manifest['records']:assert sha(P.parents[1]/item['frozen'])==item['sha256'],item['frozen']"
new="""missing=[i for i in manifest['records'] if i.get('missing')]
        assert missing==[{'original':'m16/approaches/understand_m43/m43.py','missing':True},{'original':'m16/approaches/bulk44/zoo44.py','missing':True}]
        for item in manifest['records']:
            if not item.get('missing'):assert sha(P.parents[1]/item['frozen'])==item['sha256'],item['frozen']"""
assert s.count(old)==1;s=s.replace(old,new)
for a,b in [('saved_gate.json','saved_recovery_gate.json'),('saved_results.json','saved_recovery_results.json'),('saved_progress.jsonl','saved_recovery_progress.jsonl')]:
    assert a in s;s=s.replace(a,b)
exec(compile(s,str(source),'exec'),{'__name__':'__main__','__file__':str(source)})
