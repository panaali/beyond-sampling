"""Independent saved aggregate arithmetic for unchanged real-basis operator."""
import json,math
from pathlib import Path
path=Path('/workspace/work/recover-small_h4_4/m16/approaches/anisotropic_harmonic/operator_cycle/ensemble.json')
data=json.loads(path.read_text());out={}
for family in ('H2','H4'):
    rows=[r for r in data['rows'] if r['family']==family]
    results={}
    for method in ('dense','real_operator'):
        n=len(rows)
        results[method]=dict(rmse=math.sqrt(sum((r[method]['estimate']-r['reference'])**2 for r in rows)/n),
            mean_cpu=sum(r[method]['stats']['cpu_seconds'] for r in rows)/n,
            mean_accounted_arithmetic=sum(r[method]['stats']['accounted_scalar_arithmetic_upper'] for r in rows)/n)
        for key,value in results[method].items():
            assert abs(value/data['aggregate'][family][method][key]-1)<1e-13
    out[family]=dict(**results,maximum_parity_error=max(abs(r['dense']['estimate']-r['real_operator']['estimate']) for r in rows))
Path(__file__).with_name('real_operator_summary_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
