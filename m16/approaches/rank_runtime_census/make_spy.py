"""Generate explicit pass-through ABI wrappers; no numerical expressions changed."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'rank_cost_review/counted_spy.c'
base=BASE.read_text();sha=hashlib.sha256(BASE.read_bytes()).hexdigest()
# Ledger is allocated statically and bounded. No per-call malloc or model values.
ledger='''
#define ROWS 1024
static I records[ROWS][6];static I used,overflow;
static void record(I id,I m,I n,I query,I failed){
 for(I i=0;i<used;i++)if(records[i][0]==id&&records[i][1]==m&&records[i][2]==n&&records[i][3]==query){records[i][4]++;records[i][5]+=failed;return;}
 if(used==ROWS){overflow++;return;}I *r=records[used++];r[0]=id;r[1]=m;r[2]=n;r[3]=query;r[4]=1;r[5]=failed;
}
void census_reset(void){memset(records,0,sizeof(records));used=0;overflow=0;audit_reset();nested_reset();}
I census_size(void){return used;}
I census_overflow(void){return overflow;}
void census_get(I*out){memcpy(out,records,used*6*sizeof(I));}
'''
# Forward declarations needed because baseline dgesdd is above ledger definition.
base=base.replace('static I stats[12];','static void record(I,I,I,I,I);\nstatic I stats[12];')
base=base.replace('stats[8]+=(*info==0);','record(0,*m,*n,*lwork==-1,*info!=0);\n stats[8]+=(*info==0);')
base=base.replace('I before=*iter;', 'I before=*iter;I active_before=*n0-*i0+1;I parity_before=*pp;')
base=base.replace('I delta=*iter-before;', 'record(1,active_before,parity_before,0,0);I delta=*iter-before;')
specs=[
 ('dgeqrf','I*m,I*n,double*a,I*lda,double*tau,double*work,I*lwork,I*info','*m','*n','*lwork==-1','*info!=0','void'),
 ('dgelqf','I*m,I*n,double*a,I*lda,double*tau,double*work,I*lwork,I*info','*m','*n','*lwork==-1','*info!=0','void'),
 ('dgebrd','I*m,I*n,double*a,I*lda,double*d,double*e,double*tauq,double*taup,double*work,I*lwork,I*info','*m','*n','*lwork==-1','*info!=0','void'),
 ('dgeqr2','I*m,I*n,double*a,I*lda,double*tau,double*work,I*info','*m','*n','0','*info!=0','void'),
 ('dgelq2','I*m,I*n,double*a,I*lda,double*tau,double*work,I*info','*m','*n','0','*info!=0','void'),
 ('dgebd2','I*m,I*n,double*a,I*lda,double*d,double*e,double*tauq,double*taup,double*work,I*info','*m','*n','0','*info!=0','void'),
 ('dlarfg','I*n,double*alpha,double*x,I*incx,double*tau','*n','*incx','0','0','void'),
 ('dnrm2','I*n,double*x,I*incx','*n','*incx','0','0','double'),
 ('dlasq5','I*i0,I*n0,double*z,I*pp,double*tau,double*sigma,double*dmin,double*dmin1,double*dmin2,double*dn,double*dnm1,double*dnm2,void*ieee,double*eps','*n0-*i0+1','*pp','0','0','void'),
 ('dlasq6','I*i0,I*n0,double*z,I*pp,double*dmin,double*dmin1,double*dmin2,double*dn,double*dnm1,double*dnm2','*n0-*i0+1','*pp','0','0','void')]
wrappers=[];names={0:'dgesdd',1:'dlasq3'}
for idx,(name,args,m,n,query,failed,ret) in enumerate(specs,2):
 names[idx]=name;types=','.join(x[:x.index('*')+1] for x in args.split(','));vals=','.join(x.split('*')[-1] for x in args.split(','));symbol='scipy_'+name+'_64_'
 call=f'real({vals});';call=('double answer='+call) if ret=='double' else call
 end='return answer;' if ret=='double' else ''
 wrappers.append(f'{ret} {symbol}({args}){{typedef {ret}(*fn)({types});static fn real=NULL;if(!real)real=(fn)resolve("{symbol}");{call}record({idx},{m},{n},{query},{failed});{end}}}')
(HERE/'spy.c').write_text(base+ledger+'\n'+'\n'.join(wrappers)+'\n')
(HERE/'spy_manifest.json').write_text(json.dumps(dict(base=str(BASE),base_sha256=sha,kernel_ids=names),indent=2)+'\n')
