"""Independent contractions/pivot identity and enumerable author-cross checks."""
import hashlib,importlib.util,itertools,json,time
from pathlib import Path
import numpy as np
from .algebra import direct_integral,product_weights,reconstruct,contract,weighted_svd
HERE=Path(__file__).resolve().parent
SOURCE=Path('/workspace/clean-neurons/m16/approaches/tensor_cross/estimator.py')

def author():
    spec=importlib.util.spec_from_file_location('reviewed_author_cross',SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def main():
    start=time.process_time();C=author();shape=(7,3,3,3)
    factors=np.meshgrid(*[np.linspace(.1,.9,n) for n in shape],indexing='ij')
    product=np.prod([1+x for x in factors],axis=0)+.3*np.prod([2-x for x in factors],axis=0)
    spike=np.zeros(shape);spike[3,1,1,1]=1.
    cases={'product_sum':product,'constant':np.ones(shape),'zero':np.zeros(shape),'localized_rank1_spike':spike}
    weights=[np.arange(1,n+1,dtype=float)/sum(range(1,n+1)) for n in shape]
    points=np.asarray(list(np.ndindex(shape)));out=dict(cases=[],author_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest())
    for name,tensor in cases.items():
        exact=direct_integral(tensor,weights);cores,stats=weighted_svd(tensor,weights,2)
        rebuilt=reconstruct(cores);svd_error=float(np.max(abs(rebuilt-tensor)))
        assert svd_error<1e-11 and abs(contract(cores,weights)-exact)<1e-11
        queried=set()
        def oracle(ps):
            queried.update(map(tuple,ps));return np.array([tensor[tuple(p)] for p in ps])
        fit=C.TensorCross(oracle,list(shape),2,2);pred=fit.predict(points).reshape(shape)
        entry_error=float(np.max(abs(pred-tensor)))
        contraction_error=abs(fit.integral(weights)-direct_integral(pred,weights))
        reconstruction_error=float(np.max(abs(reconstruct(fit.cores)-pred)))
        scalar_error=abs(contract(fit.cores,weights)-fit.integral(weights))
        assert max(contraction_error,reconstruction_error,scalar_error)<1e-11
        if name!='localized_rank1_spike':assert entry_error<1e-10
        out['cases'].append(dict(name=name,exact_integral=exact,svd_max_error=svd_error,
            author_max_error=entry_error,author_integral=fit.integral(weights),
            author_unique_queries=len(queried),independent_contraction_error=contraction_error,
            scalar_contraction_error=scalar_error,pivots=fit.conditions))
    # Independent CUR identity: exhaustive maximum-area minor, no author pivots.
    a=np.column_stack((np.ones(7),np.arange(7.)))
    b=np.vstack((np.arange(5.)+1,(np.arange(5.)+.5)**2));matrix=a@b
    candidates=[(abs(np.linalg.det(matrix[np.ix_(i,j)])),i,j)
        for i in itertools.combinations(range(7),2) for j in itertools.combinations(range(5),2)]
    _,i,j=max(candidates)
    cur=matrix[:,j]@np.linalg.solve(matrix[np.ix_(i,j)],matrix[list(i),:])
    error=float(np.max(abs(cur-matrix)));assert error<1e-11
    out.update(independent_CUR_max_error=error,CUR_rows=i,CUR_columns=j,cpu_seconds=time.process_time()-start)
    (HERE/'algebra_results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))

if __name__=='__main__':main()
