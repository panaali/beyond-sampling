"""Controlled source extraction and independent scalar endpoint reference."""
import ast
from pathlib import Path
from fractions import Fraction
import struct
import numpy as np

FUNCTIONS={'prefix_stats','_get','_age','base_vector','cell_index','design','design_h'}
ASSIGNMENTS={'THETA','FAMILIES','HFAMILIES'}


def extract(path):
    tree=ast.parse(Path(path).read_text());nodes=[];functions=set();assignments=set()
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in FUNCTIONS:nodes.append(node);functions.add(node.name)
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ASSIGNMENTS:
            nodes.append(node);assignments.add(node.targets[0].id)
    assert functions==FUNCTIONS and assignments==ASSIGNMENTS
    namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace,ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)


def design(ns,x,t,f,h):
    return ns['design'](ns['FAMILIES'][f],x,t) if f<8 else ns['design_h'](ns['HFAMILIES'][f-8],x,t,h)


def choices(scores):
    return np.argmax(scores,axis=0),np.argmax(scores[:8],axis=0)


def endpoint(a,c,w):
    n=a.shape[1];chosen=[];hidden=[];logits=[];labels=[]
    for r in range(n):
        row=[struct.unpack('f',struct.pack('f',float(a[int(c[j]),r,j])))[0] for j in range(16)]
        h=[max(0.,v) for v in row]
        values=[float(sum((Fraction.from_float(float(w[k,j]))*Fraction.from_float(h[j]) for j in range(16)),Fraction(0))) for k in range(10)]
        first=max(range(10),key=values.__getitem__)
        chosen.append(row);hidden.append(h);logits.append(values);labels.append(first)
    return {'chosen':np.array(chosen,dtype=np.float32),'hidden':np.array(hidden,dtype=np.float64),'logits':np.array(logits,dtype=np.float64),'prediction':np.array(labels,dtype=np.int64)}
