"""Authorized original-size continuation; source pilot is unchanged."""
import sys
from noise_reference import run
start,stop=map(int,sys.argv[1:]);assert 1<=start<stop<=100 and stop-start<=10
for index in range(start,stop):run(index)
