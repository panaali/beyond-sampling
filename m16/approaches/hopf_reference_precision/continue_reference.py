"""Fixed remaining reference blocks after independent pilot/resource gate."""
import argparse,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert hashlib.sha256((HERE/'reference.py').read_bytes()).hexdigest()=='fafe8022b87de770a49b1ee9a2694907b05f47eb660c99175e909fd3de8f874a'
assert hashlib.sha256((HERE/'pilot_review.json').read_bytes()).hexdigest()=='1f9232a71d868183bde679b68422a6986ef2f0687416ec2461e96bdde534d471'
pilot=json.loads((HERE/'pilot_review.json').read_text())
assert pilot['projection_gate_pass'] and pilot['projected_reference_cpu_seconds']<=600 and not pilot['accuracy_used_for_gate']
assert hashlib.sha256((HERE/'s0_c0/block000.json').read_bytes()).hexdigest()==pilot['pilot_sha256']
import reference

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--copy',type=int,required=True);args=ap.parse_args()
 assert 0<=args.seed<5 and 0<=args.copy<5
 first=1 if (args.seed,args.copy)==(0,0) else 0
 reference.run(args.seed,args.copy,first,99)
if __name__=='__main__':main()
