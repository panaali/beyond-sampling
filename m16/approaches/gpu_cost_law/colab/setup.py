import subprocess as s
r=s.run("cd /content && mkdir -p cn && tar xzf t192code.tgz -C cn && pip install -q git+https://github.com/alignment-research-center/alg-zoo 2>&1|tail -1; nproc; free -g|head -2; nvidia-smi -L",shell=True,capture_output=True,text=True);print(r.stdout[-1500:],r.stderr[-800:])
