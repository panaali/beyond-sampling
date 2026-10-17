"""OS resource enforcement added after preserved SIGPROF overrun on index12."""
import resource,runpy,signal,sys
from pathlib import Path

def cap(*_):raise TimeoutError('OS soft105/hard110 CPU-second limit')
resource.setrlimit(resource.RLIMIT_CPU,(105,110));signal.signal(signal.SIGXCPU,cap)
script=Path(__file__).with_name('component.py');sys.argv=[str(script)]+sys.argv[1:]
runpy.run_path(str(script),run_name='__main__')
