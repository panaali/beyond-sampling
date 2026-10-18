# T192/T234 keep-alive + job runner, executed as a BLOCKING `colab exec --timeout` so the kernel stays busy (DRI memo:
# CLI >= 0.7.3 has no keep-alive ping; idle runtimes are reclaimed). Starts any /content/jobs/*.sh uploaded meanwhile.
import glob, os, subprocess, time
os.makedirs("/content/jobs", exist_ok=True)
end = time.time() + 2900
while time.time() < end:
    for f in sorted(glob.glob("/content/jobs/*.sh")):
        os.rename(f, f + ".started")
        subprocess.Popen(f"cd /content && nohup bash {f}.started > {f}.out 2>&1 &", shell=True)
    with open("/content/keepalive.beat", "w") as fh:
        fh.write(time.strftime("%H:%M:%S"))
    time.sleep(20)
print("alive", time.strftime("%H:%M:%S"))
