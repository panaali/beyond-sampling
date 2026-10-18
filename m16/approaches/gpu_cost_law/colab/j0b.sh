for p in $(pgrep -f "q_t27[0].sh"); do kill $p; done; sleep 1; cd /content && nohup bash q_t270.sh 0 "q_cm[p].sh" > /dev/null 2>&1 &
