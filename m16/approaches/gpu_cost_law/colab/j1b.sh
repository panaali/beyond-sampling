for p in $(pgrep -f "q_t27[0].sh"); do kill $p; done; sleep 1; cd /content && nohup bash q_t270.sh 1 "q_l[b].sh" > /dev/null 2>&1 &
