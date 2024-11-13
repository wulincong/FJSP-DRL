rsync -avze 'ssh -p 65061' \
    --exclude '__pycache__' \
    --exclude 'old_runs' \
    --exclude '.git' \
    --exclude 'runs' \
    --exclude 'trained_network' \
    /home/wlc/FJSP-DRL-MAML/ lxx_hzau@xh5.hpccube.com:/work/home/lxx_hzau/project/FJSP-DRL-main/