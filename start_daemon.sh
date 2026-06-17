#!/bin/bash
pkill -f daemon.py 2>/dev/null
sleep 1
mkdir -p /run/aios /var/log/aios
nohup python3 /root/aios-core/daemon.py > /tmp/aios_daemon.log 2>&1 &
echo $! > /run/aios/aios.pid
echo "Started PID $!"
