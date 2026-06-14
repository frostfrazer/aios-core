#!/bin/bash
for f in /sys/fs/cgroup/*/cgroup.procs /sys/fs/cgroup/*/*/cgroup.procs /sys/fs/cgroup/*/*/*/cgroup.procs /sys/fs/cgroup/cgroup.procs; do
  n=$(wc -l < "$f" 2>/dev/null)
  if [ "$n" != "0" ] && [ -n "$n" ]; then
    echo "$f: $n"
  fi
done
