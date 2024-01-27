#!/bin/bash
if [ $# -gt 0 ]; then
    cpu_list=$1
fi

taskset_prefix="taskset -c $cpu_list"
$taskset_prefix python3 benchmark.py -method proposed -l 800 -n 16 -m 3
$taskset_prefix python3 benchmark.py -method bo -l 60 -n 16 -m 3
$taskset_prefix python3 benchmark.py -method proposed -l 1200 -n 32 -m 6
$taskset_prefix python3 benchmark.py -method bo -l 80 -n 32 -m 6
$taskset_prefix python3 benchmark.py -method proposed -l 2000 -n 64 -m 12
$taskset_prefix python3 benchmark.py -method bo -l 100 -n 64 -m 12
