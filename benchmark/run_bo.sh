#!/bin/bash
if [ $# -gt 0 ]; then
    cpu_list=$1
fi

taskset_prefix="taskset -c $cpu_list"
episode=50
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 16 -m 3
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 64 -m 3
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 256 -m 3
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 16 -m 6
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 64 -m 6
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 256 -m 6
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 16 -m 12
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 64 -m 12
$taskset_prefix python3 benchmark.py -method bo -l $episode -n 256 -m 12
