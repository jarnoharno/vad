#!/bin/python
import os
from sys import argv
from subprocess import call
import multiprocessing

soundfiles = os.listdir(argv[1])

def call_algorithm(cmd):
    print(" ".join(cmd))
    call(cmd, stderr=nullf)

tasks = []
for f in soundfiles:
    if os.path.splitext(f)[1] == ".flac":
        res_name = argv[2]+"/sos_"+os.path.basename(os.path.splitext(f)[0])+".txt"
        nullf = open(os.devnull, 'w')
        cmd = ["julia", "ceesr-vad/ceesr.jl", "segments", argv[1]+f, res_name]
        tasks.append(cmd)

pool = multiprocessing.Pool(None)
r = pool.map_async(call_algorithm, tasks)
r.wait()
