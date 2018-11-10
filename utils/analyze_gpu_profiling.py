#!/usr/bin/env python2
from __future__ import print_function
import sys


with open(sys.argv[1]) as f:
    lines = [l.strip().split() for l in f.readlines()]

events = [(l[1], int(l[3]) - int(l[2])) for l in lines if len(l) > 0 and l[0] == 'EVT']
names = list(set([name for name, time in events]))

tot_times = [sum([time for name, time in events if name == n]) for n in names]

info = list(zip(names, tot_times))
info.sort(key=lambda x: x[1], reverse=True)

ml = max(len(n) for n in names) + 2
for name, time in info:
    print(name.rjust(ml), time*1e-9)

print("May ignore setup/transfer times")
