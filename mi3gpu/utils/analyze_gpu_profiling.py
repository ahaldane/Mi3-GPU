#!/usr/bin/env python3
import sys

with open(sys.argv[1]) as f:
    lines = [l.strip().split() for l in f.readlines()]

units = 1e-9  # nanoseconds
events = [(l[1], int(l[2])*units, int(l[3])*units)
          for l in lines if len(l) > 0 and l[0] == 'EVT']
names = sorted(set([name for name, start, end in events]))
t0 = events[0][1]

tot_times = [sum([end-start for name, start, end in events if name == n])
             for n in names]

info = list(zip(names, tot_times))
info.sort(key=lambda x: x[1], reverse=True)

ml = max(len(n) for n in names) + 2
for name, time in info:
    print(name.rjust(ml), time*1e-9)

print("May ignore setup/transfer times")

import pylab as plt
import matplotlib.lines as mlines
from matplotlib import cm

t0 = events[0][1]
colors = {name: cm.gist_ncar(n/len(names)) for n,name in enumerate(names)}
for n, (name, start, end) in enumerate(events):
    plt.plot([start, end], [n,n], color=colors[name])

plt.legend([mlines.Line2D([], [], color=colors[name], label=name)
            for name in names], names)
plt.show()

