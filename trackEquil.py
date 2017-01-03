#!/usr/bin/env python2
#
#Copyright 2016 Allan Haldane.

#This file is part of IvoGPU.

#IvoGPU is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#IvoGPU is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with IvoGPU.  If not, see <http://www.gnu.org/licenses/>.

#Contact: allan.haldane _AT_ gmail.com
from scipy import *
import numpy as np
import sys, os, glob, re
import argparse

def sumarr(arrlist):
    #low memory usage (rather than sum(arrlist, axis=0))
    tot = arrlist[0].copy()
    for a in arrlist[1:]:
        np.add(tot, a, tot)
    return tot

def meanarr(arrlist):
    return sumarr(arrlist)/len(arrlist)
window = 64

parser = argparse.ArgumentParser(description='Track RMSD')
parser.add_argument('rundir')
parser.add_argument('bimargfile')
parser.add_argument('pc', type=float)
args = parser.parse_args(sys.argv[1:])

eqfiles = glob.glob(os.path.join(args.rundir, 'equilibration', '*'))
pc = float(args.pc)
k = load(args.bimargfile)

eqfiles.sort(key=lambda s: int(re.match('.*bimarg_(\d+).npy$', s).groups()[0]))
bimarg = [load(fn) for fn in eqfiles]
kk = k+pc
kk = kk/sum(kk,axis=1)[:,newaxis]

print repr([sum((x - kk)**2) for x in bimarg])
print repr(log10([sqrt(mean((x - kk)**2)) for x in bimarg]))

#runningbimarg = [meanarr(bimarg[i:i+window]) for i in range(len(bimarg)-window)]
#print repr([sum((x - kk)**2) for x in runningbimarg])
#print repr(log10([sqrt(mean((x - kk)**2)) for x in runningbimarg]))
