#!/usr/bin/env python2
from scipy import *

a = [load('outO4/run_1/equilibration/bimarg_{}.npy'.format(n)) for n in range(0,128)]
#k = load('nogapuDchop8_phy0.4.bimarg.npy')
k = load('KinaseHH_192p8l.bimarg.npy')
kk = k+0.000001
kk = kk/sum(kk,axis=1)[:,newaxis]
print repr(log10([sqrt(mean((x - kk)**2)) for x in a]))
