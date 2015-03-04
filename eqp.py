#!/usr/bin/env python2
from scipy import *

a = [load('outPh5/run_29/equilibration/bimarg_{}.npy'.format(n)) for n in range(0,32)]
k = load('nogapuDchop8_phy0.4.bimarg.npy')
kk = k+0.000001
kk = kk/sum(kk,axis=1)[:,newaxis]
print repr(log10([sqrt(mean((x - kk)**2)) for x in a]))
