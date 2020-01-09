#!/usr/bin/env python
#
#Copyright 2019 Allan Haldane.

#This file is part of Mi3-GPU.

#Mi3-GPU is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, version 3 of the License.

#Mi3-GPU is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License
#along with Mi3-GPU.  If not, see <http://www.gnu.org/licenses/>.

#Contact: allan.haldane _AT_ gmail.com
import numpy as np

def getL(size):
    return int(((1+np.sqrt(1+8*size))//2) + 0.5)

def getLq(J):
    return getL(J.shape[0]), int(np.sqrt(J.shape[1]) + 0.5)

def getUnimarg(ff):
    L, q = getLq(ff)
    ff = ff.reshape((L*(L-1)//2, q, q))
    marg = np.array([np.sum(ff[0], axis=1)] + 
                    [np.sum(ff[n], axis=0) for n in range(L-1)])
    return marg/(np.sum(marg, axis=1)[:,None]) # correct any fp errors

def indepF(fab):
    L, q = getLq(fab)
    fabx = fab.reshape((fab.shape[0], q, q))
    fa1, fb2 = np.sum(fabx,axis=2), np.sum(fabx,axis=1)
    fafb = np.array([np.outer(fa, fb).flatten() for fa,fb in zip(fa1, fb2)])
    return fafb

def getM(x, diag_fill=0):
    L = getL(len(x))
    M = np.zeros((L,L))
    M[np.triu_indices(L,k=1)] = x
    M = M + M.T
    M[np.diag_indices(L)] = diag_fill
    return M
