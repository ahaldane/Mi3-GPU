#!/usr/bin/env python3
#
#Copyright 2020 Allan Haldane.

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

alpha20 = "ACDEFGHIKLMNPQRSTVWY"
alpha21 = '-' + alpha20

def getL(size):
    return int(((1+np.sqrt(1+8*size))//2) + 0.5)

def getLq(J):
    return getL(J.shape[0]), int(np.sqrt(J.shape[1]) + 0.5)

def pairs(L):
    return ((i,j) for i in range(L-1) for j in range(i+1,L))

def bimarg_to_unimarg(ff):
    L, q = getLq(ff)
    ff = ff.reshape((L*(L-1)//2, q, q))
    marg = np.array([np.sum(ff[0], axis=1)] +
                    [np.sum(ff[n], axis=0) for n in range(L-1)])
    return marg/(np.sum(marg, axis=1)[:,None]) # correct any fp errors
getUnimarg = bimarg_to_unimarg

def indep_bimarg(fab):
    L, q = getLq(fab)
    fabx = fab.reshape((fab.shape[0], q, q))
    fa1, fb2 = np.sum(fabx,axis=2), np.sum(fabx,axis=1)
    fafb = np.array([np.outer(fa, fb).ravel() for fa,fb in zip(fa1, fb2)])
    return fafb
indepF = indep_bimarg

def validate_bimarg(ff, eps=1e-5):
    """
    Check that unimarg computed from different bimarg rows are consistent
    with each other and sum to 1.
    """
    L, q = getLq(ff)
    ff = ff.reshape((L*(L-1)//2, q, q))
    uni = np.array([np.sum(ff[0], axis=1)] +
                   [np.sum(ff[n], axis=0) for n in range(L-1)])
    delta = np.abs(np.sum(uni, axis=1) - 1)
    if np.max(delta) > eps:
        raise ValueError(("unimarg for pos {} does not sum to 1: delta {}"
                         ).format(np.argmax(delta), np.max(delta)))

    for n,(i,j) in enumerate(pairs(L)):
        fab = ff[n]
        fi, fj = np.sum(fab, axis=1), np.sum(fab, axis=0)
        deltai = np.abs(uni[i] - fi)
        m = np.argmax(deltai)
        if deltai[m] > eps:
            raise ValueError(("inconsitent i unimarg {:.5f} with delta {:.5f} "
                "for bimarg {} ({},{})").format(uni[i][m], deltai[m], n, i, j))
        deltaj = np.abs(uni[j] - fj)
        m = np.argmax(deltaj)
        if deltaj[m] > eps:
            raise ValueError(("inconsitent j unimarg {:.5f} with delta {:.5f} "
                "for bimarg {} ({},{})").format(uni[j][m], deltaj[m], n, i, j))

def getC(fab):
    L, q = getLq(fab)
    fabx = fab.reshape((fab.shape[0], q, q))
    fa1, fb2 = np.sum(fabx,axis=2), np.sum(fabx,axis=1)
    fafb = np.array([np.outer(fa, fb).ravel() for fa,fb in zip(fa1, fb2)])

    C = fab - fafb

    ss = np.array([np.outer(fa*(1-fa), fb*(1-fb)).ravel()
                   for fa,fb in zip(fa1, fb2)])
    rho = C/np.sqrt(ss)

    return C, rho

def getM(x, diag_fill=0):
    L = getL(x.shape[0])
    M = np.zeros((L,L) + x.shape[1:], dtype=x.dtype)
    i,j = np.triu_indices(L,k=1)
    M[i,j,...] = x
    M = M + M.swapaxes(0,1)
    i,j = np.diag_indices(L)
    M[i,j,...] = diag_fill
    return M

def getXij(J, fab):
    L, q = getLq(fab)
    npr = fab.shape[0]

    fqq = fab.reshape((npr, q, q))
    Jqq = J.reshape((npr, q, q))

    fi, fj = np.sum(fqq,axis=2), np.sum(fqq,axis=1)
    fafb = np.array([np.outer(fa, fb).ravel() for fa,fb in zip(fi, fj)])

    tij = np.sum(J*fafb, axis=1)[:,None,None]
    ti = np.sum(Jqq*fi[:,:,None], axis=1)[:,None,:]
    tj = np.sum(Jqq*fj[:,None,:], axis=2)[:,:,None]
    t = Jqq

    Xij = -np.sum((fab - fafb)*J, axis=1)
    Xijab = (-tij + ti + tj - t).reshape((npr, q*q))
    return Xij, Xijab

    ## sanity check
    #print(Xij)
    #print(np.sum(Xijab*fab, axis=1))
    #print(np.sum(Xij), np.sum(Xijab*fab))  # should be equal

def get_ddE(Jrow):
    # we only have enough memory for a few rows of a J matrix ddE
    if len(Jrow.shape) == 1:
        q = int(np.sqrt(Jrow.shape[0] + 0.5))
        assert(Jrow.shape[0] == q*q)
    elif len(Jrow.shape == 2):
        q = Jrow.shape[0]
        assert(q == Jrow.shape[1])
    else:
        raise ValueError('Jrow must be 1d or 2d')

    J = Jrow.reshape((q,q))
    ddE = np.zeros((q, q, q, q))

    b, a = np.meshgrid(np.arange(q), np.arange(q))
    gi = [(a+g)%q for g in range(1,q)]
    di = [(b+d)%q for d in range(1,q)]

    for g in gi:
        for d in di:
            ddE[a,b,g,d] = -J[a,b] + J[a,d] + J[g,b] - J[g,d]

    return ddE

def getRddE(J):
    L, q = getLq(J)
    J = J.reshape((L*(L-1)//2, q, q))

    R = np.zeros(J.shape[0])
    dR = np.zeros(J.shape)

    b,a = np.meshgrid(np.arange(q), np.arange(q))

    for g in range(1,q):
        jr = J[...,a,b] - J[...,(a+g)%q,b]
        for d in range(1,q):
            jc = -J[...,a,(b+d)%q] + J[...,(a+g)%q,(b+d)%q]
            ddE = jr + jc
            R += np.sum(np.abs(ddE), axis=(1,2))
            dR += np.sign(ddE)
    dR = dR.reshape((L*(L-1)//2, q*q))/((q-1)**2)
    R = R/(4*(q-1)**2)
    return R, dR

def printsome(a, prec=4):
    return np.array2string(a.flatten()[:5], precision=prec, sign=' ')[1:-1]
