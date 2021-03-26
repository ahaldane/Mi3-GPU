#!/usr/bin/env python
import sys, argparse
import numpy as np
from scipy.special import rel_entr
import mi3gpu.utils.changeGauge as changeGauge
from mi3gpu.utils import (getLq, pairs, getXij, bimarg_to_unimarg, indepF, 
                          getM, getRddE)

def DI(fi, fj, J):
    q = int(np.sqrt(J.size))
    fij = np.exp(-J).reshape((q, q))

    nrmlz = lambda x: x/np.sum(x)

    gi = nrmlz(np.ones(q))
    gj = nrmlz(np.ones(q))

    #iterate self-consistent equations for gi and gj
    d = np.inf
    while d > 1e-8:
        gip = nrmlz(fi/np.sum(fij   * gj, axis=1))
        gjp = nrmlz(fj/np.sum(fij.T * gi, axis=1))
        d = np.max([np.max(gip - gi), np.max(gjp - gj)])
        gi, gj = gip, gjp

    pij = nrmlz(fij * gi[:,None] * gj[None,:])

    return np.sum(rel_entr(pij, np.outer(fi, fj)))

def score(score, ff, J=None)
    if J is None and score not in ['MI', 'maxX', 'TVDC']:
        raise ValueError("J must be supplied to compute {}".format(score))

    if score == 'fb':
        h0, J0 = changeGauge.zeroGauge(None, J)
        return np.sqrt(np.sum(J0**2, axis=1))
    elif score == 'fbw':
        w = ff
        hw, Jw = changeGauge.zeroGauge(None, J, weights=w)
        return np.sqrt(np.sum((Jw*w)**2, axis=1))
    elif score == 'fbwsqrt':
        w = np.sqrt(ff)
        hw, Jw = changeGauge.zeroGauge(None, J, weights=w)
        return np.sqrt(np.sum((Jw*w)**2, axis=1))
    elif score == 'fbwsi':
        w = np.sqrt(indepF(ff))
        hw, Jw = changeGauge.zeroGauge(None, J, weights=w)
        return np.sqrt(np.sum((Jw*w)**2, axis=1))
    elif score == 'fbwexp':
        w = 1-np.exp(-4*ff)
        hw, Jw = changeGauge.zeroGauge(None, J, weights=w)
        return np.sqrt(np.sum((Jw*w)**2, axis=1))
    elif score == 'Xij':
        C = ff - indepF(ff)
        X = -np.sum(C*J, axis=1)
        return X
    elif score == 'absXij':
        Xij, Xijab = getXij(J, ff)
        return np.sum(np.abs(Xijab), axis=1)
    elif score == 'absfXij':
        Xij, Xijab = getXij(J, ff)
        return np.sum(np.abs(ff*Xijab), axis=1)
    elif score == 'MI':
        return np.sum(rel_entr(ff, indepF(ff)), axis=-1)
    elif score == 'DI':
        h0, J0 = changeGauge.zeroGauge(None, J)
        f = bimarg_to_unimarg(ff)
        return np.array([DI(f[i,:], f[j,:], J0[n])
                               for n,(i,j) in enumerate(pairs(L))])
    elif score == 'maxC':
        C = ff - indepF(ff)
        return np.max(np.abs(C), axis=1)
    elif score == 'TVDC':
        C = ff - indepF(ff)
        return 0.5*np.sum(np.abs(C), axis=1)
    elif score == 'RddE':
        R, dR = getRddE(J)
        return R
    else:
        raise Exception("Not yet implemented")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bimarg')
    parser.add_argument('J', nargs='?', default=None)
    parser.add_argument('score', choices=['fb', 'fbw', 'fbwsqrt', 'fbwsi', 
        'Xij', 'absXij', 'absfXij', 'MI', 'DI', 'maxC', 'TVDC', 'RddE', 
        'fbwexp'])
    parser.add_argument('-o', '--out', default='score.npy')
    parser.add_argument('--binary', action='store_true')
    parser.add_argument('--pcN', type=float, default=0.0)
    args = parser.parse_args()

    ff = np.load(args.bimarg)
    ff = ff + args.pcN
    ff = ff/np.sum(ff, axis=1, keepdims=True)

    J = None
    if args.J is not None:
        J = np.load(args.J)
    elif args.score not in ['MI', 'maxX', 'TVDC']:
        raise ValueError("J must be supplied to compute {}".format(args.score))
    
    pottsScore = score(args.score, ff, J)

    pottsScore = pottsScore.astype('f4')
    if args.binary:
        getM(pottsScore)[::-1,:].tofile(args.out)
    else:
        np.save(args.out, pottsScore)

if __name__ == '__main__':
    main()
