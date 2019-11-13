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
import sys, argparse
import numpy as np

def getLq(J):
    L = int(((1+np.sqrt(1+8*J.shape[0]))//2) + 0.5)
    q = int(np.sqrt(J.shape[1]) + 0.5)
    return L, q

def getCouplingMatrix(couplings):
    #compute the blocks that make up Ciajb, that is, compute the block Cij
    L, q = getLq(couplings)
    coupleinds = [(a,b) for a in range(L-1) for b in range(a+1, L)]

    C = np.empty((L,q,L,q))
    C.fill(np.nan)
    for n,(i,j) in enumerate(coupleinds):
        block = couplings[n].reshape(q,q)
        C[i,:,j,:] = block
        C[j,:,i,:] = block.T
    return C

def zeroJGauge(hs, Js, weights=None):
    """
    Changes to a gauge where $\sum_a J^{ij}_{ab} = 0$ for all $i, j, b$,
    and updates field terms so that sequence energies are preserved. The
    mean resulting coupling value is 0. This gauge is not fully contrained
    because the resulting h values depend on the gauge of the input,
    though the coupling values will be independent of the input gauge. This
    gauge minimizes the Frobenius norm of the couplings.

    If weights is provided, the gauge is instead defined by
    $\sum_a w^{ij}_{ab} J^{ij}_{ab} = 0$, and this gauge instead minimizes the
    weighted Frobenius norms, $\sqrt{ \sum_{ab} (w_{ab} J_{ab})^2 }$.

    Parameters
    ----------
    hs : numpy array of shape (L, q) or None.
    Js : numpy array of shape (L*(L-1), q*q) or None.
    weights : numpy array of shape (L*(L-1), q*q) or None.

    If either hs or Js are None (but not both), it will be imputed as an array
    of the right shape filled with zeros.
    """
    L, q, hs, Js = impute_params(hs, Js)
    if np.any(np.isinf(Js)) or np.any(np.isinf(hs)):
        raise ValueError("Error: Cannot convert to zero gauge because "
                         "of infinities")

    L, q = hs.shape
    Jx = Js.reshape((L*(L-1)//2, q, q))

    if weights is None:
        mJ1, mJ2 = np.mean(Jx, axis=1), np.mean(Jx, axis=2)
        mJ = np.mean(mJ1, axis=1)
    else:
        mJ1 = np.average(Jx, weights=weightsx, axis=1),
        mJ2 = np.average(Jx, weights=weightsx, axis=2),
        mJ = np.average(Jx, weights=weightsx, axis=(1,2))

    J0 = Jx - mJ1[:,None,:] - mJ2[:,:,None] + mJ[:,None,None]
    J0 = J0.reshape((J0.shape[0], q**2))

    h0 = hs - np.sum(mJ)/L
    i,j = np.triu_indices(L, k=1)
    np.add.at(h0, j, mJ1)
    np.add.at(h0, i, mJ2)

    return h0, J0

def zeroGauge(hs, Js, weights=None):
    """
    Changes to a gauge where $\sum_a J^{ij}_{ab} = 0$ for all $i, j, b$,
    and $\sum_a h^i_a = 0$ for all $i$.  Does *not* preserve sequence energies.
    Makes the average energy all possible sequences equal 0. This gauge is
    "fully constrained" meaning the result is independent of the gauge of the
    inputs.  This gauge minimizes the Frobenius norm of the couplings.

    If weights is provided, the gauge is instead defined by
    $\sum_a w^{ij}_{ab} J^{ij}_{ab} = 0$ and $\sum_a h^i_a = 0$, and this gauge
    instead minimizes the weighted Frobenius norm,
    $\sqrt{ \sum_{ab} (w_{ab} J_{ab})^2 }$.

    Parameters
    ----------
    hs : numpy array of shape (L, q) or None.
    Js : numpy array of shape (L*(L-1), q*q) or None.
    weights : numpy array of shape (L*(L-1), q*q) or None.

    If either hs or Js are None (but not both), it will be imputed as an array
    of the right shape filled with zeros.
    """

    h0, J0 = zeroJGauge(hs, Js, weights)
    h0 -= np.mean(h0, axis=1, keepdims=True)
    return h0, J0

def fieldlessGaugeDistributed(hs, Js, weights=None):
    """
    Converts to a fieldless gauge by evenly distributing each field value h^i_a
    among the (L-1) corresponding J^{i,j}_{a,b} values, for each i,a. This
    preserves sequence energies, but is not fully contrained meaning that the
    resulting coupling values depend on the gauge of the input.

    Parameters
    ----------
    hs : numpy array of shape (L, q) or None.
    Js : numpy array of shape (L*(L-1), q*q) or None.
    weights : dummy for signature consistency, no effect

    If either hs or Js are None (but not both), it will be imputed as an array
    of the right shape filled with zeros.
    """
    L, q, hs, Js = impute_params(hs, Js)
    J0 = Js.copy()
    hd = hs/(L-1)
    for n,(i,j) in enumerate([(i,j) for i in range(L-1) for j in range(i+1,L)]):
        J0[n,:] += np.repeat(hd[i,:], q)
        J0[n,:] += np.tile(hd[j,:], q)
    return np.zeros(hs.shape), J0

def fieldlessGaugeEven(hs, Js, weights=None):
    """
    Converts to a fieldless gauge by first transforming to the zero-mean gauge
    and them evenly distributing the resulting field values among the
    corresponding (L-1) coupling values for each position. See
    fieldlessGaugeDistributed. Does *not* preserve sequence energies. Makes
    the average energy all possible sequences equal 0. This gauge is "fully
    constrained" meaning the result is independent of the gauge of the inputs.

    Parameters
    ----------
    hs : numpy array of shape (L, q) or None.
    Js : numpy array of shape (L*(L-1), q*q) or None.
    weights : numpy array of shape (L*(L-1), q*q) or None.

    If either hs or Js are None (but not both), it will be imputed as an array
    of the right shape filled with zeros.
    """
    return fieldlessGaugeDistributed(*zeroGauge(hs, Js, weights))

def fieldlessGauge(hs, Js, weights=None):
    """
    Converts to a fieldless gauge by moving field values into the J^{0,i}
    entries (couplings to position 0). Preserves sequence energies.
    This gauge is not fully constrained, meaning that the resulting values
    depend on the gauge of the input.

    Parameters
    ----------
    hs : numpy array of shape (L, q) or None.
    Js : numpy array of shape (L*(L-1), q*q) or None.
    weights : dummy for signature consistency, no effect

    If either hs or Js are None (but not both), it will be imputed as an array
    of the right shape filled with zeros.
    """
    L, q, hs, Js = impute_params(hs, Js)

    J0 = Js.copy()
    J0[0,:] += np.repeat(hs[0,:], q)
    for i in range(L-1):
        J0[i,:] += np.tile(hs[i+1,:], q)
    return np.zeros(hs.shape), J0

def test_transform(L, q, func):
    np.random.seed(1234)
    J = np.random.rand(L*(L-1)//2, q*q)
    h = np.random.rand(L, q)

    seqs = np.random.randint(q, size=(20, L))

    from getSeqEnergies import energies

    e1 = energies(seqs, J) + np.sum(h[np.arange(L),seqs], axis=1)

    hp, Jp = func(h, J)

    ep = energies(seqs, Jp) + np.sum(hp[np.arange(L),seqs], axis=1)

    np.set_printoptions(threshold=10, suppress=True)
    print('dE', e1 - ep)
    print('h', hp)
    print('J', Jp)
    print('mh', np.mean(hp, axis=1))
    print('mJ', np.mean(Jp, axis=1))

def tryload(fn):
    if fn is None:
        return None

    try:
        return np.load(fn)
    except:
        return np.loadtxt(fn)

def impute_params(hin, Jin, err=ValueError, log=lambda x: None):
    if hin is None and Jin is None:
        err("Must supply either hin or Jin (or both)")

    if hin is not None:
        hL, hq = hin.shape

    if Jin is not None:
        jL, jq = getLq(Jin)

    if hin is None:
        log("No h supplied, assuming h = 0")
        L, q = jL, jq
        hin = np.zeros((L,q))
    elif Jin is None:
        log("No J supplied, assuming J = 0")
        L, q = hL, hq
        Jin = np.zeros((L*(L-1)//2,q*q))
    else:
        if hL != jL or hq != jq:
            err("Error: Size of h does not match size of J")
            err("Imputed (L, q) of h ({}, {}) does not match J ({}, {})".format(
                                                                hL, hq, jL, jq))
        L, q = jL, jq

    return L, q, hin, Jin

def main():
    parser = argparse.ArgumentParser(
        description='Convert Potts parameters from one gauge to another')
    parser.add_argument('gauge', choices=['fieldless', 'fieldlessEven',
                                 'weighted', 'zero', 'minJ', 'zeroJ'])
    parser.add_argument('-hin')
    parser.add_argument('-Jin')
    parser.add_argument('-hout')
    parser.add_argument('-Jout')
    parser.add_argument('-weights', help='only needed for weighted gauge')
    parser.add_argument('--txt', action='store_true',
                        help='save in text format')
    args = parser.parse_args()


    L, q, hin, Jin = impute_params(tryload(args.hin), tryload(args.Jin),
                                   err=parser.error, log=print)

    weights = None
    if args.weights != None:
        weights = np.load(args.weights)

    gfuncs = {'fieldless':     fieldlessGauge,
              'fieldlessEven': fieldlessGaugeEven,
              'zero':          zeroGauge,
              'zeroJ':         zeroJGauge}
    h1, J1 = gfuncs[args.gauge](hin, Jin, weights)

    savefunc = np.savetxt if args.txt else np.save
    if args.hout:
        savefunc(args.hout, h1)
    if args.Jout:
        savefunc(args.Jout, J1)

if __name__ == '__main__':
    main()

