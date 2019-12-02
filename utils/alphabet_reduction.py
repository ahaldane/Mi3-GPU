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
import scipy
from scipy import *
import numpy as np
from numpy.random import randint, permutation
import sys, os, argparse
from Bio.Alphabet import IUPAC
from scipy.stats import spearmanr, pearsonr
from scipy.special import entr, rel_entr
from potts_common import getLq

def MI(ffij):
    fi = np.sum(ffij, axis=1)
    fj = np.sum(ffij, axis=0)
    return np.sum(rel_entr(ffij, np.outer(fi,fj)))

class PairData:
    def __init__(self, pairvals):
        self.data = pairvals
        L = getL(len(pairvals))
        inds, pairs = zip(*enumerate((a,b) for a in range(L-1)
                                           for b in range(a+1, L)))
        self.coords = dict(zip(pairs, inds))

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, pair):
        if isinstance(pair, int):
            return self.data[pair]
        i,j = pair
        if i < j:
            return self.data[self.coords[(i,j)]]
        else:
            return transpose(self.data[self.coords[(j,i)]])

    def __setitem__(self, pair, val):
        if isinstance(pair, int):
            self.data[pair] = val
            return
        i,j = pair
        if i < j:
            self.data[self.coords[(i,j)]] = val
        else:
            self.data[self.coords[(j,i)]] = transpose(val)

    def copy(self):
        return PairData([copy(x) for x in self.data])


def msqerr(mi1, mi2):
    return sum((array(list(mi1)) - array(list(mi2)))**2)

def pearsonGoodness(mi1, mi2):
    return pearsonr(array(list(mi1)), array(list(mi2)))[0]

def calcGoodness(mi1, mi2):
    return -msqerr(mi1, mi2)

def best_merge(L, q, uni, mis21, mis, ffs, pos):
    """
    Finds the best pair of letters to merge at positions pos.

    For each letter pair A,B it computes all the MI values which change, and
    adds up the total change, to compute a new msqerr. If then finds the choice
    with smallest msqerr.

    It returns the best msqerr, and the changed MIs (a list of length L)
    """
    ffp = [ffs[pos,j] for j in range(L) if j != pos]
    entrp = np.array([sum(entr(x), axis=1) for x in ffp])
    mip = np.array([mis[pos,j] for j in range(L) if j != pos])
    mi21p = np.array([mis21[pos,j] for j in range(L) if j != pos])
    unientr = entr(uni)

    goodness = inf
    for A in range(q-1):
        ffA = [ff[A,:] for ff in ffp]
        eA = entrp[:,A]

        for B in range(A+1,q):
            unidelta = entr(uni[A] + uni[B]) - unientr[A] - unientr[B]
            eC = np.array([sum(entr(ffa + ff[B,:])) for ff,ffa in zip(ffp,ffA)])
            eB = entrp[:,B]
            #            vvvvvvvvvvvvvvvvvvvvvvvvv   change in MI due to merge
            newmis = mip - eC + eA + eB + unidelta

            g = sum((newmis - mi21p)**2)
            if g < goodness:
                goodness, bestA, bestB, newmi = g, A, B, newmis
    
    return goodness, bestA, bestB

def mergeBimarg(ff, A, B):
    if A > B:
        A,B = B,A
    # copy over old bimarg except for column B
    newff = empty((ff.shape[0]-1, ff.shape[1]))
    newff[:B,:] = ff[:B,:]
    newff[B:,:] = ff[B+1:,:]
    newff[A,:] += ff[B,:]
    return newff

def reduceAPos(L, ffs, uni, mis21, mis, pos, alpha):
    q = len(alpha)

    goodness, A, B = best_merge(L, q, uni, mis21, mis, ffs, pos)
    
    # update the bimarg and MI
    for j in range(L):
        if j == pos:
            continue
        
        newff = mergeBimarg(ffs[pos,j], A, B)
        ffs[pos,j] = newff
        mis[pos,j] = MI(newff)
    
    alpha[A] = alpha[A] + alpha[B]
    del alpha[B]

    uni[A] = uni[A] + uni[B]
    del uni[B]

def printReduction(f, q, mis, mis21, alphas):
    goodness = pearsonGoodness(mis21, mis)
    print("-----------------------------------------------", file=f)
    print("{} Alphabet length: {}".format(q, q), file=f)
    print("{} Mean Sq Error: {}".format(q, msqerr(mis21, mis)), file=f)
    print("{} Pearson Correlation: {}".format(q, goodness), file=f)
    amap = [" ".join(a + ['*']*(q - len(a))) for a in alphas]
    print("\n".join("ALPHA{} {}".format(q, a) for a in amap), file=f)

def reduceSeq(L, q, alphas, ffs, uni):
    mis = PairData([MI(ff) for ff in ffs])
    mis21 = mis.copy()

    for i in range(q, 2, -1):
        for pos in permutation(L):
            # skip positions which are already reduced past the point we want
            if len(alphas[pos]) < i:
                continue

            f = len(alphas[pos])
            reduceAPos(L, ffs, uni[pos], mis21, mis, pos, alphas[pos])

        printReduction(sys.stdout, i-1, mis, mis21, alphas)

    return ffs, alphas

def getUnimarg(ff):
    L = getL(ff.shape[0])
    marg = array([sum(ff[0],axis=1)] + [sum(ff[n],axis=0) for n in range(L-1)])
    return marg/(sum(marg,axis=1)[:,newaxis]) # correct any fp errors

def mergeUnseen(ffs, letters, L):
    """
    Initial pass which combines all "unobserved" residues at each
    position into a combined letter at the end.
    """
    alphas = [list(letters[:]) for i in range(L)]

    uni = getUnimarg(ffs)
    # first go through and get the letter mappings for each position
    mappings = {}
    newalpha = []
    for i in range(L):
        kept, missing = [], []
        for n,f in enumerate(uni[i,:]):
            if f == 0:
                missing.append(n)
            else:
                kept.append(n)
        mappings[i] = (kept, missing)
        
        # note: here we combine all missing letters into one new letter.
        # But we could also merge them onto (eg) the least common existing
        # letter, without affecting MI. Upside: one less letter. Downside:
        # missing letters and least common letter indistinguishable.
        missed = ["".join(letters[j] for j in missing)] if missing != [] else []
        newalpha.append([letters[j] for j in kept] + missed)
    
    #now do the actual mapping of all the bimarg
    ffs = PairData(list(ffs))
    for i,j in [(i,j) for i in range(L-1) for j in range(i+1,L)]:
        (ki,mi), (kj,mj) = mappings[i], mappings[j]
        qi, qj = len(ki), len(kj)
        padi, padj = (mi != []), (mj != [])
        # skip positions with no missing letters
        if not padi and not padj:
            continue
        
        # take the non-missing bimarg, and pad end with 0 if any missing
        ff = ffs[i,j]
        newff = np.zeros((qi+padi, qj+padj), dtype=ff.dtype)
        newff[:qi,:qj] = ff[np.ix_(ki,kj)]
        ffs[i,j] = newff

    newuni = {}
    for i in range(L):
        k, m = mappings[i]
        pad = [0] if m != [] else []
        newuni[i] = [uni[i,ki] for ki in k] + pad
    
    return ffs, newuni, newalpha

def main():
    helpstr = """Typical usage: 
  $ ./alphabet_reduction.py bimarg21.npy >alpha_reductions
  $ grep ALPHA8 alpha_reductions >map8  # select 8 letter reduction
  $ ./apply_alphamap.py seq21 map8 >seq8
"""
    parser = argparse.ArgumentParser(description='Optimal alphabet reduction',
                                     epilog=helpstr,
                          formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('marginals')
    parser.add_argument('-alpha', default='protgap')

    args = parser.parse_args(sys.argv[1:])
    alphabets = {'protein': IUPAC.protein.letters, 
                 'protgap': '-' + IUPAC.protein.letters, 
                 'charge': '0+-', 
                 'nuc': "ACGT"}
    letters = alphabets.get(args.alpha, args.alpha)

    q = len(letters)
    try: 
        ff = scipy.load(args.marginals)
    except:
        ff = loadtxt(args.marginals)

    ff = ff.reshape((ff.shape[0], q, q))
    L = getL(ff.shape[0])

    ffs, uni, alphas = mergeUnseen(ff, letters, L)
    
    newffs, alphas = reduceSeq(L, q, alphas, ffs, uni)

if __name__ == '__main__':  
    main()
