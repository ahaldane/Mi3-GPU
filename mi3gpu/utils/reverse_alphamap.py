#!/usr/bin/env python
import numpy as np
import sys, argparse
from mi3gpu.utils import alpha21, getLq, changeGauge, seqload, bimarg_to_unimarg


def nrmlz(x):
    return x/np.sum(x, axis=-1, keepdims=True)

def reverse_J(Jin, unimarg, alphamap, out_q):
    L, in_q = getLq(Jin)

    Jout = np.empty((Jin.shape[0], out_q, out_q), dtype=float)
    hout = np.empty((L, out_q), dtype=float)
    
    # construct reverse alphamap and h
    out2in = np.empty((L, out_q), dtype=int)
    for l in range(L):
        for n,chars in enumerate(alphamap[l]):
            out2in[l,chars] = n
            hout[l,chars] = -np.log(nrmlz(unimarg[l,chars]))
    
    # first map Js, without accounting for unimarg
    pairs = ((i,j) for i in range(L-1) for j in range(i+1,L))
    for n,(i,j) in enumerate(pairs):
        Jout[n,...] = Jin[n, np.add.outer(in_q*out2in[i], out2in[j])]
    Jout = Jout.reshape((Jin.shape[0], out_q*out_q))
    
    # at this point, Jout applied to the unreduced MSA will give *exactly*
    # the same energies as Jin applied to the reduced MSA
    
    # now put in the unimarg biases
    _, Jout = changeGauge.fieldlessGaugeEven(hout, Jout)
    return Jout

def reverse_seqs(seqs, unimarg, alphamap, in_q):
    rng = np.random.default_rng()
    for n, (amapi, uni) in enumerate(zip(alphamap, unimarg)):
        si = seqs[:,n]
        siorig = si.copy()
        for a, amap in enumerate(amapi):
            inds = np.where(siorig == a)[0]
            if len(amap) == 1:
                si[inds] = amap[0]
            else:
                uni_ai = uni[amap]
                Z = np.sum(uni_ai)
                uni_ai = None if Z == 0 else uni_ai/Z
                si[inds] = rng.choice(amap, p=uni_ai, size=len(inds))

def main():
    parser = argparse.ArgumentParser(
        description='reverse alphabet reduction for J or MSA files',
        epilog="J reversal outputs a J in the larger alphabet that should "
               "reproduce the uni/bivariate marginals of the unreduced MSA. \n"
               "MSA reversal replaces each letter in a reduced MSA by "
               "randomly choosing one of the unmapped letters in proportion "
               "to their unreduced univariate frequencies.")
    parser.add_argument('alphamap')
    parser.add_argument('orig_bimarg')
    parser.add_argument('mode', choices=['J', 'seq'])
    parser.add_argument('infile')
    parser.add_argument('outfile')
    parser.add_argument('--inalpha', default="ABCDEFGHIJKLMNOPQRSTUVQWXYZ",
        help='reduced alphabet, needed for sequence reversal only')
    parser.add_argument('--outalpha', default='protgap',
        help='reduced alphabet, needed for sequence reversal only')

    args = parser.parse_args(sys.argv[1:])
    
    alphabets = {'protein': alpha21[1:],
                 'protgap': alpha21,
                 'charge': '0+-', 
                 'nuc': "ACGT"}

    outalpha = alphabets.get(args.outalpha, args.outalpha)
    inalpha = alphabets.get(args.inalpha, args.inalpha)
    with open(args.alphamap) as f:
        alphamap = [a.split()[1:] for a in f.readlines()]
    in_q = len(alphamap[0])
    out_q = len(outalpha)

    ooa = alphamap

    # convert alphamap to numpy arrays
    alphamap = [[np.array([outalpha.index(c) for c in grp]) for grp in mapa]
                for mapa in alphamap]

    bimarg = np.load(args.orig_bimarg)
    unimarg = bimarg_to_unimarg(bimarg)

    if args.mode == 'J':
        Jin = np.load(args.infile)
        Jout = reverse_J(Jin, unimarg, alphamap, out_q)
        np.save(args.outfile, Jout)
    elif args.mode == 'seq':
        inalpha = args.inalpha[:in_q]
        seqs, ids, _ = seqload.loadSeqs(args.infile, alpha=inalpha)
        reverse_seqs(seqs, unimarg, alphamap, in_q)
        seqload.writeSeqs(args.outfile, seqs, ids=ids, alpha=outalpha)

if __name__ == '__main__':
    main()
