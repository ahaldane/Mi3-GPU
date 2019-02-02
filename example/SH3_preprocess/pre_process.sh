#!/usr/bin/env bash
export PYTHONPATH=../../utils:$PYTHONPATH
export PATH=../../utils:$PATH

echo "--> Downloading SH3 MSA from Pfam..."
wget 'https://pfam.xfam.org/family/PF00018/alignment/full/format?format=fasta&alnType=full&order=t&case=l&gaps=default&download=1' -O PF00018_full.txt

echo "--> convert FASTA to flat MSA format"
python <<EOF
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re

alpha = '-' + IUPAC.protein.letters

with open("PF00018_full.txt", "r") as fin:
    seqs = [re.sub('[a-z.]', '', str(r.seq))
            for r in SeqIO.parse(fin, "fasta")]

# remove sequences with ambigious residues
seqs = [s for s in seqs if all(c in alpha for c in s)]

with open("seqs21_raw", "wt") as fout:
    fout.write("\n".join(seqs))
EOF

echo "--> remove gapped columns and sequences"
python <<EOF
import seqload
import numpy as np

seqs = seqload.loadSeqs('seqs21_raw')[0]
nseq, L = seqs.shape

# only keep columns with < 10% gaps
col_gap_pct = np.sum(seqs == 0, axis=0)/float(nseq)
seqs = seqs[:, col_gap_pct < 0.1]

# only keep sequences with < 10% gaps
seq_gap_pct = np.sum(seqs == 0, axis=1)/float(L)
seqs = seqs[seq_gap_pct < 0.1, :]
print("N: {}   L: {}".format(*seqs.shape))

seqload.writeSeqs('seqs21', seqs)
EOF

phy=0.2
q=8
alpha=ABCDEFGHIJKLMNOPQRSTUVWXYZ
alpha=${alpha:0:$q}

echo "--> get phylogenetic weights and 21-letter bivariate marginals"
phyloWeights.py $phy seqs21 weights$phy >Neff$phy
getSeqBimarg.py --weights weights${phy}.npy seqs21 bim21
pseudocount.py bim21.npy $(cat Neff$phy) --mode jeffreys -o bim21Jeff.npy

echo "--> reduce alphabet (takes a minute)"
alphabet_reduction.py bim21Jeff.npy >alphamaps
grep ALPHA$q alphamaps >map$q
apply_alphamap.py seqs21 map$q >seqs$q

echo "--> compute final bimarg: reduced, weighted, pseudocounted, regularized"
getSeqBimarg.py --alpha $alpha --weights weights${phy}.npy seqs$q bim$q
pseudocount.py bim$q.npy $(cat Neff$phy) -o bim${q}Jeff.npy
pre_regularize.py bim${q}Jeff.npy $(cat Neff$phy) bimSH3_Reg
