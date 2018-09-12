#!/usr/bin/env bash
export PYTHONPATH=../../utils:$PYTHONPATH
export PATH=../../utils:$PATH

#echo "--> Downloading SH3 MSA from Pfam..."
#wget 'https://pfam.xfam.org/family/PF00018/alignment/full/format?format=fasta&alnType=full&order=t&case=l&gaps=default&download=1' -O PF00018_full.txt

echo "--> convert FASTA to flat MSA format"
python2 <<EOF
from Bio import SeqIO
from Bio.Alphabet import IUPAC
import re

alpha = '-' + IUPAC.protein.letters

with open("PF00018_full.txt", "rU") as fin:
    seqs = [re.sub('[a-z.]', '', str(r.seq))
            for r in SeqIO.parse(fin, "fasta")]

# remove sequences with ambigious residues
seqs = [s for s in seqs if all(c in alpha for c in s)]

with open("seqs21_raw", "wt") as fout:
    fout.write("\n".join(seqs))
EOF

echo "--> remove gapped columns and sequences"
python2 <<EOF
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
print "N: {}   L: {}".format(*seqs.shape)

seqload.writeSeqs('seqs21', seqs)
EOF

echo "--> get phylogenetic weights and 21-letter bivariate marginals"
phyloWeights.py 0.2 seqs21 weights0.2 >Neff0.2
getSeqBimarg.py --weights weights0.2.npy seqs21 bim21

echo "--> reduce alphabet (takes a minute)"
alphabet_reduction.py bim21.npy >alphamaps
grep ALPHA8 alphamaps >map8
apply_alphamap.py seqs21 map8 >seqs8

echo "--> compute final bimarg: reduced, weighted, pseudocounted, regularized"
getSeqBimarg.py --alpha ABCDEFGH --weights weights0.2.npy seqs8 bim8
pseudocount.py bim8.npy 1e-8 -o bim8PC1en8.npy
pre_regularize.py bim8PC1en8.npy 3908 bimSH3_Reg
