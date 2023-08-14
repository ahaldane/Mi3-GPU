#!/usr/bin/env bash

fail() {
    echo 'failed' ; exit 1; 
}


if [ -f PF00018_full.txt ]; then
    echo "--> SH3 MSA already downloaded from Pfam."
else
    echo -e "\n--> Downloading SH3 MSA from Pfam..."
    wget 'https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/PF00018/?annotation=alignment:full&download' -O PF00018_full.txt.gz || fail
    gunzip PF00018_full.txt.gz
fi



echo -e "\n--> convert FASTA to flat MSA format"
python3 <<EOF || fail
import re

alpha = '-ACDEFGHIKLMNPQRSTVWY'

with open("PF00018_full.txt", "r") as fin:
    seqs = (r.split()[-1] for r in fin 
            if not (r.startswith('#') or r.startswith('/')))
    seqs = [re.sub('[a-z.]', '', r) for r in seqs]

# remove sequences with ambigious residues
seqs = [s for s in seqs if all(c in alpha for c in s)]

with open("seqs21_raw", "wt") as fout:
    fout.write("\n".join(seqs))
EOF

echo -e "\n--> remove gapped columns and sequences"
python3 <<EOF || fail
import mi3gpu.utils.seqload as seqload
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



echo -e "\n--> get phylogenetic weights and 21-letter bivariate marginals"
phy=0.4
alpha=ABCDEFGHIJKLMNOPQRSTUVWXYZ
phyloWeights.py $phy seqs21 weights$phy >Neff$phy || fail
echo -e "Effective Number of Sequences:  $(cat Neff$phy)"
getMarginals.py --weights weights${phy}.npy seqs21 bim21 || fail

echo -e "\n--> Apply small pseudocount"
pseudocount.py bim21.npy $(cat Neff$phy) --mode jeffreys -o bim21Jeff.npy || fail

echo -e "\n The bim21Jeff.npy file may now be used to infer a Potts model on GPUs."
echo -e "\n For instance, the following inference options will run a first round of inference:"
#
echo '
alpha=-ACDEFGHIKLMNPQRSTVWY
bim=bim21Jeff.npy
export PYTHONUNBUFFERED=1
Mi3.py infer --bimarg $bim \
          --mcsteps 128 \
          --nwalkers 262144 \
          --alpha=" $alpha" \
          --init_model independent \
          --reseed independent \
          --damping 0.01 \
          --reg l1z:0.0002 \
          --outdir A1 \
          --log
'
#
# A slightly more converged model can be produced by continuing the inference
# with nwalkers=1048576 for 16 mcsteps.


# exit now to skip alphabet reduction example
exit 0

# As an optional extra step, one may do "alphabet reduction" to reduce the
# model from 21 letters to fewer. This can make the inference faster or easier
# and reduce overfitting, but it is *optional* and many studies do not use it.
echo -e "\n\n*** Optional: alphabet reduction"
q=8
alpha=${alpha:0:$q}

echo -e "\n--> reduce alphabet (takes a minute)"
alphabet_reduction.py bim21Jeff.npy >alphamaps || fail
grep ALPHA$q alphamaps >map$q || fail
apply_alphamap.py seqs21 map$q >seqs$q || fail

echo -e "\n--> compute final bimarg: reduced, weighted, pseudocounted"
getMarginals.py --alpha $alpha --weights weights${phy}.npy seqs$q bim$q || fail
pseudocount.py bim$q.npy $(cat Neff$phy) -o bim${q}Jeff.npy || fail
