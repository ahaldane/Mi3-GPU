IvoGPU
======

Biophysical Potts-Model inference on the GPU using OpenCL.

This program solves the "Inverse Ising problem" using a parallelized Monte-Carlo sequence generation algorithm. That is, it solves for the real-valued coupling parameters `J^{ij}_{\alpha \beta}` of an infinite-range q-state Potts model with Hamiltonian `H(s) = \sum_{i < j}^L J^{ij}_{s_i s_j}`, given bivariate marginals which might be obtained from a multiple sequence alignment containing sequences `s` with length `L` and `q` residue types.

For more information see the supplementary text in the publication:

> Haldane, Allan, William F. Flynn, Peng He, R. S. K. Vijayan, and Ronald M. Levy (2016). 
> Structural propensities of kinase family proteins from a Potts model of residue co-variation. 
> Protein Science, 25, 1378-1384. DOI: 10.1002/pro.2954.

Licensed under GPLv3, see source for contact info. The branch `v1` represents the version used in the publication above. The branch `dev` is an updated version which includes features such as parallel tempering.

Rough Usage Guide
=================

Required packages: Python2, scipy/numpy, pyopencl, mwc64x.

The helper module "seqtools" in the utils directory must be compiled with `make seqtools` (requires gcc).  The `mwc64x` package must be obtained (available at http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html) and placed in a directory named `mwc64x` next to the `IvoGPU.py` script.

The `IvoGPU.py` script can then be run in a variety of modes, incuding inverse ising inference, Monte-Carlo sequence generation, and Potts energy evaluation. Run `IvoGPU.py -h` to see a list of modes (see "positional arguments"). To get more info on each mode, use the `-h` option:

    ./IvoGPU.py inverseIsing -h

To check that the script is correctly detecting the system's GPUs, run:

    ./IvoGPU.py --clinfo

An example PBS script showing a typical set of arguments for inverse ising inference is in the file `example_pbs.sh`, which will fit the bivariate marginals from the file `example_bimarg_pc.npy`, computed from an HIV dataset for sequences of length length 93 with 4 residue types. The log file will contain many details about how the program is running. The script attempts to deduce some arguments from other supplied arguments (eg the sequence length L can be deduced from any supplied sequence file). An easy way to monitor progress is to do `grep Ferr logfile`.

The `seqmodel` argument needs more detail: If set to the string 'independent' it will initialize the coupling values accorging to the uncorrelated (logscore) model and generate corresponding initial sequences. It may also be set to a directory containing the output of a previous run from which it will load the couplings and sequences. 

Helper scripts are also included: 
 * `getSeqBimarg.py` and `getSeqEnergies.py` compute bivariate marginals and sequence statistical energies given an MSA (and Potts couplings, for energies).
 * `pseudocount.py` adds different forms of pseudocount to the bivariate marginals.
 * `phyloWeights.py` computes the weights to account for phylogenetic relationships between sequences in an MSA, using the weighting strategy described in most covariation studies.
 * `alphabet_reduction.py` reduces the 20-letter amino acid alphabet (plus gap) to a reduced alphabet, in a way which preserves the MSA correlations. `apply_alphamap.py` applies this mapping to an MSA.
 * `changeGauge.py` transforms the Potts parameters between different gauges.

Most of these scripts expect MSA files formatted to contain only the aligned sequences, one sequence per line. You must convert any FASTA files to this format yourself, by stripping the sequence header lines and joining the sequences into one line each, which can be achieved using the BioPython module, for example, or by simple shell commands. 

These scripts expect Potts parameter files to be numpy files (`.npy`) containing the couplings in a fieldless gauge, so no fields are stored. The couplings are stored as a (L choose 2) by q^2 array of single-precision floats, where the first dimension stores couplings for pairs (0,1), (0,2), (0,3) ... (1,2) ... and so on.

Most testing has been done on Nvidia graphics cards. It is confirmed to run on systems with Nvidia Tesla K80, P100, and V100 and GTX 580 and Titan X, with 4 GPUs per node.
