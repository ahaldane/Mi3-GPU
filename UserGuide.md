
Mi-3-GPU User Guide
===================

Mi3-GPU ("Mee-three") solves the "inverse Ising problem" using a GPU-parallelized Monte-Carlo sequence generation algorithm, to infer Pott models parameters for analyzing coevolutionary mutation patterns in Multiple Sequence Alignements (MSAs).

This software is meant for generation of high-accuracy Potts models of an MSA, in the sense that few analytic approximations are used and the model can be used to generate synthetic MSAs whose mutational frequencies match the dataset MSA frequencies with very low statistical error.

More precisely, this program solves for the real-valued coupling parameters `J^{ij}_{\alpha \beta}` of an infinite-range q-state Potts model with Hamiltonian `H(s) = \sum_{i < j}^L J^{ij}_{s_i s_j}` which best model the pairwise residue-frequencies obtained from an MSA containing sequences `s` with length `L` and `q` residue types.

 * Installation
 * Overview
 * Tutorial
 * File Formats
 * Usage and Options

Installation and Requirements
-----------------------------

Requirements: 

 * Python3 with the scipy, numpy, and pyopencl modules.
 * OpenCL drivers, ideally accompanied by multiple fast GPUs
 * mwc64x (see below)
 * Optional: gcc (for helper utils only)
 * Optional: mpi4py

After cloning this repository, the `mwc64x` software must be extracted into the `mwc64x` directory. This can be done automatically by running the script `dld_mwc64x.sh`. Alternatively, you can download `mwc64x` manually from http://cas.ee.ic.ac.uk/people/dt10/research/rngs-gpu-mwc64x.html and unpack.

The helper module "seqtools" in the utils directory must be compiled with `make seqtools` (requires a C compiler). This will speed up writing and loading sequences from file, and also implements phylogenetic weighting tools.

To check that the script is correctly detecting the system's OpenCL installation and GPUs, run:

    ./Mi3.py --clinfo

which should output information on the available GPUs. XXX with MPI
Mi3-GPU was developed and tested on Linux systems using the Anaconda3 python package manager. Note that it is best to install mpi4py using pip and not using conda, to avoid overriding the system MPI installation.

Overview of Functionality
-------------------------

The `Mi3.py` script can be run in different modes:

 * `infer` : Perform inverse Ising inference of a Potts model given bivariate marginals
 * `gen` : Generate new sequences given a Potts model
 * `energies` : Compute Potts energies of sequences in an MSA
 * `bimarg` : Compute bivariate residue frequencies (marginals) of an MSA
 * `subseq` : Estimate long subsequences frequencies
 * `benchmark` : Estimate computational speed of the MCMC generation

The mode is given as first argument to `Mi3.py`. To see further options for each mode, use the `-h` option:

    ./Mi3.py infer -h

A number of additional helper scripts are available in the `utils` directory
which are used for pre- and post- processing of MSAs and Potts model files.
The main ones are:

 * `changeGauge.py` : Contains methods for transforming a Potts model between different gauges.
 * `phyloWeights.py` : Efficiently compute phylogenetic weights for an MSA using the standard downweighting strategy used in covariation analysis.
 * `pseudocount.py` : Apply the small-sample pseudocount to a bivariate marginal file
 * `pre_regularize.py` : Apply a regularization pseudocount to a bivariate marginal file
 * `getSeqBimarg.py` : Compute bivariate residue frequencies for an MSA (with weights)
 * `getSeqEnergies.py` : Compute Potts energies for sequences in an MSA
 * `exploreParam.py` : Visualize the Potts model
 * `alphabet_reduction.py` : Find reduction from 21 letters to fewer for and MSA.
 * `apply_alphamap.py` : Given an alphabt mapping, convert an MSA to the reduced alphabet.

Tutorial
--------

### Inverse Ising Inference

A PBS script showing a typical set of arguments for inverse Ising inference is in the file `example/HIV_inference/pbs.sh`. When executed in its directory, this script will fit a Potts model to the bivariate marginals from the supplied file `example_bimarg_pc.npy` (derived from an HIV dataset for sequences of length length 93 with 4 residue types) and put the results in a directory `hiv_pr_inference`. The log file `hiv_pr_inference.log` contains details about how the program is running. 

It is useful to understand the stages of the inverse Ising inference. After initial startup, there are two alternating phases: 1. An MCMC sequence-generation phase in which a synthetic MSA is generated from a trial Potts model. 2. A parameter update phase, in which the Potts parameters are updated based on the discrepancy between the simulated marginals and the dataset marginals you supplied.

As seen in `pbs.sh`, you initially need to supply a bivariate marginal file (`--bim`), an initial trial Potts model (`--seqmodel`), an output directory (`--outdir`), the sequence alphabet (`--alpha`) and the synthetic MSA size (`--nwalkers`). You may later want to modify other options from their defaults. A sensible initial trial model is an independent model, (`--seqmodel independent`), and `--nwalkers` should be at least 2^15, larger is better. See "helper scripts" below on how to generate a bivariate marginal file from an MSA.

Once inference has complete the inferred Potts model can be found in the final "run" directory in the output directory, in `hiv_pr_inference/run_63/` with Potts couplings `J.npy` stored using a fieldless gauge (see glossary and file formats, below), along with other data such as the synthetic MSA (see below) in `seqs`, the bivariate marginals of that MSA in `bimarg.npy`, the Potts energies of the sequences, in `energies.npy` and more.

To monitor progress and check for convergence, a simple way is to do `grep Error hiv_pr_inference.log`. This will output lines of the following form:

    run_04 J-Step   4092  Error: SSR:   0.206  Ferr:  0.00992  X: -119.709 (-122.201) 
    run_05 J-Step   5115  Error: SSR:   0.204  Ferr:  0.00984  X: -120.186 (-122.374) 
    run_06 J-Step   6138  Error: SSR:   0.203  Ferr:  0.00971  X: -120.275 (-122.534)

This shows the result of each round of synthetic MSA generation, one round per line, with the following information from left to right: First, the "run_04" shows the round number. Next, "Jstep 6138" shows the number of Potts parameter-updates performed so far (many may be performed per round as described further below). Next, the sum-of-square-residuals (SSR) between the synthetic bivariate marginals and the dataset mbivariate marginals is shown, followed by the average bivariate marginal relative error "Ferr", computed as the sum of the relative errors |f^data - f^synth|/f^data limited to marginals $f^data > 0.01$. Finally, the "correlated energy" X is displayed, computed in two different ways. See PRE for further description of this value.

The correlated energy X should be the primary way of evaluating convergence. This value should initially become more and more negative, but after enough steps should roughly level off to a negative value, and the inference can be considered finished. X is calculated as $sum_{ij\alpha\beta} J^ij_ab C^ij_ab$ using correlation terms $C^ij_ab = f^ij_ab - f^i_a f^j_b$. In the output above the first value after "X: " is computed using the bivariate marginals of the synthetic MSA, and the value in parentheses is computed using the dataset bivariate marginals. Once near convergence these two values should become fairly close to each other and it should not matter which is tracked. The value of X give you information about how much correlated effects influence your MSA: If the abolute magnitude of X is small or 0 compared to the typical statistical energy of sequences (visible with `grep mean hiv_pr_inference.log`) this means correlated effects contribute little to mutational statistics.

Near convergence, the SSR and Ferr should become small as well, although these values are also influenced by various additional statistical errors and are not reliable indicators of convergence, and may level off prematurely due to finite-sampling effects, or occasionally increase slightly between rounds due to statistical fluctuations. Ferr is meant to be a simple and intuitive measure of the percent error in the bivariate marginals.

### Sequence Generation Arguments

 `--nwalkers` is the most important option in sequence generation and controls the size of the synthetic MSA, which is a main determinant of the level of statistical error (see discussion in PRE). The synthetic MSA is generated by having each GPU work-unit perform MCMC on a single sequence, "walking" that sequence through sequence space until equilibrium is reached. It is best to make `--nwalkers` a power of 2 to optimize GPU occupancy. As discussed in PRE, For most proteins it is desirable to use large synthetic MSAs, and we recommended at minimum 2^15 (32768), and have commonly used 2^20 and 2^22 particularly when refining a model which is already well optimized. Larger values also make better use of GPU resources (latency hiding).

The next useful argument is `--reseed`, which controls how the walker sequences are initialized in each round of MCMC sequence generation. Mi3 runs the GPU walkers until it detects that Markov equilibrium is reached by measuring the time-autocorrelation of the sequence energies. Ideally, how the walkers are initialized should not matter, but in pathological cases (eg, golf course or very rugged landscapes, glassy phases) it can. The options are to reset all walkers to the same sequence which may either generated by an independent model (`single indep`), to a previously generated sequence (randomly, `single_random`, or lowest enrgy, `single_best`), to skip resetting the sequences between rounds (`none`), to reset to sequences from a provided MSA (`msa`) specified with the `--seedmsa` option, or to reset to sequences generated by the independent model (`independent`). We recommend either `single_indep` (the default) because it helps highlight pathological behavior by starting off in a highly nonequilibrium state, or `msa` using the dataset MSA as these are likely to be local energy minima in the Potts landscape due to overfitting effects (see PRE) and this overfitting will be highlighted by a failure to reach equilibrium.

It is possible to override the automatic equilibration-detection by specifying `--equiltime`



The `seqmodel` argument is "intelligent": If set to the string 'independent' it will initialize the coupling values accorging to the uncorrelated (logscore) model and generate corresponding initial sequences. It may also be set to a directory containing the output of a previous run from which it will load the couplings and sequences. 


### Potts Parameter Fitting Arguments

Once a synthetic MSA has been generated using the trial set of coupling parameters, Mi3 enters into a parameter update phase. 

distribute_jstep


### MSA preprocessing helper-scripts

Helper scripts are also included: 
 * `getSeqBimarg.py` and `getSeqEnergies.py` compute bivariate marginals and sequence statistical energies given an MSA (and Potts couplings, for energies).
 * `pseudocount.py` adds different forms of pseudocount to the bivariate marginals.
 * `phyloWeights.py` computes the weights to account for phylogenetic relationships between sequences in an MSA, using the weighting strategy described in most covariation studies.
 * `alphabet_reduction.py` reduces the 20-letter amino acid alphabet (plus gap) to a reduced alphabet, in a way which preserves the MSA correlations. `apply_alphamap.py` applies this mapping to an MSA.
 * `changeGauge.py` transforms the Potts parameters between different gauges.

Most of these scripts expect MSA files formatted to contain only the aligned sequences, one sequence per line. You must convert any FASTA files to this format yourself, by stripping the sequence header lines and joining the sequences into one line each, which can be achieved using the BioPython module, for example, or by simple shell commands. 

Most testing has been done on Nvidia graphics cards. It is confirmed to run on systems with Nvidia Tesla K80, P100, and V100 and GTX 580 and Titan X, with 4 GPUs per node.


### File Formats

The Potts model coupling files and the bivariate marginal files are stored in the `npy` data format as 2-dimensional `float32` arrays of dimension `(L*(L-1)/2, q*q)`. The first dimension corresponds to position-pairs i,j, ordered as in the python code `[(i,j) for i in range(L-1) for j in range(i+1,L)]`. The second dimension corresponds to residue (letter) pairs, ordered as in `[(a+'i', b+'j') for a in alpha for b in alpha]` for alphabet string `alpha`.

The Potts couplings are stored according to the following sign conventions: `P(S) = exp(-E(S))`, `E(S) = \sum_{ij} J^{ij}_{s_i s_j}`.

Mi3-GPU performs almost all calculations in a "fieldless" gauge, so in general the couplings file fully specifies the model and no fields are needed. The `changeGauge.py` script can output fields, and these are output as a 2-dimensional `float32` array of dimension `(L, q)`.

MSAs are stored in a custom format to help optimize writing to and from disk, which is simply the ASCII sequences, one sequence per line, with all sequence ID information stripped. The sequences must have the same length. These MSA files can sometimes contain header lines starting with `#`. In some cases GIIIM outputs sequence files compressed with bzip2.
