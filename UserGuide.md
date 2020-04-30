
Mi3-GPU User Guide
===================

Mi3-GPU ("Mee-three", for **M**arkov-Chain **I**nverse **I**sing **I**nference) solves the inverse Ising problem for application in protein covariation analysis. The goal is to infer "coupling" parameters between positions in a Multiple Sequence Alignment (MSA) of a protein family, with many applications including protein-contact prediction and fitness prediction.

More precisely, this program solves for the real-valued coupling parameters `J^{ij}_{\alpha \beta}` of an infinite-range q-state Potts model with Hamiltonian `H(s) = \sum_{i < j}^L J^{ij}_{s_i s_j}` which best model the pairwise residue-frequencies obtained from an MSA containing sequences `s` with length `L` and `q` residue types.

Mi3-GPU solves the inverse Ising problem with few approximations using Markov-Chain Monte-Carlo methods with quasi-Newton optimization, and the implementation is highly parallelized using GPUs with ~250x speedup on typical problems. This enables the construction of "generative" models which reproduce the covariation patterns of the observed MSA with very high statistical precision. The statistically accurate model and marginals produced by this method are particularly suited for studying sequence variation on a sequence-by-sequence basis and detailed MSA statistics related to higher order marginals, but can also be used in other common applications of covariation analysis.

This package also provides tools for analysis and preparation of protein-family MSAs to account for finite-sampling issues, which are a major source of error or bias in inverse Ising inference.

## References

[1] Mi3-GPU: MCMC-based Inverse Ising Inference on GPUs for protein covariation analysis. Allan Haldane, Ronald M. Levy.  Computer Physics Communications 2020. https://doi.org/10.1016/j.cpc.2020.107312

[2] Influence of multiple-sequence-alignment depth on Potts statistical models of protein covariation
Allan Haldane, Ronald M. Levy. PRE 2019. http://dx.doi.org/10.1103/PhysRevE.99.032405

Installation and Requirements
-----------------------------

Requirements: 

 * Python3 with the scipy, numpy, and pyopencl modules, with a C compiler.
 * OpenCL drivers, ideally accompanied by multiple fast GPUs
 * For extra functionality: mpi4py

This package supports [python "setuptools"](https://packaging.python.org/tutorials/installing-packages/) for installation. The simplest way to install using setuptools is to use `pip` to install from the cloned source directory:

    $ pip install /path/to/Mi3-GPU

and uninstall by "pip uninstall mi3gpu". You may consider installing Mi3-GPU inside a python environment, for example using virtualenv or conda env.

After installation using pip you should be able to run the Mi3-GPU script `Mi3.py` in your shell as well as the helper scripts such as `getMarginals.py`, assuming that you have your pip's `/bin` installation directory in your shell's `PATH` environment.

To check that the script is correctly detecting the system's OpenCL installation and GPUs, run `Mi3.py --clinfo` which should output information on the available platforms and GPUs. The GPUs should be listed as entries like `Device 'GeForce GTX TITAN X'`. Mi3-GPU was developed and tested on Linux systems using the Anaconda3 python package manager, see below for recommended hardware setup.

Overview of Functionality
-------------------------

The `Mi3.py` script can be run in different modes:

 * `infer` : Perform inverse Ising inference of a Potts model given bivariate marginals
 * `gen` : Generate new sequences given a Potts model
 * `energies` : Compute Potts energies of sequences in an MSA
 * `subseq` : Estimate long subsequences frequencies
 * `benchmark` : Estimate computational speed of the MCMC generation

The mode is given as first argument to `Mi3.py`. To see further options for each mode, use the `-h` option:

    Mi3.py infer -h

A number of additional helper scripts are available which are used for pre- and
post- processing of MSAs and Potts model files, which are:

 * `getMarginals.py` : Compute univariate/bivariate residue frequencies for an MSA (with optional weights)
 * `pseudocount.py` : Apply pseudocounts to a bivariate marginal file
 * `phyloWeights.py` : Compute phylogenetic weights for an MSA using a standard downweighting strategy
 * `getSeqEnergies.py` : Compute Potts energies for sequences in an MSA
 * `changeGauge.py` : Transform a Potts model between different gauges.
 * `getXij.py` : Compute covariance energy X for an inferred model
 * `exploreParam.py` : Visualize the Potts model
 * `alphabet_reduction.py` : Find reduction from 21 letters to fewer for and MSA.
 * `apply_alphamap.py` : Given an alphabet mapping, convert an MSA to the reduced alphabet.
 * `pre_regularize.py` : Apply a regularization pseudocount to a bivariate marginal file

Most of these scripts expect files in the formats described further below. An example of typical usage of these preprocessing scripts to prepare a protein-family MSA from pfam for analysis is described further below in "Preprocessing example" and in the script `example/SH3_preprocess/pre_process.sh`.

Additionally, in your own python scripts you can use much of the Mi3-GPU functionality by importing the `mi3gpu` module, in particular the submodules in `mi3gpu.utils`.

Tutorial
--------

The reader is referred to Ref. [1] for the main description of the algorithm. This guide gives a more practical description and tutorial. The recommended hardware setup is multiple GPUs on a single computer, as described in the recommended hardware section below.

### Inverse Ising Inference

The primary purpose of Mi3 is to infer a set of Potts model parameters based on observed site-covariation by performing inverse Ising inference.  An example script showing how to do this is in the file `example/HIV_inference/pbs.sh`, and the main command in this script is
```shell
Mi3.py infer \
  --init_model  independent \
  --bimarg      $margfile \
  --alpha       ABCD \
  --nwalkers    262144 \
  --mcsteps     64 \
  --damping     0.01 \
  --reg         l1z:0.0001 \
  --outdir $outdir >$logfile
```
This command starts Mi3 in "infer" mode to fit a Potts model, and initializes the Potts model to parameters corresponding to a site-independent (uncorrelated) model with 4 letters "ABCD". The model will be fit to the bivariate marginal file specified by the `--bimarg` option. The MCMC inference will run in 64 MCMC rounds using 262144 (= 2^18) Markov-Chain "walkers", which will be distributed across all detected GPUs. A quasi-Newton damping parameter of 0.01 will be used, with l1 regularization of strength 0.0001. The output will be written in the directory specified by "outdir" and progress logged to a logfile.

When executed in its directory, this script will fit a Potts model to the bivariate marginals from the supplied file `example_bimarg_pc.npy`, which is derived from an HIV dataset for sequences of length length 93 with 4 residue types, and put the results in a directory `hiv_pr_inference`. The log file `hiv_pr_inference.log` contains details about how the program is running. 

Once inference has complete the inferred Potts model can be found in the final "run_" directory in the output directory `hiv_pr_inference/run_63/` with Potts couplings `J.npy`, the synthetic MSA in `seqs`, the bivariate marginals of that MSA in `bimarg.npy`, the Potts energies of the sequences, in `energies.npy` and more. See file formats below.

#### Monitoring Progress

While the log file contains many details about the inference progress, a simple way to monitor progress and check for convergence is to do `grep Error hiv_pr_inference.log`. This will output lines of the following form:

    run_04 J-Step   4092  Error: SSR:   0.206  Ferr:  0.00992  X: -119.709 (-122.201) 
    run_05 J-Step   5115  Error: SSR:   0.204  Ferr:  0.00984  X: -120.186 (-122.374) 
    run_06 J-Step   6138  Error: SSR:   0.203  Ferr:  0.00971  X: -120.275 (-122.534)

This shows the result of each round of synthetic MSA generation, one round per line, with the following information from left to right: First, the "run_04" shows the MCMC round number. Next, "Jstep 6138" shows the number of Potts parameter-updates performed so far, using the "Zwanzig reweighting" strategy described in Ref [1]. Next, the sum-of-square-residuals (SSR) between the synthetic bivariate marginals and the dataset mbivariate marginals is shown, followed by the average bivariate marginal relative error "Ferr", computed as the sum of the relative errors `|f^data - f^synth|/f^data` limited to marginals `f^data > 0.01`. Finally, the "covariance energy" X (see Refs [1,2]) is displayed, computed in two different ways.

The correlated energy X should be the primary way of evaluating convergence, and its definition is given in Refs [1,2]. This value should initially become more and more negative, but after enough steps should roughly level off to a negative value, and the inference can be considered finished. The output above also shows an extra value in parentheses, which reflects a similar value to X computed the same way but with the dataset bivariate marginals rather than the model marginals, which can sometimes more reliably show progress when the inference is far from the solution. Once near convergence these two values should become closer to each other and it should not matter which is tracked. The value of X gives you information about the strength of covariance in your MSA: If the abolute magnitude of X is small or 0 compared to the typical variation in statistical energy of sequences (visible with `grep std hiv_pr_inference.log`) this means correlated effects contribute little to mutational statistics.

Near convergence, the SSR and Ferr should become small as well, although these values are also influenced by various additional statistical errors and are not reliable indicators of convergence, and may level off prematurely due to finite-sampling effects, or occasionally increase slightly between rounds due to statistical fluctuations, as described in Ref [1]. Ferr is meant to be a simple and intuitive measure of the percent error in the bivariate marginals. For the HIV example, on step 64 (run_63) the error line should look similar to:

    Error: SSR:   0.010  Ferr:  0.00856  X: -10.118 (-10.878)

meaning there is a 0.9% error in the significant marginals, and the X values appear to be converging to a value close to -12. This model is roughly converged, though further refinement can be perfomed as noted in the comment in the script.

### Useful Mi3 Parameters

Mi3 supports a number of command line options. Here are details on important ones you may wish to change.

First,  `--nwalkers`  controls the size of the synthetic MSA, which is a main determinant of the level of statistical error as discussed in more detail in Ref [1]. The synthetic MSA is generated by having each GPU work-unit perform MCMC on a single sequence, "walking" that sequence through sequence space until equilibrium is reached. It is best to make `--nwalkers` a power of 2 to optimize GPU occupancy. For most proteins it is desirable to use large synthetic MSAs, and in Ref [1] we recommended at minimum 2^15 (32768), and have commonly used 2^20 and 2^22 particularly when refining a model which is already well optimized. Increasing `--nwalkers` allows a more accurate quasi-Newton step direction and makes it possible to fit the dataset marginals more precisely.

Next, `--init_model` specifies how to initialize the Potts model parameters. If set to the string 'independent' it will initialize the coupling values accorging to the uncorrelated (logscore) model and generate corresponding initial sequences. It may also be used to continue a previous inference, by setting it to a directory containing the output of a previous run from which it will load the couplings and sequences. Related to this is the `--preopt` argument-flag, which if given causes the Zwanzig-Reweighting phase of inference to be performed before the MCMC phase, starting from the sequences and couplings loaded using `--init_model`, rather than after regenerating a new set of sequences from the given couplings as would happen otherwise. This is sometimes useful as a speedup to skip the first MCMC phase. The initial couplings can also be specified using the `--couplings` argument, and the initial sequences using `--seqs`.

Next, the `--reg` argument specifies optional regularization strengths. The main two types of regularization which may be specified are l1 and l2 regularization on the coupling parameters in the zero-mean gauge, as described in Ref [1]. These are specified in the form `--reg l1z:0.001` or `--reg l2z:0.001`, for example, with the regularization strength parameter after the colon. Regularization of the field terms is not directly supported, as in the Mi3 workflow the fields are instead effectively regularized by applying an appropriate pseudocount to the univariate marginals of the dataset using the pseudocount.py helper script. The "covariance energy" regularization described in Refs [1,2] is implemented as a helper script "pre_regularize.py" described below rather than as an Mi3.py option.

#### Quasi-Newton Step Parameters

A number of options control the quasi-Newton step direction and size often do not need to be touched, except in cases of numerical divergence, sometimes caused by rugged landscapes of glassy phases due to overfitting effects. Mi3 is not designed to fit Potts models in glassy phases, but these options may help explore behavior in such cases.

First, `--gamma` controls the quasi-Newton step size. It is 0.0004 by default which we find works well for protein families we have tested. Increasing this value may lead to faster convergence, but increasing too much can cause numerical instability. The exact value of this parameter is not too important as long as it is not very large, because the Zwanzig-Reweighting scheme of Ref [1] will compensate by changing the number of coupling-update steps, such that the total change in value of the couplings per MCMC round will be the same no matter gamma, though with somewhat less computation time for higher gamma.

Next, `--damping` determines the size of the damping parameter used in the quasi-Newton step direction. Smaller values such as 0.01 or 0.001 generally lead to faster convergence and more accurate step directions, but larger values of the damping parameter such as 0.5 are sometimes initially needed if the Potts landscape is more rugged, as can happen due to overfitting for small dataset MSAs as discussed in Ref [2]. Again, the Zwanzig Reweighting scheme typically compensates for this parameter except if it is very small. If you encounter increasing SSR or Ferr, or if Mi3 detects step size-divergence and raises an Error, try increasing this value. Once the inference has progressed some steps with a higher damping parameter and the system is closer to a solution, with lower marginal residuals, it can typically be lowered to a smaller value.

Next, `--reseed`, controls how the walker sequences are initialized in each round of MCMC sequence generation. Mi3 runs the GPU walkers until it detects that Markov equilibrium is reached by measuring the time-autocorrelation of the sequence energies. Ideally, how the walkers are initialized should not matter, but in pathological cases (eg, golf course or very rugged landscapes, glassy phases) it might. The options are to reset all walkers to the same single sequence which may either generated by an independent model (`single indep`), to a previously generated sequence (randomly, `single_random`, or lowest enrgy, `single_best`), to skip resetting the sequences between rounds (`none`), to reset to sequences from a provided MSA (`msa`) specified with the `--seedmsa` option, or to reset to sequences generated by the independent model (`independent`). By default, Mi3 uses the `independent` initialization. We find this option has no effect on convergence of the algorithm except in extreme glassy phases.

If you do encounter numerical instabilities, besides trying to tune the quasi-newton step size parameters above, note that instability can be due to overfitting effects as often caused by fitting marginals computed from MSAs with too few sequences as described in Ref [2], which lead to glassy or rugged landscapes due to spurious correlations caused by finite-sampling error. In that case, the instability is best corrected by appying stronger regularization or pseudocounts.

### Recommended Parameters for Protein Covariation Analysis

This software has been used to infer models for a number of protein families by the authors, such as the SH3 family whose pre-processing is shown in the examples directory.

Usually, we use 262144 (2^18) walkers in a first round of inference minimization with 64 MCMC steps, with damping parameter 0.1, or occasionally 0.01 if there is little covariation. This first round typically greatly reduces the SSR and Ferr and makes X a larger negative value, particularly in the first 16 MCMC steps. An example first and second roung for HIV datasets is shown in the HIV example folder, and an example first round command for the SH3 Pfam family is shown in the SH3 example script.

For pfam-like data, the X value has typically not yet levelled off in the first round, so we then run a second (or even third) round of inference starting from the final model of the first round. In these subsequent rounds, we often decrease the damping parameter to 0.01 or 0.001 (this changes the speed of inference but does not affect the final result), and increase the number of walkers to 2^19 or 2^20. This round is run until the X value levels off enough, often 64 or 128 MCMC steps. Finally, we often run one final round with 2^22 walkers for 4 MCMC steps to minimize finite-sampling error in the model marginals, to obtain a model with highly-statistically accurate marginals and very small residuals.

We run many inferences on a system with four NVidia V100 GPUs. Fitting a model with L=232 and q=21 over multiple rounds as described here takes about 12-24 hours on this system, for models which equilibrate in 2048 MC steps per MCMC round. The fitting time depends on the regularization strength as this affects the ruggedness of the model and so the equilibration time. We find using l1 regularization strength 0.0002 typically limits equilibration to about 2048 MC steps per MCMC round with minimal bias.

### MSA Preprocessing and Script Tutorial

The example script `examples/SH3_preprocess/pre_process.sh` shows a series of commands which take the SH3 Pfam MSA and process it with pseudocounts and phylogenetic weighting to produce a bivariate file which can be used to infer a Potts model.

This example script does the following, in order:

    1. Downloads the Pfam family MSA `PF00018_full.txt`
    2. Converts it to the file format used by Mi3-GPU (see below)
    3. Filters out sequences or columns with too many gaps
    4. Computes phylogenetic weights at 40% similarity threshold.
    5. Computes weighted bivariate marginals for this data
    6. Adds a "Jeffrey's" pseudocount to the marginals.

The final file, `bim21Jeff.npy`, can now be used as input to the inference procedure as described above to produce a Potts model for the SH3 family based on the Pfam alignment.

### Recommended Hardware Setup

Mi3-GPU is meant to be run on systems with one or more GPUs, and has been tested on NVIDIA V100, Titan X, GTX 1080, GTX 580, and k80 gpus. Mi3 will automatically detect available GPUs, which can be displayed by running `Mi3.py --clinfo`.

For typical protein families, one wants to use a large value of the `--nwalkers` argument described above, often of size 2^17 or larger, as this option determines how many total GPU threads are used in the MCMC phase of the inference. These threads are then automatically distributed to the available GPUs, so that the more GPUs are used the more parallel threads will be running. However, as noted in Ref [1] there are diminishing returns once the number of walkers per GPU becomes less than 2^15 (32768) on typical problems. The number of walker per GPU should be a power of 2 to optimize GPU occupancy. The ideal setup for protein covariation analysis is then 4 or more GPUs on a single system with at least 32768 walkers per GPU.

It is also possible to use Mi3 over multiple compute nodes in a cluster as Mi3 supports MPI though the python module mpi4py. Again Mi3 will detect the available GPUs on each node. However, because the Zwanzig reweighting phase described in Ref [1] can require significant communication between GPUs this will generally be slower than if the GPUs were on the same node. To minimize inter-node communication requirements, the Zwanzig reweighting step can be carried out on only the first node using the `--distribute_jstep head_node` option, leaving the other nodes unused during this phase. Note that it is best to install mpi4py using pip and not using conda, to avoid overriding the system MPI installation.

### File Formats

The Potts model coupling files and the bivariate marginal files are stored in the `npy` data format as 2-dimensional `float32` arrays of dimension `(L*(L-1)/2, q*q)`. The first dimension corresponds to position-pairs i,j, ordered as in the python code `[(i,j) for i in range(L-1) for j in range(i+1,L)]`. The second dimension corresponds to residue (letter) pairs, ordered as in `[(a+'i', b+'j') for a in alpha for b in alpha]` for alphabet string `alpha`.

The Potts couplings are stored according to the following sign conventions: `P(S) = exp(-E(S))`, `E(S) = \sum_{ij} J^{ij}_{s_i s_j}`.

Mi3-GPU performs almost all calculations in a "fieldless" gauge, so in general the couplings file fully specifies the model and no fields are needed. The `changeGauge.py` script can output fields, and these are output as a 2-dimensional `float32` array of dimension `(L, q)`.

MSAs are stored in a custom format to help optimize writing to and from disk, which is simply the ASCII sequences, one sequence per line, with all sequence ID information stripped. The sequences must have the same length. These MSA files can sometimes contain header lines starting with `#`. Mi3-GPU typically outputs sequence files compressed with bzip2.
