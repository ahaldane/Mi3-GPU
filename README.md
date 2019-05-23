Mi3-GPU (or Montecarlo Inverse Ising Inference for GPUs)
===================================================================

Mi3-GPU (Mee-three) solves the "inverse Ising problem" using a GPU-parallelized Monte-Carlo sequence generation algorithm, to infer Potts models for analyzing coevolutionary mutation patterns in Multiple Sequence Alignements.

Given pairwise residue-frequencies obtained from an MSA containing sequences `s` with length `L` and `q` residue types, this program solves for the real-valued coupling parameters `J^{ij}_{\alpha \beta}` of an infinite-range q-state Potts model with Hamiltonian `H(s) = \sum_{i < j}^L J^{ij}_{s_i s_j}` which best model the dataset MSA.

See the User guide for detailed information and examples.

For more information see the following publications:

> Haldane, Allan, William F. Flynn, Peng He, R. S. K. Vijayan, and Ronald M. Levy (2016). 
> Structural propensities of kinase family proteins from a Potts model of residue co-variation. 
> Protein Science, 25, 1378-1384. DOI: 10.1002/pro.2954.

Licensed under GPLv3, see source for contact info.
