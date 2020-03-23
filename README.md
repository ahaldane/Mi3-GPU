Mi3-GPU (Markov-Chain Inverse Ising Inference for GPU)
==============================================================

## Program Description

Mi3-GPU ("Mee-three", for **M**arkov-Chain **I**nverse **I**sing **I**nference) solves the inverse Ising problem for application in protein covariation analysis. The goal is to infer ``coupling" parameters between positions in a Multiple Sequence Alignment (MSA) of a protein family, with many applications including protein-contact prediction and fitness prediction.

This package also provides tools for analysis and preparation of protein-family MSAs to account for finite-sampling issues, which are a major source of error or bias in inverse Ising inference.

## Solution Method and Applications

Mi3-GPU solves the inverse Ising problem with few approximations using Markov-Chain Monte-Carlo methods with Quasi-Newton optimization, and the implementation is highly parallelized using GPUs with ~250x speedup on typical problems. This enables the construction of models which reproduce the covariation patterns of the observed MSA with very high statistical precision. The statistically accurate model and marginals produced by this method are particularly suited for studying sequence variation on a sequence-by-sequence basis and detailed MSA statistics related to higher order marginals, but can also be used in other common applications of covariation analysis.

## Further Informtion

See the [User Guide](UserGuide.md) for detailed information and examples. This software is primarily described by Ref [1], with additional details on the statistical properties of the algorithm in Ref [2].

Licensed under GPLv3, see source for contact information.

## References

[1] Mi3-GPU: MCMC-based Inverse Ising Inference on GPUs for protein covariation analysis. Allan Haldane, Ronald M. Levy. Submitted.

[2] Influence of multiple-sequence-alignment depth on Potts statistical models of protein covariation
Allan Haldane, Ronald M. Levy. PRE 2019. http://dx.doi.org/10.1103/PhysRevE.99.032405
