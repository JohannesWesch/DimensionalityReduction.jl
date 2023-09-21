# DimensionalityReduction.jl
A tool for preprocessing neural networks in order to speed up neural network verification

## Abstract
Formal verification of neural networks is essential in order to deploy them in safety-critical
settings. However, existing verification methods are not yet scalable enough to handle
practical problems with large input sets. One major reason for this is the input dimension
directly influencing the verification time. In this work, we propose a technique to address
this challenge: A dimensionality reduction approach that ensures that the verification
of the reduced network implies the verification of the original network. The reduction
approach uses the first weight matrix of the network as well as the input constraints of
the verification problem to reduce the input dimension. Our evaluation shows that our
method can speed up the verification process on certain instances. We evaluate exact
and approximate reduction techniques. The approximate approach outperforms the exact
approaches with less precomputation time as well as a faster verification time.
