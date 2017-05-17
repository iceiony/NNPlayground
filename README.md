Summary
=======
Benchmark and verity different dynamic networks for a standard XOR problem
Currently known frameworks are : dynet, chainer, pyTorch

Observations
============
For the small XOR problem, dynet has the fastest computation time for both CPU and GPU.
Tests run on [dynet-benchmarks](https://github.com/neulab/dynet-benchmark) show pyTorch as significantly faster.

However, when verifying the correctness of implementation, we observe that pyTorch does not converge as fast as dynet does.
This occurs in a situation where XOR implementations are parameterized the same for both dynet and Torch.

