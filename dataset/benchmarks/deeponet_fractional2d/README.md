# DeepONet 2D fractional-Laplacian paper validation

This opt-in benchmark validates DeepONet on the original paper's **2D**
fractional Laplacian on the unit disk. It does not use the smaller 1D
antiderivative or Caputo examples.

The released protocol uses 5,000 input functions, 225 fixed branch sensors,
225 spatial queries, and 10 fractional orders. Its expanded representation is
11.25 million triples per split. The compact HDF5 stores each branch and target
once, and the trainer obtains the exact expanded-row semantics by indexing.
No normal `Neural_Operator` loader, factory, training loop, or inference path is
modified or slowed.

## Important released-data qualification

The authors' 2D `training_set.m` and `test_set.m` both construct the same
unscrambled Sobol stream with `Skip=3` and `Leap=24`. They therefore generate
identical functions and targets. The direct paper comparison preserves this
behavior and records it in both HDF5 metadata and results. It is a reproduction
of the published/released protocol, not evidence of unseen-function
generalization.

The public artifact is generator code rather than pre-generated arrays. The
Python translation follows the released Zernike formulas, 16-point angular
Gauss-Legendre quadrature, `h=1e-3` vector Grunwald-Letnikov scheme, and the
same Sobol skip/leap sampling. Exact MATLAB R2019a versus SciPy Sobol
direction numbers are reproduced from the Joe-Kuo 2003
`joe-kuo-old.1111` table cited by MathWorks, whose SHA-256 is verified before
generation. Standard point order is generated directly from binary logical
indices, so the data no longer depend on SciPy's newer Joe-Kuo 2008 table. A
released manufactured-solution check audits the numerical operator
independently.

## Generate, train, and evaluate

From the suite root:

```powershell
python dataset/benchmarks/deeponet_fractional2d/prepare_fractional2d.py

python dataset/benchmarks/deeponet_fractional2d/train_fractional2d.py `
  --config configs/benchmarks/deeponet_fractional2d/config_train_deeponet_paper.txt

python dataset/benchmarks/deeponet_fractional2d/train_fractional2d.py `
  --config configs/benchmarks/deeponet_fractional2d/config_train_deeponet_paper.txt `
  --eval-only
```

The isolated model matches the release:

- branch `225 -> 60 -> 60 -> 60`, with `tanh, tanh, linear`;
- trunk `3 -> 60 -> 60 -> 60`, with `tanh` after every layer;
- scalar dot product plus bias;
- truncated-Xavier weights and zero biases;
- Adam at `1e-3`;
- normalized MSE `mean(error^2) / mean(target^2)`;
- 100,000 randomly shuffled expanded triples per optimizer step.

The paper reports the 2D result only as Figure 2e, not as a numeric table. Its
test curve ends at roughly `1.2e-3` normalized MSE after 5,000 **epochs**.
There are `floor(11,250,000 / 100,000) = 112` optimizer updates per epoch, so
the direct run performs about 560,000 updates. Results must be compared to a
plot-derived tolerance, not presented against a falsely exact scalar.

Primary sources:

- Paper: <https://doi.org/10.1038/s42256-021-00302-5>
- Supplement: <https://static-content.springer.com/esm/art%3A10.1038%2Fs42256-021-00302-5/MediaObjects/42256_2021_302_MOESM1_ESM.pdf>
- Authors' code: <https://github.com/lululxvi/deeponet>
- MATLAB Sobol definition: <https://www.mathworks.com/help/stats/sobolset.html>
- Authors' Joe-Kuo 2003 direction table: <https://web.maths.unsw.edu.au/~fkuo/sobol/joe-kuo-old.1111>
