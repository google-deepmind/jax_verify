# jax_verify: Neural Network Verification in JAX

[![tests status](https://travis-ci.com/deepmind/jax_verify.svg?branch=master)](https://travis-ci.com/deepmind/jax_verify)
[![docs: latest](https://img.shields.io/badge/docs-stable-blue.svg)](https://jax-verify.readthedocs.io)

Jax_verify is a library containing JAX implementations of many widely-used neural network verification techniques.

## Overview

If you just want to get started with using jax_verify to verify your neural
networks, the main thing to know is we provide a simple, consistent interface
for a variety of verification algorithms:

```python
output_bounds = jax_verify.verification_technique(network_fn, input_bounds)
```

Here, `network_fn` is any JAX function, `input_bounds` define bounds over
possible inputs to `network_fn`, and `output_bounds` will be the computed bounds
over possible outputs of `network_fn`. `verification_technique` can be one of
many algorithms implemented in `jax_verify`, such as `interval_bound_propagation`
or `crown_bound_propagation`.

The overall approach is to use JAX’s powerful [program transformation system](https://jax.readthedocs.io/en/latest/notebooks/Writing_custom_interpreters_in_Jax.html), 
which allows us to analyze general network structures defined by `network_fn`
and then to define corresponding functions for calculating
verified bounds for these networks.

## Verification Techniques

The methods currently provided by `jax_verify` include:

* SDP-FO (first-order SDP verification, [Dathathri et al 2020](https://arxiv.org/abs/2010.11645))
* Non-convex ([Bunel et al 2020](https://arxiv.org/abs/2010.14322))
* Interval Bound Propagation ([Gowal et al 2018](https://arxiv.org/pdf/1810.12715.pdf), [Mirman et al 2018](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf))
* Backward Lirpa bounds such as CAP ([Wong and Kolter 2017](https://arxiv.org/pdf/1711.00851.pdf)), FastLin([Weng et al 2018](https://arxiv.org/pdf/1804.09699.pdf)) or CROWN ([Zhang et al 2018](https://arxiv.org/pdf/1811.00866.pdf))
* Forward Lirpa bounds ([Xu et al 2020](https://arxiv.org/pdf/2002.12920.pdf))
* CROWN-IBP ([Zhang et al 2019](https://arxiv.org/abs/1906.06316))
* Planet (also known as the "LP" or "triangle" relaxation, [Ehlers 2017](https://arxiv.org/abs/1705.01320)), currently using [CVXPY](https://github.com/cvxgrp/cvxpy) as the LP solver
* MIP encoding ([Cheng et al 2017](https://arxiv.org/pdf/1705.01040.pdf), [Tjeng et al 2019](https://arxiv.org/pdf/1711.07356.pdf))

## Installation

**Stable**: Just run `pip install jax_verify` and you can `import jax_verify` from any of your Python code.

**Latest**: Clone this directory and run `pip install .` from the directory root.

## Getting Started

We suggest starting by looking at the minimal examples in the `examples/` directory.
For example, all the bound propagation techniques can be run with the `run_boundprop.py` script:

```bash
cd examples/
python3 run_boundprop.py --boundprop_method=interval_bound_propagation
```

For documentation, please refer to the [API reference page](https://jax-verify.readthedocs.io/en/latest/api.html).

## Notes

Contributions of additional verification techniques are very welcome. Please open
an issue first to let us know.

This is not an official Google product.



