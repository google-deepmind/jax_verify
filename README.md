# Neural Network Verification in JAX

This directory contains simple implementations of several neural network verification techniques in JAX, including:

* The SDP verification approach in Dathathri et al 2020
* The nonconvex formulation from Hinder et al 2020 (to be added soon)
* Bound propagation techniques including Interval Bound Propagation ([Gowal et al 2018](https://arxiv.org/pdf/1810.12715.pdf), [Mirman et al 2018](http://proceedings.mlr.press/v80/mirman18b/mirman18b.pdf)), Fast-Lin ([Wong and Kolter 2017](https://arxiv.org/pdf/1711.00851.pdf), [Weng et al 2018](https://arxiv.org/pdf/1804.09699.pdf)), and CROWN ([Zhang et al 2018](https://arxiv.org/pdf/1811.00866.pdf))

This is not an official Google product.

## Installation

Just run `pip install jax_verify` and you can `import jax_verify` from any of your Python code.

We include several minimal examples in the `examples/` directory.
For example, all the bound propagation techniques can be run with the `run_boundprop.py` script:

```bash
cd examples/
python3 run_boundprop.py --boundprop_method=interval_bound_propagation
```

## Usage

Please refer to the [official documentation pages](link to readthedocs).


