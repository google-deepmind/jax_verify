# Functional Lagrangian Neural Network Verification

This directory provides an implementation of the Functional Lagrangian framework from [Berrada et al 2021](https://arxiv.org/abs/2102.09479).

The `run` sub-directory contains the necessary code to reproduce the results of our paper, namely configuration files (in `run/configs/`) to specify the verification problem at hand and its various parameters, and a script (`run_functional_lagragian.py`) to solve that problem (importing code from the rest of the codebase).

## Running the Code

First make sure that:

1. you have installed the `jax_verify` package.
2. your current directory is `extensions/functional_lagrangian/run`.

Then the results of our paper can be reproduced using the commands provided below. Note that each command verifies a single sample for a single label; the full paper results can be obtained by iterating over the samples and labels.

**Note:** for each experiment, the required data and model parameters are downloaded to `/tmp/jax_verify` by default. This can be changed by modifying `config.assets_dir` in the config files.

### Robust OOD Detection on Stochastic Neural Networks

#### MLP on MNIST

```bash
python3 run_functional_lagrangian.py --config=configs/config_ood_stochastic_model.py:mnist_mlp_2_256
```

#### LeNet on MNIST

```bash
python3 run_functional_lagrangian.py --config=configs/config_ood_stochastic_model.py:mnist_cnn
```

#### VGG on CIFAR

Example for a VGG-32 (other variants also implemented):

```bash
python3 run_functional_lagrangian.py --config=configs/config_ood_stochastic_model.py:cifar_vgg_32
```

### Adversarial Robustness for Stochastic Neural Networks

Example for an MLP with 2 layers and 256 neurons (other variants also implemented):

```bash
python3 run_functional_lagrangian.py --config=configs/config_adv_stochastic_model.py:mnist_mlp_2_256
```

### Distributionally Robust OOD Detection

```bash
python3 run_functional_lagrangian.py --config=configs/config_adv_stochastic_input.py
```

## Citing

If you find this code useful, we would appreciate if you cite our paper:

```
@article{berrada2021funclag,
  title={Make Sure You're Unsure: A Framework for Verifying Probabilistic Specifications},
  author={Berrada, Leonard and Dathathri, Sumanth and Dvijotham, Krishnamurthy and Stanforth, Robert and Bunel, Rudy and Uesato, Jonathan and Gowal, Sven and Kumar, M. Pawan},
  journal={NeurIPS},
  year={2021}
}
```
