################
SDP Verification
################

The ``sdp_verify`` directory contains a largely self-contained implementation of
the SDP-FO (first-order SDP verification) algorithm described in Dathathri et al
2020. We *encourage* projects building off this code to fork this directory,
though contributions are also welcome!

The core solver is contained in ``sdp_verify.py``. The main function is
``dual_fun(verif_instance, dual_vars)``, which defines the dual upper bound from
Equation (5). For any feasible ``dual_vars`` this provides a valid bound. It is
written amenable to autodiff, such that ``jax.grad`` with respect to
``dual_vars`` yields a valid subgradient.

We also provide ``solve_sdp_dual_simple(verif_instance)``, which implements the
optimization loop (SDP-FO). This initializes the dual variables using our
proposed scheme, and performs projected subgradient steps.

Both methods accept a ``SdpDualVerifInstance`` which specifies (1) the
Lagrangian, (2) interval bounds on the primal variables, and (3) dual variable
shapes.

As described in the paper, the solver can easily be applied to other
input/output specifications or network architectures for any QCQP. This involves
defining the corresponding QCQP Lagrangian and creating a
``SdpDualVerifInstance``. In ``examples/run_sdp_verify.py`` we include an
example for certifying adversarial L_inf robustness of a ReLU convolutional
network image classifier.

API Reference
=============

.. currentmodule:: jax_verify.sdp_verify

.. autofunction:: dual_fun

.. autofunction:: solve_sdp_dual

.. autofunction:: solve_sdp_dual_simple

.. autoclass:: SdpDualVerifInstance
