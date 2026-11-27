"""cvjax: continuous-variable quantum neural networks in the Fock basis, in JAX.

Inspired by PennyLane's `CVNeuralNetLayers` and the Strawberry Fields Fock backend (Xanadu),
The Walrus, CV-QuForge and QuIara (Tiago de S. Farias and Alexandre C. Ricardo) and the
regression QNN of de Lima, Farias, Ricardo and Boas. See README.md (Créditos).
"""
from . import gates, layers, measure, models
from .fock import apply_diag, apply_one, apply_two, fock_state, norm, vacuum
from .layers import cv_neural_net_layers, run_pennylane_ops
from .models import CVQNN

__version__ = "0.1.0"
__all__ = ["gates", "layers", "measure", "models", "vacuum", "fock_state", "apply_one", "apply_diag",
           "apply_two", "norm", "cv_neural_net_layers", "run_pennylane_ops", "CVQNN"]
