from typing import Optional, List
import numpy as np
from qiskit_ibm_runtime import (EstimatorV2 as Estimator, EstimatorOptions, IBMBackend, 
                                RuntimeJobV2 as RuntimeJob)
from qiskit_ibm_runtime.utils.noise_learner_result import LayerError
from qiskit.primitives import PrimitiveResult

from .mirror_pub import MirrorPubOptions

def submit_mirror_test(backend: IBMBackend,
                       num_gates: int=4986,
                       num_qubits: int=100,
                       theta: float=0,
                       path: Optional[tuple[int, ...]] = None,
                       path_strategy: str = "eplg_chain",
                       noise_model: Optional[List[LayerError]] = None) -> RuntimeJob:
    """
    Constructs a mirror circuit test based upon a 1D Ising model simulation. The 1D model
    is executed on a line of qubits. The particular line can be selected automatically by
    passing `num_qubits` along with a `path_strategy`, or manually by specifying a `path`
    as a list of edges (q_i, q_j) in the connectivity graph of the backend. `num_gates`
    will control the number of distinct time-steps in the Trotter evolution of the model.
    The `theta` parameter controls the rotation angle of the layer of 1Q gates inserted
    between the 2Q gate layers. Non-zero values of `theta` will ensure entanglement growth
    in successive time steps.

    You can avoid re-learning noise models by passing in an already learned `noise_model`
    from a prior `NoiseLearner` execution.

    Args:
        backend: the IBM backend to submit the benchmark to.
        num_gates: proxy for number of Trotter time steps in the 1D Ising model circuit.
        num_qubits: determines the width of the benchmark circuit.
        theta: Controls rotation angle of 1Q gates in Trotter step. Non-zero values will
            spread entanglement.
        path: a list of edges (q_i, q_j) in the connectivity graph of the backend that
            defines the 1D chain of the Ising model
        path_strategy: one of "eplg_chain", "vf2_optimal", or None. "eplg_chain" will use
            the same chain as found by the EPLG benchmark. "vf2_optimal" will choose a
            chain using the same heuristics as the vf2 layout pass in Qiskit (also known
            as "mapomatic"). A value of None will simply select an appropriate length chain
            from the longest possible chain on the device.
        noise_model: A noise model from a prior NoiseLearner or Estimator job on the same
            layers as used in the benchmark circuit.

    Returns:
        A RuntimeJob corresopnding to the Estimator query of the benchmark.
    """
    pub_options = MirrorPubOptions()
    pub_options.num_qubits = num_qubits
    pub_options.target_num_2q_gates = num_gates
    pub_options.theta = theta
    if path is not None:
        pub_options.path = path
        pub_options.path_strategy = None
    else:
        pub_options.path_strategy = path_strategy

    pubs = pub_options.get_pubs(backend)

    options = EstimatorOptions()
    # turn on T-REX and ZNE
    options.resilience_level = 2

    # dynamical decoupling
    options.dynamical_decoupling.enable = True
    options.dynamical_decoupling.sequence_type = "XpXm"

    # twirling
    options.twirling.enable_gates = True
    options.twirling.num_randomizations = 1000
    options.twirling.shots_per_randomization = 64

    # PEA
    options.resilience.zne.amplifier = "pea"
    options.resilience.zne.noise_factors = [1, 1.6, 1.9, 2.8]

    if noise_model is not None:
        options.resilience.layer_noise_model = noise_model
    else:
        options.resilience.layer_noise_learning.shots_per_randomization = 64
        options.resilience.layer_noise_learning.num_randomizations = 1000

    # experimental options
    options.experimental = {
        "execution": {"fast_parametric_update": True}
    }

    estimator = Estimator(backend, options=options)
    return estimator.run(pubs)


def analyze_mirror_result(result: PrimitiveResult, accuracy_threshold: float=0.1):
    assert len(result) == 1, "Expected a length 1 PrimitiveResult"
    evs = result[0].data.evs
    evs_shape = evs.shape
    # we expect a shape of the form (1, 1, N)
    assert len(evs_shape) == 3, "Failed data shape check"
    assert evs_shape[0] == 1, "Failed data shape check"
    assert evs_shape[1] == 1, "Failed data shape check"

    evs = evs.flatten()
    N = len(evs)
    ev_errors = np.abs(1 - evs)
    ev_errors.sort()

    median_error = np.median(ev_errors)
    mean_error = np.average(ev_errors)
    print(f"Median error: {median_error}")
    print(f"Mean error: {mean_error}")

    # find fraction within the accuracy threshold of the ideal value
    fraction = np.argmax(ev_errors > accuracy_threshold) / N
    print(f"Fraction within {int(accuracy_threshold * 100)}% of ideal value: {fraction}")
    
    return median_error, mean_error, fraction
