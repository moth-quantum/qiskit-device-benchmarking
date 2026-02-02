# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Mirror QA Topo Experiment class.
"""

from .mirror_qa import MirrorQA
from qiskit_aer import AerSimulator

class MirrorQATopo(MirrorQA):
    """
    An experiment to measure gate infidelity on the square lattice using mirrored circuit
    layers sampled from a defined distribution.
    """
    
    
