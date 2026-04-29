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

from qiskit.transpiler import CouplingMap

from .mirror_qa import MirrorQA
from qiskit_device_benchmarking.utilities.sampling_utils import TopoSampler

class MirrorQATopo(MirrorQA):
    def __init__(self, physical_qubits, lengths, sampling_algorithm='topo', mode='full', ffw=0.93, **kwargs):
        super().__init__(
            physical_qubits,
            lengths,
            sampling_algorithm = sampling_algorithm,
            sampler_opts = {
                'legit': len(list(physical_qubits)),
                'mode': mode, # either 'full' or 'random'
                'ffw': ffw,
            },
            **kwargs
        )

"""
Utility functions for topological MQA.
"""

class TopoUtil():
    def __init__():
        None

    @staticmethod
    def makeCouple(num_rows, num_cols, faqe=2, backend="ibm"):
        """Make the customised coupling map for topological MQA.

        Two fake qubits ('faqes') are grafted onto the left and right columns of an
        m×n square lattice. Both n×n and m×n aspect ratios are supported.
        Faqes are connected to each other and to every qubit in their respective
        boundary column.

        Args:
            num_rows (int): Number of rows in the grid (≥ 2). Passed as first arg
                to CouplingMap.from_grid, which uses row-major indexing:
                node = row * num_cols + col.
            num_cols (int): Number of columns in the grid (≥ 2). Passed as second
                arg to CouplingMap.from_grid.
            faqe (int): Number of fake qubits. Only 2 is supported.
            backend (str): Device layout style. 'ibm' uses CouplingMap.from_grid.

        Returns:
            coop (qiskit.transpiler.CouplingMap): Symmetric coupling map with genuine
                indices 0…num_rows*num_cols-1 and fake indices num_rows*num_cols
                (nl, left boundary) and num_rows*num_cols+1 (nr, right boundary).

        Raises:
            ValueError: If dimensions < 2, num_rows*num_cols is odd, or faqe != 2.
            NotImplementedError: If backend != 'ibm'.
        """
        if faqe != 2:
            raise ValueError(f"Only faqe=2 is supported; got faqe={faqe}.")
        if num_rows < 2 or num_cols < 2:
            raise ValueError(
                f"Grid must be at least 2×2; got {num_rows}×{num_cols}."
            )
        n_legit = num_rows * num_cols
        if n_legit % 2:
            raise ValueError(
                f"Grid has {n_legit} genuine qubits (odd number); "
                f"MWPM requires an even count."
            )

        if backend == "ibm":
            # from_grid(num_rows, num_cols) uses row-major indexing:
            #   node index = row * num_cols + col
            coop = CouplingMap.from_grid(num_rows, num_cols, bidirectional=True)

            first_faqe  = n_legit      # nl — connected to left column  (col 0)
            second_faqe = n_legit + 1  # nr — connected to right column (col num_cols-1)
            coop.add_physical_qubit(first_faqe)
            coop.add_physical_qubit(second_faqe)

            coop.add_edge(first_faqe, second_faqe)

            # Left column:  (row, col=0)         → node = row * num_cols
            # Right column: (row, col=num_cols-1) → node = num_cols - 1 + row * num_cols
            for row in range(num_rows):
                coop.add_edge(first_faqe,  row * num_cols)
                coop.add_edge(second_faqe, num_cols - 1 + row * num_cols)

        elif backend == "iqm":
            raise NotImplementedError("IQM backend layout is not yet implemented.")

        coop.make_symmetric()
        return coop

    @staticmethod
    def checkCouple(cmap):
        """
        Input:
            - cmap: (qiskit.transpiler.CouplingMap)

        Process:
            Just prints the coupling map via PIL.

        Output:
            prints stuff
        """
        return cmap.draw()  # draws the manual coupling map for topological MQA.