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

import networkx as nx
from qiskit.transpiler import CouplingMap

from .mirror_qa import MirrorQA, QuantumAwesomeness
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

    def _sample_sequences(self):
        # Reset sampler outcome log before each run so stale data never leaks.
        self._distribution._all_call_outcomes = []
        sequences = super()._sample_sequences()
        # With full_sampling=False, the sampler is called once per sample (not per
        # circuit). _all_call_outcomes has num_samples entries; _pairs has
        # num_samples * num_lengths entries. Replicate each sample's outcome across
        # all circuits of that sample so _topo_outcomes[i] aligns with _pairs[i].
        num_lengths = len(self.experiment_options.lengths)
        self._topo_outcomes = [
            call[0]
            for call in self._distribution._all_call_outcomes
            for _ in range(num_lengths)
        ]
        return sequences

    def bot(self, data):
        """Topological MWPM bot. Classifies each circuit as topological (f2f) or not (f2g).

        Mirrors the sampler's nl–nr graph structure: genuine edges are weighted by MI,
        fake boundary edges (nl→left_nodes, nr→right_nodes) are weighted by the circuit's
        average MI, and the nl–nr edge is weighted by ffw × average MI. This preserves
        the ffw calibration at all noise levels — when MI collapses to zero at high depth,
        the bot's classification converges to ~50% topological (random).

        Does NOT use circuit metadata (exp._pairs / exp._singles).

        Args:
            data: list of circuit result dicts from experiment_data.data().

        Returns:
            is_topo (list[bool]): True if nl–nr in MWPM matching (topological) per circuit.
            boundary (tuple): (left_nodes, right_nodes) from the sampler.
        """
        import numpy as np

        sampler = self._distribution
        left_nodes = sampler._left_nodes
        right_nodes = sampler._right_nodes
        ffw = sampler.ffw
        nl = sampler.legit
        nr = sampler.legit + 1

        genuine_edges = list(sampler._2q.coupling_map.get_edges())
        qa = QuantumAwesomeness(genuine_edges)

        is_topo = []
        for circ_data in data:
            mi_dict = qa.mutual_info([circ_data])[0]  # {(j,k): float} for j < k

            avg_mi = np.mean(list(mi_dict.values())) if mi_dict else 1.0

            G = nx.Graph()
            for (q0, q1), mi_val in mi_dict.items():
                G.add_edge(q0, q1, weight=mi_val)

            # Inject fake boundary nodes — same topology as the sampler.
            # Fake edge weights scale with avg_mi so the ffw ratio is preserved.
            G.add_edge(nl, nr, weight=ffw * avg_mi)
            for j in left_nodes:
                G.add_edge(nl, j, weight=avg_mi)
            for k in right_nodes:
                G.add_edge(nr, k, weight=avg_mi)

            matching = nx.max_weight_matching(G, maxcardinality=True, weight='weight')
            is_topo.append(frozenset({nl, nr}) in {frozenset(e) for e in matching})

        return is_topo, (left_nodes, right_nodes)

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

            first_faqe  = n_legit      # nl — connected to first shorter boundary
            second_faqe = n_legit + 1  # nr — connected to second shorter boundary
            coop.add_physical_qubit(first_faqe)
            coop.add_physical_qubit(second_faqe)

            coop.add_edge(first_faqe, second_faqe)

            # Connect fake qubits to the shorter boundary sides.
            # node = row * num_cols + col  (row-major indexing from CouplingMap.from_grid)
            # For square grids, _find_boundary_nodes picks top/bottom rows (sorted corner
            # (0,3) found before (0,12)); makeCouple must match that choice.
            if num_rows < num_cols:
                # Strictly shorter sides are left/right columns
                for row in range(num_rows):
                    coop.add_edge(first_faqe,  row * num_cols)
                    coop.add_edge(second_faqe, num_cols - 1 + row * num_cols)
            else:
                # num_rows > num_cols: strictly shorter top/bottom rows
                # num_rows == num_cols: square — matches _find_boundary_nodes (top/bottom rows)
                for col in range(num_cols):
                    coop.add_edge(first_faqe,  col)
                    coop.add_edge(second_faqe, (num_rows - 1) * num_cols + col)

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


