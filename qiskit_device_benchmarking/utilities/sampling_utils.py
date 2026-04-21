# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Utilities for sampling layers in randomized benchmarking experiments
"""

import warnings
import math
from abc import ABC, abstractmethod
from typing import Optional, Union, List, Tuple, Sequence, NamedTuple, Dict, Iterator
from collections import defaultdict
from numpy.random import Generator, default_rng, BitGenerator, SeedSequence
import numpy as np
import networkx as nx

from qiskit.circuit import Instruction
from qiskit.circuit.gate import Gate
from qiskit.exceptions import QiskitError
from qiskit.transpiler import CouplingMap

from qiskit_experiments.library.randomized_benchmarking.clifford_utils import (
    CliffordUtils,
    _CLIFF_SINGLE_GATE_MAP_1Q,
)


class GenericClifford(Gate):
    """Representation of a generic multi-qubit Clifford gate for sampling."""

    def __init__(self, n_qubits):
        super().__init__("generic_clifford", n_qubits, [])


class GenericPauli(Gate):
    """Representation of a generic multi-qubit Pauli gate for sampling."""

    def __init__(self, n_qubits):
        super().__init__("generic_pauli", n_qubits, [])


class GateInstruction(NamedTuple):
    """Named tuple class for sampler output."""

    # the list of qubits to apply the operation on
    qargs: tuple
    # the operation to apply
    op: Instruction


class GateDistribution(NamedTuple):
    """Named tuple class for sampler input distribution."""

    # probability with which to sample the instruction
    prob: float
    # the instruction to include in sampling
    op: Instruction


class BaseSampler(ABC):
    """Base class for samplers that generate circuit layers based on a defined
    algorithm and gate set. Subclasses must implement the ``__call__()`` method
    which outputs a number of circuit layers."""

    def __init__(
        self,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
    ) -> None:
        """Initializes the sampler.

        Args:
            seed: Seed for random generation.
            gate_distribution: The gate distribution for sampling.
        """
        self.seed = seed

    @property
    def seed(self) -> Union[int, SeedSequence, BitGenerator, Generator]:
        """The seed for random generation."""
        return self._rng

    @seed.setter
    def seed(self, seed) -> None:
        self._rng = default_rng(seed)

    @property
    def gate_distribution(self) -> List[GateDistribution]:
        """The gate distribution for sampling. The distribution is a list of
        ``GateDistribution`` named tuples with field names ``(prob, op)``, where
        the probabilites must sum to 1 and ``op`` is the Instruction instance to
        be sampled. An example distribution for the edge grab sampler is

        .. parsed-literal::
            [(0.8, GenericClifford(1)), (0.2, CXGate())]
        """
        return self._gate_distribution

    @gate_distribution.setter
    def gate_distribution(self, dist: List[GateDistribution]) -> None:
        """Set the distribution of gates used in the sampler.

        Args:
            dist: A list of tuples with format ``(probability, gate)``.

        Raises:
            TypeError: If the gate distribution format is invalid.
            QiskitError: If the gate distribution probabilities don't sum to 1.
        """
        # cast to named tuple
        try:
            dist = [GateDistribution(*elem) for elem in dist]
        except TypeError as exc:
            raise TypeError(
                "The gate distribution should be a sequence of (prob, op) tuples."
            ) from exc
        if sum(list(zip(*dist))[0]) != 1:
            raise QiskitError("Gate distribution probabilities must sum to 1.")
        for gate in dist:
            if not isinstance(gate.op, Instruction):
                raise TypeError(
                    "The only allowed gates in the distribution are Instruction instances."
                )
        self._gate_distribution = dist

    def _probs_by_gate_size(self, distribution: Sequence[GateDistribution]) -> Dict:
        """Converts gate distribution into an easy-to-use format.

        Returns:
            A dictionary of gates and their probabilities indexed by the size of the gate

        Raises:
            QiskitError: If Cliffords or Paulis are too large."""

        gate_probs = defaultdict(list)

        for gate in distribution:
            if gate.op.name == "generic_clifford":
                if gate.op.num_qubits == 1:
                    gateset = list(range(CliffordUtils.NUM_CLIFFORD_1_QUBIT))
                    probs = [
                        gate.prob / CliffordUtils.NUM_CLIFFORD_1_QUBIT
                    ] * CliffordUtils.NUM_CLIFFORD_1_QUBIT
                elif gate.op.num_qubits == 2:
                    gateset = list(range(CliffordUtils.NUM_CLIFFORD_2_QUBIT))
                    probs = [
                        gate.prob / CliffordUtils.NUM_CLIFFORD_2_QUBIT
                    ] * CliffordUtils.NUM_CLIFFORD_2_QUBIT
                else:
                    raise QiskitError(
                        "Generic Cliffords larger than 2-qubit are not currently supported."
                    )
            elif gate.op.name == "generic_pauli":
                if gate.op.num_qubits == 1:
                    gateset = [
                        _CLIFF_SINGLE_GATE_MAP_1Q[("id", (0,))],
                        _CLIFF_SINGLE_GATE_MAP_1Q[("x", (0,))],
                        _CLIFF_SINGLE_GATE_MAP_1Q[("y", (0,))],
                        _CLIFF_SINGLE_GATE_MAP_1Q[("z", (0,))],
                    ]
                    probs = [gate.prob / len(gateset)] * len(gateset)
                else:
                    raise QiskitError(
                        "Generic Paulis larger than 1-qubit are not currently supported."
                    )
            else:
                gateset = [gate.op]
                probs = [gate.prob]
            if len(gate_probs[gate.op.num_qubits]) == 0:
                gate_probs[gate.op.num_qubits] = [gateset, probs]
            else:
                gate_probs[gate.op.num_qubits][0].extend(gateset)
                gate_probs[gate.op.num_qubits][1].extend(probs)
        return gate_probs

    @abstractmethod
    def __call__(
        self, qubits: Sequence, length: int = 1
    ) -> Iterator[Tuple[GateInstruction, ...]]:
        """Samplers should define this method such that it returns sampled layers
        given the input parameters. Each layer is represented by a list of
        ``GateInstruction`` namedtuples, where ``GateInstruction.op`` is the gate to be
        applied and ``GateInstruction.qargs`` is the tuple of qubit indices to
        apply the gate to.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The number of layers to generate. Defaults to 1.

        Returns:
            A generator of layers consisting of GateInstruction objects.
        """
        raise NotImplementedError


class SingleQubitSampler(BaseSampler):
    """A sampler that samples layers of random single-qubit gates from a specified gate set."""

    @BaseSampler.gate_distribution.setter
    def gate_distribution(self, dist: List[GateDistribution]) -> None:
        """Set the distribution of gates used in the sampler.

        Args:
            dist: A list of tuples with format ``(probability, gate)``.

        Raises:
            QiskitError: If the distribution isn't all single-qubit gates.
        """
        super(SingleQubitSampler, type(self)).gate_distribution.fset(self, dist)

        gateset = self._probs_by_gate_size(self.gate_distribution)
        if not math.isclose(sum(gateset[1][1]), 1):
            raise QiskitError(
                "The distribution for SingleQubitSampler should be all single qubit gates."
            )

    def __call__(
        self,
        qubits: Sequence,
        length: int = 1,
    ) -> Iterator[Tuple[GateInstruction]]:
        """Samples random single-qubit gates from the specified gate set. The
        input gate distribution must consist solely of single qubit gates.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The length of the sequence to output.

        Yields:
            A ``length``-long iterator of :class:`qiskit.circuit.QuantumCircuit`
            layers over ``qubits``. Each layer is represented by a list of
            ``GateInstruction`` tuples where ``GateInstruction.op`` is the gate
            to be applied and ``GateInstruction.qargs`` is the tuple of qubit
            indices to apply the gate to. Single-qubit Cliffords are represented
            by integers for speed.
        """

        gateset = self._probs_by_gate_size(self._gate_distribution)

        samples = self._rng.choice(
            np.array(gateset[1][0], dtype=object),
            size=(length, len(qubits)),
            p=gateset[1][1],
        )

        for samplelayer in samples:
            yield tuple(
                GateInstruction(*ins)
                for ins in zip(((j,) for j in qubits), samplelayer)
            )


class EdgeGrabSampler(BaseSampler):
    r"""A sampler that uses the edge grab algorithm [1] for sampling gate layers.

    The edge grab sampler, given a list of :math:`w` qubits, their connectivity
    graph, and the desired two-qubit gate density :math:`\xi_s`, outputs a layer
    as follows:

    1. Begin with the empty set :math:`E` and :math:`E_r`, the set of all edges
    in the connectivity graph. Select an edge from :math:`E_r` at random and
    add it to :math:`E`, removing all edges that share a qubit with the edge
    from :math:`E_r`.

    2. Select edges from :math:`E` with the probability :math:`w\xi/2|E|`. These
    edges will have two-qubit gates in the output layer.

    This produces a layer with an expected two-qubit gate density :math:`\xi`. In the
    default mirror RB configuration where these layers are dressed with single-qubit
    Pauli layers, this means the overall expected two-qubit gate density will be
    :math:`\xi_s/2=\xi`. The actual density will converge to :math:`\xi_s` as the
    circuit size increases.

    .. ref_arxiv:: 1 2008.11294

    """

    @BaseSampler.gate_distribution.setter
    def gate_distribution(self, dist: List[GateDistribution]) -> None:
        """Set the distribution of gates used in the sampler.

        Args:
            dist: A list of tuples with format ``(probability, gate)``.

        Raises:
            QiskitError: If invalid gates are specified.
        """
        super(EdgeGrabSampler, type(self)).gate_distribution.fset(self, dist)

        gateset = self._probs_by_gate_size(self.gate_distribution)

        try:
            norm1q = sum(gateset[1][1])
            norm2q = sum(gateset[2][1])
        except IndexError as exc:
            raise QiskitError(
                "The edge grab sampler requires 1-qubit and 2-qubit gates to be specified."
            ) from exc
        if not np.isclose(norm1q + norm2q, 1):
            raise QiskitError(
                "The edge grab sampler only supports 1- and 2-qubit gates."
            )

    def __init__(
        self,
        gate_distribution=None,
        coupling_map: Optional[Union[List[List[int]], CouplingMap]] = None,
        seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
        matching=False,
    ) -> None:
        """Initializes the sampler.

        Args:
            seed: Seed for random generation.
            gate_distribution: The gate distribution for sampling.
            coupling_map: The coupling map between the qubits.
        """
        super().__init__(seed)
        self._gate_distribution = gate_distribution
        self.coupling_map = coupling_map
        self._matching = matching

    @property
    def coupling_map(self) -> CouplingMap:
        """The coupling map of the system to sample over."""
        return self._coupling_map

    @coupling_map.setter
    def coupling_map(self, coupling_map: Union[List[List[int]], CouplingMap]) -> None:
        try:
            self._coupling_map = CouplingMap(coupling_map)
        except (ValueError, TypeError) as exc:
            raise TypeError("Invalid coupling map provided.") from exc

    def _select_edges(self) -> List:
        """Select a set of non-overlapping edges from the coupling map.

        Uses the edge grab algorithm (random greedy) when ``self._matching``
        is False, or maximum weight matching when True.

        Returns:
            A list of ``(qubit_a, qubit_b)`` edge tuples.
        """
        if not self._matching:
            all_edges = self.coupling_map.get_edges()[:]
            selected_edges = []
            while all_edges:
                rand_edge = all_edges.pop(self._rng.integers(len(all_edges)))
                selected_edges.append(rand_edge)
                old_all_edges = all_edges[:]
                all_edges = []
                for edge in old_all_edges:
                    if rand_edge[0] not in edge and rand_edge[1] not in edge:
                        all_edges.append(edge)
        else:
            G = nx.Graph()
            edges = self.coupling_map.get_edges()
            for u, v in edges:
                w = self._rng.integers(1, 100)
                G.add_edge(u, v, weight=w)
            matching = nx.algorithms.matching.max_weight_matching(
                G, maxcardinality=True
            )
            selected_edges = []
            for edge in matching:
                if edge in edges:
                    selected_edges.append(edge)
                else:
                    selected_edges.append(edge[::-1])
        return selected_edges

    def __call__(
        self,
        qubits: Sequence,
        length: int = 1,
    ) -> Iterator[Tuple[GateInstruction]]:
        """Sample layers using the edge grab algorithm.

        Args:
            qubits: A sequence of qubits to generate layers for.
            length: The length of the sequence to output.

        Raises:
            Warning: If the coupling map has no connectivity or
                ``two_qubit_gate_density`` is too high.
            TypeError: If invalid gate set(s) are specified.
            QiskitError: If the coupling map is invalid.

        Yields:
            A ``length``-long iterator of :class:`qiskit.circuit.QuantumCircuit`
            layers over ``num_qubits`` qubits. Each layer is represented by a
            list of ``GateInstruction`` named tuples which are in the format
            (qargs, gate). Single-qubit Cliffords are represented by integers
            for speed. Here's an example with the default choice of Cliffords
            for the single-qubit gates and CXs for the two-qubit gates:

            .. parsed-literal::
                (((1, 2), CXGate()), ((0,), 12), ((3,), 20))

            This represents a layer where the 12th Clifford is performed on qubit 0,
            a CX is performed with control qubit 1 and target qubit 2, and the 20th
            Clifford is performed on qubit 3.

        """
        num_qubits = len(qubits)
        gateset = self._probs_by_gate_size(self._gate_distribution)
        norm1q = sum(gateset[1][1])
        norm2q = sum(gateset[2][1])

        two_qubit_gate_density = norm2q / (norm1q + norm2q)

        for _ in range(length):
            selected_edges = self._select_edges()

            two_qubit_prob = 0
            try:
                # need to divide by 2 since each two-qubit gate spans two lattice sites
                two_qubit_prob = (
                    num_qubits * two_qubit_gate_density / 2 / len(selected_edges)
                )
            except ZeroDivisionError:
                warnings.warn(
                    "Device has no connectivity. All gates will be single-qubit."
                )
            if two_qubit_prob > 1 and not np.isclose(two_qubit_prob, 1):
                warnings.warn(
                    "Mean number of two-qubit gates is higher than the number of selected edges. "
                    + "Actual density of two-qubit gates will likely be lower than input density."
                )

            put_1q_gates = set(qubits)
            # put_1q_gates is a list of qubits that aren't assigned to a 2-qubit gate
            # 1-qubit gates will be assigned to these edges
            layer = []
            for edge in selected_edges:
                if self._rng.random() < two_qubit_prob:
                    # with probability two_qubit_prob, place a two-qubit gate from the
                    # gate set on edge in selected_edges
                    if len(gateset[2][0]) == 1:
                        layer.append(GateInstruction(tuple(edge), gateset[2][0][0]))
                    else:
                        layer.append(
                            GateInstruction(
                                tuple(edge),
                                self._rng.choice(
                                    np.array(gateset[2][0], dtype=Instruction),
                                    p=[x / norm2q for x in gateset[2][1]],
                                ),
                            ),
                        )
                    if not self._matching:
                        # remove these qubits from put_1q_gates
                        put_1q_gates.remove(edge[0])
                        put_1q_gates.remove(edge[1])
            for q in put_1q_gates:
                if sum(gateset[1][1]) > 0:
                    layer.append(
                        GateInstruction(
                            (q,),
                            self._rng.choice(
                                np.array(gateset[1][0], dtype=Instruction),
                                p=[x / norm1q for x in gateset[1][1]],
                            ),
                        ),
                    )
                else:  # edge case of two qubit density of 1 where we still fill gaps
                    layer.append(
                        GateInstruction(
                            (q,),
                            self._rng.choice(
                                np.array(gateset[1][0], dtype=Instruction)
                            ),
                        ),
                    )
            yield tuple(layer)

class MatchingSampler(EdgeGrabSampler):
    r"""A sampler that uses maximum weight matching for sampling gate layers.

    Given a list of :math:`w` qubits, their connectivity graph, and the desired
    two-qubit gate density :math:`\xi_s`, this algorithm outputs a layer as follows:

    1. Perform a maximum weight matching :math:`E` of the graph defined by the
    coupling map, using randomly chosen edges to acheive a random result.

    2. Select edges from :math:`E` with the probability :math:`w\xi/2|E|`. These
    edges will have two-qubit gates in the output layer.

    3. Random single qubit gates are then assigned to all qubits.

    This produces a layer with an expected two-qubit gate density :math:`\xi`. In the
    default mirror RB configuration where these layers are dressed with single-qubit
    Pauli layers, this means the overall expected two-qubit gate density will be
    :math:`\xi_s/2=\xi`. The actual density will converge to :math:`\xi_s` as the
    circuit size increases.

    """

    def __init__(
        self,
        gate_distribution=None,
        coupling_map=None,
        seed=None,
    ):
        super().__init__(
            gate_distribution=gate_distribution,
            coupling_map=coupling_map,
            seed=seed,
            matching=True,
        )

class NewSampler(MatchingSampler):
    """Produces a strict 2Q-1Q-2Q-1Q-... alternating pattern.

    Even rounds (0, 2, ...): pure 2Q layer — CX on every max-matched edge, no 1Q dressing.
    Odd rounds (1, 3, ...): pure 1Q layer — SingleQubitSampler Cliffords on all qubits.
    Layer 0 is always 2Q, so after symmetric truncation the outermost Clifford is always 2Q.
    """

    def __init__(self, seed=None, **kwargs):
        self._2q = MatchingSampler(seed=seed, **kwargs)  # used only for _select_edges()
        self._1q = SingleQubitSampler(seed=seed)
        self._two_q_gate = None

    @property
    def seed(self):
        return self._2q.seed

    @seed.setter
    def seed(self, val):
        self._2q.seed = val
        self._1q.seed = val

    @property
    def coupling_map(self):
        return self._2q.coupling_map

    @coupling_map.setter
    def coupling_map(self, val):
        self._2q.coupling_map = val
        # SingleQubitSampler needs no coupling map

    @property
    def gate_distribution(self):
        return self._2q.gate_distribution

    @gate_distribution.setter
    def gate_distribution(self, dist):
        two_q_gate = None
        one_q_gate = GenericClifford(1)
        for d in dist:
            gd = GateDistribution(*d) if not isinstance(d, GateDistribution) else d
            if gd.op.num_qubits == 2:
                two_q_gate = gd.op
            elif gd.op.num_qubits == 1:
                one_q_gate = gd.op
        self._two_q_gate = two_q_gate
        # Pass the original dist to _2q so its EdgeGrabSampler setter validation
        # passes (requires both 1Q and 2Q gates). We only call _2q._select_edges().
        self._2q.gate_distribution = dist
        self._1q.gate_distribution = [GateDistribution(1.0, one_q_gate)]

    def __call__(self, qubits, length=1):
        for i in range(length):
            if i % 2 == 0:
                # Pure 2Q layer: CX on ALL matched edges, no 1Q dressing
                edges = self._2q._select_edges()
                yield tuple(GateInstruction(tuple(e), self._two_q_gate) for e in edges)
            else:
                # Pure 1Q layer: independent Clifford on every qubit
                yield from self._1q(qubits, 1)

class TopoSampler(NewSampler):
    """Like NewSampler, this also produces 2q-1q-2q-1q-... pattern.
    
    This sampler's goal is making the increasing depth of collectible circuits
    that contains multiple layers, but if they are odd indices, they have
    1q operations only. For the even indices, they have 2q gates only, and,
    like NewSampler, its connectivity must be fully used, which means that all
    possible pairs must be selected. e.g. 4x4 -> 8 pairs in the outermost.
    
    However, TopoSampler is the advanced version of NewSampler.
    Now, it will contain false qubits and their connections toward the edge of
    left and right side of the device map. This one will intentionally exclude
    None (full matching) or one qubit on the edge, making the fake connection.
    
    c.f. That connection will be ignored throughout the ._pairs.
    
    **Modes**
        - 'g2g' (genuine to genuine connection): MWPM on 'true' qubit only.
        Eventually the same as NewSampler. All n / 2 pairs will store in ._pairs.
        - 'f2f' (fa'q'e to faqe connection): Original square lattice + two fake qubits
        It will include the fake qubits' connection along with 'g2g', but it won't
        be included in ._pairs.
        - 'f2g' (faqe to genuine): MWPM will automatically connect all pairs possible
        based on the coupling map including fake qubits connected to L/R boundary.
        Abandoned(?) genuine qubits will be either left on left or right boundary.
    """
    
    def __init__(self, legit, mode='g2g', seed=None, **kwargs):
        super().__init__(seed=seed, **kwargs)
        if mode not in ('g2g', 'f2f', 'f2g'):
            raise ValueError(f'Check the documentation to set the correct mode.')
        self.legit = legit
        self.mode = mode
        self._nl = None
        self._nr = None
        self._left_nodes = None
        self._right_nodes = None

    @staticmethod
    def _graph2cmap(g, rng):
        """ Change the coupling maps to the graph for sampling.
        Not only change it but also add fake qubits (faqe) to the graph
        so that behind-the-scene tricks can happen.
        """
        corners = [n for n in g.nodes if g.degree(n) == 2]
        if len(corners) != 4:
            raise ValueError(f'Coupling map is not a square lattice with even number of qubits.')
        
        corners_dist = {}
        for j in range(len(corners)):
            for k in range(j + 1, len(corners)):
                corners_dist[corners[j], corners[k]] = nx.dijkstra_path_length(
                    g, corners[j], corners[k]
                )
        corner_dist = {
            ns: d
            for ns, d in corners_dist.items()
            if max(corners_dist, key=corners_dist.get)[0] in ns
            }
        min_d = min(corner_dist.values())
        candidates = [pair for pair, d in corner_dist.items() if d == min_d]
        
        # determine the left boundary qubits
        left_corners = candidates[0]
        for pair in candidates:
            path = nx.dijkstra_path(g, pair[0], pair[1])
            diffs = [abs(path[i + 1] - path[i]) for i in range(len(path) - 1)]
            if max(diffs) > 1: # vertical edge -> column path
                left_corners = pair
                break
        
        right_corners = tuple(n for n in corners if n not in left_corners)
        
        left_nodes = nx.dijkstra_path(g, left_corners[0], left_corners[1])
        right_nodes = nx.dijkstra_path(g, right_corners[0], right_corners[1])
        
        nl = max(g.nodes) + 1 # faqe on the left side
        nr = nl + 1 # faqe on the right side
        
        for n in left_nodes:
            g.add_edge(nl, n, weight=int(rng.integers(0, 101)))
        for n in right_nodes:
            g.add_edge(nr, n, weight=int(rng.integers(0, 101)))
        
        return nl, nr, left_nodes, right_nodes
    
    def _select_edges_topo(self):
        all_edges = self._2q.coupling_map.get_edges()
        all_edges_set = set(map(tuple, all_edges))
        legit_edges = [(u, v) for u, v in all_edges
                       if u < self.legit and v < self.legit]
        
        G = nx.Graph()
        seen = set()
        
        for u, v in legit_edges:
            key = (min(u, v), max(u, v))
            if key not in seen:
                seen.add(key)
                G.add_edge(u, v)
        
        if self.mode == 'f2g': 
            self._nl, self._nr, self._left_nodes, self._right_nodes = self._graph2cmap(G, self._2q._rng)
        elif self.mode == 'f2f':
            G_tmp = G.copy()
            _, _, self._left_nodes, self._right_nodes = self._graph2cmap(G_tmp, self._2q._rng)
        
        for u, v, d in G.edges(data=True):
            if u < self.legit and v < self.legit:
                d['weight'] = int(self._2q._rng.integers(0, 101))
        matching = nx.max_weight_matching(G, maxcardinality=True)
        
        return [(u, v) if (u, v) in all_edges_set else (v, u)
                for u, v in matching
                if u< self.legit and v < self.legit]
    
    def __call__(self, qubits, length=1):
        legits = [q for q in qubits if q < self.legit]
        for i in range(length):
            if i % 2 == 0:
                edges = self._select_edges_topo()
                yield tuple(GateInstruction(tuple(e), self._two_q_gate) for e in edges)
            else:
                yield from self._1q(legits, 1)
        