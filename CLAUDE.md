# CLAUDE.md — Quantum Awesomeness Bot & MWPM Pairing

## 1. Original Bot Logic (Quantum Awesomeness)

### What the bot is solving

Given a set of qubits and a connectivity graph, the bot must identify which qubits are **entangled pairs** based solely on measurement statistics. The game applies hidden CX-based entangling gates to a subset of adjacent qubit pairs, then the bot reads single-qubit measurement statistics and reconstructs the correct pairing.

### Data Structures

```python
# pairs dict — all possible pairings on the device (adjacency graph)
pairs = {'A': [1, 0], 'B': [2, 0], 'C': [2, 1], 'D': [4, 2], ...}
#         pair_name  [control_qubit, target_qubit]

# oneProb list — fraction of shots where each qubit measured |1⟩
oneProb = [0.11, 0.09, 0.49, 0.47, 0.31]   # one float per qubit

# sameProb dict — fraction of shots where both qubits gave the same bit
sameProb = {'A': 0.92, 'B': 0.51, 'C': 0.87, ...}   # one float per pair

# gates list — applied entangling gate fracs (hidden from the bot during play)
gates = [{'A': 0.42, 'C': 0.31}, {'A': -0.42, 'C': -0.31}, ...]
```

### Raw Bitstrings → `oneProb` and `sameProb`

```python
# QuantumAwesomeness.py:388-436
def processResults(resultsRaw, num, pairs, sim, shots):
    oneProb  = [0] * num
    sameProb = {p: 0 for p in pairs}

    # oneProb: accumulate fraction of |1⟩ outcomes per qubit
    for bitString in strings:
        for v in range(num):
            if bitString[v] == "1":
                oneProb[v] += results[bitString]

    # sameProb: accumulate fraction of shots where both qubits in a pair matched
    for bitString in strings:
        for p in pairs:
            if bitString[pairs[p][0]] == bitString[pairs[p][1]]:
                sameProb[p] += results[bitString]

    return oneProb, sameProb, results
```

---

## 2. The Three Metrics: `oneProb` vs `sameProb` vs Mutual Information

These three quantities are **not interchangeable** — they measure different things and are used in different parts of the bot. Understanding this distinction is critical for porting the logic to the new MirrorQA experiment.

---

### `oneProb` — marginal single-qubit statistic

**What it is:** `oneProb[q]` = P(qubit q measures |1⟩). Purely a per-qubit quantity; no pair information.

**What it captures physically:** After a `Rx(frac·π)` rotation followed by measurement, `P(|1⟩) = sin²(frac·π/2)`. If two qubits were entangled under the same hidden angle `frac`, they will both have similar `oneProb` values. The bot inverts this to reconstruct `frac`:

```python
# QuantumAwesomeness.py:541-556
def calculateFrac(oneProb):
    # P(|1⟩) = sin(frac·π/2)²  →  frac = asin(√p)·2/π
    oneProb = max(0, min(1, oneProb))
    frac = math.asin(math.sqrt(oneProb)) * 2 / math.pi
    return frac
```

**Pairwise weight from `oneProb`:** Two qubits with nearly equal `frac` are likely entangled partners. The MWPM weight is the *negative* angular distance (higher = better match):

```python
# QuantumAwesomeness.py:770-804
def calculateFracDifference(frac1, frac2):
    delta = max(frac1, frac2) - min(frac1, frac2)
    delta = min(delta, 1 - delta)   # frac=0 and frac=1 are equivalent (periodicity)
    return delta

# weight for edge (q0, q1):
weight[p] = -calculateFracDifference(
    calculateFrac(oneProb[pairs[p][0]]),
    calculateFrac(oneProb[pairs[p][1]])
)
```

**Where it is used in the bot:** Directly in `move='B'` (the primary bot mode). Only `oneProb` feeds the MWPM:

```python
# QuantumAwesomeness.py:945-946
if move == "B":
    guessedPairs = getDisjointPairs(pairs, oneProb, {})
```

**Limitation:** Relies on both qubits seeing the same hidden rotation angle. This is specific to the original game's circuit design. MirrorRB circuits do not have a single persistent `frac` per qubit — random Clifford sequences wash this out. **`oneProb`-based weighting cannot be ported directly to the new experiment.**

---

### `sameProb` — per-pair joint correlation (classical)

**What it is:** `sameProb[p]` = P(qubit q0 and qubit q1 give the same measurement outcome, whether "00" or "11").

**What it captures physically:** A crude classical correlation. If two qubits are maximally entangled in the Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2, then `sameProb = 1.0`. If they are uncorrelated, `sameProb ≈ 0.5`. **It discards the asymmetry** between "00" and "11" outcomes, and conflates states like |Φ+⟩ and |Ψ-⟩ = (|00⟩−|11⟩)/√2.

**Is it used directly for pairing?** No. `sameProb` is **never fed directly into the MWPM**. It is only used as input to `calculateMutual()`, which converts it to Mutual Information:

```python
# QuantumAwesomeness.py:631-648
def calculateMutual(oneProb, sameProb, pairs):
    I = {}
    for p in sameProb.keys():
        p0 = oneProb[pairs[p][0]]
        p1 = oneProb[pairs[p][1]]

        # Reconstruct the joint probability distribution {|00⟩, |01⟩, |10⟩, |11⟩}
        # from the marginals (p0, p1) and the correlation (sameProb[p])
        expect = calculateExpect([p0, p1, 1 - sameProb[p]])

        prob = [0] * 4
        prob[0] = (1 + expect[0] + expect[1] + expect[2]) / 4   # P(|00⟩)
        prob[1] = (1 - expect[0] + expect[1] - expect[2]) / 4   # P(|01⟩)
        prob[2] = (1 + expect[0] - expect[1] - expect[2]) / 4   # P(|10⟩)
        prob[3] = (1 - expect[0] - expect[1] + expect[2]) / 4   # P(|11⟩)

        # MI = H(marginal_q0) + H(marginal_q1) - H(joint)
        I[p] = (calculateEntropy([1 - p0, p0])
                + calculateEntropy([1 - p1, p1])
                - calculateEntropy(prob))

        # Normalize by the minimum marginal entropy
        if I[p] > 1e-3:
            I[p] = I[p] / min(calculateEntropy([1-p0, p0]), calculateEntropy([1-p1, p1]))
    return I
```

**Where it appears in the bot:** Only in the secondary `correlatedPairs` path (used for display, not move='B'):

```python
# QuantumAwesomeness.py:922-923
I = calculateMutual(oneProb, sameProb, pairs)
correlatedPairs = getDisjointPairs(pairs, [], I)
```

Here `I` (not `sameProb`) is passed as the `weight` dict to `getDisjointPairs()`.

---

### Mutual Information (MI) — the principled correlation metric

**What it is:**

```
MI(j, k) = H(Pⱼ) + H(Pₖ) − H(Pⱼ,ₖ)

where H = Shannon entropy (base 2),
Pⱼ    = marginal distribution of qubit j  → [P(0), P(1)]
Pₖ    = marginal distribution of qubit k  → [P(0), P(1)]
Pⱼ,ₖ  = joint distribution              → [P(00), P(01), P(10), P(11)]
```

**What it captures physically:** How much knowing one qubit's outcome reduces uncertainty about the other. `MI = 0` means the qubits are statistically independent. `MI = 1` means knowing one qubit perfectly predicts the other (maximum classical or quantum correlation). MI distinguishes cases that `sameProb` cannot — e.g., a |Ψ+⟩ = (|01⟩+|10⟩)/√2 state has `sameProb ≈ 0` but high MI.

**Original game implementation** (uses `sameProb` as a proxy to reconstruct the joint distribution, then computes MI — see `calculateMutual()` above).

**New experiment implementation** (computes MI directly from raw counts, no intermediate `sameProb`):

```python
# mirror_qa.py:259-286
def mutual_info(self, data: np.ndarray):
    mutual_infos = []
    for circ_data in data:
        counts = circ_data["counts"]
        shots  = sum(counts.values())

        # Build joint distribution per coupling-map edge
        p = {}
        for j, k in self._coupling_map:
            p[j, k] = {"00": 0, "01": 0, "10": 0, "11": 0}
            for string in counts:
                ss = string[-1 - j] + string[-1 - k]
                p[j, k][ss] += counts[string]
            for ss in p[j, k]:
                p[j, k][ss] /= shots

        # MI = H(marginal_j) + H(marginal_k) - H(joint)
        mi = {}
        for j, k in self._coupling_map:
            if j < k:
                ps_l = [p[j,k][b+"0"] + p[j,k][b+"1"] for b in ["0", "1"]]   # P(j)
                ps_r = [p[j,k]["0"+b] + p[j,k]["1"+b] for b in ["0", "1"]]   # P(k)
                mi[j, k] = -entropy(list(p[j, k].values()), base=2)           # −H(joint)
                for ps in [ps_l, ps_r]:
                    mi[j, k] += entropy(ps, base=2)                           # +H(marginals)
        mutual_infos.append(mi)

    return mutual_infos   # list[dict[(j,k) -> float]]
```

**Grouping MI by pair status** (paired vs cross-pair vs singles):

```python
# mirror_qa.py:288-308
def mean_mutual_info(self, data, pairs):
    mutual_infos = self.mutual_info(data)
    mean_mi = {"paired": [], "unpaired": [], "singles": []}

    for c, mi in enumerate(mutual_infos):
        all_paired = set(itertools.chain.from_iterable(pairs[c]))
        for pair, value in mi.items():
            if tuple(pair) in pairs[c] or tuple(pair[::-1]) in pairs[c]:
                mean_mi["paired"][-1].append(value)      # edge was an actual pair
            elif not set(pair).intersection(all_paired):
                mean_mi["singles"][-1].append(value)     # both qubits unpaired
            else:
                mean_mi["unpaired"][-1].append(value)    # cross-pair edge
        ...
    return mean_mi
    # {
    #   "paired":   [float per circuit]  — mean MI of actual pairs
    #   "unpaired": [float per circuit]  — mean MI for edges crossing pair boundaries
    #   "singles":  [float per circuit]  — mean MI for unentangled qubits (≈ NaN)
    # }
```

---

### Comparison Table

| Property | `oneProb` | `sameProb` | Mutual Information |
|----------|-----------|------------|-------------------|
| Granularity | Per qubit | Per pair | Per pair |
| Formula | P(\|1⟩) | P(q0 bit == q1 bit) | H(Pⱼ) + H(Pₖ) − H(Pⱼ,ₖ) |
| Captures | Single-qubit rotation angle | Classical same-outcome correlation | Full pairwise correlation (joint entropy) |
| Conflates | Nothing (qubit-level) | \|Φ+⟩ with \|Ψ-⟩ (same `sameProb`, diff phase) | Nothing — distinguishes all cases |
| Used for MWPM in original? | **Yes** (move='B' directly) | **No** (only as proxy to compute MI) | **Yes** (via `calculateMutual()` in `correlatedPairs`) |
| Portable to MirrorRB/MirrorQA? | **No** — frac assumption breaks | **No** — same limitation | **Yes** — formula works on any circuit's counts |
| Range | [0, 1] | [0, 1] | [0, 1] (normalized) |

---

## 3. New Experiment: MirrorQA + NewSampler (Topological Approach)

### Architecture

| File | Role |
|------|------|
| `mirror_rb_experiment.py` | Base MirrorRB — circuit generation, metadata, sampling dispatch |
| `mirror_qa.py` | MirrorQA — adds MI analysis on top of MirrorRB |
| `mirror_qa_topo.py` | MirrorQATopo — topology-aware subclass (minimal, inherits MirrorQA) |
| `sampling_utils.py::NewSampler` | Strict 2Q-1Q-2Q-1Q alternating sampler |
| `common.ipynb` | Comparison notebook: edge_grab / matching / new samplers on 4×4 grid |

### NewSampler Design

```python
# sampling_utils.py:520-579
class NewSampler(MatchingSampler):
    """Strict 2Q-1Q alternation.
    Even rounds (0,2,...): pure 2Q layer — CX on every max-matched edge, no 1Q dressing.
    Odd rounds (1,3,...): pure 1Q layer — SingleQubitSampler Cliffords on all qubits.
    Layer 0 is always 2Q, so after symmetric truncation outermost Clifford is always 2Q.
    """

    def __call__(self, qubits, length=1):
        for i in range(length):
            if i % 2 == 0:
                edges = self._2q._select_edges()  # full max-matching on topology
                yield tuple(GateInstruction(tuple(e), self._two_q_gate) for e in edges)
            else:
                yield from self._1q(qubits, 1)   # independent Clifford per qubit
```

### Circuit Metadata (Per-Circuit Pair Access)

```python
# mirror_rb_experiment.py:434-444
self._pairs = []
self._singles = []
for s, sequence in enumerate(sequences):
    self._pairs.append([])
    self._singles.append([])
    for gate in sequences[s][1]:        # layer [1] = first Clifford layer
        if len(gate.qargs) == 2:
            self._pairs[s].append(gate.qargs)       # 2-tuple (q0, q1)
        else:
            self._singles[s].append(gate.qargs[0])

# Access per-circuit pair assignment:
exp._pairs[0]   # → [(4,0), (10,11), (5,1), (13,9), (7,6), (8,12), (3,2), (14,15)]
exp._singles[0] # → []  (all 16 qubits paired when using NewSampler on 4×4)
```

### Accessing Pair Data — New vs Original

| Concept | Original (QA game) | New (MirrorQA) |
|---------|-------------------|----------------|
| Connectivity graph | `pairs` dict | `coupling_map` list of tuples |
| Raw counts | `resultsRaw` bitstring dict | `experiment_data.data()[i]["counts"]` |
| Per-qubit stat | `oneProb[q]` = P(\|1⟩) | Marginal: `p[j,k]["0b"] + p[j,k]["1b"]` |
| Per-pair correlation | `sameProb[p]` = P(same bit) | `p[j,k]["00"] + p[j,k]["11"]` |
| Per-pair score for MWPM | `−fracDifference(frac0, frac1)` | `MI(q0, q1)` from `mutual_info()` |
| Pair assignment per circuit | `gates[2*(round-1)].keys()` | `exp._pairs[circuit_idx]` |

---

## 4. Bringing MWPM to the New Bot

### Why MI is the right weight for the new bot

In MirrorRB circuits, random Clifford sequences overwrite any persistent single-qubit rotation angle, so `oneProb`-based frac matching no longer works. Mutual Information computed directly from the joint measurement distribution is valid for any circuit — it quantifies how much information the qubits share, regardless of the gate sequence used.

### Strategy

Build a weighted graph on the coupling map where edge weight = mean MI across all circuits that tested that pair. Run MWPM to find the globally optimal perfect matching.

### MI Score Aggregation Across Circuits

```python
from collections import defaultdict
import numpy as np

# Step 1: collect MI per edge across all circuits where that edge appeared as an active pair
edge_mi = defaultdict(list)
for i, pairs_in_circuit in enumerate(exp._pairs):
    mi_dict = qa.mutual_info(data[i])   # {(q0, q1): float}
    for (q0, q1) in pairs_in_circuit:
        key = (min(q0, q1), max(q0, q1))
        mi_val = mi_dict.get((q0, q1), mi_dict.get((q1, q0), float('nan')))
        edge_mi[key].append(mi_val)

# Step 2: average MI per edge
mean_mi_per_edge = {edge: np.nanmean(vals) for edge, vals in edge_mi.items()}
```

### MWPM Call (using `networkx`)

```python
import networkx as nx

def mwpm_from_mutual_info(coupling_map, mi_scores: dict) -> list:
    """
    coupling_map : list of (q0, q1) tuples — device adjacency
    mi_scores    : dict (q0, q1) → float  — mean MI per edge
    Returns      : list of (q0, q1) matched pairs
    """
    G = nx.Graph()
    for (q0, q1) in coupling_map:
        key = (min(q0,q1), max(q0,q1))
        mi  = mi_scores.get(key, 0.0)
        G.add_edge(q0, q1, weight=mi)   # higher MI → preferred pair

    matching = nx.max_weight_matching(G, maxcardinality=True, weight='weight')
    return list(matching)

best_pairs = mwpm_from_mutual_info(coupling_map, mean_mi_per_edge)
```

### MWPM Call (using original `mwmatching.py`, pure Python)

```python
import mwmatching as mw

def mwpm_from_mutual_info(coupling_map, mi_scores):
    edges = []
    for q0, q1 in coupling_map:
        key = (min(q0,q1), max(q0,q1))
        w   = mi_scores.get(key, 0.0)
        edges.append((q0, q1, w))

    match = mw.maxWeightMatching(edges, maxcardinality=True)
    return [(v, match[v]) for v in range(len(match)) if match[v] > v]
```

---

## 5. What Is the Blossom Algorithm?

The **Blossom algorithm** (Edmonds, 1965) solves the **Maximum Weight Perfect Matching** problem on general (non-bipartite) graphs in polynomial time.

### The problem

Given graph G = (V, E) with edge weights w(e), find a subset M ⊆ E such that:

- Every vertex in V is incident to exactly one edge in M (perfect matching)
- The total weight Σ w(e) for e ∈ M is maximized

### Why it's non-trivial

Bipartite matching (e.g., the Hungarian algorithm) fails on odd-length cycles. A **blossom** is an odd-length cycle. Edmonds' key insight: contract a blossom into a pseudo-node, solve on the contracted graph, then expand the solution back — preserving optimality.

### Key facts

| Property | Value |
|----------|-------|
| Time complexity | O(n³) where n = \|V\| |
| Space complexity | O(n²) |
| Graph type | General (non-bipartite) |
| Application here | Qubit pairing on arbitrary device topology |

### How `mwmatching.py` is called

```python
# Input: list of (node_i, node_j, weight) tuples — all candidate edges
edges = [(0, 1, 0.9), (0, 2, 0.3), (1, 3, 0.7), (2, 3, 0.8)]

match = mw.maxWeightMatching(edges, maxcardinality=True)
# match[i] = j  means node i is matched to node j
# maxcardinality=True forces as many matches as possible, even at lower total weight
```

### Why MWPM is the right algorithm for qubit pairing

The qubit coupling map is a sparse general graph (not bipartite) — device topologies like heavy-hex or square lattices contain odd cycles. Greedy approaches fail because a locally optimal pair can block globally optimal solutions. The Blossom algorithm guarantees the globally optimal perfect matching.

---

## 6. End-to-End Bot Logic: Original vs New

| Step | Original Bot (move='B') | New Bot (MirrorQA + NewSampler) |
|------|------------------------|---------------------------------|
| Circuit | Hidden CX(frac) on paired qubits, then measure | MirrorRB with random Cliffords, `initial_entangling_angle=π/2` |
| Raw data | `resultsRaw` = `{bitstring: probability}` | `experiment_data.data()[i]["counts"]` = `{bitstring: int}` |
| Metric | `oneProb[q]` → `frac` → `−fracDifference` | `MI(q0,q1)` from `mutual_info()` |
| Why this metric? | Entangled qubits share the same hidden `frac` | Entangled pairs have high joint entropy correlation |
| Weight sign | Negative distance (maximize closeness) | Positive MI (maximize correlation) |
| MWPM input | `edges = [(q0, q1, −fracDiff) ...]` | `edges = [(q0, q1, mean_MI) ...]` |
| MWPM call | `mw.maxWeightMatching(edges, maxcardinality=True)` | Same, or `nx.max_weight_matching` |
| Output | `matchingPairs` list of pair name strings | List of `(q0, q1)` integer tuples |
