# An topological, non-local approach to our `mirror-qa` and `mirror-qb`

The `topo` branch is introduced to store codes for `Mirror Quantum Awesomeness`' topological approacth to quantum device benchmarking. Its goals are two.
- Use the square lattice for benchmarking (Rigetti for the most)
- Upgrade the algorithm to track global error rate and 'topologically-protected' information
  
The process is this: Until now, we focused on making as many random pairs as possible across lattices. There are multiple ways that I can pair them. But in topological approach, we would like to pair things EXCEPT the things we don't want to pair.

After almost everything is paired up, the benchmarking algorithm's goal is to find the qubit or qubits which is/are abandoned. They are information that we must discover. Those qubits always locate either side of a lattice boundary. Find them. (Give the lattice a choice of leaving two qubits on the either side of the boundary. Or WE can choose which indices will we exclude. We must deal with automatically-changing qubit indices.)

c.f. Definition of 'making pairs' is the process of selecting graph edges randomly.

### Installation

```
git clone git@github.com:qiskit-community/qiskit-device-benchmarking.git
cd qiskit-device-benchmarking
pip install .
```

### Run Tests

```
pip install pytest
pytest
```

### Lint

```
pip install ruff
ruff check      # Lint files
ruff format     # Format files
```

# License

[Apache License 2.0](LICENSE.txt)

# Acknowledgements

Portions of the code in this repository was developed via sponsorship by the Army Research Office ``QCISS Program'' under Grant Number W911NF-21-1-0002. 
