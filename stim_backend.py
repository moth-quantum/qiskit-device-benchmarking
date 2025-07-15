from qiskit import transpile, QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit.providers import JobV1
from qiskit.providers.jobstatus import JobStatus
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import *

from qiskit_aer.noise import pauli_error

import uuid


def bitflip(p):
    return pauli_error(list(zip(['I', 'X'],[1-p, p])))

def depol_1(p):
    return pauli_error(list(zip(['I', 'X', 'Y', 'Z'],[1-3*p/4, p/4, p/4, p/4])))

def depol_2(p):
    return pauli_error(list(zip(
            [
                'II', 'IX', 'IY', 'IZ',
                'XI', 'XX', 'XY', 'XZ',
                'YI', 'YX', 'YY', 'YZ',
                'ZI', 'ZX', 'ZY', 'ZZ'
            ],
            [1-15*p/16] + [p/16]*15
        )))

def get_results(qcs, counts_list):
    if isinstance(qcs, QuantumCircuit):
        qcs = [qcs]
    if isinstance(counts_list, Dict):
        counts_list = [counts_list]  
    experiment_results = []
    for j, counts in enumerate(counts_list):
        shots = sum(counts.values())
        data = ExperimentResultData(counts=counts, quasi_dists={key: val / shots for key, val in counts.items()})
        result = ExperimentResult(
            shots=sum(counts.values()),
            success=True,
            status='DONE',
            meas_level=2,
            memory=None,
            data=data,
            header={'name': qcs[j].name}
    )
    experiment_results.append(result)
    result = Result(
        backend_name='stim_simulator',
        backend_version='1.0',
        qobj_id=str(uuid.uuid4()),
        job_id=str(uuid.uuid4()),
        success=True,
        results=experiment_results
    )
    return result

class StimJob(JobV1):
    def __init__(self, backend, job_id, result):
        super().__init__(backend, job_id)
        self._result = result

    def result(self):
        return self._result

    def status(self):
        return JobStatus.DONE

    def submit(self):
        pass

class NoisyStimBackend(GenericBackendV2):
    def __init__(
        self,
        num_qubits: int,
        basis_gates: list[str] | None = None,
        coupling_map: list[list[int]] = None,
        p1: float = 0,
        p2: float = 0,
        pm: float = 0,
    ):
        
        self.p = (p1,p2,pm)

        if coupling_map is None:
            coupling_map = []
            for j in range(num_qubits):
                for k in range(j+1, num_qubits):
                    coupling_map.append((j,k))

        one_qubit_props = {(q,): InstructionProperties() for q in range(num_qubits)}
        two_qubit_props = {}
        for pair in coupling_map:
            two_qubit_props[tuple(pair)] = InstructionProperties()

        target = Target(num_qubits=num_qubits)
        target.add_instruction(Measure(), one_qubit_props)
        if 'id' in basis_gates:
            target.add_instruction(IGate(), one_qubit_props)
        if 'x' in basis_gates:
            target.add_instruction(XGate(), one_qubit_props)
        if 'y' in basis_gates:
            target.add_instruction(YGate(), one_qubit_props)
        if 'z' in basis_gates:
            target.add_instruction(ZGate(), one_qubit_props)
        if 'h' in basis_gates:
            target.add_instruction(HGate(), one_qubit_props)
        if 's' in basis_gates:
            target.add_instruction(SGate(), one_qubit_props)
        if 'sdg' in basis_gates:
            target.add_instruction(SdgGate(), one_qubit_props)
        if 'cx' in basis_gates:
            target.add_instruction(CXGate(), two_qubit_props)
        if 'cz' in basis_gates:
            target.add_instruction(CZGate(), two_qubit_props)

        super().__init__(
            num_qubits,
            basis_gates,
            coupling_map=coupling_map,
            noise_info = (p1>0 or p2>0 or pm>0)
            )
        
        self._target = target

    @property
    def target(self):
        return self._target
            
    def _noisify_circuits(self, qcs):

        if isinstance(qcs, QuantumCircuit):
            qcs = [qcs]
        
        noisy_circuits = []
        for qc in qcs:
            noisy_qc = QuantumCircuit()
            for qreg in qc.qregs:
                noisy_qc.add_register(qreg)
            for creg in qc.cregs:
                noisy_qc.add_register(creg)

            for gate in qc:
                # add gate before the error if it is a reset
                if gate.name!='barrier':
                    if gate.operation.name=="reset":
                        noisy_qc.append(gate)
                    # then the error
                    if gate.operation.name=="measure":
                        noisy_qc.append(bitflip(self.p[2]), gate.qubits)
                    else:
                        if len(gate.qubits)==1:
                            noisy_qc.append(depol_1(self.p[0]), gate.qubits)
                        elif len(gate.qubits)==2:
                            noisy_qc.append(depol_2(self.p[1]), gate.qubits)
                    # add gate after the error if not a reset
                    if not gate.operation.name=="reset":
                        noisy_qc.append(gate)
            noisy_circuits.append(noisy_qc)

        if len(qcs)==1:
            noisy_circuits = noisy_circuits[0]

        return noisy_circuits

    def run(self, qcs, **opts):
        qcs = transpile(qcs, self)
        qcs = self._noisify_circuits(qcs)
        counts = get_counts_via_stim(qcs, shots=int(opts['shots']))
        result = get_results(qcs, counts)
        job_id = str(uuid.uuid4())
        return StimJob(self, job_id, result)


# Adapted from https://github.com/qiskit-community/qiskit-qec/edit/main/src/qiskit_qec/utils/stim_tools.py

# -*- coding: utf-8 -*-

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

# pylint: disable=invalid-name, disable=no-name-in-module, disable=unused-argument

"""Tools to use functionality from Stim."""
from typing import Union, List, Dict
from stim import Circuit as StimCircuit
from stim import target_rec as StimTarget_rec

import numpy as np
import rustworkx as rx

from qiskit import QuantumCircuit
from qiskit_aer.noise.errors.base_quantum_error import QuantumChannelInstruction

def get_stim_circuits(
    circuit: Union[QuantumCircuit, List],
    detectors: List[Dict] = None,
    logicals: List[Dict] = None,
):
    """Converts compatible qiskit circuits to stim circuits.
       Dictionaries are not complete. For the stim definitions see:
       https://github.com/quantumlib/Stim/blob/main/doc/gates.md

    Args:
        circuit: Compatible gates are Paulis, controlled Paulis, h, s,
        and sdg, swap, reset, measure and barrier. Compatible noise operators
        correspond to a single or two qubit pauli channel.
        detectors: A list of measurement comparisons. A measurement comparison
        (detector) is either a list of measurements given by a the name and index
        of the classical bit or a list of dictionaries, with a mandatory clbits
        key containing the classical bits. A dictionary can contain keys like
        'qubits', 'time', 'basis' etc.
        logicals: A list of logical measurements. A logical measurement is a
        list of classical bits whose total parity is the logical eigenvalue.
        Again it can be a list of dictionaries.

    Returns:
        stim_circuits, stim_measurement_data
    """

    if detectors is None:
        detectors = [{}]
    if logicals is None:
        logicals = [{}]

    if len(detectors) > 0 and isinstance(detectors[0], List):
        detectors = [{"clbits": det, "qubits": [], "time": 0} for det in detectors]

    if len(logicals) > 0 and isinstance(logicals[0], List):
        logicals = [{"clbits": log} for log in logicals]

    stim_circuits = []
    stim_measurement_data = []
    if isinstance(circuit, QuantumCircuit):
        circuit = [circuit]
    for circ in circuit:
        stim_circuit = StimCircuit()

        qiskit_to_stim_dict = {
            "id": "I",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "h": "H",
            "s": "S",
            "sdg": "S_DAG",
            "cx": "CX",
            "cy": "CY",
            "cz": "CZ",
            "swap": "SWAP",
            "reset": "R",
            "measure": "M",
            "barrier": "TICK",
        }
        pauli_error_1_stim_order = {
            "id": 0,
            "I": 0,
            "X": 1,
            "x": 1,
            "Y": 2,
            "y": 2,
            "Z": 3,
            "z": 3,
        }
        pauli_error_2_stim_order = {
            "II": 0,
            "IX": 1,
            "IY": 2,
            "IZ": 3,
            "XI": 4,
            "XX": 5,
            "XY": 6,
            "XZ": 7,
            "YI": 8,
            "YX": 9,
            "YY": 10,
            "YZ": 11,
            "ZI": 12,
            "ZX": 13,
            "ZY": 14,
            "ZZ": 15,
        }

        measurement_data = []
        qreg_offset = {}
        creg_offset = {}
        prevq_offset = 0
        prevc_offset = 0

        for instruction in circ.data:
            inst = instruction.operation
            qargs = instruction.qubits
            cargs = instruction.clbits
            for qubit in qargs:
                if qubit._register.name not in qreg_offset:
                    qreg_offset[qubit._register.name] = prevq_offset
                    prevq_offset += qubit._register.size
            for bit in cargs:
                if bit._register.name not in creg_offset:
                    creg_offset[bit._register.name] = prevc_offset
                    prevc_offset += bit._register.size

            qubit_indices = [
                qargs[i]._index + qreg_offset[qargs[i]._register.name] for i in range(len(qargs))
            ]

            if isinstance(inst, QuantumChannelInstruction):
                qerror = inst._quantum_error
                pauli_errors_types = qerror.to_dict()["instructions"]
                pauli_probs = qerror.to_dict()["probabilities"]
                if pauli_errors_types[0][0]["name"] in pauli_error_1_stim_order:
                    probs = 4 * [0.0]
                    for pind, ptype in enumerate(pauli_errors_types):
                        probs[pauli_error_1_stim_order[ptype[0]["name"]]] = pauli_probs[pind]
                    stim_circuit.append("PAULI_CHANNEL_1", qubit_indices, probs[1:])
                elif pauli_errors_types[0][0]["params"][0] in pauli_error_2_stim_order:
                    # here the name is always 'pauli' and the params gives the Pauli type
                    probs = 16 * [0.0]
                    for pind, ptype in enumerate(pauli_errors_types):
                        probs[pauli_error_2_stim_order[ptype[0]["params"][0]]] = pauli_probs[pind]
                    stim_circuit.append("PAULI_CHANNEL_2", qubit_indices, probs[1:])
                else:
                    raise Exception("Unexpected operations: " + str([inst, qargs, cargs]))
            else:
                # Gates and measurements
                if inst.name in qiskit_to_stim_dict:
                    if len(cargs) > 0:  # keeping track of measurement indices in stim
                        measurement_data.append([cargs[0]._register.name, cargs[0]._index])

                    if qiskit_to_stim_dict[inst.name] == "TICK":  # barrier
                        stim_circuit.append("TICK")
                    else:  # gates/measurements acting on qubits
                        stim_circuit.append(qiskit_to_stim_dict[inst.name], qubit_indices)
                else:
                    raise Exception("Unexpected operations: " + str([inst, qargs, cargs]))

        if detectors != [{}]:
            for det in detectors:
                stim_record_targets = []
                for reg, ind in det["clbits"]:
                    stim_record_targets.append(
                        StimTarget_rec(measurement_data.index([reg, ind]) - len(measurement_data))
                    )
                if det["time"] != []:
                    stim_circuit.append(
                        "DETECTOR", stim_record_targets, det["qubits"] + [det["time"]]
                    )
                else:
                    stim_circuit.append("DETECTOR", stim_record_targets, [])
        if logicals != [{}]:
            for log_ind, log in enumerate(logicals):
                stim_record_targets = []
                for reg, ind in log["clbits"]:
                    stim_record_targets.append(
                        StimTarget_rec(measurement_data.index([reg, ind]) - len(measurement_data))
                    )
                stim_circuit.append("OBSERVABLE_INCLUDE", stim_record_targets, log_ind)

        stim_circuits.append(stim_circuit)
        stim_measurement_data.append(measurement_data)

    return stim_circuits, stim_measurement_data


def get_counts_via_stim(
    circuits: Union[List, QuantumCircuit],
    shots: int = 4000,
):
    """Returns a qiskit compatible dictionary of measurement outcomes

    Args:
        circuits: Qiskit circuit compatible with `get_stim_circuits` or list thereof.
        shots: Number of samples to be generated.
        noise_model: Pauli noise model for any additional noise to be applied.

    Returns:
        counts: Counts dictionary in standard Qiskit form or list thereof.
    """

    single_circuit = isinstance(circuits, QuantumCircuit)
    if single_circuit:
        circuits = [circuits]

    counts = []
    for circuit in circuits:
        stim_circuits, stim_measurement_data = get_stim_circuits(circuit)
        stim_circuit = stim_circuits[0]
        measurement_data = stim_measurement_data[0]

        stim_samples = stim_circuit.compile_sampler().sample(shots=shots)
        qiskit_counts = {}
        for stim_sample in stim_samples:
            prev_reg = measurement_data[-1][0]
            qiskit_count = ""
            for idx, meas in enumerate(measurement_data[::-1]):
                reg, _ = meas
                if reg != prev_reg:
                    qiskit_count += " "
                qiskit_count += str(int(stim_sample[-idx - 1]))
                prev_reg = reg
            if qiskit_count in qiskit_counts:
                qiskit_counts[qiskit_count] += 1
            else:
                qiskit_counts[qiskit_count] = 1
        counts.append(qiskit_counts)

    if single_circuit:
        counts = counts[0]

    return counts