from typing import Dict, List, Tuple
import numpy as np
from pprint import pprint
import os,hashlib
from time import time

from qiskit_aer.primitives import EstimatorV2
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import SamplerV2, Batch
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_addon_cutting.qpd import QPDBasis
from qiskit.circuit.library.standard_gates import CXGate
from qiskit import QuantumCircuit

def bitarray_to_labels(bitarray):
    arr = bitarray.array
    n_bits = bitarray.num_bits if hasattr(bitarray, "num_bits") else arr.shape[-1]
    # If arr is shape (n_shots, n_bits), treat each row as a bit array
    if len(arr.shape) == 2:
        # If each entry is 0 or 1, join as bits
        if arr.max() <= 1:
            return ["".join(str(int(b)) for b in row) for row in arr]
        # If each entry is an integer representing a bitstring, convert each to bitstring and join
        else:
            return [
                "".join(format(int(b), f"0{n_bits}b") for b in row)
                for row in arr
            ]
    # If arr is shape (n_shots,), convert each int to bitstring
    elif len(arr.shape) == 1:
        return [format(int(b), f"0{n_bits}b") for b in arr]
    else:
        raise ValueError("Unexpected BitArray shape")


def bit_parity(bits):
    return (-1) ** (sum(int(b) for b in bits) % 2)

def weighted_obs_counts(yields_list, coeffs, obs_qubits):
    """Compute the weighted observed counts with local-to-global mapping."""
    weighted = []
    for counts, (coeff, *_) in zip(yields_list, coeffs):
        obs_counts = Counter()
        for bitstring, count in counts.items():
            # Assume bitstring is from local subcircuit: index 0 is qubit 0 in that circuit
            n_qubits = len(bitstring)
            local_indices = list(range(n_qubits))

            # Map local bits to their corresponding global qubit indices
            local_to_global = {i: obs_qubits[i] for i in range(len(obs_qubits))}

            # Bits we care about for reconstruction
            obs_bits = ''.join(bitstring[i] for i in local_to_global.keys())
            qpd_bits = ''.join(bitstring[i] for i in local_indices if i not in local_to_global)

            sign = (-1) ** (sum(int(b) for b in qpd_bits) % 2) if qpd_bits else 1
            obs_counts[obs_bits] += coeff * sign * count
        weighted.append(obs_counts)
    return weighted

def reconstruct_global_bitstring(sub_bits_list: List[str],
                                 sub_qubit_maps: List[List[int]],
                                 total_qubits: int) -> str:
    """Construct full global bitstring from partitioned sub-bitstrings."""
    bitstring = ['0'] * total_qubits
    for bits, qubits in zip(sub_bits_list, sub_qubit_maps):
        for bit, idx in zip(bits, qubits):
            bitstring[idx] = bit
    return ''.join(bitstring)

def reconstruct_global_counts(partition_data: List[Tuple[List[Dict[str, int]],
                                                          List[Tuple[float]],
                                                          List[int]]],
                              total_qubits: int) -> Dict[str, int]:
    """
    Reconstruct global bitstring distribution from partitioned results.
    
    Each partition data item is (yields_list, coeffs, obs_qubit_indices)
    """
    weighted_partitions = [
        weighted_obs_counts(yields_list, coeffs, obs_qubits)
        for (yields_list, coeffs, obs_qubits) in partition_data
    ]
    
    from itertools import product
    global_counts = Counter()

    for subexperiment_combo in product(*weighted_partitions):
        for bitstring_combo in product(*[counts.items() for counts in subexperiment_combo]):
            sub_bits = [bits for bits, _ in bitstring_combo]
            weights = [weight for _, weight in bitstring_combo]
            value = 1
            for w in weights:
                value *= w
            global_bitstring = reconstruct_global_bitstring(
                sub_bits,
                [q for (_, _, q) in partition_data],
                total_qubits
            )
            global_counts[global_bitstring] += value

    return {k: max(0, int(round(v))) for k, v in global_counts.items() if v > 0}

def find_sparse_cut_indices(circuit: QuantumCircuit,
                             address_qubits: List[int],
                             data_qubits: List[int],
                             max_cuts: int = 6) -> List[int]:
    """
    Cut only long-distance CX gates crossing address-data boundary,
    constrained to keep total sampling overhead below 3^max_cuts.
    """
    cross_gates = []

    for i, inst in enumerate(circuit.data):
        if inst.operation.name == 'cx':
            q0 = circuit.qubits.index(inst.qubits[0])
            q1 = circuit.qubits.index(inst.qubits[1])
            # print(f"CX gate between qubits {q0} and {q1}")
            if ((q0 in address_qubits and q1 in data_qubits) or
                (q1 in address_qubits and q0 in data_qubits)):
                dist = abs(q0 - q1)
                # print(f"  -> Crosses partition (distance {dist})")
                cross_gates.append((i, dist))

    if not cross_gates:
        print("Warning: No CX gates cross the address-data boundary. Check your qubit indices.")

    # Sort cross gates by distance (descending), prioritizing longer connections
    sorted_gates = sorted(cross_gates, key=lambda x: -x[1])

    # Select top max_cuts gates
    cut_indices = [idx for idx, _ in sorted_gates[:max_cuts]]

    return sorted(cut_indices)
