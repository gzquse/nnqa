#!/usr/bin/env python3
"""Test different IBM Quantum channels to find all available backends."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qiskit_ibm_runtime import QiskitRuntimeService

token = os.getenv('IBM_QUANTUM_TOKEN')

print("=" * 60)
print("Testing IBM Quantum Channels")
print("=" * 60)

# Test ibm_cloud channel (current)
print("\n=== Channel: ibm_cloud ===")
try:
    instance = os.getenv('QISKIT_IBM_INSTANCE')
    service = QiskitRuntimeService(channel='ibm_cloud', token=token, instance=instance)
    backends = service.backends()
    print(f"Backends available: {len(backends)}")
    for b in sorted(backends, key=lambda x: x.num_qubits, reverse=True):
        print(f"  {b.name:20s} | {b.num_qubits:3d} qubits")
except Exception as e:
    print(f"Error: {e}")

# Test ibm_quantum channel (legacy/open)
print("\n=== Channel: ibm_quantum ===")
try:
    # For ibm_quantum channel, we need different setup
    # First check if there's a saved account
    try:
        service = QiskitRuntimeService(channel='ibm_quantum')
    except:
        # Try to save and connect
        QiskitRuntimeService.save_account(
            channel='ibm_quantum',
            token=token,
            overwrite=True
        )
        service = QiskitRuntimeService(channel='ibm_quantum')
    
    backends = service.backends()
    print(f"Backends available: {len(backends)}")
    for b in sorted(backends, key=lambda x: x.num_qubits, reverse=True):
        print(f"  {b.name:20s} | {b.num_qubits:3d} qubits")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Note: ibm_quantum channel is for IBM Quantum Network members")
print("ibm_cloud channel is for IBM Cloud pay-as-you-go plans")
print("=" * 60)


