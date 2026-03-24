#!/usr/bin/env python3
"""List available IBM Quantum backends."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qiskit_ibm_runtime import QiskitRuntimeService

token = os.getenv('IBM_QUANTUM_TOKEN')
channel = os.getenv('QISKIT_IBM_CHANNEL', 'ibm_cloud')
instance = os.getenv('QISKIT_IBM_INSTANCE')

print('Connecting to IBM Quantum...')
# Use explicit credentials for reliable connection
service = QiskitRuntimeService(
    channel=channel,
    token=token,
    instance=instance
)

print('\nAvailable backends:')
print('-' * 60)
backends = service.backends()
for b in sorted(backends, key=lambda x: x.num_qubits, reverse=True):
    try:
        status = 'online' if b.status().operational else 'offline'
    except:
        status = 'unknown'
    name = b.name
    nq = b.num_qubits
    print(f'  {name:25s} qubits={nq:3d}  status={status}')

print('-' * 60)
print(f'Total: {len(backends)} backends')

