#!/usr/bin/env python3
"""Check IBM Quantum job status and credentials."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qiskit_ibm_runtime import QiskitRuntimeService

# Check credentials
token = os.getenv('IBM_QUANTUM_TOKEN')
channel = os.getenv('QISKIT_IBM_CHANNEL')
instance = os.getenv('QISKIT_IBM_INSTANCE')

print("=== Credentials Check ===")
print(f"Token: {'set' if token else 'NOT SET'}")
print(f"Channel: {channel}")
if instance:
    print(f"Instance: {instance[:60]}...")
else:
    print("Instance: NOT SET")

print("\n=== Connecting to IBM Quantum ===")
try:
    # Try to get existing service first
    service = QiskitRuntimeService()
    print(f"Connected successfully!")
    
    # Check active account
    print(f"\n=== Active Account ===")
    print(f"Channel: {service.channel}")
    
    # List all backends
    print("\n=== All Available Backends ===")
    backends = service.backends()
    print(f"Total backends: {len(backends)}")
    for b in sorted(backends, key=lambda x: x.num_qubits, reverse=True):
        try:
            status = 'online' if b.status().operational else 'offline'
        except:
            status = 'unknown'
        print(f"  {b.name:20s} | {b.num_qubits:3d} qubits | {status}")
    
    # Check recent jobs
    print("\n=== Recent Jobs (last 10) ===")
    jobs = service.jobs(limit=10)
    if not jobs:
        print("  No jobs found!")
    for job in jobs:
        try:
            backend_name = job.backend().name
        except:
            backend_name = 'N/A'
        print(f"  {job.job_id()} | {str(job.status()):10s} | {backend_name}")
    
    # Optional: set IBM_JOB_IDS="id1,id2" to probe specific jobs
    extra = os.getenv('IBM_JOB_IDS', '').strip()
    if extra:
        print("\n=== Checking job IDs (IBM_JOB_IDS) ===")
        for jid in [x.strip() for x in extra.split(',') if x.strip()]:
            try:
                job = service.job(jid)
                print(f"  {jid}: {job.status()}")
            except Exception as e:
                print(f"  {jid}: NOT FOUND - {e}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


