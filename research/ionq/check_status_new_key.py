from qiskit_ionq import IonQProvider

from ionq_config import get_ionq_api_key

provider = IonQProvider(get_ionq_api_key())

target_backends = ['qpu.forte-1', 'qpu.forte', 'qpu.aria-1', 'qpu.aria-2', 'qpu.harmony', 'simulator']

print("Checking backend status with new key:")
for name in target_backends:
    try:
        if name == 'simulator':
            backend = provider.get_backend(name)
            print(f"  - {name}: FOUND ({backend.name()}) - Status: {backend.status().status_msg}")
        else:
            # For QPUs, we might not have access, so wrap in try/except
            backend = provider.get_backend(name)
            status = backend.status()
            print(f"  - {name}: FOUND ({backend.name()}) - Status: {status.status_msg} - Pending Jobs: {status.pending_jobs}")
    except Exception as e:
        print(f"  - {name}: ERROR/NOT FOUND ({e})")
