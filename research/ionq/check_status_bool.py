from qiskit_ionq import IonQProvider

from ionq_config import get_ionq_api_key

provider = IonQProvider(get_ionq_api_key())

target_backends = ['qpu.forte', 'qpu.forte-1', 'qpu.aria-1', 'qpu.aria-2', 'qpu.harmony', 'simulator']

print("Checking backend availability (bool):")
for name in target_backends:
    try:
        backend = provider.get_backend(name)
        status = backend.status()
        print(f"  - {name}: {status}")
    except Exception as e:
        print(f"  - {name}: Error ({e})")
