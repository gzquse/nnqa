from qiskit_ionq import IonQProvider

from ionq_config import get_ionq_api_key

provider = IonQProvider(get_ionq_api_key())

print("Inspecting available backends:")
for backend in provider.backends():
    print(f"\nBackend: {backend.name}")
    print(f"  Version: {backend.version}")
    try:
        # Some versions of qiskit-ionq might have different attributes
        if hasattr(backend, 'calibration'):
            print(f"  Has calibration data: Yes")
    except:
        pass

# Try to get 'qpu' generic backend
try:
    qpu = provider.get_backend('ionq_qpu')
    print(f"\nGeneric 'ionq_qpu' maps to: {qpu.name}")
except Exception as e:
    print(f"\nCould not get 'ionq_qpu': {e}")

# Try to get 'qpu.forte' specifically
try:
    forte = provider.get_backend('qpu.forte')
    print(f"\n'qpu.forte' object found: {forte.name}")
except Exception as e:
    print(f"\nCould not get 'qpu.forte': {e}")
