from qiskit_ionq import IonQProvider

from ionq_config import get_ionq_api_key

try:
    provider = IonQProvider(get_ionq_api_key())
    print("Listing all available backends for this key:")
    for backend in provider.backends():
        print(f"  - {backend.name}")
        try:
            status = backend.status()
            print(f"    Status object type: {type(status)}")
            print(f"    Status msg: {status.status_msg}")
        except Exception as e:
            print(f"    Could not get status: {e}")

except Exception as e:
    print(f"Provider error: {e}")
