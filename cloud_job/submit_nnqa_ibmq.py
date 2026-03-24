#!/usr/bin/env python3
"""
Submit Neural-Native Quantum Arithmetic jobs to IBM Quantum Cloud

Uses IBM Heron 3 QPUs (ibm_boston, ibm_pittsburgh) for polynomial recovery experiments.
Saves all intermediate data as H5 for reproducibility.

Usage:
    ./submit_nnqa_ibmq.py -E --backend ibm_boston --numSample 5 --numShot 4096
    ./submit_nnqa_ibmq.py -E --backend ibm_pittsburgh --polynomial 0.1 0.3 -0.1 0.2

Credentials should be in .env file (see .env.template)
"""

import sys
import os
import hashlib
import numpy as np
from pprint import pprint
from time import time, localtime
from datetime import datetime

# Add parent path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.options.sampler_options import SamplerOptions
from qiskit_aer import AerSimulator

from toolbox.Util_H5io4 import write4_data_hdf5, read4_data_hdf5
from toolbox.Util_IOfunc import dateT2Str
from toolbox.Util_ibm import harvest_circ_transpMeta

import argparse


# ==============================================================================
# QUANTUM ARITHMETIC PRIMITIVES
# ==============================================================================

def data_to_angle(x):
    """Convert x in [-1,1] to rotation angle."""
    x = np.clip(x, -1 + 1e-7, 1 - 1e-7)
    return np.arccos(x)


def weight_to_alpha(w):
    """Convert weight w in [0,1] to alpha angle."""
    w = np.clip(w, 1e-7, 1 - 1e-7)
    return np.arccos(1 - 2*w)


def build_polynomial_circuit(x, coefficients):
    """
    Build circuit for polynomial evaluation.
    Encodes the polynomial value into a qubit state.
    """
    # Evaluate polynomial classically
    y = sum(coefficients[i] * (x ** i) for i in range(len(coefficients)))
    y_clipped = np.clip(y, -1 + 1e-6, 1 - 1e-6)
    
    # Build circuit
    qr = QuantumRegister(1, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(y_clipped), 0)
    qc.measure(0, 0)
    
    return qc, y_clipped


def build_weighted_sum_circuit(x0, x1, w):
    """Build circuit for weighted sum: y = w*x0 + (1-w)*x1"""
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(x0), 0)
    qc.ry(data_to_angle(x1), 1)
    qc.barrier()
    
    alpha = weight_to_alpha(w)
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    qc.ry(alpha/2, 0)
    qc.cx(1, 0)
    qc.ry(-alpha/2, 0)
    
    qc.measure(0, 0)
    
    tEV = w * x0 + (1 - w) * x1
    return qc, tEV


def build_multiplication_circuit(x0, x1):
    """Build circuit for multiplication: y = x0 * x1"""
    qr = QuantumRegister(2, 'q')
    cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, cr)
    
    qc.ry(data_to_angle(x0), 0)
    qc.ry(data_to_angle(x1), 1)
    qc.barrier()
    
    qc.rz(np.pi/2, 1)
    qc.cx(0, 1)
    
    qc.measure(1, 0)
    
    tEV = x0 * x1
    return qc, tEV


# ==============================================================================
# JOB MANAGEMENT
# ==============================================================================

def get_service():
    """Initialize IBM Quantum service from environment."""
    token = os.getenv("IBM_QUANTUM_TOKEN")
    channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_cloud")
    instance = os.getenv("QISKIT_IBM_INSTANCE")
    
    if not token or not instance:
        raise ValueError(
            "Missing credentials. Create .env file with:\n"
            "IBM_QUANTUM_TOKEN=your_token\n"
            "QISKIT_IBM_CHANNEL=ibm_cloud\n"
            "QISKIT_IBM_INSTANCE=your_crn"
        )
    
    # Always use explicit credentials to ensure correct account
    service = QiskitRuntimeService(
        channel=channel,
        token=token,
        instance=instance
    )
    
    return service


def build_payload_meta(args):
    """Build metadata for the job payload."""
    pd = {
        'polynomial': args.polynomial,
        'degree': len(args.polynomial) - 1,
        'num_sample': args.numSample,
        'test_type': args.testType,
        'rnd_seed': args.rndSeed,
    }
    
    sbm = {
        'num_shots': args.numShot,
        'random_compilation': args.useRC,
        'dynamical_decoupling': args.useDD,
    }
    
    tmd = {
        'transp_seed': args.transpSeed,
    }
    
    pom = {}
    
    md = {'payload': pd, 'submit': sbm, 'transpile': tmd, 'postproc': pom}
    return md


def harvest_submit_meta(job_id, md, args):
    """Record submission metadata."""
    sd = md['submit']
    sd['job_id'] = job_id
    sd['backend'] = args.backend
    sd['date'] = dateT2Str(localtime())
    sd['unix_time'] = int(time())
    sd['provider'] = 'IBMQ_cloud'
    
    if args.expName is None:
        md['hash'] = job_id.replace('-', '')[3:9]
        tag = args.backend.split('_')[1]
        md['short_name'] = f'{tag}_{md["hash"]}'
    else:
        md['hash'] = hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['short_name'] = args.expName


def construct_test_inputs(md, verb=1):
    """Generate test inputs based on test type."""
    pmd = md['payload']
    np.random.seed(pmd['rnd_seed'])
    
    n_samples = pmd['num_sample']
    coefficients = np.array(pmd['polynomial'])
    
    # Store data in H5-compatible flat arrays
    bigD = {}
    
    if pmd['test_type'] == 'polynomial':
        # Test polynomial at random x values
        x_values = np.random.uniform(-0.9, 0.9, n_samples)
        theoretical = np.array([
            np.clip(sum(coefficients[i] * (x ** i) for i in range(len(coefficients))),
                   -1 + 1e-6, 1 - 1e-6)
            for x in x_values
        ])
        bigD['x_values'] = x_values
        bigD['theoretical'] = theoretical
        
    elif pmd['test_type'] == 'weighted_sum':
        x0 = np.random.uniform(-0.9, 0.9, n_samples)
        x1 = np.random.uniform(-0.9, 0.9, n_samples)
        w = np.random.uniform(0.1, 0.9, n_samples)
        theoretical = w * x0 + (1 - w) * x1
        bigD['x0'] = x0
        bigD['x1'] = x1
        bigD['w'] = w
        bigD['theoretical'] = theoretical
        
    elif pmd['test_type'] == 'multiplication':
        x0 = np.random.uniform(-0.9, 0.9, n_samples)
        x1 = np.random.uniform(-0.9, 0.9, n_samples)
        theoretical = x0 * x1
        bigD['x0'] = x0
        bigD['x1'] = x1
        bigD['theoretical'] = theoretical
    
    else:
        raise ValueError(f"Unknown test type: {pmd['test_type']}")
    
    return bigD


def build_circuits(md, bigD):
    """Build circuits based on test type."""
    pmd = md['payload']
    n_samples = pmd['num_sample']
    coefficients = np.array(pmd['polynomial'])
    
    circuits = []
    
    if pmd['test_type'] == 'polynomial':
        for i in range(n_samples):
            qc, _ = build_polynomial_circuit(bigD['x_values'][i], coefficients)
            qc.metadata = {'sample_idx': i, 'x': float(bigD['x_values'][i])}
            circuits.append(qc)
            
    elif pmd['test_type'] == 'weighted_sum':
        for i in range(n_samples):
            qc, _ = build_weighted_sum_circuit(bigD['x0'][i], bigD['x1'][i], bigD['w'][i])
            qc.metadata = {'sample_idx': i}
            circuits.append(qc)
            
    elif pmd['test_type'] == 'multiplication':
        for i in range(n_samples):
            qc, _ = build_multiplication_circuit(bigD['x0'][i], bigD['x1'][i])
            qc.metadata = {'sample_idx': i}
            circuits.append(qc)
    
    return circuits


def harvest_sampler_results(job, md, bigD, T0=None):
    """Harvest results from sampler job."""
    pmd = md['payload']
    qa = {}
    jobRes = job.result()
    
    if T0 is not None:
        elaT = time() - T0
        print(f' job done, elaT={elaT/60.:.1f} min')
        qa['running_duration'] = elaT
    
    nCirc = len(jobRes)
    jstat = str(job.status())
    
    # Extract counts
    countsL = [jobRes[i].data.c.get_counts() for i in range(nCirc)]
    
    # Compute measured expectation values
    measured = np.zeros(nCirc)
    measured_err = np.zeros(nCirc)
    
    for i, counts in enumerate(countsL):
        n0 = counts.get('0', 0)
        n1 = counts.get('1', 0)
        shots = n0 + n1
        mprob = n1 / shots
        measured[i] = 1 - 2 * mprob
        measured_err[i] = 2 * np.sqrt(mprob * (1 - mprob) / shots)
    
    qa['status'] = jstat
    qa['num_circ'] = nCirc
    qa['shots'] = jobRes[0].data.c.num_shots
    
    try:
        jobMetr = job.metrics()
        qa['quantum_seconds'] = jobMetr['usage']['quantum_seconds']
    except:
        pass
    
    print('job QA:')
    pprint(qa)
    md['job_qa'] = qa
    
    bigD['measured'] = measured
    bigD['measured_err'] = measured_err
    
    # Store counts as separate arrays (H5 compatible)
    # counts_0[i] = number of '0' outcomes for circuit i
    # counts_1[i] = number of '1' outcomes for circuit i
    counts_0 = np.array([c.get('0', 0) for c in countsL], dtype=np.int32)
    counts_1 = np.array([c.get('1', 0) for c in countsL], dtype=np.int32)
    bigD['counts_0'] = counts_0
    bigD['counts_1'] = counts_1
    
    return bigD


# ==============================================================================
# ARGUMENT PARSER
# ==============================================================================

def commandline_parser():
    parser = argparse.ArgumentParser(
        description='Submit NNQA jobs to IBM Quantum Cloud'
    )
    parser.add_argument("-v", "--verb", type=int, default=1,
                       help="verbosity level")
    parser.add_argument("--basePath", default='cloud_job/out',
                       help="output directory")
    parser.add_argument("--expName", default=None,
                       help="experiment name (auto-generated if not provided)")
    
    # Test configuration
    parser.add_argument('-t', '--testType', default='polynomial',
                       choices=['polynomial', 'weighted_sum', 'multiplication'],
                       help='type of quantum arithmetic test')
    parser.add_argument('-p', '--polynomial', default=[0.1, 0.3, -0.1, 0.2],
                       nargs='+', type=float,
                       help='polynomial coefficients [a0, a1, a2, ...]')
    parser.add_argument('-i', '--numSample', default=10, type=int,
                       help='number of test samples')
    parser.add_argument('--rndSeed', default=42, type=int,
                       help='random seed for reproducibility')
    
    # Execution
    parser.add_argument('-n', '--numShot', type=int, default=4096,
                       help='shots per circuit')
    parser.add_argument('-b', '--backend', default='ibm_boston',
                       help='IBM backend (ibm_boston, ibm_pittsburgh, ibm_fez, ibm_marrakesh, ibm_kingston, ibm_torino, ibm_miami)')
    parser.add_argument('--transpSeed', default=42, type=int,
                       help='transpiler seed')
    parser.add_argument("--useRC", action='store_true', default=True,
                       help="enable randomized compilation")
    parser.add_argument("--useDD", action='store_true', default=False,
                       help="enable dynamical decoupling")
    parser.add_argument("-E", "--executeCircuit", action='store_true', default=False,
                       help="actually submit the job")
    
    args = parser.parse_args()
    
    for arg in vars(args):
        print(f'myArgs: {arg} = {getattr(args, arg)}')
    
    return args


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    args = commandline_parser()
    
    np.set_printoptions(precision=4)
    
    # Build metadata and inputs
    expMD = build_payload_meta(args)
    expD = construct_test_inputs(expMD, args.verb)
    
    if args.verb > 1:
        pprint(expMD)
    
    # Build circuits
    circuits = build_circuits(expMD, expD)
    nCirc = len(circuits)
    
    print(f'\nBuilt {nCirc} circuits for {args.testType} test')
    if args.verb > 1:
        print(circuits[0].draw('text'))
    
    # Setup output paths
    jobsPath = os.path.join(args.basePath, 'jobs')
    measPath = os.path.join(args.basePath, 'meas')
    os.makedirs(jobsPath, exist_ok=True)
    os.makedirs(measPath, exist_ok=True)
    
    # Initialize backend
    runLocal = True
    
    if 'ibm_' in args.backend:
        print('Connecting to IBM Quantum Cloud...')
        service = get_service()
        
        if 'fake' not in args.backend:
            backend = service.backend(args.backend)
            print(f'Using HW backend: {backend.name}')
            runLocal = False
        else:
            real_backend = args.backend.replace('fake_', 'ibm_')
            hw_backend = service.backend(real_backend)
            backend = AerSimulator.from_backend(hw_backend)
            print(f'Using fake noisy backend: {backend.name}')
    else:
        backend = AerSimulator()
        print(f'Using ideal simulator: {backend.name}')
    
    # Transpile circuits
    print('Transpiling circuits...')
    circuits_t = transpile(circuits, backend, optimization_level=3,
                          seed_transpiler=args.transpSeed)
    
    harvest_circ_transpMeta(circuits_t[0], expMD, backend.name)
    
    if not args.executeCircuit:
        pprint(expMD)
        print('\nNO execution. Use -E to submit the job.\n')
        sys.exit(0)
    
    # Submit job
    print(f'\nSubmitting {nCirc} circuits to {args.backend}...')
    
    options = SamplerOptions()
    options.default_shots = args.numShot
    
    if args.useRC:
        options.twirling.enable_gates = True
        options.twirling.enable_measure = True
        options.twirling.num_randomizations = 60
        print('Enabled Randomized Compilation')
    
    if args.useDD:
        options.dynamical_decoupling.enable = True
        options.dynamical_decoupling.sequence_type = 'XX'
        options.dynamical_decoupling.extra_slack_distribution = 'middle'
        options.dynamical_decoupling.scheduling_method = 'alap'
        print('Enabled Dynamical Decoupling')
    
    sampler = Sampler(mode=backend, options=options)
    T0 = time()
    job = sampler.run(tuple(circuits_t))
    
    harvest_submit_meta(job.job_id(), expMD, args)
    
    if args.verb > 0:
        pprint(expMD)
    
    if runLocal:
        # Wait for local results
        harvest_sampler_results(job, expMD, expD, T0=T0)
        print('Got results')
        
        outF = os.path.join(measPath, expMD['short_name'] + '.meas.h5')
        write4_data_hdf5(expD, outF, expMD)
        
        print(f"\n  ./retrieve_nnqa_ibmq.py --basePath {args.basePath} --expName {expMD['short_name']}\n")
    else:
        # Save job info for later retrieval
        outF = os.path.join(jobsPath, expMD['short_name'] + '.ibm.h5')
        write4_data_hdf5(expD, outF, expMD)
        
        print(f"\nJob submitted: {job.job_id()}")
        print(f"Retrieve with:")
        print(f"  ./retrieve_nnqa_ibmq.py --basePath {args.basePath} --expName {expMD['short_name']}\n")

