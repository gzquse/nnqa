#!/usr/bin/env python3
"""
Retrieve NNQA job results from IBM Quantum Cloud

Usage:
    ./retrieve_nnqa_ibmq.py --basePath cloud_job/out --expName boston_abc123
"""

import sys
import os
import numpy as np
from pprint import pprint
from time import time, sleep

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from qiskit_ibm_runtime import QiskitRuntimeService
from toolbox.Util_H5io4 import read4_data_hdf5, write4_data_hdf5
from submit_nnqa_ibmq import harvest_sampler_results, get_service

import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description='Retrieve NNQA job results from IBM Quantum'
    )
    parser.add_argument("-v", "--verb", type=int, default=1,
                       help="verbosity level")
    parser.add_argument("--basePath", default='cloud_job/out',
                       help="base directory")
    parser.add_argument('-e', "--expName", required=True,
                       help='experiment name from submission')
    parser.add_argument("--timeout", type=int, default=3600,
                       help="max wait time in seconds")
    
    args = parser.parse_args()
    
    args.inpPath = os.path.join(args.basePath, 'jobs')
    args.outPath = os.path.join(args.basePath, 'meas')
    
    for arg in vars(args):
        print(f'myArgs: {arg} = {getattr(args, arg)}')
    
    return args


if __name__ == "__main__":
    args = get_parser()
    
    # Read submitted job info
    inpF = args.expName + '.ibm.h5'
    expD, expMD = read4_data_hdf5(os.path.join(args.inpPath, inpF), verb=args.verb)
    
    pprint(expMD['submit'])
    
    if args.verb > 1:
        pprint(expMD)
    
    jid = expMD['submit']['job_id']
    
    # Connect to IBM
    print('Connecting to IBM Quantum Cloud...')
    service = get_service()
    
    print(f'Retrieving job: {jid}')
    job = service.job(jid)
    
    T0 = time()
    i = 0
    while True:
        jstat = job.status()
        elaT = time() - T0
        print(f'i={i}  status={jstat}, elaT={elaT:.1f} sec')
        
        if jstat == 'DONE':
            break
        if jstat == 'ERROR':
            print('Job failed!')
            sys.exit(99)
        if elaT > args.timeout:
            print('Timeout waiting for job')
            sys.exit(98)
        
        i += 1
        sleep(20)
    
    print('Got results')
    
    # Harvest results
    harvest_sampler_results(job, expMD, expD)
    
    if args.verb > 2:
        pprint(expMD)
    
    # Save results
    os.makedirs(args.outPath, exist_ok=True)
    outF = os.path.join(args.outPath, expMD['short_name'] + '.meas.h5')
    write4_data_hdf5(expD, outF, expMD)
    
    print(f"\nResults saved to: {outF}")
    print(f"Plot with:")
    print(f"  ./plot_nnqa_accuracy.py --basePath {args.basePath} --expName {expMD['short_name']}\n")


