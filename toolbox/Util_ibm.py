# Utility functions for IBM Quantum

import hashlib
import os
from pathlib import Path
import sys
from time import time, localtime
from pprint import pprint
from qiskit.result.utils import marginal_distribution
import numpy as np
from toolbox.Util_Qiskit import pack_counts_to_numpy
from toolbox.Util_IOfunc import dateT2Str
from qiskit_ibm_runtime import QiskitRuntimeService
#...!...!....................
def harvest_ibmq_backRun_submitMeta(job,md,args):
    sd=md['submit']
    sd['job_id']=job.job_id()
    sd['backend']= job.backend().name # V2

    t1=localtime()
    sd['date']=dateT2Str(t1)
    sd['unix_time']=int(time())
    #sd['provider']=args.provider
    #sd['api_type']='circuit-runner' #IBM-spceciffic , the  alternative is 'sampler'

    if args.expName==None:
        # the  6 chars in job id , as handy job identiffier
        md['hash']=sd['job_id'].replace('-','')[3:9] # those are still visible on the IBMQ-web
        name='ibm_'+md['hash']
        md['short_name']=name
    else:
        myHN=hashlib.md5(os.urandom(32)).hexdigest()[:6]
        md['hash']=myHN
        md['short_name']=args.expName

def harvest_backRun_results(job,md,bigD):
    jobRes = job.result()
    resL=jobRes.results
    nCirc=len(resL)  # number of circuit in the job

    #print('jr:'); pprint(jobRes)
    #1qc=job.circuits()[ic]  # transpiled circuit
    #1ibmMD=jobRes.metadata ; print('tt nC',nCirc,type(ibmMD))

    #nqc=len(resL)  # number of circuit in the job
    countsL=jobRes.get_counts()
    jstat=str(job.status())
    res0=resL[0]
    if nCirc==1:
        countsL=[countsL]  # this is poor design
    qa={}
    qa['status']=jstat
    qa['num_circ']=nCirc
    #print('ccc1b',countsL[0])
    #print('meta:'); pprint(res0._metadata)
    try :
    # collect job performance info
        ibmMD=res0.metadata
        for x in ['num_clbits','device','method','noise']:
            #print(x,ibmMD[x])
            qa[x]=ibmMD[x]

        qa['shots']=res0.shots
        qa['time_taken']=res0.time_taken
    except:
        print('MD1 partially missing')

    if 'num_clbits' not in qa:  # use alternative input
        head=res0.header
        #print('head2');  pprint(head)
        qa['num_clbits']=len(head.creg_sizes)
    print('job QA'); pprint(qa)
    md['job_qa']=qa
    pack_counts_to_numpy(md,bigD,countsL)
    return bigD

def harvest_SamplerRun_results(job,md,bigD):
    jobRes = job.result()                      # SamplerPubResult
    
    resL=[jobRes]
    nCirc=len(resL)  # number of circuit in the job

    #print('jr:'); pprint(jobRes)
    #1qc=job.circuits()[ic]  # transpiled circuit
    #1ibmMD=jobRes.metadata ; print('tt nC',nCirc,type(ibmMD))

    #nqc=len(resL)  # number of circuit in the job
    countsL = [
        result.data[list(result.data.keys())[0]].get_counts()
        for result in jobRes
    ]
    jstat=str(job.status())
    res0=resL[0]
    if nCirc==1:
        countsL=[countsL]  # this is poor design
    qa={}
    qa['status']=jstat
    qa['num_circ']=nCirc
    #print('ccc1b',countsL[0])
    #print('meta:'); pprint(res0._metadata)
    try :
    # collect job performance info
        ibmMD=res0.metadata
        for x in ['num_clbits','device','method','noise']:
            #print(x,ibmMD[x])
            qa[x]=ibmMD[x]

        qa['shots']=res0.shots
        qa['time_taken']=res0.time_taken
    except:
        print('MD1 partially missing')

    if 'num_clbits' not in qa:  # use alternative input
        head=res0.header
        #print('head2');  pprint(head)
        qa['num_clbits']=len(head.creg_sizes)
    print('job QA'); pprint(qa)
    md['job_qa']=qa
    pack_counts_to_numpy(md,bigD,countsL)
    return bigD

def qcrank_reco_from_yields( countsL,nq_addr,nq_data):
        '''Reconstructs data from measurement counts.
        Args:
            countsL: list
                List of measurement counts from the instantiated circuits.        
        Returns:
            rec_udata: numpy array
                Reconstructed un-normalized data with shape 
                (num_addr, nq_data, number of circuits).
        '''
        addrBitsL = [nq_data + i for i in range(nq_addr)]
        print('QRFY: nq_addr,nq_data=',nq_addr,nq_data,' addrBitsL:',addrBitsL)
        nCirc = len(countsL)
        num_addr=1<<nq_addr
        rec_udata = np.zeros((num_addr, nq_data, nCirc))  # To match input indexing
        rec_udataErr = np.zeros_like(rec_udata)   
        for ic in range(nCirc):
            counts = countsL[ic]
            #print('ic:',ic,counts)
            for jd in range(nq_data):
                ibit = nq_data - 1 - jd                
                valV,valErV = marginalize_qcrank_EV(addrBitsL, counts, dataBit=ibit)
                rec_udata[:, jd, ic] = valV
                rec_udataErr[:, jd, ic] = valErV
        return rec_udata,rec_udataErr
    

#...!...!....................
def harvest_cutRun_results(job,md,bigD):  # many circuits
    nCirc=1
    countsL=job['counts']
    shots=job['shots']
    time_taken=job['time_taken']
    pmd=md['payload']
    jstat='Done'
    countsL=[countsL]  # this is poor design
    qa={}
    qa['status']=jstat
    qa['num_circ']=nCirc
    try :
        qa['shots']=shots
        qa['time_taken']=time_taken
    except:
        print('MD1 partially missing')
    print('job QA'); pprint(qa)
    md['job_qa']=qa
    # pack_counts_to_numpy(md,bigD,countsL)
    bigD['rec_udata'], bigD['rec_udata_err'] =  qcrank_reco_from_yields(countsL,pmd['nq_addr'],pmd['nq_data'])
    return bigD


#...!...!....................
# def marginalize_qcrank_EV(  addrBitsL, probsB,dataBit):
#     #print('MQCEV inp bits:',dataBit,addrBitsL)
#     # ... marginal distributions for 2 data qubits, for 1 circuit
#     assert dataBit not in addrBitsL
#     bitL=[dataBit]+addrBitsL
#     #print('MQCEV bitL:',bitL)
#     probs=marginal_distribution(probsB,bitL)

#     #.... comput probabilities for each address
#     nq_addr=len(addrBitsL)
#     seq_len=1<<nq_addr
#     mdata=np.zeros(seq_len)
#     fstr='0'+str(nq_addr)+'b'
#     for j in range(seq_len):
#         mbit=format(j,fstr)
#         mbit0=mbit+'0'; mbit1=mbit+'1'
#         m1=probs[mbit1] if mbit1 in probs else 0
#         m0=probs[mbit0] if mbit0 in probs else 0
#         m01=m0+m1
#         #print(j,mbit,'sum=',m01)
#         p=m1/m01 if m01>0 else 0
#         mdata[j]=p
#     return 1-2*mdata

def marginalize_qcrank_EV(  addrBitsL, probsB, dataBit):
    #print('MQCEV inp bits:',dataBit,addrBitsL)
    # ... marginal distributions for 2 data qubits, for 1 circuit
    assert dataBit not in addrBitsL
    bitL=[dataBit]+addrBitsL
    #print('MQCEV bitL:',bitL,len(addrBitsL))
    probs=marginal_distribution(probsB,bitL)
    
    #.... for each address comput probabilities,stat error, EV, EV_err 
    nq_addr=len(addrBitsL)
    seq_len=1<<nq_addr
    prob=np.zeros(seq_len)
    probEr=np.zeros(seq_len)
    fstr='0'+str(nq_addr)+'b' 
    for j in range(seq_len):
        mbit=format(j,fstr)
        if nq_addr==0: mbit=''  # special case , when no address qubits are measured
        mbit0=mbit+'0'; mbit1=mbit+'1'
        m1=probs[mbit1] if mbit1 in probs else 0
        m0=probs[mbit0] if mbit0 in probs else 0
        m01=m0+m1
        #print('j:',j,mbit,'sum=',m01)
        if m01>0 :
            p=m1/m01
            pErr=np.sqrt( p*(1-p)/m01) if m0*m1>0 else 1/m01
        else:
            p=0; pErr=1
        prob[j]=p
        probEr[j]=pErr
        
    return 1-2*prob, 2*probEr


#...!...!....................
def harvest_circ_transpMeta(qc,md,transBackN):
    qc = qc[0] if isinstance(qc, list) else qc  # handle single or list of circuits
    nqTot=qc.num_qubits

    try:  # works for qiskit  0.45.1
        physQubitLayout = qc._layout.final_index_layout(filter_ancillas=True)
    except:
        physQubitLayout =[ i for i in range(nqTot)]

    #print('physQubitLayout'); print(physQubitLayout)

    #.... cycles & depth ...
    len2=qc.depth(filter_function=lambda x: x.operation.num_qubits == 2 )

    #.....  gate count .....
    opsOD=qc.count_ops()  # ordered dict
    opsD={ k:opsOD[k] for k in opsOD}

    n1q_g=0
    for xx in ['ry','h','r','u2','u3']:
        if xx not in opsD: continue
        n1q_g+=opsD[xx]

    n2q_g=0
    for xx in ['cx','cz','ecr']:
        if xx not in opsD: continue
        n2q_g+=opsD[xx]

    #... store results
    tmd={'num_qubit': nqTot,'phys_qubits':physQubitLayout}
    tmd['transpile_backend']=transBackN

    tmd['2q_gate_depth']=len2
    tmd['1q_gate_count']=n1q_g
    tmd[ '2q_gate_count']= n2q_g
    md['transpile']=tmd

    md['payload'].update({'num_qubit':nqTot})
    print('circ_transpMeta:'); pprint(tmd)


    #...!...!....................
from dotenv import load_dotenv
load_dotenv()
def login():
    token = os.getenv("IBM_QUANTUM_TOKEN")
    channel = os.getenv("QISKIT_IBM_CHANNEL")
    instance = os.getenv("QISKIT_IBM_INSTANCE")
    service = QiskitRuntimeService.save_account(
    channel=channel, # `channel` distinguishes between different account types
    token=token, # Your token is confidential.
    instance=instance, 
    name="quantum_account", # Optionally name this set of account credentials.
    set_as_default=True, # Optionally set these as your default credentials.
    overwrite=True, # Optionally overwrite any existing credentials.
    )
    service = QiskitRuntimeService()
    return service
