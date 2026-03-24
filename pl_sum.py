#!/usr/bin/env python3
import argparse
import numpy as np
from toolbox.Util_ibm import harvest_backRun_results, login
from toolbox.logger import log as logger       # ← your logger

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch IBM job results and analyze expectation values"
    )
    parser.add_argument('-j', '--job_id', required=True,
                        help="IBM Cloud job id")
    return parser.parse_args()


def ana_exp_prod(bigD):
    counts = bigD['counts']
    n0, n1 = counts.get('0', 0), counts.get('1', 0)
    shots = n0 + n1
    mprob = n1 / shots

    meta = bigD['metadata']['circuit_metadata']
    X = np.array(meta['X'])
    raw_W = meta['W'][0] if hasattr(meta['W'], '__getitem__') else meta['W']
    W = float(raw_W)
    tag = meta['tag']

    logger.debug(f"Input:  X = {X.tolist()}, W = {W:.3f}, tag = {tag}")

    tag_funcs = {
        0: lambda X, W: (1 - W) * X[1] + W * X[0],
        1: lambda X, W: X[1] * X[0],
        2: lambda X, W: np.sqrt(W * (1 - W)) * (X[0] - X[1]),
        3: lambda X, W: np.sqrt(1 - W) * np.sqrt(1 - X[0]**2),
        4: lambda X, W: np.sqrt(1 - X[0]**2) * np.sqrt(1 - X[1]**2),
        5: lambda X, W: np.sqrt(W) * np.sqrt(1 - X[1]**2),
    }

    try:
        tEV = tag_funcs[tag](X, W)
    except KeyError:
        logger.error(f"Unknown tag {tag!r}; expected one of {list(tag_funcs)}")
        return

    mEV = 1 - 2 * mprob
    delta = tEV - mEV

    if abs(delta) < 0.03:
        status = 'PASS'
        level = logger.info
    elif abs(delta) < 0.10:
        status = 'POOR'
        level = logger.warning
    else:
        status = 'FAILED'
        level = logger.error

    level(f"Eval: mprob={mprob:.3f} | tEV={tEV:.3f} | mEV={mEV:.3f} | "
          f"delta={delta:.3f} → {status}")


def main():
    args = parse_args()
    logger.info(f"Starting analysis for job {args.job_id}")

    service = login()
    job = service.job(args.job_id)

    md, bigD = {}, {}
    harvest_backRun_results(job, md, bigD)

    ana_exp_prod(bigD)
    shots = bigD['metadata']['circuit_metadata'].get('shots', 'unknown')
    logger.info(f"Done. Total shots processed: {shots}")

if __name__ == "__main__":
    main()
