#!/usr/bin/env python3
"""
NNQA comparison benchmarks
===========================

This script runs three fixed experiments and writes one JSON report.

1) **Sine target, deep NNQA**  
   Trains ``DeepPolynomialNN`` (degree 6, hidden [24, 24]) on
   ``TargetFunction(sin, frequency=1.25, amplitude=0.45)``. Evaluates RMSE of
   classical NN outputs and of mapped quantum circuits (Aer, finite shots) vs
   the analytic sine on a grid in ``[-0.85, 0.85]``. NN and quantum outputs are
   multiplied by the same ``y_scale`` as ``Trainer.generate_data`` (max |y| on
   ``linspace(-0.9, 0.9, 600)``) so they match physical target units.

2) **Same sine target: two-qubit VQA vs mapped NNQA**  
   VQA: ``Ry(arccos x)`` on q0, then ``n_layers`` blocks of ``Ry, Ry, CX`` on
   two qubits; training minimizes MSE of **exact** statevector ``<Z_0>`` vs
   **raw** ``y`` on 40 train points in ``[-0.85, 0.85]`` (L-BFGS-B). Evaluation
   uses **the same shot count per** ``x`` for VQA and for NNQA mapped circuits
   on 21 test points. NNQA side uses the same rescaling as in (1).

3) **Polynomial degree and test domain**  
   For each degree 1..``max_degree``, trains ``PolynomialNN`` on fixed
   coefficients (tabulated in this file), then reports RMSE of NN and of
   ``quantum_polynomial_direct`` vs the true polynomial on 25 points. Two test
   domains: ``narrow`` (``x in [-0.5, 0.5]``) and ``wide`` (``x in [-0.9, 0.9]``).

Usage (from repo root)::

    python research/scripts/nnqa_comparison_benchmarks.py --output-dir results/nnqa_comparison_benchmarks
    python research/scripts/nnqa_comparison_benchmarks.py --quick --shots 2048

Dependencies: nnqa package, qiskit, qiskit-aer, torch, scipy, numpy.
"""

import argparse
import json
import os
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List

import numpy as np
import torch
from scipy.optimize import minimize

# Repo root (neuro_synthesis/)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Pauli, Statevector
from qiskit_aer import AerSimulator

from nnqa.mapper import NNToQuantumMapper, quantum_polynomial_direct
from nnqa.models import DeepPolynomialNN, PolynomialNN
from nnqa.quantum_polynomial import data_to_angle
from nnqa.trainer import TargetFunction, Trainer


# ---------------------------------------------------------------------------
# Two-qubit VQA (fixed encoding + alternating RY layers + CNOT)
# ---------------------------------------------------------------------------


def build_vqa_circuit(x: float, params: np.ndarray, n_layers: int) -> QuantumCircuit:
    """RY(arccos x) on q0, then n_layers of (RY, RY, CX)."""
    x = float(np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(1, "c")
    qc = QuantumCircuit(qr, cr)
    qc.ry(data_to_angle(x), qr[0])
    p = np.asarray(params, dtype=float).reshape(-1)
    expected = 2 * n_layers
    if p.size != expected:
        raise ValueError(f"VQA expects {expected} params, got {p.size}")
    k = 0
    for _ in range(n_layers):
        qc.ry(p[k], qr[0])
        k += 1
        qc.ry(p[k], qr[1])
        k += 1
        qc.cx(qr[0], qr[1])
    qc.measure(qr[0], cr[0])
    return qc


def vqa_expectation_statevector(x: float, params: np.ndarray, n_layers: int) -> float:
    """Exact <Z0> for training loss (no measurement noise)."""
    x = float(np.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))
    qr = QuantumRegister(2, "q")
    qc = QuantumCircuit(qr)
    qc.ry(data_to_angle(x), qr[0])
    p = np.asarray(params, dtype=float).reshape(-1)
    k = 0
    for _ in range(n_layers):
        qc.ry(p[k], qr[0])
        k += 1
        qc.ry(p[k], qr[1])
        k += 1
        qc.cx(qr[0], qr[1])
    sv = Statevector(qc)
    return float(np.real(sv.expectation_value(Pauli("ZI"))))


def vqa_expectation_shots(x: float, params: np.ndarray, n_layers: int, shots: int) -> float:
    """Sample estimate of <Z0>, same shot semantics as CircuitExecutor."""
    qc = build_vqa_circuit(x, params, n_layers)
    backend = AerSimulator()
    qc_t = transpile(qc, backend, optimization_level=1)
    job = backend.run(qc_t, shots=shots)
    counts = job.result().get_counts()
    n0, n1 = counts.get("0", 0), counts.get("1", 0)
    return (n0 - n1) / shots


def train_vqa_statevector_mse(
    train_x: np.ndarray,
    train_y: np.ndarray,
    n_layers: int,
    seed: int,
) -> np.ndarray:
    """Fit VQA angles by minimizing MSE of exact <Z0> vs train_y."""
    rng = np.random.default_rng(seed)
    n_param = 2 * n_layers
    theta0 = rng.uniform(-np.pi, np.pi, size=n_param)

    def obj(theta: np.ndarray) -> float:
        t = np.asarray(theta, dtype=float)
        preds = np.array([vqa_expectation_statevector(x, t, n_layers) for x in train_x])
        mse = np.mean((preds - train_y) ** 2)
        return mse

    res = minimize(
        obj,
        theta0,
        method="L-BFGS-B",
        bounds=[(-np.pi, np.pi)] * n_param,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    return np.asarray(res.x, dtype=float)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _training_y_scale(target: TargetFunction, num_samples: int = 600, x_range=(-0.9, 0.9)) -> float:
    """Match Trainer.generate_data normalization: y / max(|y|) on a linspace grid."""
    xs = np.linspace(x_range[0], x_range[1], num_samples, dtype=np.float64)
    y = np.asarray(target(xs), dtype=float).reshape(-1)
    ymax = float(np.max(np.abs(y)))
    return ymax if ymax > 1e-8 else 1.0


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _result(name, metrics, meta):
    return {"name": name, "metrics": metrics, "meta": meta}


def task_non_polynomial_nnqa(
    shots: int,
    epochs: int,
    seed: int,
    work_dir: str,
):
    """Deep model on sin target; quantum path via mapped circuit + shots."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    target = TargetFunction(func_type="sin", frequency=1.25, amplitude=0.45)
    model = DeepPolynomialNN(degree=6, hidden_dims=[24, 24], activation="tanh")
    os.makedirs(work_dir, exist_ok=True)
    trainer = Trainer(model, target, output_dir=work_dir, device="cpu")
    trainer.train(
        epochs=epochs,
        lr=0.05,
        num_samples=600,
        scheduler_type="step",
        log_interval=max(epochs, 1),
        save_checkpoints=False,
    )

    test_x = np.linspace(-0.85, 0.85, 21, dtype=float)
    y_true = target(test_x)
    y_scale = _training_y_scale(target, num_samples=600, x_range=(-0.9, 0.9))

    model.eval()
    nn_preds = []
    with torch.no_grad():
        for xv in test_x:
            t = torch.tensor([[xv]], dtype=torch.float32)
            nn_preds.append(model(t).squeeze().item())
    nn_preds = np.array(nn_preds, dtype=float) * y_scale

    executor = AerSimulator()
    mapper = NNToQuantumMapper(model)
    q_preds = []
    for xv in test_x:
        qc = mapper.build_mapped_circuit(float(xv))
        qc_t = transpile(qc, executor, optimization_level=1)
        job = executor.run(qc_t, shots=shots)
        counts = job.result().get_counts()
        n0, n1 = counts.get("0", 0), counts.get("1", 0)
        q_preds.append((n0 - n1) / shots)
    q_preds = np.array(q_preds, dtype=float) * y_scale

    return _result(
        "non_polynomial_sin_deep_nnqa",
        {
            "rmse_nn_vs_target": rmse(nn_preds, y_true),
            "rmse_quantum_vs_target": rmse(q_preds, y_true),
            "rmse_quantum_vs_nn": rmse(q_preds, nn_preds),
        },
        {
            "target": target.get_description(),
            "model": "DeepPolynomialNN degree=6 hidden=[24,24]",
            "test_points": len(test_x),
            "eval_shots_per_circuit": shots,
            "y_scale_applied": y_scale,
            "note": "NN/quantum outputs rescaled from Trainer's normalized targets to physical units.",
        },
    )


def task_vqa_vs_nnqa_sin(
    shots: int,
    seed: int,
    n_layers: int,
):
    """Same sin target; VQA vs NNQA with identical eval shots per x."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    target = TargetFunction(func_type="sin", frequency=1.25, amplitude=0.45)
    train_x = np.linspace(-0.85, 0.85, 40, dtype=float)
    train_y = target(train_x)

    theta_vqa = train_vqa_statevector_mse(train_x, train_y, n_layers=n_layers, seed=seed)

    model = DeepPolynomialNN(degree=6, hidden_dims=[24, 24], activation="tanh")
    wd = tempfile.mkdtemp(prefix="nnqa_cmp_vqa_train_")
    trainer = Trainer(model, target, output_dir=wd, device="cpu")
    trainer.train(
        epochs=120,
        lr=0.05,
        num_samples=600,
        scheduler_type="step",
        log_interval=9999,
        save_checkpoints=False,
    )

    test_x = np.linspace(-0.85, 0.85, 21, dtype=float)
    y_true = target(test_x)
    y_scale = _training_y_scale(target, num_samples=600, x_range=(-0.9, 0.9))

    vqa_test = np.array(
        [vqa_expectation_shots(x, theta_vqa, n_layers, shots) for x in test_x]
    )

    model.eval()
    mapper = NNToQuantumMapper(model)
    backend = AerSimulator()
    nnqa_test = []
    with torch.no_grad():
        for xv in test_x:
            qc = mapper.build_mapped_circuit(float(xv))
            qc_t = transpile(qc, backend, optimization_level=1)
            job = backend.run(qc_t, shots=shots)
            counts = job.result().get_counts()
            n0, n1 = counts.get("0", 0), counts.get("1", 0)
            nnqa_test.append((n0 - n1) / shots)
    nnqa_test = np.array(nnqa_test, dtype=float) * y_scale

    return _result(
        "vqa_vs_nnqa_sin_same_eval_shots",
        {
            "rmse_vqa_vs_target": rmse(vqa_test, y_true),
            "rmse_nnqa_vs_target": rmse(nnqa_test, y_true),
        },
        {
            "vqa_layers": n_layers,
            "vqa_params": int(2 * n_layers),
            "vqa_training": "L-BFGS-B on exact statevector <Z0> MSE",
            "eval_shots_per_circuit_per_method": shots,
            "y_scale_applied_to_nnqa": y_scale,
            "note": (
                "VQA trained on raw target values; NNQA outputs rescaled from Trainer normalization. "
                "Evaluation uses equal shots per x for both; optimization cost differs."
            ),
        },
    )


# Bounded polynomial coefficients by degree (aligned with research_config style)
_POLY_COEFFS: Dict[int, List[float]] = {
    1: [0.0, 0.5],
    2: [0.3, 0.4, -0.2],
    3: [0.1, 0.3, -0.1, 0.2],
    4: [0.2, 0.2, -0.15, 0.1, -0.05],
    5: [0.1, 0.25, -0.1, 0.15, -0.08, 0.05],
    6: [0.15, 0.2, -0.1, 0.1, -0.06, 0.04, -0.02],
}


def poly_value(coeffs: List[float], x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    p = np.ones_like(x, dtype=float)
    for c in coeffs:
        out += c * p
        p *= x
    return out


def task_degree_scaling_ablation(
    shots: int,
    max_degree: int,
    seed: int,
):
    """PolynomialNN + direct quantum vs truth; narrow vs wide x interval."""
    rows = []
    for span_label, x_max in [("narrow", 0.5), ("wide", 0.9)]:
        test_x = np.linspace(-x_max, x_max, 25, dtype=float)
        for deg in range(1, max_degree + 1):
            torch.manual_seed(seed + deg)
            np.random.seed(seed + deg)
            coeffs = _POLY_COEFFS[deg]
            target_fn: Callable[[np.ndarray], np.ndarray] = lambda t, c=coeffs: poly_value(c, t)
            y_true = target_fn(test_x)

            model = PolynomialNN(degree=deg)
            tf = TargetFunction(func_type="polynomial", coefficients=coeffs)
            wd = tempfile.mkdtemp(prefix="nnqa_cmp_deg{}_{}_".format(deg, span_label))
            trainer = Trainer(model, tf, output_dir=wd, device="cpu")
            trainer.train(
                epochs=200,
                lr=0.08,
                num_samples=400,
                scheduler_type="step",
                log_interval=9999,
                save_checkpoints=False,
            )
            learned = model.get_coefficients()
            q_pred = np.array(
                [quantum_polynomial_direct(float(xv), learned, shots) for xv in test_x]
            )
            nn_pred = np.array([float(model(torch.tensor([[xv]])).item()) for xv in test_x])
            rows.append(
                {
                    "x_span": span_label,
                    "x_max_abs": x_max,
                    "degree": deg,
                    "rmse_quantum_vs_truth": rmse(q_pred, y_true),
                    "rmse_nn_vs_truth": rmse(nn_pred, y_true),
                }
            )

    return _result(
        "polynomial_degree_xspan_ablation",
        {"rows": rows},
        {
            "eval_shots_per_circuit": shots,
            "narrow_domain": "test x uniform on [-0.5, 0.5], 25 points",
            "wide_domain": "test x uniform on [-0.9, 0.9], 25 points",
        },
    )


def _serialize(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    return obj


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run NNQA comparison benchmarks (see module docstring).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--output-dir", type=str, default="results/nnqa_comparison_benchmarks")
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs-nonpoly", type=int, default=180)
    p.add_argument("--vqa-layers", type=int, default=3)
    p.add_argument("--max-degree", type=int, default=6)
    p.add_argument(
        "--quick",
        action="store_true",
        help="Smaller training budget and max degree 4 for a fast smoke run",
    )
    args = p.parse_args()

    epochs = min(args.epochs_nonpoly, 80) if args.quick else args.epochs_nonpoly
    max_deg = min(4, args.max_degree) if args.quick else args.max_degree
    shots = min(args.shots, 1024) if args.quick else args.shots

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    results = []  # type: List[Dict[str, Any]]

    wp = os.path.join(args.output_dir, "work_nonpoly")
    results.append(task_non_polynomial_nnqa(shots, epochs, args.seed, wp))
    results.append(task_vqa_vs_nnqa_sin(shots, args.seed, n_layers=args.vqa_layers))
    results.append(task_degree_scaling_ablation(shots, max_deg, args.seed))

    out_path = os.path.join(args.output_dir, "comparison_benchmark_report.json")
    payload = {
        "generated_seconds": round(time.time() - t0, 3),
        "config": vars(args),
        "results": [
            {
                "name": r["name"],
                "metrics": _serialize(r["metrics"]),
                "meta": _serialize(r["meta"]),
            }
            for r in results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
