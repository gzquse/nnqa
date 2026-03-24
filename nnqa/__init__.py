"""
Neural-Native Quantum Arithmetic (NNQA)
========================================

A toolkit for mapping neural networks to quantum circuits using
quantum arithmetic primitives.

Modules:
    models: Neural network architectures
    quantum_circuits: Quantum circuit builders
    mapper: NN to Quantum translation
    trainer: Training utilities
    evaluator: Evaluation and benchmarking

Example:
    from nnqa import create_trainer, Evaluator
    
    # Train a polynomial NN
    trainer = create_trainer(
        model_type='polynomial',
        degree=3,
        target_type='polynomial',
        coefficients=[0.1, 0.3, -0.2, 0.5]
    )
    history = trainer.train(epochs=200)
    
    # Evaluate and compare with quantum
    evaluator = Evaluator(trainer.model)
    results = evaluator.run_benchmark()
"""

from .models import (
    PolynomialNN,
    DeepPolynomialNN,
    QuantumCompatibleMLP,
    FunctionApproximator,
)

from .quantum_circuits import (
    data_to_angle,
    weight_to_alpha,
    add_weighted_sum_block,
    add_multiplication_block,
    QuantumPolynomialCircuit,
    DeepQuantumCircuit,
    CircuitExecutor,
)

from .mapper import (
    NNToQuantumMapper,
    BatchMapper,
    quantum_polynomial_direct,
    quantum_polynomial_eval,
)

from .trainer import (
    TargetFunction,
    TrainingHistory,
    Trainer,
    create_trainer,
)

from .evaluator import (
    Evaluator,
    evaluate_trained_model,
)

__version__ = '0.1.0'

__all__ = [
    # Models
    'PolynomialNN',
    'DeepPolynomialNN', 
    'QuantumCompatibleMLP',
    'FunctionApproximator',
    # Quantum Circuits
    'data_to_angle',
    'weight_to_alpha',
    'add_weighted_sum_block',
    'add_multiplication_block',
    'QuantumPolynomialCircuit',
    'DeepQuantumCircuit',
    'CircuitExecutor',
    # Mapper
    'NNToQuantumMapper',
    'BatchMapper',
    'quantum_polynomial_direct',
    'quantum_polynomial_eval',
    # Trainer
    'TargetFunction',
    'TrainingHistory',
    'Trainer',
    'create_trainer',
    # Evaluator
    'Evaluator',
    'evaluate_trained_model',
]

