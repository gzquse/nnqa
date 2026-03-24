#!/usr/bin/env python3
"""
Neural Network Models for Quantum Arithmetic
=============================================

This module defines neural network architectures that are designed
for direct mapping to quantum circuits using quantum arithmetic primitives.

Models:
- PolynomialNN: Direct polynomial coefficient learning
- DeepPolynomialNN: Deep NN with polynomial feature expansion
- QuantumCompatibleMLP: MLP with quantum-friendly activations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolynomialNN(nn.Module):
    """
    Neural network that learns polynomial coefficients directly.
    
    For degree n polynomial: a_n*x^n + a_{n-1}*x^{n-1} + ... + a_1*x + a_0
    
    This architecture is designed to match quantum arithmetic decomposition,
    where each coefficient maps to a rotation angle in the quantum circuit.
    
    Attributes:
        degree (int): Polynomial degree
        coefficients (nn.Parameter): Learnable polynomial coefficients
    """
    
    def __init__(self, degree=3, init_scale=0.1):
        """
        Args:
            degree: Maximum polynomial degree
            init_scale: Scale for coefficient initialization
        """
        super().__init__()
        self.degree = degree
        self.coefficients = nn.Parameter(
            torch.randn(degree + 1) * init_scale
        )
        
    def forward(self, x):
        """
        Compute polynomial: sum_{i=0}^{degree} a_i * x^i
        
        Args:
            x: Input tensor of shape (batch,) or (batch, 1)
            
        Returns:
            Polynomial output of same shape as input
        """
        # Ensure x is on the same device as model parameters
        device = self.coefficients.device
        if x.device != device:
            x = x.to(device)
            
        if x.dim() == 2:
            x = x.squeeze(-1)
            
        result = torch.zeros_like(x)
        x_power = torch.ones_like(x)
        
        for i in range(self.degree + 1):
            result = result + self.coefficients[i] * x_power
            x_power = x_power * x
            
        return result
    
    def get_coefficients(self):
        """Return learned coefficients as numpy array."""
        return self.coefficients.detach().cpu().numpy()
    
    def set_coefficients(self, coeffs):
        """Set coefficients from numpy array."""
        with torch.no_grad():
            self.coefficients.copy_(torch.tensor(coeffs, dtype=torch.float32))


class DeepPolynomialNN(nn.Module):
    """
    Deep neural network with quantum-compatible layer structure.
    
    Each layer performs operations that map to quantum primitives:
    - Linear combinations (weighted sums)
    - Polynomial feature expansion
    - Bounded activations (tanh) for quantum mapping
    
    This architecture enables hierarchical function approximation
    while maintaining quantum circuit compatibility.
    """
    
    def __init__(self, input_dim=1, hidden_dims=[16, 16], output_dim=1, 
                 degree=3, activation='tanh'):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            degree: Polynomial feature degree
            activation: Activation function ('tanh', 'sigmoid', 'softplus')
        """
        super().__init__()
        self.degree = degree
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Feature expansion: x -> [1, x, x^2, ..., x^degree]
        self.feature_dim = degree + 1
        
        # Activation selection
        activations = {
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'softplus': nn.Softplus(),
        }
        self.activation = activations.get(activation, nn.Tanh())
        
        # Build network layers
        layers = []
        prev_dim = self.feature_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def polynomial_features(self, x):
        """
        Expand input to polynomial features [1, x, x^2, ..., x^n].
        
        Args:
            x: Input tensor of shape (batch, 1)
            
        Returns:
            Feature tensor of shape (batch, degree+1)
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
            
        features = []
        x_power = torch.ones_like(x)
        
        for i in range(self.degree + 1):
            features.append(x_power)
            x_power = x_power * x
            
        return torch.cat(features, dim=-1)
    
    def forward(self, x):
        """
        Forward pass with polynomial feature expansion.
        
        Args:
            x: Input tensor
            
        Returns:
            Network output
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        features = self.polynomial_features(x)
        return self.network(features)
    
    def extract_weights(self):
        """
        Extract all weights for quantum circuit mapping.
        
        Returns:
            Dict of weight matrices and biases
        """
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.detach().cpu().numpy()
        return weights
    
    def get_layer_weights(self, layer_idx):
        """Get weights for specific layer."""
        layer_name = f'network.{layer_idx * 2}'  # Account for activation layers
        weights = {}
        for name, param in self.named_parameters():
            if layer_name in name:
                weights[name] = param.detach().cpu().numpy()
        return weights


class QuantumCompatibleMLP(nn.Module):
    """
    Multi-layer perceptron with quantum-compatible constraints.
    
    Features:
    - Bounded weights for quantum angle encoding
    - Symmetric activations for sign-invariance
    - Layer normalization for stable training
    """
    
    def __init__(self, layer_dims, bounded_weights=True, weight_bound=1.0):
        """
        Args:
            layer_dims: List of layer dimensions [input, hidden1, ..., output]
            bounded_weights: If True, constrain weights to [-weight_bound, weight_bound]
            weight_bound: Maximum absolute weight value
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.bounded_weights = bounded_weights
        self.weight_bound = weight_bound
        
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:  # No activation on last layer
                layers.append(nn.LayerNorm(layer_dims[i+1]))
                layers.append(nn.Tanh())
                
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        if self.bounded_weights:
            self._clamp_weights()
        return self.network(x)
    
    def _clamp_weights(self):
        """Clamp weights to bounded range."""
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight.clamp_(-self.weight_bound, self.weight_bound)
                    if m.bias is not None:
                        m.bias.clamp_(-self.weight_bound, self.weight_bound)
    
    def get_normalized_weights(self):
        """
        Get weights normalized to [0, 1] for quantum angle encoding.
        
        Returns:
            Dict of normalized weight arrays
        """
        weights = {}
        for name, param in self.named_parameters():
            w = param.detach().cpu().numpy()
            w_norm = (w - w.min()) / (w.max() - w.min() + 1e-8)
            weights[name] = w_norm
        return weights


class FunctionApproximator(nn.Module):
    """
    Generic function approximator for various target functions.
    
    Supports:
    - Polynomial targets
    - Trigonometric targets
    - Custom function targets
    """
    
    def __init__(self, model_type='polynomial', **kwargs):
        """
        Args:
            model_type: 'polynomial', 'deep', or 'mlp'
            **kwargs: Model-specific arguments
        """
        super().__init__()
        
        if model_type == 'polynomial':
            self.model = PolynomialNN(**kwargs)
        elif model_type == 'deep':
            self.model = DeepPolynomialNN(**kwargs)
        elif model_type == 'mlp':
            self.model = QuantumCompatibleMLP(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        self.model_type = model_type
        
    def forward(self, x):
        return self.model(x)
    
    def get_quantum_params(self):
        """
        Extract parameters suitable for quantum circuit encoding.
        
        Returns:
            Dict of parameter arrays
        """
        if hasattr(self.model, 'get_coefficients'):
            return {'coefficients': self.model.get_coefficients()}
        elif hasattr(self.model, 'extract_weights'):
            return self.model.extract_weights()
        elif hasattr(self.model, 'get_normalized_weights'):
            return self.model.get_normalized_weights()
        else:
            return {name: p.detach().cpu().numpy() 
                    for name, p in self.model.named_parameters()}

