#!/usr/bin/env python3
"""
Neural Network Trainer for Quantum-Compatible Models
=====================================================

This module provides training utilities for neural networks
that will be mapped to quantum circuits.

Features:
- Training loop with logging and checkpointing
- Support for various target functions
- Training history tracking
- Model saving/loading
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from time import time
from datetime import datetime

from .models import PolynomialNN, DeepPolynomialNN, FunctionApproximator


class TargetFunction:
    """
    Target function generator for training data.
    
    Supports:
    - Polynomial functions
    - Trigonometric functions (sin, cos)
    - Gaussian functions
    - Custom functions
    """
    
    def __init__(self, func_type='polynomial', **kwargs):
        """
        Args:
            func_type: 'polynomial', 'sin', 'cos', 'gaussian', 'custom'
            **kwargs: Function-specific parameters
        """
        self.func_type = func_type
        self.kwargs = kwargs
        
        if func_type == 'polynomial':
            self.coefficients = kwargs.get('coefficients', [0.1, 0.3, -0.2, 0.5])
            self.func = self._polynomial
        elif func_type == 'sin':
            self.frequency = kwargs.get('frequency', 1.0)
            self.amplitude = kwargs.get('amplitude', 1.0)
            self.func = self._sin
        elif func_type == 'cos':
            self.frequency = kwargs.get('frequency', 1.0)
            self.amplitude = kwargs.get('amplitude', 1.0)
            self.func = self._cos
        elif func_type == 'gaussian':
            self.mu = kwargs.get('mu', 0.0)
            self.sigma = kwargs.get('sigma', 0.3)
            self.func = self._gaussian
        elif func_type == 'custom':
            self.func = kwargs.get('func')
            if self.func is None:
                raise ValueError("Custom function must be provided")
        else:
            raise ValueError(f"Unknown function type: {func_type}")
            
    def _polynomial(self, x):
        result = np.zeros_like(x)
        x_power = np.ones_like(x)
        for c in self.coefficients:
            result += c * x_power
            x_power *= x
        return result
    
    def _sin(self, x):
        return self.amplitude * np.sin(2 * np.pi * self.frequency * x)
    
    def _cos(self, x):
        return self.amplitude * np.cos(2 * np.pi * self.frequency * x)
    
    def _gaussian(self, x):
        return np.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)
    
    def __call__(self, x):
        return self.func(x)
    
    def get_description(self):
        """Return human-readable description of function."""
        if self.func_type == 'polynomial':
            terms = [f"{c:.3f}*x^{i}" for i, c in enumerate(self.coefficients)]
            return "F(x) = " + " + ".join(terms)
        elif self.func_type == 'sin':
            return f"F(x) = {self.amplitude:.2f} * sin(2*pi*{self.frequency:.2f}*x)"
        elif self.func_type == 'cos':
            return f"F(x) = {self.amplitude:.2f} * cos(2*pi*{self.frequency:.2f}*x)"
        elif self.func_type == 'gaussian':
            return f"F(x) = exp(-0.5*((x-{self.mu:.2f})/{self.sigma:.2f})^2)"
        else:
            return "Custom function"


class TrainingHistory:
    """
    Tracks training metrics over epochs.
    """
    
    def __init__(self):
        self.epochs = []
        self.losses = []
        self.learning_rates = []
        self.timestamps = []
        self.metrics = {}
        
    def log(self, epoch, loss, lr=None, **extra_metrics):
        """Log metrics for an epoch."""
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.learning_rates.append(lr)
        self.timestamps.append(datetime.now().isoformat())
        
        for key, value in extra_metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)
            
    def get_best_epoch(self):
        """Return epoch with lowest loss."""
        return self.epochs[np.argmin(self.losses)]
    
    def get_final_loss(self):
        """Return final training loss."""
        return self.losses[-1] if self.losses else None
    
    def to_dict(self):
        """Convert history to dictionary."""
        return {
            'epochs': self.epochs,
            'losses': self.losses,
            'learning_rates': self.learning_rates,
            'timestamps': self.timestamps,
            'metrics': self.metrics,
        }
    
    def save(self, filepath):
        """Save history to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath):
        """Load history from JSON file."""
        history = cls()
        with open(filepath, 'r') as f:
            data = json.load(f)
        history.epochs = data['epochs']
        history.losses = data['losses']
        history.learning_rates = data.get('learning_rates', [])
        history.timestamps = data.get('timestamps', [])
        history.metrics = data.get('metrics', {})
        return history


class Trainer:
    """
    Training manager for quantum-compatible neural networks.
    
    Features:
    - Configurable training loop
    - Learning rate scheduling
    - Checkpointing
    - Progress logging
    """
    
    def __init__(self, model, target_func, output_dir='results', device='cpu'):
        """
        Args:
            model: PyTorch model to train
            target_func: TargetFunction instance or callable
            output_dir: Directory for saving results
            device: Training device ('cpu' or 'cuda')
        """
        self.model = model
        self.target_func = target_func
        self.output_dir = output_dir
        self.device = device
        
        self.model.to(device)
        self.history = TrainingHistory()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def generate_data(self, num_samples, x_range=(-0.9, 0.9), normalize_y=True):
        """
        Generate training data.
        
        Args:
            num_samples: Number of training samples
            x_range: Tuple of (min, max) for input range
            normalize_y: If True, normalize outputs to [-1, 1]
            
        Returns:
            tuple: (X_train, y_train, y_scale)
        """
        # Generate on CPU first
        X = torch.linspace(x_range[0], x_range[1], num_samples, device='cpu').unsqueeze(-1)
        
        if callable(self.target_func):
            y = torch.tensor(self.target_func(X.numpy()), dtype=torch.float32, device='cpu')
        else:
            y = self.target_func(X)
            if not isinstance(y, torch.Tensor):
                y = torch.tensor(y, dtype=torch.float32, device='cpu')
            
        y_scale = 1.0
        if normalize_y:
            y_max = torch.max(torch.abs(y))
            if y_max > 1e-8:
                y_scale = y_max.item()
                y = y / y_max
        
        # Ensure y has correct shape
        if y.dim() == 3:
            y = y.squeeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
                
        # Move to target device
        X = X.to(self.device)
        y = y.to(self.device)
        
        return X, y, y_scale
    
    def train(self, epochs=200, lr=0.05, num_samples=500, 
              scheduler_type='step', log_interval=20, 
              save_checkpoints=True, checkpoint_interval=50):
        """
        Train the model.
        
        Args:
            epochs: Number of training epochs
            lr: Initial learning rate
            num_samples: Number of training samples
            scheduler_type: 'step', 'cosine', or None
            log_interval: Epochs between log messages
            save_checkpoints: Whether to save model checkpoints
            checkpoint_interval: Epochs between checkpoints
            
        Returns:
            TrainingHistory: Training history object
        """
        # Generate training data
        X_train, y_train, y_scale = self.generate_data(num_samples)
        
        # Setup optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Setup scheduler
        if scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//4, gamma=0.5)
        elif scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            scheduler = None
            
        loss_fn = nn.MSELoss()
        
        print("=" * 60)
        print("Training Neural Network for Quantum Mapping")
        print("=" * 60)
        if hasattr(self.target_func, 'get_description'):
            print(f"Target: {self.target_func.get_description()}")
        print(f"Model: {type(self.model).__name__}")
        print(f"Epochs: {epochs}, LR: {lr}, Samples: {num_samples}")
        print(f"Device: {self.device}")
        print("-" * 60)
        
        start_time = time()
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            predictions = self.model(X_train)
            
            # Ensure predictions and targets have matching shapes
            if predictions.dim() == 1:
                predictions = predictions.unsqueeze(-1)
            if y_train.dim() == 1:
                y_train = y_train.unsqueeze(-1)
                
            loss = loss_fn(predictions, y_train)
            
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = lr
                
            # Log history
            self.history.log(epoch, loss.item(), current_lr)
            
            # Print progress
            if epoch % log_interval == 0 or epoch == epochs - 1:
                if hasattr(self.model, 'get_coefficients'):
                    coeffs = self.model.get_coefficients()
                    print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}, LR = {current_lr:.6f}")
                    print(f"          Coeffs: {coeffs}")
                else:
                    print(f"Epoch {epoch:4d}: Loss = {loss.item():.6f}, LR = {current_lr:.6f}")
                    
            # Save checkpoint
            if save_checkpoints and (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch, optimizer)
                
        elapsed = time() - start_time
        
        print("-" * 60)
        print(f"Training completed in {elapsed:.2f} seconds")
        print(f"Final Loss: {loss.item():.6f}")
        
        # Save final model and history
        self.save_model('final_model.pt')
        self.history.save(os.path.join(self.output_dir, 'training_history.json'))
        
        # Save training metadata
        self._save_metadata(epochs, lr, num_samples, elapsed, y_scale)
        
        return self.history
    
    def save_checkpoint(self, epoch, optimizer):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history.to_dict(),
        }, checkpoint_path)
        
    def save_model(self, filename):
        """Save model weights."""
        model_path = os.path.join(self.output_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
    def load_model(self, filename):
        """Load model weights."""
        model_path = os.path.join(self.output_dir, filename)
        self.model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from: {model_path}")
        
    def _save_metadata(self, epochs, lr, num_samples, elapsed, y_scale):
        """Save training metadata."""
        metadata = {
            'model_type': type(self.model).__name__,
            'epochs': epochs,
            'learning_rate': lr,
            'num_samples': num_samples,
            'training_time_seconds': elapsed,
            'y_scale': y_scale,
            'final_loss': self.history.get_final_loss(),
            'best_epoch': self.history.get_best_epoch(),
            'device': str(self.device),
            'timestamp': datetime.now().isoformat(),
        }
        
        if hasattr(self.model, 'degree'):
            metadata['polynomial_degree'] = self.model.degree
            
        if hasattr(self.model, 'get_coefficients'):
            metadata['learned_coefficients'] = self.model.get_coefficients().tolist()
            
        if hasattr(self.target_func, 'get_description'):
            metadata['target_function'] = self.target_func.get_description()
            
        metadata_path = os.path.join(self.output_dir, 'training_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)


def create_trainer(model_type='polynomial', degree=3, target_type='polynomial',
                   output_dir='results', **kwargs):
    """
    Factory function to create trainer with model and target.
    
    Args:
        model_type: 'polynomial' or 'deep'
        degree: Polynomial degree
        target_type: Target function type
        output_dir: Output directory
        **kwargs: Additional arguments for target function
        
    Returns:
        Trainer: Configured trainer instance
    """
    # Create model
    if model_type == 'polynomial':
        model = PolynomialNN(degree=degree)
    elif model_type == 'deep':
        hidden_dims = kwargs.pop('hidden_dims', [16, 16])
        model = DeepPolynomialNN(degree=degree, hidden_dims=hidden_dims)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    # Create target function
    target = TargetFunction(func_type=target_type, **kwargs)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return Trainer(model, target, output_dir=output_dir, device=device)

