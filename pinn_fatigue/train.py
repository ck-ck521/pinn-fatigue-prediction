"""
Training functions for PINN model with physics constraints.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Tuple

from .model import EnhancedPINNFatigueModel as PhysicsInformedNeuralNetwork
from .config import Config


def physics_loss(model: PhysicsInformedNeuralNetwork,
                 X: torch.Tensor,
                 y_pred: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute physics-informed loss with gradient constraints.

    Enforces:
        ∂log10(N)/∂w0 > 0
        ∂log10(N)/∂w45 > 0
        ∂log10(N)/∂w90 < 0
        ∂log10(N)/∂S < 0

    Args:
        model: PINN model
        X: Input tensor
        y_pred: Model predictions (unused but kept for interface)

    Returns:
        Tuple of (loss, grad_w0, grad_w45, grad_w90, grad_S)
    """
    X.requires_grad_(True)
    y_pred = model(X)

    # Compute gradients
    grads = torch.autograd.grad(
        outputs=y_pred,
        inputs=X,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Extract key gradients
    grad_N_w0 = grads[:, 0]  # ∂log10(N)/∂w0 (should be >0)
    grad_N_w45 = grads[:, 1]  # ∂log10(N)/∂w45 (should be >0)
    grad_N_w90 = grads[:, 2]  # ∂log10(N)/∂w90 (should be <0)
    grad_N_S = grads[:, 4]  # ∂log10(N)/∂S (should be <0)

    # Huber loss for smooth constraints
    huber_loss = nn.HuberLoss(delta=1.0)

    # Gradient constraints
    loss_w0 = huber_loss(grad_N_w0, torch.full_like(grad_N_w0, 0.1)) * 0.1
    loss_w45 = huber_loss(grad_N_w45, torch.full_like(grad_N_w45, 0.1)) * 15
    loss_w90 = huber_loss(grad_N_w90, torch.full_like(grad_N_w90, -0.1)) * 5
    loss_S = huber_loss(grad_N_S, torch.full_like(grad_N_S, -0.1)) * 0.1

    # Extra constraints
    loss_w45_extra = torch.relu(-grad_N_w45).mean() * 8
    loss_w90_extra = torch.relu(grad_N_w90).mean() * 5

    # Parameter sign constraints
    loss_param_alpha = torch.relu(-model.alpha + 1e-6) ** 2
    loss_param_beta1 = torch.relu(-model.beta1 + 1e-6) ** 2
    loss_param_beta2 = torch.relu(-model.beta2 + 1e-6) ** 2
    loss_param_beta3 = torch.relu(model.beta3 + 1e-6) ** 2  # β₃ should be negative
    loss_param_gamma = torch.relu(-model.gamma + 1e-6) ** 2
    loss_param = (loss_param_alpha + loss_param_beta1 + loss_param_beta2 +
                  loss_param_beta3 + loss_param_gamma) * 5

    # Combine all constraints
    loss_phys = (loss_w0 + loss_w45 + loss_w90 + loss_S +
                 loss_w45_extra + loss_w90_extra + loss_param)

    return loss_phys, grad_N_w0, grad_N_w45, grad_N_w90, grad_N_S


def enhanced_physics_loss(model: PhysicsInformedNeuralNetwork,
                          X: torch.Tensor,
                          y_pred: torch.Tensor) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Enhanced physics loss with monotonicity constraints.

    Args:
        model: PINN model
        X: Input tensor
        y_pred: Model predictions

    Returns:
        Tuple of (loss, grad_w0, grad_w45, grad_w90, grad_S)
    """
    # Base physics constraints
    base_loss, grad_w0, grad_w45, grad_w90, grad_S = physics_loss(model, X, y_pred)

    # Monotonicity constraint: higher fiber content -> higher life
    X_clone = X.clone().detach().requires_grad_(True)
    X_high_fiber = X_clone.clone()
    X_high_fiber[:, 0] += 0.1  # Increase 0° content
    X_high_fiber[:, 1] += 0.2  # Increase ±45° content
    X_high_fiber[:, 2] -= 0.1  # Decrease 90° content (reduces negative effect)
    y_pred_high = model(X_high_fiber)

    monotonic_loss = torch.mean(torch.relu(y_pred - y_pred_high + 1e-6)) * 0.2

    return base_loss + monotonic_loss, grad_w0, grad_w45, grad_w90, grad_S


def adaptive_physics_weight(epoch: int, total_epochs: int) -> float:
    """
    Adaptively adjust physics loss weight during training.

    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs

    Returns:
        Physics loss weight for current epoch
    """
    if epoch < total_epochs * 0.3:
        return 0.01
    elif epoch < total_epochs * 0.6:
        return 0.01
    else:
        return 0.5


def train_model(model: PhysicsInformedNeuralNetwork,
                data: Dict,
                config: Config,
                model_name: str = "pinn_model") -> Tuple[PhysicsInformedNeuralNetwork, Dict]:
    """
    Train the PINN model.

    Args:
        model: PINN model instance
        data: Data dictionary from load_data()
        config: Configuration object
        model_name: Name for saving model and results

    Returns:
        Tuple of (trained model, training history dictionary)
    """
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     patience=100, factor=0.5)
    criterion = nn.HuberLoss()

    use_physics = config.lambda_phys > 0
    print(f"[{model_name}] Physical constraints enabled: {use_physics}")

    # Initialize history tracking
    train_losses = []
    test_losses = []
    physics_losses = []
    data_losses = []
    learning_rates = []
    best_test_loss = float('inf')
    test_loss = float('inf')

    # Gradient tracking
    grad_w0_means, grad_w45_means, grad_w90_means, grad_S_means = [], [], [], []
    grad_w0_stds, grad_w45_stds, grad_w90_stds, grad_S_stds = [], [], [], []

    # Physical parameters history
    params_history = {
        'alpha': [], 'beta1': [], 'beta2': [], 'beta3': [], 'gamma': []
    }

    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data['X_train_tensor'] = data['X_train_tensor'].to(device)
    data['y_train_tensor'] = data['y_train_tensor'].to(device)
    data['X_test_tensor'] = data['X_test_tensor'].to(device)
    data['y_test_tensor'] = data['y_test_tensor'].to(device)

    # Training loop
    for epoch in tqdm(range(config.epochs), desc=f"Training {model_name}"):
        model.train()
        train_loss = 0.0
        epoch_phys_loss = 0.0
        epoch_data_loss = 0.0

        # Gradient tracking for this epoch
        epoch_grad = {'w0': 0.0, 'w45': 0.0, 'w90': 0.0, 'S': 0.0}
        epoch_grad_std = {'w0': 0.0, 'w45': 0.0, 'w90': 0.0, 'S': 0.0}
        grad_batch_count = 0

        current_lambda = adaptive_physics_weight(epoch, config.epochs) if use_physics else 0.0

        for batch_X, batch_y in data['train_loader']:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            y_pred = model(batch_X)
            loss_data = criterion(y_pred, batch_y)

            if use_physics:
                loss_phys, grad_w0, grad_w45, grad_w90, grad_S = enhanced_physics_loss(model, batch_X, y_pred)

                # Update gradient statistics
                epoch_grad['w0'] += grad_w0.mean().item()
                epoch_grad['w45'] += grad_w45.mean().item()
                epoch_grad['w90'] += grad_w90.mean().item()
                epoch_grad['S'] += grad_S.mean().item()
                epoch_grad_std['w0'] += grad_w0.std().item()
                epoch_grad_std['w45'] += grad_w45.std().item()
                epoch_grad_std['w90'] += grad_w90.std().item()
                epoch_grad_std['S'] += grad_S.std().item()
                grad_batch_count += 1

                epoch_phys_loss += loss_phys.item() * batch_X.size(0)
            else:
                loss_phys = torch.tensor(0.0, device=device)

            loss = loss_data + current_lambda * loss_phys * 2.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            train_loss += loss.item() * batch_X.size(0)
            epoch_data_loss += loss_data.item() * batch_X.size(0)

        # Record learning rate
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Record physical parameters
        params_history['alpha'].append(model.alpha.item())
        params_history['beta1'].append(model.beta1.item())
        params_history['beta2'].append(model.beta2.item())
        params_history['beta3'].append(model.beta3.item())
        params_history['gamma'].append(model.gamma.item())

        # Compute average gradient statistics
        if grad_batch_count > 0:
            grad_w0_means.append(epoch_grad['w0'] / grad_batch_count)
            grad_w45_means.append(epoch_grad['w45'] / grad_batch_count)
            grad_w90_means.append(epoch_grad['w90'] / grad_batch_count)
            grad_S_means.append(epoch_grad['S'] / grad_batch_count)
            grad_w0_stds.append(epoch_grad_std['w0'] / grad_batch_count)
            grad_w45_stds.append(epoch_grad_std['w45'] / grad_batch_count)
            grad_w90_stds.append(epoch_grad_std['w90'] / grad_batch_count)
            grad_S_stds.append(epoch_grad_std['S'] / grad_batch_count)

        # Update learning rate scheduler
        scheduler.step(test_loss)

        # Compute average losses
        n_train = len(data['train_loader'].dataset)
        train_loss /= n_train
        avg_phys_loss = epoch_phys_loss / n_train if use_physics else 0.0
        avg_data_loss = epoch_data_loss / n_train

        train_losses.append(train_loss)
        physics_losses.append(avg_phys_loss)
        data_losses.append(avg_data_loss)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred_test = model(data['X_test_tensor'])
            test_loss = criterion(y_pred_test, data['y_test_tensor']).item()
            test_losses.append(test_loss)

        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(),
                       os.path.join(config.save_dir, 'models', f'{model_name}_best.pth'))

        # Print progress
        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/{config.epochs}], LR: {current_lr:.2e}, '
                  f'Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
            print(f'  Params: α={model.alpha.item():.4f}, β₁={model.beta1.item():.4f}, '
                  f'β₂={model.beta2.item():.4f}, β₃={model.beta3.item():.4f}, γ={model.gamma.item():.4f}')

    # Load best model
    model.load_state_dict(torch.load(os.path.join(config.save_dir, 'models', f'{model_name}_best.pth')))

    # Save training history
    history = {
        'epochs': list(range(1, config.epochs + 1)),
        'train_losses': train_losses,
        'test_losses': test_losses,
        'data_losses': data_losses,
        'physics_losses': physics_losses,
        'learning_rates': learning_rates,
        'grad_w0_means': grad_w0_means,
        'grad_w45_means': grad_w45_means,
        'grad_w90_means': grad_w90_means,
        'grad_S_means': grad_S_means,
        'grad_w0_stds': grad_w0_stds,
        'grad_w45_stds': grad_w45_stds,
        'grad_w90_stds': grad_w90_stds,
        'grad_S_stds': grad_S_stds,
        'physical_params_history': params_history
    }

    # Save to JSON
    history_file = os.path.join(config.save_dir, 'loss_data', f'{model_name}_history.json')
    with open(history_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_history = {k: v if not isinstance(v, np.ndarray) else v.tolist()
                        for k, v in history.items()}
        json.dump(json_history, f, indent=4)

    # Save loss data to CSV
    loss_df = pd.DataFrame({
        'epoch': history['epochs'],
        'train_loss': train_losses,
        'test_loss': test_losses,
        'data_loss': data_losses,
        'physics_loss': physics_losses,
        'learning_rate': learning_rates
    })
    loss_df.to_csv(os.path.join(config.save_dir, 'loss_data', f'{model_name}_loss_data.csv'),
                   index=False, encoding='utf-8-sig')

    # Save gradient data
    if grad_w0_means:
        grad_df = pd.DataFrame({
            'epoch': range(1, len(grad_w0_means) + 1),
            'grad_w0_mean': grad_w0_means,
            'grad_w0_std': grad_w0_stds,
            'grad_w45_mean': grad_w45_means,
            'grad_w45_std': grad_w45_stds,
            'grad_w90_mean': grad_w90_means,
            'grad_w90_std': grad_w90_stds,
            'grad_S_mean': grad_S_means,
            'grad_S_std': grad_S_stds
        })
        grad_df.to_csv(os.path.join(config.save_dir, 'loss_data', f'{model_name}_gradient_data.csv'),
                       index=False, encoding='utf-8-sig')

    # Save physical parameters
    params_df = pd.DataFrame({
        'epoch': history['epochs'],
        'alpha': params_history['alpha'],
        'beta1': params_history['beta1'],
        'beta2': params_history['beta2'],
        'beta3': params_history['beta3'],
        'gamma': params_history['gamma']
    })
    params_df.to_csv(os.path.join(config.save_dir, 'plot_data', f'{model_name}_physical_parameters.csv'),
                     index=False, encoding='utf-8-sig')

    # Plot training curves
    _plot_training_curves(history, config.save_dir, model_name)

    return model, history


def _plot_training_curves(history: Dict, save_dir: str, model_name: str):
    """Plot training curves."""
    # Training and test loss
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_losses'], label='Training Loss', linewidth=2)
    plt.plot(history['test_losses'], label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, 'figures', f'{model_name}_training_curve.png'))
    plt.close()

    # Loss components
    plt.figure(figsize=(8, 6))
    plt.plot(history['data_losses'], label='Data Loss', linewidth=2)
    if any(h > 0 for h in history['physics_losses']):
        plt.plot(history['physics_losses'], label='Physics Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Loss Components')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, 'figures', f'{model_name}_loss_components.png'))
    plt.close()

    # Physical parameters evolution
    params = history['physical_params_history']
    epochs = range(1, len(params['alpha']) + 1)

    plt.figure(figsize=(12, 10))
    plt.subplot(2, 3, 1)
    plt.plot(epochs, params['alpha'], linewidth=2, color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('α')
    plt.title('α Evolution')
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 2)
    plt.plot(epochs, params['beta1'], linewidth=2, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('β₁')
    plt.title('β₁ Evolution')
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 3)
    plt.plot(epochs, params['beta2'], linewidth=2, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('β₂')
    plt.title('β₂ Evolution')
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 4)
    plt.plot(epochs, params['beta3'], linewidth=2, color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('β₃')
    plt.title('β₃ Evolution')
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.plot(epochs, params['gamma'], linewidth=2, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('γ')
    plt.title('γ Evolution')
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 6)
    plt.plot(epochs, params['alpha'], label='α')
    plt.plot(epochs, params['beta1'], label='β₁')
    plt.plot(epochs, params['beta2'], label='β₂')
    plt.plot(epochs, params['beta3'], label='β₃')
    plt.plot(epochs, params['gamma'], label='γ')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('All Parameters')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figures', f'{model_name}_physical_params.png'))
    plt.close()