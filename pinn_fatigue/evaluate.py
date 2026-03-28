"""
Evaluation and visualization functions for PINN model.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple

from .model import EnhancedPINNFatigueModel as PhysicsInformedNeuralNetwork


def predict(model: PhysicsInformedNeuralNetwork,
            X: np.ndarray,
            scaler_X,
            scaler_y,
            device: torch.device = None) -> np.ndarray:
    """
    Make predictions with the trained model.

    Args:
        model: Trained PINN model
        X: Input features (raw, not scaled)
        scaler_X: Feature scaler
        scaler_y: Target scaler
        device: Torch device

    Returns:
        Predicted fatigue life (original scale)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    X_scaled = scaler_X.transform(X)
    X_tensor = torch.FloatTensor(X_scaled).to(device)

    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy()

    y_pred_log = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = 10 ** y_pred_log - 1

    return y_pred.flatten()


def evaluate_model(model: PhysicsInformedNeuralNetwork,
                   data: Dict,
                   save_dir: str,
                   model_name: str = "pinn_model") -> Dict:
    """
    Evaluate model performance and generate visualizations.

    Args:
        model: Trained PINN model
        data: Data dictionary from load_data()
        save_dir: Directory to save results
        model_name: Name for output files

    Returns:
        Dictionary of evaluation metrics
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Make predictions
    with torch.no_grad():
        y_pred_scaled = model(data['X_test_tensor'].to(device)).cpu().numpy()

    y_pred_phys = 10 ** data['scaler_y'].inverse_transform(y_pred_scaled) - 1
    y_test_phys = 10 ** data['scaler_y'].inverse_transform(data['y_test']) - 1

    # Original scale metrics
    r2_orig = r2_score(y_test_phys, y_pred_phys)
    rmse_orig = np.sqrt(mean_squared_error(y_test_phys, y_pred_phys))
    mae_orig = mean_absolute_error(y_test_phys, y_pred_phys)
    error_orig = np.abs(y_pred_phys.flatten() - y_test_phys.flatten()) / y_test_phys.flatten() * 100
    mre_orig = np.mean(error_orig)

    # Log scale metrics
    y_test_log = np.log10(y_test_phys.flatten() + 1)
    y_pred_log = np.log10(y_pred_phys.flatten() + 1)
    r2_log = r2_score(y_test_log, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test_log, y_pred_log))
    mae_log = mean_absolute_error(y_test_log, y_pred_log)
    error_log = np.abs(y_pred_log - y_test_log) / y_test_log * 100
    mre_log = np.mean(error_log)

    # Save predictions
    pred_df = pd.DataFrame({
        'true_life': y_test_phys.flatten(),
        'predicted_life': y_pred_phys.flatten(),
        'true_life_log': y_test_log,
        'predicted_life_log': y_pred_log,
        'relative_error_orig': error_orig,
        'relative_error_log': error_log
    })
    pred_df.to_csv(os.path.join(save_dir, 'plot_data', f'{model_name}_predictions.csv'),
                   index=False, encoding='utf-8-sig')

    # Save metrics
    metrics = {
        'r2_original': r2_orig,
        'rmse_original': rmse_orig,
        'mae_original': mae_orig,
        'mre_original': mre_orig,
        'r2_log': r2_log,
        'rmse_log': rmse_log,
        'mae_log': mae_log,
        'mre_log': mre_log
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(save_dir, 'metrics', f'{model_name}_metrics.csv'),
                      index=False, encoding='utf-8-sig')

    print(f"\n[{model_name}] Evaluation Results:")
    print(f"  Original scale: R²={r2_orig:.4f}, RMSE={rmse_orig:.2f}, MRE={mre_orig:.2f}%")
    print(f"  Log scale: R²={r2_log:.4f}, RMSE={rmse_log:.4f}, MRE={mre_log:.2f}%")

    # Generate plots
    plot_predictions(y_test_phys, y_pred_phys, r2_orig, save_dir, model_name)
    plot_predictions_log(y_test_log, y_pred_log, r2_log, save_dir, model_name)
    plot_residuals(y_test_phys, y_pred_phys, save_dir, model_name)

    return metrics


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray,
                     r2: float, save_dir: str, model_name: str):
    """Plot predicted vs true values."""
    plt.figure(figsize=(8, 6))
    max_val = max(np.max(y_true), np.max(y_pred)) * 1.1

    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    plt.fill_between([0, max_val], [0, max_val * 0.8], [0, max_val * 1.2],
                     alpha=0.2, color='gray', label='±20% Error Band')

    plt.xlabel('True Fatigue Life (Cycles)')
    plt.ylabel('Predicted Fatigue Life (Cycles)')
    plt.title(f'{model_name}\nR² = {r2:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xscale('log')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figures', f'{model_name}_predictions.png'), dpi=300)
    plt.close()


def plot_predictions_log(y_true_log: np.ndarray, y_pred_log: np.ndarray,
                         r2: float, save_dir: str, model_name: str):
    """Plot predicted vs true values on log scale."""
    plt.figure(figsize=(8, 6))
    max_val = max(np.max(y_true_log), np.max(y_pred_log)) * 1.1

    plt.scatter(y_true_log, y_pred_log, alpha=0.6, color='blue', s=50)
    plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2)

    plt.xlabel('log10(True Fatigue Life)')
    plt.ylabel('log10(Predicted Fatigue Life)')
    plt.title(f'{model_name} (Log Scale)\nR² = {r2:.4f}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figures', f'{model_name}_predictions_log.png'), dpi=300)
    plt.close()


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray,
                   save_dir: str, model_name: str):
    """Plot residual analysis."""
    residuals = y_true.flatten() - y_pred.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.6)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Fatigue Life')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Predicted')
    axes[0].grid(alpha=0.3)

    # Residual histogram
    axes[1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'figures', f'{model_name}_residuals.png'), dpi=300)
    plt.close()