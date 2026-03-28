"""
Configuration management for PINN fatigue prediction.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Config:
    """Configuration for PINN model training."""

    # Data
    data_path: Optional[Path] = None
    test_size: float = 0.5
    random_state: int = 42

    # Model
    input_dim: int = 5
    hidden_dim: int = 32
    use_dropout: bool = False
    dropout_rate: float = 0.1

    # Training
    batch_size: int = 16
    epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    lambda_phys: float = 2.0

    # Output
    save_dir: Path = Path("results")

    def __post_init__(self):
        """Create necessary directories."""
        self.save_dir = Path(self.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        (self.save_dir / "models").mkdir(exist_ok=True)
        (self.save_dir / "figures").mkdir(exist_ok=True)
        (self.save_dir / "metrics").mkdir(exist_ok=True)
        (self.save_dir / "loss_data").mkdir(exist_ok=True)
        (self.save_dir / "plot_data").mkdir(exist_ok=True)


default_config = Config()