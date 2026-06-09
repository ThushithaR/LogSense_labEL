"""Defines the central Config dataclass with all hyperparameters and paths for LogSense."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    MODEL_NAME: str = "distilbert-base-uncased"
    MAX_LENGTH: int = 128
    BATCH_SIZE: int = 16
    LEARNING_RATE: float = 5e-6  # was 2e-5, lower = more careful fine-tuning
    NUM_EPOCHS: int = 6           # was 4, give it more time
    WEIGHT_DECAY: float = 0.01
    SEED: int = 42
    ANOMALY_THRESHOLD: float = 0.5
    CHECKPOINT_DIR: str = r"C:\Users\lenovo_p52\Documents\LAB_EL_6THF\logsense\checkpoints\logsense_checkpoint"
    SPECIAL_TOKENS: List[str] = field(
        default_factory=lambda: [
            "<IP>", "<TIME>", "<MONTH>", "<PROC>",
            "<HEX>", "<PATH>", "<LEVEL>", "<NUM>",
        ]
    )
