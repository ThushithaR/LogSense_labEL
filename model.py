"""Builds the DistilBERT tokenizer and binary classification model with custom log special tokens."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import Config


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_tokenizer(config: Config) -> AutoTokenizer:
    """Load DistilBERT tokenizer and register all log-domain special tokens from config."""
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": config.SPECIAL_TOKENS})
    return tokenizer


def build_model(config: Config, tokenizer: AutoTokenizer) -> AutoModelForSequenceClassification:
    """Load DistilBERT sequence classifier, resize embeddings for added tokens, and move to device."""
    device = get_device()
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2
    )
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device     : {device}")
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    return model
