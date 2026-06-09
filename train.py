"""Fine-tunes DistilBERT for log anomaly detection with warmup scheduling, best-F1 checkpointing, and history export."""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from config import Config
from dataset import generate_synthetic, load_hdfs, make_loaders
from model import build_model, build_tokenizer, get_device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, scheduler, device, criterion) -> float:
    """Run one full training pass; return average cross-entropy loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        loss   = criterion(output.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def validate(model, loader, device, criterion) -> tuple[float, float]:
    """Run validation without gradients; return (avg_loss, binary F1)."""
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            output     = model(input_ids=input_ids, attention_mask=attention_mask)
            loss       = criterion(output.logits, labels)
            total_loss += loss.item()
            preds      = output.logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1       = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    return avg_loss, f1


def train(config: Config, use_hdfs: bool = False, hdfs_log: str = None, hdfs_labels: str = None) -> None:
    set_seed(config.SEED)
    device    = get_device()
    tokenizer = build_tokenizer(config)
    model     = build_model(config, tokenizer)

    if use_hdfs:
        df       = load_hdfs(hdfs_log, hdfs_labels)
        anomalies = df[df.label == 1]  # all 16,838 anomaly blocks
        normals   = df[df.label == 0].sample(n=len(anomalies), random_state=42)  # match exactly
        df        = pd.concat([anomalies, normals]).sample(frac=1, random_state=42).reset_index(drop=True)
        print(f"Balanced: {len(df):,} samples | {len(anomalies):,} anomalies | {len(normals):,} normals")
    else:
        df = generate_synthetic(n=5000, seed=config.SEED)

    train_loader, val_loader, _ = make_loaders(df, tokenizer, config)

    # Weight loss heavily toward anomaly class to fix class imbalance
    criterion    = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))
    total_steps  = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = total_steps // 10
    optimizer    = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    best_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        tr_loss          = train_epoch(model, train_loader, optimizer, scheduler, device, criterion)
        val_loss, val_f1 = validate(model, val_loader, device, criterion)

        history["train_loss"].append(round(tr_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_f1"].append(round(val_f1, 6))

        print(f"Epoch {epoch}/{config.NUM_EPOCHS} | Train Loss: {tr_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(config.CHECKPOINT_DIR)
            tokenizer.save_pretrained(config.CHECKPOINT_DIR)
            print(f"  New best saved (F1={best_f1:.4f})")

    history_path = os.path.join(config.CHECKPOINT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LogSense anomaly detector")
    parser.add_argument("--hdfs",        action="store_true",                          help="Use real HDFS data instead of synthetic")
    parser.add_argument("--hdfs-log",    default="data/HDFS/HDFS.log",                help="Path to HDFS raw log file")
    parser.add_argument("--hdfs-labels", default="data/HDFS/anomaly_label.csv",       help="Path to HDFS anomaly label CSV")
    args = parser.parse_args()

    train(Config(), use_hdfs=args.hdfs, hdfs_log=args.hdfs_log, hdfs_labels=args.hdfs_labels)