"""Loads a LogSense checkpoint and evaluates it on the test split, saving metrics, confusion matrix, and training curves."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config
from dataset import generate_synthetic, make_loaders
from model import get_device


def load_model(config: Config) -> tuple:
    """Load checkpoint from config.CHECKPOINT_DIR, move to device, set eval mode."""
    device    = get_device()
    tokenizer = AutoTokenizer.from_pretrained(config.CHECKPOINT_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(config.CHECKPOINT_DIR)
    model.to(device).eval()
    return model, tokenizer


def evaluate(model, loader, device) -> dict:
    """Run inference over loader; return metrics dict with labels, preds, probs, f1, roc_auc, report, confusion_matrix."""
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].cpu().tolist()

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs  = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
            preds  = [1 if p >= 0.5 else 0 for p in probs]

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    report = classification_report(all_labels, all_preds, target_names=["Normal", "Anomaly"])
    return {
        "labels":           all_labels,
        "preds":            all_preds,
        "probs":            all_probs,
        "f1":               round(f1_score(all_labels, all_preds, average="binary", zero_division=0), 6),
        "roc_auc":          round(roc_auc_score(all_labels, all_probs), 6),
        "report":           report,
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
    }


def plot_confusion_matrix(cm: np.ndarray, save_path: str) -> None:
    """Save a seaborn Blues confusion matrix heatmap to save_path at 150 DPI."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("LogSense — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_path: str, save_path: str) -> None:
    """Load training_history.json and save a dual-axis loss/F1 curve plot to save_path at 150 DPI."""
    with open(history_path) as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(epochs, h["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, h["val_loss"],   "b--s", label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(epochs, h["val_f1"], "r-^", label="Val F1")
    ax2.set_ylabel("F1 Score", color="red")
    ax2.tick_params(axis="y", labelcolor="red")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.set_title("LogSense — Training Curves")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    config = Config()
    device = get_device()
    model, tokenizer = load_model(config)

    df = generate_synthetic(n=5000, seed=config.SEED)
    _, _, test_loader = make_loaders(df, tokenizer, config)

    results = evaluate(model, test_loader, device)
    print(results["report"])
    print(f"ROC-AUC : {results['roc_auc']:.4f}")
    print(f"F1      : {results['f1']:.4f}")

    cm_path     = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png")
    curves_path = os.path.join(config.CHECKPOINT_DIR, "training_curves.png")
    history_path = os.path.join(config.CHECKPOINT_DIR, "training_history.json")

    plot_confusion_matrix(results["confusion_matrix"], cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    if os.path.exists(history_path):
        plot_training_curves(history_path, curves_path)
        print(f"Training curves saved to {curves_path}")

    json_results = {k: v for k, v in results.items() if k != "confusion_matrix"}
    eval_path = os.path.join(config.CHECKPOINT_DIR, "eval_results.json")
    with open(eval_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Full results saved to {eval_path}")


if __name__ == "__main__":
    main()
