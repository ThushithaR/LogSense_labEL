"""Generates SHAP token-attribution and gradient-based importance explanations for LogSense anomaly predictions."""

import os
import shap
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import Config
from pretokenizer import LogAwarePreTokenizer


def build_predict_fn(model, tokenizer, pretokenizer, device):
    """Return a predict_fn(texts) -> np.ndarray of shape (N, 2) softmax probabilities."""
    def predict_fn(texts):
        normalized = pretokenizer(list(texts))
        enc = tokenizer(normalized, max_length=128, padding=True,
                        truncation=True, return_tensors="pt")
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        return torch.softmax(logits, dim=-1).cpu().numpy()
    return predict_fn


def build_explainer(predict_fn, background_texts: list[str]) -> shap.Explainer:
    """Build a SHAP Explainer using a sample of normal log lines as background."""
    masker = shap.maskers.Text(r"\s+")
    return shap.Explainer(predict_fn, masker, output_names=["Normal", "Anomaly"])


def explain_batch(explainer, texts: list[str], save_dir: str) -> list[dict]:
    """Run SHAP on a batch of texts, save per-sample PNG plots, return top-5 token attributions."""
    os.makedirs(save_dir, exist_ok=True)
    shap_values = explainer(texts)
    results = []
    for i, text in enumerate(texts):
        tokens = text.split()
        vals   = shap_values.values[i, :len(tokens), 1]  # anomaly class
        pairs  = sorted(zip(tokens, vals.tolist()), key=lambda x: abs(x[1]), reverse=True)
        top5   = pairs[:5]

        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.45), 3))
        colors  = ["#d62728" if v > 0 else "#1f77b4" for _, v in zip(tokens, vals)]
        ax.bar(tokens, vals, color=colors)
        ax.set_title(f"SHAP Attribution — sample {i}")
        ax.set_xlabel("Token"); ax.set_ylabel("SHAP (Anomaly)")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"shap_{i}.png"), dpi=150)
        plt.close(fig)

        results.append({"text": text, "top_tokens": top5})
    return results


def explain_single(model, tokenizer, pretokenizer, device, log_line: str) -> dict:
    """Return prediction, confidence, and top-5 gradient-norm token importances for one log line."""
    normalized = pretokenizer([log_line])[0]
    enc = tokenizer(normalized, max_length=128, padding="max_length",
                    truncation=True, return_tensors="pt")
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    embeddings = model.distilbert.embeddings(input_ids)
    embeddings.retain_grad()

    logits = model(inputs_embeds=embeddings, attention_mask=attention_mask).logits
    pred_class = logits.argmax(dim=-1).item()
    confidence = torch.softmax(logits, dim=-1)[0, pred_class].item()

    model.zero_grad()
    logits[0, pred_class].backward()

    grad_norms = embeddings.grad[0].norm(dim=-1).detach().cpu()
    tokens     = tokenizer.convert_ids_to_tokens(input_ids[0].cpu())
    active     = [(t, g.item()) for t, g in zip(tokens, grad_norms) if t not in ("[PAD]", "[CLS]", "[SEP]")]
    normalized_norms = [(t, g / sum(v for _, v in active)) for t, g in active]
    top5 = sorted(normalized_norms, key=lambda x: x[1], reverse=True)[:5]

    return {
        "prediction": "ANOMALY" if pred_class == 1 else "NORMAL",
        "confidence": round(confidence, 4),
        "top_tokens": top5,
    }


if __name__ == "__main__":
    cfg       = Config()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.CHECKPOINT_DIR)
    model     = AutoModelForSequenceClassification.from_pretrained(cfg.CHECKPOINT_DIR).to(device)
    model.eval()
    pre = LogAwarePreTokenizer()

    test_lines = [
        "Jun 14 15:16:01 combo sshd[19939]: Failed password for invalid user admin from 218.22.3.51 port 60788 ssh2",
        "May 01 08:22:10 server sshd[4821]: Accepted publickey for deploy from 10.0.1.22 port 51234 ssh2",
        "Mar 03 09:03:44 host kernel[0]: connection timeout after 30s retrying blk_1234567890",
    ]
    labels = ["ATTACK", "NORMAL", "AMBIGUOUS"]
    for label, line in zip(labels, test_lines):
        result = explain_single(model, tokenizer, pre, device, line)
        print(f"\n[{label}]")
        print(f"  Prediction : {result['prediction']} ({result['confidence']:.2%})")
        print(f"  Top tokens : {result['top_tokens']}")
