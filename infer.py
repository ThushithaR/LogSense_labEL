"""Provides single-line and batch inference for LogSense using the fine-tuned DistilBERT checkpoint."""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import Config
from pretokenizer import LogAwarePreTokenizer
from model import get_device

class LogSenseInferencer:
    def __init__(self, checkpoint_dir: str):
        self.config = Config()
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(self.device)
        self.model.eval()
        self.pretokenizer = LogAwarePreTokenizer()

    def predict(self, log_line: str) -> dict:
        """Predict anomaly status and confidence for a single log line."""
        normalized = self.pretokenizer.normalize(log_line)
        inputs = self.tokenizer(normalized, return_tensors="pt", truncation=True, max_length=self.config.MAX_LENGTH).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred_class = torch.max(probs, dim=-1)
        
        return {
            "label": "Anomaly" if pred_class.item() == 1 else "Normal",
            "confidence": round(confidence.item(), 4),
            "log": log_line
        }

    def predict_batch(self, log_lines: list[str]) -> list[dict]:
        """Predict anomaly status for a batch of log lines."""
        return [self.predict(line) for line in log_lines]

if __name__ == "__main__":
    cfg = Config()
    # Replace with actual checkpoint path if available
    inf = LogSenseInferencer(cfg.CHECKPOINT_DIR)
    test_log = "Jun 14 15:16:01 combo sshd[19939]: Failed password for invalid user admin from 218.22.3.51 port 60788 ssh2"
    print(inf.predict(test_log))
