"""Replaces structured log tokens (IPs, PIDs, timestamps, hex, paths) with semantic placeholders before BERT tokenization."""

import re


class LogAwarePreTokenizer:
    _PATTERNS = [
        (re.compile(r"blk_-?\d+"),                                           "blk_<ID>"),
        (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\b"),              "<IP>"),
        (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"),                   "<TIME>"),
        (re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b"), "<MONTH>"),
        (re.compile(r"\b[A-Za-z_]\w*\[\d+\]"),                              "<PROC>"),
        (re.compile(r"\b0x[0-9a-fA-F]+\b"),                                 "<HEX>"),
        (re.compile(r"\/(?:[\w\-\.]+\/)*[\w\-\.]+"),                        "<PATH>"),
        (re.compile(r"\b(?:ERROR|WARN|FATAL|INFO|DEBUG|CRITICAL)\b"),        "<LEVEL>"),
        (re.compile(r"\b(?!\d{6}\b)\d{4,}\b"),                              "<NUM>"),
    ]

    def __init__(self) -> None:
        pass

    def normalize(self, log_line: str) -> str:
        """Apply all regex replacements in order to a single log line string."""
        if not log_line:
            return log_line
        result = log_line
        for pattern, token in self._PATTERNS:
            result = pattern.sub(token, result)
        return " ".join(result.split())

    def __call__(self, texts: list[str]) -> list[str]:
        """Map normalize() over a list of raw log line strings."""
        return [self.normalize(t) for t in texts]


if __name__ == "__main__":
    pre = LogAwarePreTokenizer()

    cases = [
        (
            "SSH brute-force attack",
            "Jun 14 15:16:01 combo sshd[19939]: Failed password for invalid user admin from 218.22.3.51 port 60788 ssh2",
        ),
        (
            "Kernel OOM kill",
            "Jan 27 09:03:44 host kernel[0]: Out of memory: Kill process 14532 (java) score 892 or sacrifice child at 0xffff8801d5c0",
        ),
        (
            "Normal accepted login",
            "Mar 05 08:22:10 server sshd[4821]: Accepted publickey for deploy from 10.0.1.22 port 51234 ssh2: RSA /home/deploy/.ssh/id_rsa",
        ),
        (
            "HDFS normal block",
            "081109 203518 143 INFO dfs.DataNode$PacketResponder: blk_-1608999687919862906 received",
        ),
        (
            "HDFS anomaly block",
            "081109 203518 143 ERROR dfs.DataNode$PacketResponder: blk_-1608999687919862906 Exception in receiveBlock",
        ),
    ]

    for label, raw in cases:
        print(f"[{label}]")
        print(f"  BEFORE: {raw}")
        print(f"  AFTER : {pre.normalize(raw)}")
        print()