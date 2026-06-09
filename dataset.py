"""Loads HDFS/BGL log datasets or generates synthetic data, tokenizes with pre-normalization, and returns weighted DataLoaders."""

import random
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from pretokenizer import LogAwarePreTokenizer

_pre = LogAwarePreTokenizer()

_TEMPLATES = {
    "brute_force": [
        "Jun 14 {t} combo sshd[{pid}]: Failed password for invalid user {user} from {ip} port {port} ssh2",
        "Jun 14 {t} server sshd[{pid}]: Invalid user {user} from {ip}",
    ],
    "privilege_esc": [
        "Jan 27 {t} host sudo[{pid}]: {user} : command not allowed ; TTY=pts/0 ; PWD=/root ; USER=root",
        "Jan 27 {t} host su[{pid}]: FAILED SU (to root) {user} on /dev/pts/1",
    ],
    "crash": [
        "Mar 03 {t} host kernel[{pid}]: Out of memory: Kill process {pid2} (java) score 892 or sacrifice child",
        "Mar 03 {t} host kernel[{pid}]: segfault at 0x{hex} ip 0x{hex} sp 0x{hex} error 4",
    ],
    "port_scan": [
        "Apr 10 {t} fw kernel[{pid}]: DROP IN=eth0 SRC={ip} DST=10.0.0.1 PROTO=TCP DPT={port}",
        "Apr 10 {t} fw kernel[{pid}]: REJECT IN=eth0 SRC={ip} DST=10.0.0.2 DPT={port}",
    ],
}
_NORMAL_TEMPLATES = [
    "May 01 {t} server sshd[{pid}]: Accepted publickey for {user} from {ip} port {port} ssh2",
    "May 01 {t} server cron[{pid}]: ({user}) CMD (/usr/bin/backup.sh)",
    "May 01 {t} server systemd[{pid}]: Started Session {pid2} of user {user}.",
]
_USERS = ["admin", "deploy", "ubuntu", "ec2-user", "root", "git"]
_rng = random.Random()


def _rand_ip() -> str:
    return ".".join(str(_rng.randint(1, 254)) for _ in range(4))

def _rand_time() -> str:
    return f"{_rng.randint(0,23):02d}:{_rng.randint(0,59):02d}:{_rng.randint(0,59):02d}"

def _fill(tpl: str) -> str:
    return tpl.format(
        t=_rand_time(), ip=_rand_ip(), port=_rng.randint(1024, 65535),
        pid=_rng.randint(1000, 32000), pid2=_rng.randint(1000, 32000),
        user=_rng.choice(_USERS), hex=format(_rng.randint(0, 0xFFFFFFFF), "08x"),
    )


def load_hdfs(log_path: str, label_path: str) -> pd.DataFrame:
    with open(log_path, "r", errors="ignore") as f:
        lines = [line.strip() for line in f if line.strip()]
    
    logs = pd.DataFrame({"log": lines})
    logs["BlockId"] = logs["log"].str.extract(r"(blk_-?\d+)")
    logs = logs.dropna(subset=["BlockId"])
    
    # Group all lines per block into one document
    grouped = logs.groupby("BlockId")["log"].apply(
        lambda x: " ".join(x.tolist()[:20])  # first 20 lines per block
    ).reset_index()
    grouped.columns = ["BlockId", "log"]
    
    labels = pd.read_csv(label_path)
    labels["label"] = (labels["Label"] == "Anomaly").astype(int)
    
    df = grouped.merge(labels[["BlockId", "label"]], on="BlockId", how="inner")
    print(f"Total blocks: {len(df):,} | Normal: {(df.label==0).sum():,} | Anomaly: {(df.label==1).sum():,}")
    return df[["log", "label"]]


def load_bgl(path: str) -> pd.DataFrame:
    """Parse BGL.log where a leading '-' marks normal lines; return a log/label DataFrame."""
    rows = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue
            label = 0 if parts[0] == "-" else 1
            rows.append({"log": parts[1], "label": label})
    return pd.DataFrame(rows)


def generate_synthetic(n: int = 5000, anomaly_ratio: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic DataFrame of n log lines with the given anomaly_ratio for offline testing."""
    _rng.seed(seed)
    n_anomaly = int(n * anomaly_ratio)
    rows = []
    attack_types = list(_TEMPLATES.keys())
    for _ in range(n_anomaly):
        tpls = _TEMPLATES[_rng.choice(attack_types)]
        rows.append({"log": _fill(_rng.choice(tpls)), "label": 1})
    for _ in range(n - n_anomaly):
        rows.append({"log": _fill(_rng.choice(_NORMAL_TEMPLATES)), "label": 0})
    _rng.shuffle(rows)
    return pd.DataFrame(rows)


class LogDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length: int = 128):
        self.texts = _pre(list(texts))
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": self.labels[idx]
        }


def make_loaders(df: pd.DataFrame, tokenizer, config) -> tuple:
    """Split df 70/15/15 stratified, build WeightedRandomSampler for train, return (train, val, test) DataLoaders."""
    tr, tmp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=config.SEED)
    val, test = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=config.SEED)

    train_ds = LogDataset(tr["log"].values, tr["label"].values, tokenizer, config.MAX_LENGTH)
    val_ds   = LogDataset(val["log"].values, val["label"].values, tokenizer, config.MAX_LENGTH)
    test_ds  = LogDataset(test["log"].values, test["label"].values, tokenizer, config.MAX_LENGTH)

    counts = torch.bincount(train_ds.labels)
    weights = 1.0 / counts[train_ds.labels].float()
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=config.BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader
