# LogSense

LogSense is a log anomaly detection system using fine-tuned DistilBERT. It utilizes a custom pre-tokenizer to normalize log lines before classification and provides detailed token-level explanations.

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
```
By default, this trains on synthetic data. Use `--hdfs` to use real data.

### 3. Run the API
```bash
python api.py
```

## Dataset Download (HDFS)
To use the real HDFS dataset from Loghub (Zenodo):
1. Download the dataset from [Zenodo (Loghub HDFS)](https://zenodo.org/record/8196385).
2. Extract `HDFS_1.tar.gz`.
3. Place `HDFS.log_structured.csv` and `anomaly_label.csv` in a `data/` directory.
4. Run training with `python train.py --hdfs`.
