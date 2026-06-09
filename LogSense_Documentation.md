# LogSense: System Architecture & Documentation

## 1. System Overview
LogSense is a real-time anomaly and threat detection system designed for analyzing system logs. It leverages a fine-tuned Transformer model served via a high-performance FastAPI backend, combined with a dynamic frontend dashboard for real-time monitoring. The entire stack is built to prioritize low latency and high accuracy in identifying malicious activities or system faults.

## 2. Model Architecture & Storage
- **Model Used**: `DistilBERT` (AutoModelForSequenceClassification).
- **Why DistilBERT?**: DistilBERT was chosen over standard BERT or larger LLMs because it retains 97% of BERT's language understanding capabilities while being 40% smaller and 60% faster. This is crucial for real-time log processing where inference speed is critical. Traditional Machine Learning models (like Random Forest) lack the contextual semantic understanding of unstructured log text that Transformers provide.
- **Storage Location**: The fine-tuned model checkpoint and tokenizer are stored locally in the directory: 
  `C:\Users\lenovo_p52\Documents\LAB_EL_6THF\logsense\checkpoints\logsense_checkpoint`

## 3. Training Pipeline (`train.py`)
- **Data Balancing**: The model can be trained on synthetic data or real HDFS logs. When using HDFS logs, the dataset is dynamically balanced (sampling normal logs to exactly match the number of anomaly logs) to prevent the model from becoming biased toward the majority class (normal logs).
- **Loss Function & Weighting**: We use a class-weighted `CrossEntropyLoss` (with weights `[1.0, 3.0]`). The anomaly class is weighted 3x heavier. Why? In cybersecurity, False Negatives (missing an attack) are far more dangerous than False Positives. This weighting forces the model to heavily penalize missing an anomaly.
- **Optimization**: The `AdamW` optimizer is used, paired with a linear learning rate scheduler with a warmup phase (first 10% of steps) to ensure stable convergence.
- **Checkpointing**: The system evaluates the model at the end of each epoch and saves a new checkpoint *only* if the Validation Binary F1-score improves.

## 4. Evaluation Metrics
- **F1-Score**: The primary metric optimized during training. It balances Precision (avoiding false alarms) and Recall (catching actual threats). The target for this project was an F1 > 0.80.
- **Precision & Recall**: Individual targets are > 0.70.
- **Cross-Entropy Loss**: Tracked for both training and validation sets to monitor model convergence and detect overfitting.

## 5. Inference & API Backend (`api.py`)
- **Framework**: `FastAPI` is used as the backend framework. 
- **Why FastAPI?**: It provides asynchronous execution capabilities, extremely high throughput, and automatic OpenAPI documentation. It is significantly faster than standard synchronous frameworks like Flask, making it ideal for Machine Learning model serving.
- **Log Preprocessing**: Before reaching the model, logs pass through a `LogAwarePreTokenizer` (`pretokenizer.py`). This script normalizes dynamic elements in logs (like IP addresses, timestamps, and hex codes) into generic tokens. This prevents the model from memorizing specific IPs and forces it to learn the *structure* of an attack.

## 6. Latency Analysis
- **How is it calculated?**: Latency is calculated inside the `_infer` function of `api.py` using Python's high-resolution timer `time.perf_counter()`. It measures the total time taken from the moment the log is passed into the pre-tokenizer, through the DistilBERT inference (`probs = torch.softmax(...)`), to the final heuristic classification logic.
- **Formula**: `latency_ms = round((time.perf_counter() - t0) * 1000, 2)`
- **Is it good or bad?**: Based on real-time execution logs, latency ranges from **70ms to 175ms** per log. For a deep learning Transformer model running inference, this is **very good** for near real-time monitoring. Sub-200ms latency ensures the system can handle a moderate throughput of logs without creating a massive processing backlog.

## 7. Threat Rate (Average Threat)
- **What does it mean?**: Displayed on the dashboard and live monitor as the "Threat Rate", this metric represents the percentage of processed logs that are classified as anomalies out of the total logs analyzed (`(anomalies / total) * 100`).
- **What is it used for?**: It serves as a high-level situational awareness indicator. In a healthy system, the threat rate should remain low (e.g., < 1% to 5%). A sudden spike in the Threat Rate instantly alerts security operators to a coordinated attack (like a Brute Force or Port Scan campaign) or a cascading systemic failure (like Out of Memory crashes).

## 8. Future Scope & Modifications
To scale and improve LogSense for enterprise deployment, the following modifications can be implemented:
1. **Model Optimization (ONNX/TensorRT)**: Convert the PyTorch model to ONNX format and apply INT8 quantization. This could cut the inference latency down to `< 20ms`, making it hyper-efficient for CPU-only or edge deployments.
2. **Streaming Architecture (Kafka/WebSockets)**: Currently, logs are ingested via HTTP POST requests (`/predict`). Replacing this with a persistent WebSocket connection or having the backend directly consume logs from a Kafka stream would eliminate HTTP overhead and drastically increase throughput.
3. **Multi-Class Classification**: Currently, the exact attack type (e.g., "Brute Force", "Port Scan") is determined using a heuristic/keyword-matching function (`_classify` in `api.py`). For the future, the DistilBERT model can be retrained with a multi-class sequence classification head to natively predict both the anomaly status *and* the exact attack category.
4. **Unsupervised/Continual Learning**: Implementing clustering models to group unknown log patterns dynamically, allowing the system to detect zero-day attacks without requiring manually labeled training data.
