"""FastAPI inference server — serves model predictions and the dashboard UI."""
import time
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pretokenizer import LogAwarePreTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("logsense")

_state: dict = {}

CHECKPOINT = r"C:\Users\lenovo_p52\Documents\LAB_EL_6THF\logsense\checkpoints\logsense_checkpoint"

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading tokenizer and model...")
    try:
        _state["tok"]    = AutoTokenizer.from_pretrained(CHECKPOINT)
        _state["model"]  = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
        _state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _state["model"]  = _state["model"].to(_state["device"]).eval()
        _state["pre"]    = LogAwarePreTokenizer()
        log.info(f"Model ready on {_state['device']}")
    except Exception as e:
        raise RuntimeError(f"Could not load checkpoint from {CHECKPOINT}: {e}")
    yield
    _state.clear()


app = FastAPI(title="LogSense", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────────

class SingleReq(BaseModel):
    log_line: str

class BatchReq(BaseModel):
    log_lines: list[str]


# ── Core inference ─────────────────────────────────────────────────────────────

def _infer(log_line: str) -> dict:
    if not log_line.strip():
        raise HTTPException(422, "log_line is empty")
    t0      = time.perf_counter()
    norm    = _state["pre"]([log_line])[0]
    enc     = _state["tok"](
        norm, return_tensors="pt",
        truncation=True, max_length=128, padding="max_length"
    )
    enc     = {k: v.to(_state["device"]) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(_state["model"](**enc).logits, dim=1)[0]
    ms      = round((time.perf_counter() - t0) * 1000, 2)
    conf    = round(probs[1].item(), 4)
    label   = "ANOMALY" if conf > 0.5 else "Normal"
    return {
        "label":          label,
        "confidence":     conf,
        "latency_ms":     ms,
        "normalized_log": norm,
        "attack_type":    _classify(log_line, label),
    }


def _classify(line: str, label: str) -> str:
    if label == "Normal":
        return "Normal"
    l = line.lower()
    if "failed password" in l or "invalid user" in l:               return "Brute Force"
    if "user=root" in l or ("sudo" in l and "root" in l):           return "Privilege Escalation"
    if "out of memory" in l or "segfault" in l or "kernel panic" in l or "bug:" in l: return "System Crash"
    if "drop in=" in l or "reject in=" in l:                        return "Port Scan"
    if "fatal" in l or "appread" in l or "kernsela" in l:           return "HPC Fault"
    if "/tmp/" in l:                                                 return "Malware"
    return "Anomaly"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def ui():
    return FileResponse("dashboard.html")

@app.get("/health")
def health():
    return {"status": "ok", "device": str(_state.get("device", "—"))}

@app.post("/predict")
def predict(req: SingleReq):
    return _infer(req.log_line)

@app.post("/predict/batch")
def predict_batch(req: BatchReq):
    t0      = time.perf_counter()
    results = [_infer(l) for l in req.log_lines[:64]]
    return {"results": results, "total_ms": round((time.perf_counter()-t0)*1000, 2)}