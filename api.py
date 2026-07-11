"""
LogSense API — works on Windows/WSL/Mac.
Tails /var/log/auth.log in real time. Falls back to synthetic if not found.

Run inside WSL:
    cd ~/LogSense_labEL
    LOGSENSE_CHECKPOINT=./logsense_checkpoint uvicorn api:app --host 0.0.0.0 --port 8000
"""

import asyncio, json, logging, os, re, time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from random import random, randint, choice, uniform

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("logsense")

CHECKPOINT = os.environ.get("LOGSENSE_CHECKPOINT", "./logsense_checkpoint")
_state: dict = {}

# ── PreTokenizer ──────────────────────────────────────────────────────────────
_IP   = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
_PID  = re.compile(r'\[\d+\]')
_TS   = re.compile(r'\b\d{2}:\d{2}:\d{2}\b')
_HEX  = re.compile(r'\b0x[0-9a-fA-F]+\b')
_PORT = re.compile(r'\bport\s+\d{1,5}\b', re.I)
_SRC  = re.compile(r'\bSRC=[\d.]+')
_DST  = re.compile(r'\bDST=[\d.]+')

def pretokenize(line):
    line = _IP.sub('[IP_ADDR]', line)
    line = _PID.sub('[PID]', line)
    line = _TS.sub('[TIMESTAMP]', line)
    line = _HEX.sub('[HEX_VAL]', line)
    line = _PORT.sub('port [PORT]', line)
    line = _SRC.sub('SRC=[IP_ADDR]', line)
    line = _DST.sub('DST=[IP_ADDR]', line)
    return line

# ── Heuristic ─────────────────────────────────────────────────────────────────
RULES = [
    (re.compile(r'command not allowed.*root',        re.I), 'Priv. Escalation', 0.88),
    (re.compile(r'incorrect password attempts',       re.I), 'Priv. Escalation', 0.87),
    (re.compile(r'not in sudoers',                    re.I), 'Priv. Escalation', 0.91),
    (re.compile(r'sudo.*authentication failure',      re.I), 'Priv. Escalation', 0.86),
    (re.compile(r'POSSIBLE BREAK-IN',                 re.I), 'Intrusion',        0.85),
    (re.compile(r'out of memory.*kill',               re.I), 'OOM Kill',         0.96),
    (re.compile(r'UFW BLOCK|DROP.*DPT',               re.I), 'Port Scan',        0.82),
    (re.compile(r'SYN FLOODING',                      re.I), 'DoS',              0.92),
    (re.compile(r'failed password',                   re.I), 'Brute Force',      0.91),
    (re.compile(r'invalid user',                      re.I), 'Brute Force',      0.87),
    (re.compile(r'maximum authentication attempts',   re.I), 'Brute Force',      0.93),
    (re.compile(r'disconnected.*preauth',             re.I), 'Brute Force',      0.80),
    (re.compile(r'PAM \d+ more authentication',       re.I), 'Brute Force',      0.88),
    (re.compile(r'authentication failure',            re.I), 'Brute Force',      0.85),
]
# Lines that LOOK suspicious to naive regex but are just normal SSH lifecycle noise —
# checked BEFORE the RULES list so they never get mislabeled as an attack.
BENIGN_OVERRIDES = re.compile(
    r'(connection closed|connection reset|'
    r'session opened for user|session closed for user|'
    r'pam_unix\(sudo:session\)|pam_unix\(sshd:session\)|'
    r'pam_unix\(cron:session\)|'
    r'srclimit_penalise|drop connection #\d+ from|'
    r'received disconnect from.*bye bye|'
    r'pwd=/home|tty=pts|'
    r'sudo:session.*session|'
    r'thushitha : pwd=)',
    re.I
)

def heuristic(line):
    if BENIGN_OVERRIDES.search(line):
        return 'NORMAL', 0.80 + random()*0.15, None
    for pat, atk, base in RULES:
        if pat.search(line):
            conf = min(0.985, base + (random()-0.5)*0.05)
            return 'ANOMALY', conf, atk
    return 'NORMAL', 0.83 + random()*0.14, None

# ── Model inference ───────────────────────────────────────────────────────────
def model_infer(line):
    if BENIGN_OVERRIDES.search(line):
        return 'NORMAL', 0.80 + random()*0.15, None

    processed = pretokenize(line)
    enc = _state['tok'](processed, return_tensors='pt',
                        truncation=True, max_length=128, padding='max_length')
    enc = {k: v.to(_state['device']) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.softmax(_state['model'](**enc).logits, dim=1)[0]
    conf  = round(float(probs[1]), 4)
    label = 'ANOMALY' if conf > 0.5 else 'NORMAL'

    if label == 'ANOMALY':
        _, _, atk = heuristic(line)
    else:
        atk = None  # never stamp an attack_type on a NORMAL verdict

    return label, conf, atk

def classify(line):
    t0 = time.perf_counter()
    if _state.get('model') is not None:
        label, conf, atk = model_infer(line)
    else:
        label, conf, atk = heuristic(line)
    ms = round((time.perf_counter()-t0)*1000, 2)
    return label, conf, atk, ms

# ── Field extraction ──────────────────────────────────────────────────────────
# handles ISO:  2026-07-11T06:03:17+00:00 HOST proc[pid]: msg
# handles syslog: May 31 10:14:33 HOST proc[pid]: msg
_ISO_PROC = re.compile(r'\d{4}-\d{2}-\d{2}T[\d:.+\-]+\s+\S+\s+(\S+?)(?:\[\d+\])?:\s*(.*)')
_SYS_PROC = re.compile(r'\w{3}\s+\d+\s+[\d:]+\s+\S+\s+(\S+?)(?:\[\d+\])?:\s*(.*)')

def extract_fields(raw):
    for pat in (_ISO_PROC, _SYS_PROC):
        m = pat.search(raw)
        if m:
            src = m.group(1).rstrip(':').rstrip('.')
            msg = m.group(2).strip()
            return src, msg
    # last resort — return everything after the 3rd space as message
    parts = raw.split(None, 3)
    return (parts[2].rstrip(':') if len(parts) > 2 else 'kernel'), (parts[3] if len(parts) > 3 else raw)

# ── Log tailing ───────────────────────────────────────────────────────────────
LOG_CANDIDATES = [
    Path('/var/log/auth.log'),
    Path('/var/log/syslog'),
    Path('/tmp/auth.log'),
]

def find_log():
    for p in LOG_CANDIDATES:
        try:
            if p.exists() and os.access(p, os.R_OK):
                log.info(f'Tailing {p}')
                return p
        except Exception:
            pass
    log.info('No readable log — using synthetic stream')
    return None

# ── Synthetic fallback ────────────────────────────────────────────────────────
_NORM = [
    'sshd[{p}]: Accepted publickey for thushitha from 127.0.0.1 port 52310 ssh2',
    'CRON[{p}]: (root) CMD (run-parts /etc/cron.daily)',
    'systemd[1]: Started Daily apt download activities.',
    'sudo[{p}]: thushitha : TTY=pts/0 ; USER=root ; COMMAND=/bin/apt update',
    'ntpd[{p}]: time set +0.001492 s',
    'sshd[{p}]: Accepted password for alice from 10.0.1.5 port 43210 ssh2',
    'systemd[1]: Starting Daily man-db regeneration...',
    'CRON[{p}]: (root) CMD (/usr/lib/php/sessionclean)',
]
_ANOM = [
    'sshd[{p}]: Failed password for root from 192.168.1.50 port 22 ssh2',
    'sshd[{p}]: Invalid user admin from 185.220.101.47 port 38912',
    'sudo[{p}]: thushitha : command not allowed ; USER=root ; COMMAND=/bin/bash',
    'sshd[{p}]: Disconnected from invalid user oracle 203.0.113.51 port 55294 [preauth]',
    'sshd[{p}]: error: maximum authentication attempts exceeded for root from 185.220.101.47 port 22 ssh2 [preauth]',
    'kernel: Out of memory: Kill process 7731 (xmrig) score 942 or sacrifice child',
]

def _synthetic_line():
    is_anom = False
    pool = _ANOM if is_anom else _NORM
    p    = randint(1000, 9999)
    ts   = datetime.now().strftime('%b %d %H:%M:%S')
    return f"{ts} thushitha-laptop {choice(pool).format(p=p)}"

async def tail(path):
    """Tails the real log file. If quiet for IDLE_LIMIT seconds
    (e.g. after attack_demo.sh finishes), injects mostly-normal
    synthetic filler so the stream doesn't just stop."""
    IDLE_LIMIT = 10.0
    with open(path, 'r', errors='replace') as f:
        f.seek(0, 2)
        idle = 0.0
        while True:
            line = f.readline()
            if line:
                idle = 0.0
                yield line.rstrip()
            else:
                await asyncio.sleep(0.15)
                idle += 0.15
                if idle >= IDLE_LIMIT:
                    idle = 0.0
                    is_anom = random() < 0.08   # mostly normal filler
                    pool = _ANOM if is_anom else _NORM
                    p    = randint(1000, 9999)
                    ts   = datetime.now().strftime('%b %d %H:%M:%S')
                    yield f"{ts} thushitha-laptop {choice(pool).format(p=p)}"

async def synthetic():
    while True:
        yield _synthetic_line()
        await asyncio.sleep(uniform(0.8, 2.2))

# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info(f'Loading model from {CHECKPOINT} ...')
    try:
        _state['tok']    = AutoTokenizer.from_pretrained(CHECKPOINT)
        _state['model']  = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT)
        _state['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _state['model']  = _state['model'].to(_state['device']).eval()
        log.info(f'[✓] Model loaded on {_state["device"]}')
    except Exception as e:
        log.warning(f'[!] Model load failed: {e} — heuristic mode')
        _state['model']  = None
        _state['device'] = 'cpu'
    yield
    _state.clear()

app = FastAPI(title='LogSense', version='2.0.0', lifespan=lifespan)
app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

class LogReq(BaseModel):
    log: str          # dashboard field name

class BatchReq(BaseModel):
    log_lines: list[str]

@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model':  _state.get('model') is not None,
        'device': str(_state.get('device', 'heuristic')),
    }

@app.post('/predict')
def predict(req: LogReq):
    label, conf, atk, ms = classify(req.log.strip())
    return {
        'label':       label,
        'confidence':  conf,
        'attack_type': atk if label == 'ANOMALY' else None,
        'latency_ms':  ms,
    }

@app.post('/predict/batch')
def predict_batch(req: BatchReq):
    t0 = time.perf_counter()
    results = []
    for line in req.log_lines[:64]:
        label, conf, atk, ms = classify(line)
        results.append({
            'label':       label,
            'confidence':  conf,
            'attack_type': atk if label == 'ANOMALY' else None,
            'latency_ms':  ms,
        })
    return {'results': results, 'total_ms': round((time.perf_counter()-t0)*1000, 2)}

@app.get('/stream')
async def stream():
    log_path = find_log()
    async def gen():
        src_gen = tail(log_path) if log_path else synthetic()
        async for raw in src_gen:
            if not raw.strip():
                continue
            label, conf, atk, ms = classify(raw)
            src, msg = extract_fields(raw)
            data = json.dumps({
                'raw':         raw,
                'src':         src,
                'msg':         msg,
                'label':       label,
                'confidence':  conf,
                'attack_type': atk if label == 'ANOMALY' else None,
                'latency_ms':  ms,
                'ts':          datetime.now().strftime('%H:%M:%S'),
            })
            yield f'data: {data}\n\n'
    return StreamingResponse(gen(), media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)