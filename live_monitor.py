"""
Streams log lines to LogSense API every second.
Prints alerts to terminal with colors.
"""
import requests
import time
import random
from datetime import datetime

API  = "http://localhost:8000/predict"
RED  = "\033[91m"
GRN  = "\033[92m"
YLW  = "\033[93m"
RST  = "\033[0m"
BLD  = "\033[1m"

# ── Normal logs (varied — some obvious, some ambiguous) ────────
NORMALS = [
    # Obvious normal — model gives 99%+ normal
    "sshd[5678]: Accepted publickey for alice from 10.0.0.5 port 22 ssh2",
    "sshd[5679]: Accepted publickey for bob from 10.0.1.2 port 22 ssh2",
    "sshd[5680]: Accepted password for carol from 192.168.0.10 port 22 ssh2",
    "cron[1234]: (root) CMD (/usr/bin/apt-get update)",
    "cron[5432]: (deploy) CMD (/usr/bin/backup.sh)",
    "systemd[1]: Started Session 42 of user alice.",
    "systemd[1]: Started Daily apt download activities.",
    "systemd[1]: Reached target Multi-User System.",
    "kernel[0]: eth0: renamed from veth3a2b1c",
    "- 1117838580 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO starting kernel",
    "- 1117838590 2005.06.03 R04-M0-N3-C:J06-U01 RAS APP INFO application started",
    "sshd[5681]: Accepted password for deploy from 10.0.2.5 port 22 ssh2",
    # Slightly ambiguous normal — model gives 70-85% normal
    "sudo: alice : USER=root ; COMMAND=/usr/bin/apt-get install nginx",
    "sudo: carol : USER=root ; COMMAND=/bin/systemctl restart nginx",
    "sshd[5682]: Accepted password for root from 10.0.0.1 port 22 ssh2",
    "kernel[0]: device veth3a2b1c entered promiscuous mode",
    "kernel[0]: Out of memory: oom_kill_process started for systemd",
    "sudo: bob : USER=root ; COMMAND=/usr/sbin/service apache2 restart",
    "sshd[5683]: Accepted password for admin from 192.168.1.5 port 22 ssh2",
    "cron[2345]: (root) CMD (/usr/bin/find /tmp -mtime +7 -delete)",
]

# ── Attack logs (varied — some obvious, some borderline) ───────
ATTACKS = [
    # Obvious attacks — model gives 99%+ anomaly
    "sshd[3842]: Failed password for root from 185.220.101.45 port 54892 ssh2",
    "sshd[3843]: Failed password for invalid user admin from 45.33.32.156 port 22",
    "sshd[3844]: Failed password for invalid user ubuntu from 23.129.64.190 port 22",
    "sudo: bob : USER=root ; COMMAND=/bin/bash",
    "sudo: nobody : USER=root ; COMMAND=/bin/bash",
    "kernel[0]: Out of memory: Kill process 9821 (postgres) score 950 or sacrifice child",
    "kernel[0]: segfault at 0x00007f8b4c000000 ip 0x00007f8b4c0001a0 error 6",
    "FATAL 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL FATAL node card memtest failed",
    "kernel[0]: DROP IN=eth0 SRC=192.168.1.100 DST=10.0.0.1 PROTO=TCP DPT=23",
    "kernel[0]: REJECT IN=eth0 SRC=192.168.1.100 DST=10.0.0.1 DPT=3389",
    "su: FAILED SU (to root) alice on /dev/pts/0",
    # Medium confidence attacks — model gives 65-85% anomaly
    "sudo: www-data : USER=root ; COMMAND=/usr/bin/python3 /tmp/exploit.py",
    "kernel[0]: BUG: unable to handle kernel NULL pointer dereference",
    "sshd[3845]: Failed password for root from 203.0.113.99 port 22 ssh2",
    "APPREAD 1117838600 2005.06.03 R04-M0-N3-C:J06-U01 RAS APP FATAL application terminated",
    "sudo: backup : USER=root ; COMMAND=/bin/tar -czf /tmp/data.tar.gz /etc/passwd",
    "sshd[3846]: Failed password for invalid user oracle from 91.121.87.45 port 22",
    "kernel[0]: Oops: 0002 [#1] SMP PTI",
]

# 3 normals per 1 attack — realistic threat rate ~25%
WEIGHTED = NORMALS * 3 + ATTACKS

stats = {"total": 0, "anomalies": 0, "normal": 0}

def send_log(log_line: str):
    try:
        r      = requests.post(API, json={"log_line": log_line}, timeout=5)
        result = r.json()
        label  = result["label"]
        conf   = result["confidence"]
        ms     = result["latency_ms"]
        atype  = result.get("attack_type", "")
        ts     = datetime.now().strftime("%H:%M:%S")
        stats["total"] += 1

        if label == "ANOMALY":
            stats["anomalies"] += 1
            print(f"\n{RED}{BLD}{'='*65}")
            print(f"  🚨 ALERT [{ts}]  {atype.upper()}")
            print(f"  Confidence : {conf:.1%}  |  Latency: {ms:.1f}ms")
            print(f"  Log        : {log_line[:70]}")
            print(f"{'='*65}{RST}\n")
        else:
            # Color intensity based on confidence
            if conf < 0.7:   # borderline normal — amber warning
                color = YLW
                tag   = f"⚠ Normal ({conf:.1%})"
            else:
                color = GRN
                tag   = f"✓ Normal ({conf:.1%})"
            print(f"{color}[{ts}] {tag} | {log_line[:65]}{RST}")

        if stats["total"] % 10 == 0:
            rate = stats["anomalies"] / stats["total"] * 100
            print(f"\n{YLW}── Stats: {stats['total']} logs | "
                  f"{stats['anomalies']} anomalies ({rate:.1f}%) | "
                  f"{stats['normal']} normal ──{RST}\n")

    except requests.exceptions.ConnectionError:
        print(f"{YLW}[{datetime.now().strftime('%H:%M:%S')}] "
              f"API offline — run: uvicorn api:app --reload{RST}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print(f"{BLD}LogSense Live Monitor — Ctrl+C to stop{RST}")
    print(f"Connecting to {API}\n")
    while True:
        log = random.choice(WEIGHTED)
        send_log(log)
        time.sleep(1)