"""Poll a Studio job to completion using plain HTTP -- no browser needed.

Companion to deepjeb_submit_training.py: that script submits a job through the
real GUI and exits; this one watches it finish. Safe to background with
nohup+& since it never touches Chromium/Playwright.

Usage: python poll_job.py <job_id> [studio-url] [--interval SECONDS] [--timeout-min N]
"""

import json
import sys
import time
import urllib.request
from urllib.parse import urlsplit

JOB_ID = sys.argv[1]
STUDIO_URL = "http://127.0.0.1:8200/index.html"
INTERVAL = 30
TIMEOUT_MIN = 600

args = sys.argv[2:]
positional = []
i = 0
while i < len(args):
    if args[i] == "--interval":
        INTERVAL = int(args[i + 1])
        i += 2
    elif args[i] == "--timeout-min":
        TIMEOUT_MIN = int(args[i + 1])
        i += 2
    else:
        positional.append(args[i])
        i += 1
if positional:
    STUDIO_URL = positional[0]

origin = f"{urlsplit(STUDIO_URL).scheme}://{urlsplit(STUDIO_URL).netloc}"


def fetch_job(job_id):
    with urllib.request.urlopen(f"{origin}/api/jobs/{job_id}", timeout=15) as resp:
        return json.load(resp)


def main():
    deadline = time.time() + TIMEOUT_MIN * 60
    last_log_len = 0
    while time.time() < deadline:
        job = fetch_job(JOB_ID)
        status = job.get("status")
        log = job.get("log") or ""
        if len(log) > last_log_len:
            tail = log[last_log_len:]
            for line in tail.strip("\n").splitlines()[-20:]:
                print(f"  {line}")
            last_log_len = len(log)
        print(f"[{time.strftime('%H:%M:%S')}] status={status}", flush=True)
        if status not in ("queued", "running"):
            print(f"\nFINAL: status={status} returncode={job.get('returncode')}")
            step0 = (job.get("steps") or [{}])[0]
            print(f"results={step0.get('results')} samples={step0.get('results_samples')}")
            if status != "completed":
                print("\n--- last 4000 chars of log ---")
                print(log[-4000:])
            return 0 if status == "completed" and job.get("returncode") == 0 else 1
        time.sleep(INTERVAL)
    print(f"TIMEOUT after {TIMEOUT_MIN} minutes, job {JOB_ID} still {fetch_job(JOB_ID).get('status')}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
