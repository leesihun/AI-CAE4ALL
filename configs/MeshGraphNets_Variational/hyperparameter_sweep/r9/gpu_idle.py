#!/usr/bin/env python3
"""Why is the GPU idle a third of the time? Measures the running sweep jobs for 20 s.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r9/gpu_idle.py

Standard library only, Linux, run on the node while the jobs train (it only
reads /proc and nvidia-smi; with py-spy on PATH it also samples each job's main
thread 10 times, which pauses it for a few milliseconds each). One line per job:

    ==== TYPE THIS BACK ====
    gpu 2 util 64% | hybrid_base | main 125% | workers 4 busy: 61 60 58 57 (R 60% S 38% D 2%, disk 40 MB/s) | data-wait 7/10
        thread python (main): cpu 47% R 47% S 53% D 0%
        thread pt_autograd_0: cpu 40% R 40% S 60% D 0%
    ==== END ====

util     mean nvidia-smi utilization over the window (about DCGM GR_ACT).
main     CPU of the training process, all threads (100% = one core).
workers  DataLoader workers using >5% CPU, each one's CPU; R/S/D = share of
         0.1 s samples running / sleeping (queue full, waiting on a pipe) /
         in disk wait, pooled over all the job's workers; their disk reads.
thread   the trainer's busiest threads by name (python = forward and the
         optimizer, pt_autograd_* = backward, pin_memory runs as a python
         thread), with the same R/S/D shares.
data-wait  py-spy samples in which the main thread waited for the next batch.

Reading it:
- workers near 100% R             -> the workers cannot keep up: more num_workers.
- workers high D, disk MB/s high  -> reading the hierarchy cache is the limit.
- workers mostly S                -> batches are ready ahead; the data is not
  the limit, look at the threads.
- a non-main python thread near 100% -> the pin_memory thread: pin_memory False.
- main or pt_autograd near 100% R -> the trainer's own host work is the limit.
"""
import os
import re
import shutil
import subprocess
import time

WINDOW = 20
TICK = os.sysconf('SC_CLK_TCK')


def read(path, mode='r'):
    try:
        with open(path, mode) as f:
            return f.read()
    except OSError:
        return None


def cmdline(pid):
    raw = read(f'/proc/{pid}/cmdline', 'rb')
    return raw.replace(b'\0', b' ').decode(errors='replace').strip() if raw else ''


def ppid(pid):
    st = read(f'/proc/{pid}/stat')
    return int(st.rsplit(')', 1)[1].split()[1]) if st else None


def ticks(stat_path):
    st = read(stat_path)
    if not st:
        return 0
    f = st.rsplit(')', 1)[1].split()
    return int(f[11]) + int(f[12])          # utime + stime


def thread_ticks(pid):
    out = {}
    try:
        for tid in os.listdir(f'/proc/{pid}/task'):
            out[tid] = ticks(f'/proc/{pid}/task/{tid}/stat')
    except OSError:
        pass
    return out


def visible_gpu(pid):
    env = read(f'/proc/{pid}/environ', 'rb')
    if not env:
        return '?'
    for kv in env.split(b'\0'):
        if kv.startswith(b'CUDA_VISIBLE_DEVICES='):
            return kv.split(b'=', 1)[1].decode() or '?'
    return 'all'


def find_jobs():
    pids = [int(p) for p in os.listdir('/proc') if p.isdigit()]
    procs = {p: cmdline(p) for p in pids}
    jobs = []
    for p, cmd in procs.items():
        if 'MeshGraphNets_main.py' not in cmd or 'spawn_main' in cmd:
            continue
        m = re.search(r'--config\s+(\S+)', cmd)
        name = os.path.basename(m.group(1)) if m else '?'
        name = re.sub(r'^config_(train|infer)_|\.txt$', '', name)
        workers = [c for c, ccmd in procs.items()
                   if 'spawn_main' in ccmd and 'resource_tracker' not in ccmd and ppid(c) == p]
        jobs.append({'pid': p, 'name': name, 'gpu': visible_gpu(p), 'workers': workers})
    return sorted(jobs, key=lambda j: (j['gpu'], j['name']))


def gpu_util_sample(acc):
    if not shutil.which('nvidia-smi'):
        return
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return
    for line in out.splitlines():
        try:
            i, u = (s.strip() for s in line.split(','))
            acc.setdefault(i, []).append(float(u))
        except ValueError:
            pass


def data_wait(pid, n=10):
    """Samples in which the main thread sat in the DataLoader's next()."""
    if not shutil.which('py-spy'):
        return 'no py-spy'
    hits = got = 0
    for _ in range(n):
        try:
            out = subprocess.run(['py-spy', 'dump', '--pid', str(pid)],
                                 capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        blocks = re.split(r'\n(?=Thread )', out)
        main = next((b for b in blocks if '"MainThread"' in b), None)
        if main is None:
            continue
        got += 1
        if re.search(r'_next_data|_get_data|_try_get_data|dataloader\.py', main):
            hits += 1
        time.sleep(0.5)
    return f'{hits}/{got}' if got else 'py-spy failed (permission? try sudo)'


def state(stat_path):
    st = read(stat_path)
    return st.rsplit(')', 1)[1].split()[0] if st else None


def comm(pid, tid):
    return (read(f'/proc/{pid}/task/{tid}/comm') or '?').strip()


def read_bytes(pid):
    io = read(f'/proc/{pid}/io')
    m = re.search(r'^read_bytes:\s+(\d+)', io or '', re.M)
    return int(m.group(1)) if m else 0


def main():
    jobs = find_jobs()
    if not jobs:
        print('no running MeshGraphNets_main.py found')
        return 1
    t0 = time.time()
    before = {j['pid']: (thread_ticks(j['pid']),
                         {w: ticks(f'/proc/{w}/stat') for w in j['workers']},
                         {w: read_bytes(w) for w in j['workers']}) for j in jobs}
    # Scheduler state every 0.1 s: R running, S sleeping (a queue, a lock, a
    # CUDA wait that yields), D uninterruptible (disk). Per thread for the
    # trainer, pooled over the workers.
    tstate = {j['pid']: {} for j in jobs}
    wstate = {j['pid']: {} for j in jobs}
    util, n = {}, 0
    while time.time() - t0 < WINDOW:
        if n % 10 == 0:
            gpu_util_sample(util)
        for j in jobs:
            p = j['pid']
            try:
                tids = os.listdir(f'/proc/{p}/task')
            except OSError:
                tids = []
            for tid in tids:
                s = state(f'/proc/{p}/task/{tid}/stat')
                if s:
                    d = tstate[p].setdefault(tid, {})
                    d[s] = d.get(s, 0) + 1
            for w in j['workers']:
                s = state(f'/proc/{w}/stat')
                if s:
                    wstate[p][s] = wstate[p].get(s, 0) + 1
        n += 1
        time.sleep(0.1)
    dt = time.time() - t0
    lines = [f"gpu_idle {os.uname().nodename} cpus={os.cpu_count()} "
             f"load={os.getloadavg()[0]:.0f} window={dt:.0f}s"]

    def pct(d, k):
        tot = sum(d.values())
        return 100 * d.get(k, 0) / tot if tot else 0

    for j in jobs:
        p = j['pid']
        th0, wk0, rb0 = before[p]
        th1 = thread_ticks(p)
        per = sorted(((100 * (th1[t] - th0.get(t, 0)) / TICK / dt, t) for t in th1), reverse=True)
        wk = sorted((100 * (ticks(f'/proc/{w}/stat') - wk0[w]) / TICK / dt for w in j['workers']),
                    reverse=True)
        busy = [w for w in wk if w > 5]
        mbs = sum(read_bytes(w) - rb0[w] for w in j['workers']) / dt / 1e6
        ws = wstate[p]
        u = util.get(j['gpu'].split(',')[0], [])
        head = f"gpu {j['gpu']} util {sum(u) / len(u):.0f}%" if u else f"gpu {j['gpu']} util ?"
        lines.append(f"{head} | {j['name']} | main {sum(c for c, _ in per):.0f}% | "
                     f"workers {len(busy)} busy: {' '.join(f'{w:.0f}' for w in busy) or '-'} "
                     f"(R {pct(ws, 'R'):.0f}% S {pct(ws, 'S'):.0f}% D {pct(ws, 'D'):.0f}%, "
                     f"disk {mbs:.0f} MB/s) | data-wait {data_wait(p)}")
        for c, t in per[:5]:
            if c < 3:
                break
            d = tstate[p].get(t, {})
            lines.append(f"    thread {comm(p, t)}{' (main)' if t == str(p) else ''}: "
                         f"cpu {c:.0f}% R {pct(d, 'R'):.0f}% S {pct(d, 'S'):.0f}% D {pct(d, 'D'):.0f}%")
    print('==== TYPE THIS BACK ====')
    print('\n'.join(lines))
    print('==== END ====')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
