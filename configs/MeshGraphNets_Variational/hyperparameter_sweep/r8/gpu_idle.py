#!/usr/bin/env python3
"""Why is the GPU idle a third of the time? Measures the running sweep jobs for 20 s.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r8/gpu_idle.py

Standard library only, Linux, run on the node while the jobs train (it only
reads /proc and nvidia-smi; with py-spy on PATH it also samples each job's main
thread 10 times, which pauses it for a few milliseconds each). One line per job:

    ==== TYPE THIS BACK ====
    gpu 2 util 64% | hybrid_base | main 105% (top thread 97%) | workers 4 busy: 99 98 97 99 | data-wait 7/10
    ==== END ====

util     mean nvidia-smi utilization over the window (about DCGM GR_ACT).
main     CPU of the training process, all threads (100% = one core), and its
         busiest thread.
workers  DataLoader workers using >5% CPU (the train loader's), each one's CPU.
data-wait  py-spy samples in which the main thread waited for the next batch.

Reading it:
- workers all near 100% and data-wait high  -> the workers cannot keep up:
  more num_workers.
- workers well below 100% but data-wait high -> the batch is ready but stuck
  between worker and trainer (pin_memory thread / shared-memory hand-off):
  pin_memory False.
- top thread near 100% and data-wait low     -> the trainer's own Python thread
  is the limit (kernel launches, host-side work), not the data.
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


def main():
    jobs = find_jobs()
    if not jobs:
        print('no running MeshGraphNets_main.py found')
        return 1
    t0 = time.time()
    before = {j['pid']: (thread_ticks(j['pid']),
                         {w: ticks(f'/proc/{w}/stat') for w in j['workers']}) for j in jobs}
    util = {}
    while time.time() - t0 < WINDOW:
        gpu_util_sample(util)
        time.sleep(1)
    dt = time.time() - t0
    lines = [f"gpu_idle {os.uname().nodename} cpus={os.cpu_count()} "
             f"load={os.getloadavg()[0]:.0f} window={dt:.0f}s"]
    for j in jobs:
        th0, wk0 = before[j['pid']]
        th1 = thread_ticks(j['pid'])
        per = sorted((100 * (th1[t] - th0.get(t, 0)) / TICK / dt for t in th1), reverse=True)
        wk = sorted((100 * (ticks(f'/proc/{w}/stat') - wk0[w]) / TICK / dt for w in j['workers']),
                    reverse=True)
        busy = [w for w in wk if w > 5]
        u = util.get(j['gpu'].split(',')[0], [])
        lines.append(
            f"gpu {j['gpu']} util {sum(u) / len(u):.0f}% | " if u else f"gpu {j['gpu']} util ? | ")
        lines[-1] += (f"{j['name']} | main {sum(per):.0f}% (top thread {per[0] if per else 0:.0f}%) | "
                      f"workers {len(busy)} busy: {' '.join(f'{w:.0f}' for w in busy) or '-'} | "
                      f"data-wait {data_wait(j['pid'])}")
    print('==== TYPE THIS BACK ====')
    print('\n'.join(lines))
    print('==== END ====')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
