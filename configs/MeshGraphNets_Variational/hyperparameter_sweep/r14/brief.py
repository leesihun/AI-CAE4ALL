#!/usr/bin/env python3
"""A results block short enough to retype by hand, for B300 round 14 (the prior, GNN or
transformer, x the latent regularizer, MMD or KL 10, 100, x what the prior reads of the
geometry, prior_trunk; each on round 12's joint setting).

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r14/brief.py
    python .../r14/brief.py --long     # every number, not for typing
    python .../r14/brief.py --prior a4 # the full prior diagnosis of one setting (default a1)

Standard library only; run it with any Python 3.8+, while the jobs run or
after. It prints a column legend and a numbered job index (read them, do not
type them), then the block to type back. Jobs are numbers (#, the jobs.txt
order); R1-R3 are the base's seeds 42/43/44, runs of rounds 8-10 read where
those rounds left them. The block ends with one line per setting: its mark, w2 per
seed, the mean and, once diagnose_135.sh has run, where the error comes from.

Every setting is one test of one cause of the base's prior error. Once
diagnose_135.sh has left prior_diagnosis_<ho|tr|ts>.json per job
(methods/MeshGraphNets_Variational/misc/prior_diagnosis.py), the PRIOR
DIAGNOSIS section says per setting whether its cause is CONFIRMED, RULED OUT or
still OPEN (the rule is printed with it), and the block carries those decisions
as the `pd` and `pb` lines:

    ==== TYPE THIS BACK ====
    R14 135 31h D5/16 F0          round, node, run hours, done/jobs, failed
    floor 27 19 36                the 256 FE results split 128/128: median, 5%, 95%
    run 6:40%:290 7:38%:295       training: #, % of the epochs done, seconds of the last epoch
    inf 5                         inference running
    wait 6-7                      not started yet
    stop 7                        was running when the run died (run the round again)
    4 F:train CUDA out of memory  a failure: stage and the first error line
    d 4:tr                        a diagnosis part that failed (job:part)
    a0 66/95/75 m78 ho 27/29/78 tr 56/57/82 ts 45/46/101 g0 l100 n551 s2       the base
    a1 + 55/60 m58 ho 27/30/58 tr 56/57/60 ts 45/47/62 g0 l100 n400 s40        joint: mark, w2 per seed, mean, diagnosis
    a2 += 52/61 m57 ho 27/29/57 tr 56/57/59 ts 45/46/60 g0 l100 n90 s300       joint plus one change: marks against a0, a1
    pd - 2 3 1 1 2 2 - 2 - - 1 - 4        the decision per setting a0, a1, ... (a0..a12), then a check digit
    pb - 2 2 1 1 2 2 - 2 - - 2 - 4          the same against the base
    d0 30 4 85 9                          the decoder between two posteriors, base (then d1: joint)
    c1 1 3 8 14 24 u 12 13 5              the causes found in joint's own diagnosis
    pf 4:tr                               a prior-diagnosis part that failed (job:part)
    ==== END ====
"""
import argparse
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True   # importing prior_verdict must not leave a __pycache__ in the method
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
ROUND = os.path.basename(HERE)   # r14
sys.path.insert(0, os.path.join(REPO, "methods", "MeshGraphNets_Variational", "misc"))
try:
    import prior_verdict as PV   # standard library only: the summary, the causes, the typed lines
except ImportError:              # an older checkout: the brief still prints, without the prior diagnosis
    PV = None
# Runs of earlier rounds shown beside this round's: (number, round, job, setting, seed, diag dir under output/).
# r8's run was diagnosed by r9's watcher; r9's and r10's own jobs under output/b300_rN/<job>/diag.
REFS = [
    ("R1", "r8", "hybrid_e360", "base", 42, "b300_r9/ref_r8_hybrid_e360/diag"),
    ("R2", "r9", "hybrid_base_s43", "base", 43, "b300_r9/hybrid_base_s43/diag"),
    ("R3", "r10", "hybrid_base_s44", "base", 44, "b300_r10/hybrid_base_s44/diag"),
]
NODE = "135"
EPOCH_RE = re.compile(r"^Elapsed: ([\d.]+)s Epoch (\d+)")
LEGEND = """\
COLUMNS (do not type)   every number x100, rounded ("27" = 0.27)
  aN    one line per setting (SETTINGS below), the last lines of the block:
        mark  blank before the setting has two seeds. a1 (joint) has one, against the base (a0).
              a2-a6 have two: against the base, then against a1 (joint, the GNN prior with the MMD).
              a7, a8 (the transformer prior + KL 10, + prior_trunk decoder) have their second
              mark against a6 (the transformer prior with the MMD).
              Read a1, a2, a3 as one line (the GNN prior: MMD, KL 10, KL 100); a4, a5 as what
              the prior reads of the geometry (a4 the decoder's coarsest features, a5 its own
              multiscale trunk) against a1; a6, a7, a8 as the transformer prior: MMD, KL 10,
              the decoder's features. a9 (noise: no encoder, no prior, z ~ N(0,I), CRPS of 2
              draws) is marked against the base only; a10 (the same, latent_type global) also
              against a9. a11 (KL 10 + prior_trunk decoder) is marked against a2 (joint_kl10);
              a12 (noise, z_l read at the coarsest level only) against a9.
              a9, a10, a12 have the w2 only (no ho/tr/ts: nothing to diagnose).
              + every seed beats every seed of the other, - every seed loses, = neither
        w2    per seed, in seed order 42/43(/44). a0 is r8's hybrid_e360, r9's hybrid_base_s43
              and r10's hybrid_base_s44 (rows R1-R3). One setting's seeds differ by about 30.
        m     the mean over the seeds
        ho    the 8 held-out geometries, mean over the seeds: floor / posterior sample / prior
              (W2 per geometry, median; from posterior_vs_prior_groups.py, diagnose_135.sh)
        tr    the same on 16 training geometries x 64 draws
        ts    the same on the 15 geometries of the test split x 64 draws (never trained on,
              inside the training ranges)
        g l   how much the paired correlation (decoded draw vs its FE result, within a geometry)
              falls when z_g, or z_l, is swapped with another draw's: ~0 = the decoder ignores
              that part; large = that part carries the draw ("g-" = no z_g: latent_type local)
        n     is the latent in order: a z ~ N(0,I), never seen by the prior, decoded on the
              held-out geometries, scored like the prior (third number of ho). Equal to the
              prior's score = the decoder reads any N(0,I) draw as a valid result, the
              geometry does not move the latent, the prior has nothing to learn. The base:
              a0's n (551; r8's 240-epoch base had 578).
        s     the posterior's width over the spread of its means, z_l, held-out (x100).
              ~0 = every sample is a point with gaps between them; 100 or more = the
              samples of one geometry overlap. The base: a0's s (2; r8's 240-epoch base had 51).
        "-"   not diagnosed yet (or the part failed: a "d" line)
  run   #:% of the epochs done:seconds of the last epoch (r8 alone on a card: 277)
  #     job number (JOBS below, the jobs.txt order)
  w2    per geometry, W2 between the 1024 draws and the 256 FE results
        over the FE sd, then the RMS over the 8 held-out geometries. 0 = identical;
        ~15 a perfect sampler (estimate); 27 the floor; 100 = the exact FE mean
        with no spread, or the right shape shifted by 1 sd
  w2h   (--long) 128 draws against a random half (128) of the FE results, median of 200
        splits: compare it with the floor line (a perfect sampler sits inside it)
  b     (--long) mean offset    s  width error |sd ratio - 1|    w2^2 = b^2 + s^2 + shape^2
  sdg   (--long) geometric mean of the sd ratios (100 = right width, below = too narrow)
  h     (--long) training hours ("-" = inference only, on an earlier checkpoint)
  floor 128 of each geometry's FE results against the other 128: median, 5%, 95%
  pd    the decision per setting, a0 first (PRIOR DIAGNOSIS below): 1 CONFIRMED, 2 RULED OUT,
        3 OPEN, 4 WORSE, 5 MIXED, "-" = not diagnosed (a0 is the base: always "-").
        The last number is a check digit (the last digit of the sum of the others).
  pb    the prior against the base's, same order: 1 better, 2 no different, 4 worse, 5 mixed
  dN    the decoder halfway between two posteriors of one geometry, setting aN: its score
        (compare with the posterior's), how far its output lies outside the two ends, where it
        lies between them (0 = their average, 100 = one of them), check digit
  cN    the causes found in setting aN's own diagnosis, `u`, those that cannot be told, check digit
        (dN, cN: a0 has d only; then a1, and the CONFIRMED setting with the best prior on ho)
  pf    prior-diagnosis parts that failed (job:part; R1-R3 = the base's runs)
"""

# One aN line read out, printed under the legend. Korean where the terminal can print it.
EXAMPLE = "a0 66/95/75 m78 ho 27/29/78 tr 56/57/82 ts 45/46/101 g0 l100 n551 s2"
READING_KO = """\
ONE aN LINE (do not type)   %s
  66/95/75   seed 42/43/44 각각의 held-out W2
  m          그 평균
  ho a/b/c   held-out geometry 8개에서 floor / posterior / prior
  tr, ts     학습 geometry 16개, test split geometry 15개에서 같은 세 값 (draw 64개라 floor가 높습니다)
  읽는 순서   1) 셋째 값(prior)이 base보다 내려갔는가: ho, ts, tr 순으로 봅니다.
             2) 둘째 값(posterior)이 floor에 그대로 있는가. prior의 loss가 encoder로 들어가면
                posterior가 geometry만으로 맞히기 쉬운 쪽으로 밀립니다. 둘째 값이 floor에서
                올라갔다면 encoder가 draw의 정보를 버린 것이고, 그 설정의 prior 점수는 믿을 수 없습니다.
             3) a2~a6는 a1(joint)에 바꾼 것을 더한 설정입니다. 기호가 둘이면 첫째는 base,
                둘째는 a1과의 비교입니다 (a7, a8은 a6, 즉 transformer prior와의 비교).
                같은 설정의 seed끼리도 30쯤 다르니 기호를 먼저 봅니다.
             4) KL 설정 (a2, a3, a7)은 posterior 자체가 넓어서 둘째 값이 floor보다 조금 높은 것이
                정상입니다. 셋째 값(prior)과의 차이를 봅니다. KL 100에서 둘째 값이 100 근처까지
                올라갔다면 decoder가 z를 읽지 않게 된 것입니다 (l도 0에 가까워집니다).
             5) n, s는 latent가 정리되었는지입니다. n이 ho의 셋째 값(prior)까지 내려오면 N(0,I)에서
                아무렇게나 뽑은 z도 decoder가 제대로 읽는다는 뜻입니다. n은 내려갔는데 prior가
                그대로면 "latent 정리"가 원인이 아니고, n과 prior가 같이 내려가면 원인이 맞습니다.
             6) 세 묶음으로 읽습니다: a1, a2, a3 = GNN prior에서 MMD, KL 10, KL 100.
                a4, a5 = prior가 geometry를 읽는 부분만 바꾼 것 (a4: decoder의 coarsest 특징,
                a5: prior 자체의 multiscale trunk). a6, a7, a8 = transformer prior에서 MMD, KL 10,
                decoder의 coarsest 특징. a9, a10 = encoder도 prior도 없는 noise 방식 (z ~ N(0,I),
                draw 2개의 CRPS로 학습; a10은 latent_type global, a12는 z_l을 coarsest level에만).
                w2만 있고 base와 비교합니다. a11 = KL 10 + decoder 특징 (a2와 비교).
             7) a4, a5, a8 (prior_trunk)은 tr, ts, ho의 셋째 값(prior)을 a1 (a8은 a6)과 비교합니다.
                a4와 a5가 같이 내려가면 "prior가 geometry를 넓게 본 것"이 이유이고, a4만 내려가면
                "decoder가 쓰는 특징"이 이유입니다. tr만 내려가고 ts, ho가 그대로면 학습 geometry를
                외운 것입니다. 둘 다 그대로면 prior의 입력이 원인이 아닙니다.
""" % EXAMPLE
READING_EN = """\
ONE aN LINE (do not type)   %s
  66/95/75   the held-out W2 of seeds 42/43/44
  m          their mean
  ho a/b/c   on the 8 held-out geometries: floor / posterior / prior
  tr, ts     the same three on 16 training geometries and on the 15 test-split geometries
             (64 draws each, so their floors are high)
  ORDER      1) is the third number (the prior) below the base's: ho, then ts, then tr.
             2) is the second number (the posterior) still at the floor. With the prior's loss
                in the encoder the posterior is pushed towards what the geometry alone predicts.
                A second number above the floor = the encoder dropped the draw's information,
                and that setting's prior score cannot be trusted.
             3) a2-a6 are a1 (joint) plus a change. Of two marks the first is against
                the base, the second against a1 (a7, a8: against a6, the transformer prior).
                One setting's seeds differ by about 30, so read the marks first.
             4) The KL settings (a2, a3, a7) have a wider posterior by design, so their second
                number sits somewhat above the floor; read the gap to the third. A second number
                near 100 at KL 100 = the decoder stopped reading z (l falls towards 0 too).
             5) n and s say whether the latent is in order. n down at the third number of ho
                (the prior) = the decoder reads any N(0,I) draw correctly. n down with the
                prior unchanged = the order of the latent is not the cause; both down = it is.
             6) Read three groups: a1, a2, a3 = MMD, KL 10, KL 100 with the GNN prior.
                a4, a5 = only what the prior reads of the geometry changes (a4: the decoder's
                coarsest features, a5: the prior's own multiscale trunk). a6, a7, a8 = the
                transformer prior with MMD, KL 10, the decoder's coarsest features. a9, a10 = the
                noise way, no encoder and no prior (z ~ N(0,I), trained on the CRPS of 2 draws;
                a10 latent_type global, a12 z_l at the coarsest level only): the w2 only, read
                against the base. a11 = KL 10 + the decoder's features, read against a2.
             7) a4, a5, a8 (prior_trunk): compare the third number (the prior) on tr, ts, ho
                with a1's (a8: with a6's). a4 and a5 both down = the wider view of the geometry
                is the reason; a4 alone down = the decoder's own features are. tr down with ts
                and ho unchanged = it memorized the training geometries. Neither down = the
                prior's input is not the cause.
""" % EXAMPLE


def reading():
    """The Korean block, or the English one where stdout cannot encode Korean."""
    try:
        READING_KO.encode(getattr(sys.stdout, "encoding", None) or "ascii")
    except (UnicodeError, LookupError):
        return READING_EN
    return READING_KO


def cfg(path):
    """Flat `key value` config -> {lowercased key: first token}, comments stripped."""
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.split("%", 1)[0].split("#", 1)[0].strip()
            parts = line.split(None, 1)
            if len(parts) == 2:
                out.setdefault(parts[0].lower(), parts[1].split(",")[0].split()[0])
    return out


def role(path):
    try:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("% Role:"):
                    return line[len("% Role:"):].strip()
    except OSError:
        pass
    return ""


def method_path(value):
    """A config path is relative to methods/<Name>/ (the native cwd), so '../../x' is REPO/x."""
    value = value.replace("\\", "/")
    return os.path.join(REPO, value[6:] if value.startswith("../../") else value)


def x100(v):
    """"-" = not there yet; "nan" = the number exists and is NaN or infinite (type it as it is)."""
    if v is None:
        return "-"
    if v != v or v in (float("inf"), float("-inf")):
        return "nan"
    return str(int(round(v * 100)))


def load_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_tail(path, size=65536):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - size))
            return fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read().strip()
    except OSError:
        return None


def last_epoch(log):
    """(epochs done, seconds) from the last `Elapsed: Xs Epoch k` line (k is 0-based)."""
    for line in reversed(read_tail(log)):
        m = EPOCH_RE.match(line)
        if m:
            return int(m.group(2)) + 1, float(m.group(1))
    return None, None


def epoch_seconds(log):
    """Seconds of the last epoch: the difference of the last two `Elapsed:` lines."""
    found = []
    for line in reversed(read_tail(log)):
        m = EPOCH_RE.match(line)
        if m:
            found.append(float(m.group(1)))
            if len(found) == 2:
                return found[0] - found[1]
    return None


def pid_alive(pid_file):
    pid = read_text(pid_file)
    if not pid:
        return False
    if os.name != "posix":
        return True   # os.kill(pid, 0) on Windows terminates the process; assume alive
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


class Job:
    """One train/infer config pair plus everything its run left on disk."""

    def __init__(self, num, jid, train_cfg, infer_cfg, status_dir):
        ci = cfg(infer_cfg if os.path.isabs(infer_cfg) else os.path.join(REPO, infer_cfg))
        self.infer_only = train_cfg == "-"
        c = ci if self.infer_only else cfg(train_cfg if os.path.isabs(train_cfg) else os.path.join(REPO, train_cfg))
        self.num, self.jid = num, jid
        self.role = role(infer_cfg if os.path.isabs(infer_cfg) else os.path.join(REPO, infer_cfg))
        self.model = method_path(c["modelpath"])
        self.log = method_path(c["log_file_dir"])
        self.infer_dir = method_path(ci["inference_output_dir"])
        self.epochs = int(c.get("training_epochs", 0))
        self.status = {}
        for key in ("state", "err", "train_start", "train_end", "inferfail"):
            self.status[key] = read_text(os.path.join(status_dir, f"{jid}.{key}"))

    def hours(self):
        if self.infer_only:
            return None
        s, e = self.status.get("train_start"), self.status.get("train_end")
        try:
            if s:
                return ((float(e) if e else time.time()) - float(s)) / 3600
        except ValueError:
            pass
        secs = last_epoch(self.log)[1]
        return None if secs is None else secs / 3600

    def state(self):
        st = self.status.get("state")
        if st:
            return st
        if os.path.isfile(os.path.join(self.infer_dir, "spread_metrics.json")):
            return "done"
        return "queued"

    def failure(self):
        st = self.state()
        if st.startswith("fail"):
            stage = st.split()[1] if len(st.split()) > 1 else "?"
            return stage, (self.status.get("err") or "").replace("\n", " ")
        if self.status.get("inferfail"):
            return "infer", (self.status.get("err") or "").replace("\n", " ")
        return None

    def percent(self):
        done = last_epoch(self.log)[0]
        if done is None or not self.epochs:
            return None
        return int(100 * done / self.epochs)


def ranges(nums):
    """[3, 4, 5, 9] -> '3-5 9'."""
    out, nums = [], sorted(nums)
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(str(nums[i]) if j == i else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return " ".join(out)


# The settings, a0 = the base. setting -> the one it is marked against (after the base).
SETTINGS = {
    "base": None,
    "joint": "base",                    # prior_freeze_epoch 0 + prior_grad_to_encoder 1 (GNN prior, MMD)
    "joint_kl10": "joint",              # ... latent_reg kl, kl_weight 10
    "joint_kl100": "joint",             # ... kl_weight 100
    "joint_dec": "joint",               # ... prior_trunk decoder (the simulator's descending arm at z = 0)
    "joint_ms": "joint",                # ... prior_trunk multiscale (the prior's own descending arm)
    "joint_tfk8": "joint",              # ... transformer prior, 8 flow draws, z_l at the coarsest level
    "joint_tfk8_kl10": "joint_tfk8",    # tf_k8 + kl_weight 10
    "joint_tfk8_dec": "joint_tfk8",     # tf_k8 + prior_trunk decoder
    "noise": "base",                    # stochastic_mode noise: no encoder, no prior, CRPS of 2 N(0,I) draws
    "noise_global": "noise",            # ... latent_type global (no z_l)
    "joint_kl10_dec": "joint_kl10",     # KL 10 + prior_trunk decoder (added with 3 jobs per card)
    "noise_coarse": "noise",            # noise, z_l at the coarsest level only (local_inject coarsest)
}
# Settings with no posterior and no prior (stochastic_mode noise): diagnose_135.sh skips them,
# so their aN line has the w2 only and they take no part in the prior diagnosis.
NO_PRIOR = ("noise", "noise_global", "noise_coarse")
JID_RE = re.compile(r"^(?:hybrid_)?(.+)_s(\d+)$")   # hybrid_<setting>_s<seed>

# Each setting tests one cause of the base's prior error (the keys of prior_verdict.CAUSES):
# setting -> (cause, True = its change holds by construction / False = whether the cause is gone
#             is read from the setting's own diagnosis, what it changes against its reference).
PRIOR_TESTS = {
    "joint": ("frozen", True, "the prior trained together with the encoder"),
    "joint_kl10": ("points", False, "KL 10 in place of MMD (a wider posterior)"),
    "joint_kl100": ("points", False, "KL 100 in place of MMD (a wider posterior)"),
    "joint_dec": ("unseen", False, "the prior reads the decoder's coarsest-level features"),
    "joint_ms": ("unseen", False, "the prior's own trunk goes down to the coarsest level"),
    "joint_tfk8": ("fit", True, "a transformer prior trained on 8 flow draws per sample"),
    "joint_tfk8_kl10": ("points", False, "KL 10 in place of MMD, with the transformer prior"),
    "joint_tfk8_dec": ("unseen", False, "the decoder's coarsest-level features, with the transformer prior"),
    "joint_kl10_dec": ("unseen", False, "the decoder's coarsest-level features, with KL 10"),
}
CODES = {1: "CONFIRMED", 2: "RULED OUT", 3: "OPEN", 4: "WORSE", 5: "MIXED"}
SETS = ("tr", "ts", "ho")   # the order of every tr/ts/ho triple below
SEED_HALF = 0.15            # half the seed range assumed for a setting with one seed (the base: 66/95/75)
PRIOR_LEGEND = """\
PRIOR DIAGNOSIS (do not type)   each setting is one test of one cause of the base's prior error.
  prior      the prior's score (W2 over the FE sd, x100, seed mean) on tr / ts / ho; lower is better
  marks      against its reference, then against the base (a0), on tr ts ho:
             + lower by more than `need`, - higher by more than `need`, = neither, . no data
  need       the larger of the floor's scatter and half the seed range of either side (15 where
             a side has one seed), x100, on ho. A gain below it cannot be seen with two seeds.
  result     from the marks against the reference. better = + on all sets but at most one and - on
             none; worse = the same with - and +.
             CONFIRMED  better: acting on this cause helps
             RULED OUT  not better, and the change did what it aims at: this is not what limits the
                        prior (within `need`)
             OPEN       not better, and the cause still shows in this setting's own diagnosis: the
                        change did not remove it, nothing is learned about the cause
             WORSE      worse      MIXED  the sets disagree
  cause      the cause in this setting's own diagnosis: gone / shows / ?; set = the change holds
             by construction
  post       the posterior's score above the floor on ho: above `need` = the encoder dropped part
             of the draw, and the prior's score is no longer only the prior's
"""


def diag_scores(path):
    """(floor, posterior sample, prior W2 medians, {g_shuffle drop, l_shuffle drop},
    {n: W2 median of a decoded z ~ N(0,I), s: sigma / sd(means) of z_l}) of one diagnosis."""
    d = load_json(path)
    if not d:
        return None
    sc = d.get("scores") or {}
    w = [((sc.get(e) or {}).get("w2") or {}).get("p50") for e in ("floor", "posterior_sample", "prior")]
    pair = d.get("paired_within_geometry") or {}
    r_mu = (pair.get("posterior_mean") or {}).get("corr")
    drop = {p: (r_mu - pair[f"{p}_shuffle"]["corr"]) if r_mu is not None and f"{p}_shuffle" in pair
            else None for p in ("g", "l")}
    org = {"n": ((sc.get("std_normal") or {}).get("w2") or {}).get("p50"),
           "s": ((d.get("latent_geomean_over_groups") or {}).get("l") or {}).get("sigma_over_sd_means")}
    return w, drop, org


def mean(vals):
    vals = [v for v in vals if v is not None and v == v]
    return sum(vals) / len(vals) if vals else None


def setting_lines(settings):
    """settings: {setting: [(seed, w2 or None, ho, tr, ts)]} -> the aN lines."""
    def w2s(setting):
        return [r[1] for r in sorted(settings.get(setting, [])) if r[1] is not None]

    def mark(w, other):
        if len(w) < 2 or len(other) < 2 or any(v != v for v in w + other):   # a NaN seed: no mark
            return " "
        return "+" if max(w) < min(other) else "-" if min(w) > max(other) else "="

    out = []
    for k, (setting, anchor) in enumerate(SETTINGS.items()):
        rows = sorted(settings.get(setting, []))
        if not rows:
            continue
        w = w2s(setting)
        marks = ""
        if setting != "base":
            marks = mark(w, w2s("base")) + (mark(w, w2s(anchor)) if anchor != "base" else "")
            marks = marks.rstrip() + " " if marks.strip() else ""
        m = float("nan") if any(v != v for v in w) else mean(w)   # a NaN seed: no mean either
        line = f"a{k} {marks}" + "/".join(x100(r[1]) for r in rows) + f" m{x100(m)}"
        for tag, idx in (("ho", 2), ("tr", 3), ("ts", 4)):
            ds = [r[idx] for r in rows if r[idx]]
            line += f" {tag} " + "/".join(x100(mean([d[0][i] for d in ds])) for i in range(3))
            if tag == "ho":
                gl = " ".join(f"{p}{x100(mean([d[1][p] for d in ds]))}" for p in ("g", "l"))
                gl += " " + " ".join(f"{p}{x100(mean([d[2][p] for d in ds]))}" for p in ("n", "s"))
        out.append(f"{line} {gl}")
    return out


# ───────────────────────────────────────────────────────────── the prior diagnosis, per setting

def prior_by_setting(refs, jobs, out_root):
    """{setting: ({part: seed-mean summary}, [seeds])} from each run's prior_diagnosis_<part>.json."""
    every = PV.PARTS + (PV.JOINT,)
    got = {}
    for j in refs + jobs:
        if j in refs:
            setting, seed, ddir = j.setting, j.seed, j.ddir
        else:
            m = JID_RE.match(j.jid)
            if not m:
                continue
            setting, seed, ddir = m.group(1), int(m.group(2)), os.path.join(out_root, j.jid, "diag")
        parts = {}
        for part in every:
            doc = load_json(os.path.join(ddir, f"prior_diagnosis_{part}.json"))
            try:
                parts[part] = PV.summarize(doc) if doc else None
            except (KeyError, TypeError, AttributeError, ValueError):
                parts[part] = None   # cut off while writing
        if any(parts[p] for p in PV.PARTS):
            got.setdefault(setting, []).append((seed, parts))
    out = {}
    for setting, rows in got.items():
        rows.sort(key=lambda r: r[0])
        merged = {part: PV.merge([p[part] for _, p in rows]) for part in every}
        out[setting] = ({k: v for k, v in merged.items() if v}, [seed for seed, _ in rows])
    return out


def seed_half(s):
    q = [v for v in (s.get("seed_Q") or []) if PV.fin(v)]
    return (max(q) - min(q)) / 2 if len(q) > 1 else SEED_HALF


def against(mine, other):
    """{set: (prior here, prior there, need, -1 lower / 0 neither / +1 higher)} over the sets both have."""
    out = {}
    for part in SETS:
        s, r = (mine or {}).get(part), (other or {}).get(part)
        qs, qr = ((s or {}).get("sc") or {}).get("Q"), ((r or {}).get("sc") or {}).get("Q")
        if not (PV.fin(qs) and PV.fin(qr)):
            continue
        need = max(s["noise"], r["noise"], seed_half(s), seed_half(r))
        out[part] = (qs, qr, need, -1 if qs < qr - need else 1 if qs > qr + need else 0)
    return out


def outcome(cmp):
    """better / worse / same / mixed from the marks of against(); None with no set to compare."""
    if not cmp:
        return None
    lower = sum(1 for v in cmp.values() if v[3] < 0)
    higher = sum(1 for v in cmp.values() if v[3] > 0)
    most = max(1, len(cmp) - 1)
    if lower >= most and not higher:
        return "better"
    if higher >= most and not lower:
        return "worse"
    return "same" if not lower and not higher else "mixed"


def cause_state(setting, parts):
    """Is the cause this setting acts on gone in its own diagnosis: set / gone / shows / ?"""
    key, by_construction, _what = PRIOR_TESTS[setting]
    if by_construction:
        return "set"
    status = {k: st for _n, k, _name, st, _why in PV.causes(parts)}.get(key)
    return {"no": "gone", "YES": "shows"}.get(status, "?")


def marks(cmp):
    return "".join("." if p not in cmp else "+" if cmp[p][3] < 0 else "-" if cmp[p][3] > 0 else "="
                   for p in SETS)


def prior_report(prior, fails):
    """(the PRIOR DIAGNOSIS section, the lines of the typed block) from prior_by_setting()."""
    names = list(SETTINGS)
    num = {s: k for k, s in enumerate(names)}
    get = lambda setting: (prior.get(setting) or ({}, []))[0]   # noqa: E731
    triple = lambda parts: "/".join(x100(((parts.get(p) or {}).get("sc") or {}).get("Q")) for p in SETS)  # noqa: E731

    def post(parts):
        s = parts.get("ho") or parts.get("ts") or parts.get("tr") or {}
        sc = s.get("sc") or {}
        return x100(sc["P"] - sc["floor"]) if PV.fin(sc.get("P")) and PV.fin(sc.get("floor")) else "-"

    text = [PRIOR_LEGEND]
    pd, pb, groups, waiting = [None], [None], {}, []
    width = max(len(s) for s in names)
    base = get("base")
    seeds = "/".join(str(s) for s in (prior.get("base") or ({}, []))[1])
    text.append(f"  a0  {'base':<{width}}  prior {triple(base)}  post {post(base)}"
                + (f"  seeds {seeds}" if base else "  (not diagnosed yet)"))
    for setting in names[1:]:
        k, ref = num[setting], SETTINGS[setting]
        if setting in NO_PRIOR:   # no prior to diagnose: "-" in pd and pb
            pd.append(None)
            pb.append(None)
            text.append(f"  a{k:<2} {setting:<{width}}  no prior (stochastic_mode noise): the w2 line only")
            continue
        key, _by, what = PRIOR_TESTS[setting]
        parts = get(setting)
        cause = f"{PV.NUMBER[key]} {key}"
        head = f"  a{k:<2} {setting:<{width}}  vs a{num[ref]:<2} {cause:<10}"
        cmp, cmp_base = against(parts, get(ref)), against(parts, base)
        res = outcome(cmp)
        if res is None:
            pd.append(None)
            pb.append({"better": 1, "same": 2, "worse": 4, "mixed": 5}.get(outcome(cmp_base)))
            waiting.append(f"a{k}")
            why = "not diagnosed yet" if not parts else f"its reference a{num[ref]} is not diagnosed yet"
            text.append(f"{head} -          ({why})")
            continue
        state = cause_state(setting, parts)
        code = {"better": 1, "worse": 4, "mixed": 5}.get(res) or (2 if state in ("gone", "set") else 3)
        pd.append(code)
        pb.append({"better": 1, "same": 2, "worse": 4, "mixed": 5}.get(outcome(cmp_base)))
        groups.setdefault(code, []).append(f"a{k} {setting} ({cause})")
        need = (cmp.get("ho") or next(iter(cmp.values())))[2]
        text.append(f"{head} {CODES[code]:<10} prior {triple(parts)}  ref {triple(get(ref))}  "
                    f"{marks(cmp)} {marks(cmp_base)}  need {x100(need)}  cause {state:<5}  post {post(parts)}"
                    f"   {what}")
    text.append("")
    for code in sorted(CODES):
        if groups.get(code):
            text.append(f"  {CODES[code]:<10} " + ", ".join(groups[code]))
    if waiting:
        text.append(f"  {'no result':<10} " + " ".join(waiting))
    scored = [(s["sc"]["Q"], setting) for setting in names
              for s in [get(setting).get("ho")] if s and PV.fin(s["sc"].get("Q"))]
    if scored:
        q, best = min(scored)
        text.append(f"  best prior on ho: a{num[best]} {best} {x100(q)}"
                    + "".join(f", {other} {x100(get(other)['ho']['sc']['Q'])}" for other in ("base", "joint")
                              if other != best and (get(other).get("ho") or {}).get("sc", {}).get("Q") is not None))
    text.append("")

    typed = []
    if any(v is not None for v in pd[1:]):
        typed += [PV.checked("pd", pd), PV.checked("pb", pb)]
    confirmed = sorted((get(s)["ho"]["sc"]["Q"] if get(s).get("ho") else float("inf"), s)
                       for s in names[2:] if pd[num[s]] == 1)
    shown = [s for s in ("base", "joint") if get(s)] + [s for _q, s in confirmed[:1]]   # the best CONFIRMED one
    for setting in shown:
        parts, k = get(setting), num[setting]
        label = f"a{k} {setting} (seed {'/'.join(str(s) for s in prior[setting][1])})"
        full = PV.verdict(parts, label)
        if setting != "base":   # the base's own diagnosis is round 11's
            text += PV.verdict(parts, label, short=True)
        for i, line in enumerate(full):
            if line.startswith("decoder "):
                text += ([f"  a{k} {setting}:"] if setting == "base" else []) + full[i:i + 2]
        text.append("")
        for line in PV.block(parts):
            if line.startswith("de "):
                typed.append(f"d{k} " + line[3:])
            elif line.startswith("cs ") and setting != "base":
                typed.append(f"c{k} " + line[3:])
    if fails:
        text.append("  failed parts of the prior diagnosis: " + " ".join(fails)
                    + f"   (reasons: output/b300_{ROUND}/_diag/prior_fail.txt)")
        text.append("")
        typed.append("pf " + " ".join(fails))
    return text, typed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", action="store_true", help="also print shape and crps, and every metric per job")
    ap.add_argument("--prior", nargs="?", const="a1", metavar="aN",
                    help="the full prior diagnosis of one setting (default a1, joint)")
    ap.add_argument("--refs", action="store_true",
                    help="for diagnose_135.sh: number, infer config, diag dir of the base's runs")
    args = ap.parse_args()
    if args.refs:
        for num, rnd, name, _setting, _seed, ddir in REFS:
            infer = os.path.join(HERE, "..", rnd, f"config_infer_{name}.txt")
            print(num, os.path.relpath(infer, REPO).replace(os.sep, "/"), f"output/{ddir}")
        return 0

    table = os.path.join(HERE, "jobs.txt")
    if not os.path.isfile(table):
        raise SystemExit(f"no jobs.txt ({table}); generate it with make_sweep.py")
    out_root = os.path.join(REPO, "output", f"b300_{ROUND}")
    status_dir = os.path.join(out_root, "run_logs", NODE, "status")
    tag = ROUND.upper()
    rows = []
    with open(table, encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5 and not parts[0].startswith("#") and parts[0] == NODE:
                rows.append(parts[:5])
    jobs = [Job(k, j, t, i, status_dir) for k, (_n, j, _m, t, i) in enumerate(rows, 1)]
    # R1-R3: the base's runs of rounds 8-10, read where those rounds left them.
    refs = []
    for num, rnd, name, setting, seed, ddir in REFS:
        rdir = os.path.join(HERE, "..", rnd)
        r = Job(num, f"{rnd}:{name}", os.path.join(rdir, f"config_train_{name}.txt"),
                os.path.join(rdir, f"config_infer_{name}.txt"),
                os.path.join(REPO, "output", f"b300_{rnd}", "run_logs", NODE, "status"))
        r.setting, r.seed, r.ddir = setting, seed, os.path.join(REPO, "output", *ddir.split("/"))
        r.role = f"{setting}, seed {seed}: round {rnd[1:]}'s {name} (not run here)"
        refs.append(r)

    # The prior diagnosis must never take the brief down: without it the block still prints.
    prior, prior_err = {}, None if PV else "methods/MeshGraphNets_Variational/misc/prior_verdict.py is missing"
    if PV:
        try:
            prior = prior_by_setting(refs, jobs, out_root)
        except Exception as exc:   # noqa: BLE001
            prior_err = f"{type(exc).__name__}: {exc}"
    pfail = read_text(os.path.join(out_root, "_diag", "prior_fail.txt")) or ""
    nums = {j.jid: j.num for j in jobs}
    pfails = [f"{nums.get(p[1], p[1])}:{p[2]}" for p in (l.split() for l in pfail.splitlines())
              if len(p) > 2 and p[0] == "FAIL"]

    if args.prior:
        names = list(SETTINGS)
        try:
            k = int(args.prior.lstrip("aA"))
            setting = names[k]
        except (ValueError, IndexError):
            raise SystemExit(f"--prior takes a0..a{len(names) - 1}")
        if prior_err:
            raise SystemExit(f"the prior diagnosis cannot be read: {prior_err}")
        if setting not in prior:
            print(f"no prior_diagnosis_*.json of a{k} {setting} yet (diagnose_135.sh writes them)")
            return 0
        parts, seeds = prior[setting]
        for line in PV.verdict(parts, f"a{k} {setting} (seed {'/'.join(str(s) for s in seeds)})"):
            print(line)
        print()
        print(PV.BLOCK_LEGEND)
        print("==== TYPE THIS BACK ====")
        print(f"{tag}P {NODE} a{k} S{len(seeds)}")
        for line in PV.block(parts):
            print(line)
        print("==== END ====")
        return 0

    print(LEGEND)
    print(reading())
    print("JOBS (do not type)")
    width = max(len(j.jid) for j in refs + jobs)
    for j in refs + jobs:
        print(f"  {j.num:>2} {j.jid:<{width}}  {j.role}")
    print()
    print("SETTINGS (do not type)")
    for k, (setting, anchor) in enumerate(SETTINGS.items()):
        print(f"  a{k} {setting}" + (f"   (vs a{list(SETTINGS).index(anchor)} {anchor})" if anchor else ""))
    print()

    abort = read_text(os.path.join(status_dir, "_abort"))
    if abort:
        print("==== TYPE THIS BACK ====")
        print(f"{tag} {NODE} ABORT {abort.splitlines()[0]}")
        print("==== END ====")
        return 0

    alive = any(pid_alive(os.path.join(out_root, f)) for f in os.listdir(out_root)
                if f.startswith(f"run_{NODE}") and f.endswith(".pid")) if os.path.isdir(out_root) else False
    ranked, infl, floor, k_ref = [], [], None, None
    settings = {}
    run, inf, wait, stop, fails = [], [], [], [], []
    n_done = 0
    for j in refs + jobs:
        is_ref = j in refs
        if is_ref and j.state() != "done":
            continue
        f = j.failure()
        st = j.state()
        if f:
            fails.append(f"{j.num} F:{f[0]} {f[1][:40]}".rstrip())
            if st != "done":
                continue
        if st in ("train", "infer"):
            if not alive:
                stop.append(j.num)
            elif st == "train":
                p, sec = j.percent(), epoch_seconds(j.log)
                run.append(f"{j.num}:{'-' if p is None else p}%:{'-' if sec is None else f'{sec:.0f}'}")
            else:
                inf.append(j.num)
            continue
        if st != "done":
            wait.append(j.num)
            continue
        n_done += f is None and not is_ref
        full = load_json(os.path.join(j.infer_dir, "spread_w2.json")) or {}
        mj = JID_RE.match(j.jid)
        setting, seed = (j.setting, j.seed) if is_ref else (mj.group(1), int(mj.group(2))) if mj else (None, None)
        if setting:
            ddir = j.ddir if is_ref else os.path.join(out_root, j.jid, "diag")
            settings.setdefault(setting, []).append(
                (seed, (full.get("summary") or {}).get("w2"),
                 *(diag_scores(os.path.join(ddir, f"posterior_vs_prior_groups_{t}.json")) for t in ("ho", "tr", "ts"))))
        m = load_json(os.path.join(j.infer_dir, "spread_metrics.json")) or {}
        w2, noise = full.get("summary") or {}, full.get("noise") or {}
        if noise.get("floor") and floor is None:
            floor, k_ref = noise["floor"].get("w2"), noise.get("k")
        half = ((noise.get("model") or {}).get("w2") or {}).get("p50")
        h = j.hours()
        cols = [x100(w2.get("w2")), x100(half), x100(w2.get("bias")), x100(w2.get("sd_gap")),
                x100(w2.get("sd_ratio_geomean")), "-" if h is None else f"{h:.0f}"]
        if args.long:
            cols += [x100(w2.get("shape")), x100(m.get("crps_norm"))]
        shown = cols if args.long else cols[:1]
        ranked.append((w2.get("w2", float("inf")), str(j.num), f"{j.num} " + " ".join(shown) + ("" if w2 else " noplot")))
        for lam, b in sorted((full.get("by_inflation") or {}).items(), key=lambda kv: float(kv[0])):
            s = b.get("summary") or b
            bh = (((b.get("noise") or {}).get("model") or {}).get("w2") or {}).get("p50")
            icols = [x100(s.get("w2")), x100(bh), x100(s.get("bias")), x100(s.get("sd_gap")),
                     x100(s.get("sd_ratio_geomean"))]
            infl.append(f"{j.num}@{lam} " + " ".join(icols if args.long else icols[1:2]))
        if args.long:
            print(f"  {j.num:>2}: " + " ".join(f"{k}={v:.4f}" for k, v in sorted(w2.items())
                                                     if isinstance(v, (int, float)))
                  + " | " + " ".join(f"{k}={v:.4f}" for k, v in sorted(m.items()) if isinstance(v, (int, float))))

    s, e = read_text(os.path.join(status_dir, "_start")), read_text(os.path.join(status_dir, "_end"))
    if alive:   # the first run's _end while a second run (LANE) still goes
        e = None
    try:
        hours = f"{((float(e) if e else time.time()) - float(s)) / 3600:.0f}" if s else "-"
    except ValueError:
        hours = "-"
    ptyped = []
    if prior and not prior_err:
        try:
            ptext, ptyped = prior_report(prior, pfails)
            print()
            for line in ptext:
                print(line)
        except Exception as exc:   # noqa: BLE001
            prior_err, ptyped = f"{type(exc).__name__}: {exc}", []
    if prior_err:
        print(f"PRIOR DIAGNOSIS (do not type)   could not be read: {prior_err}\n")
    elif not prior:
        print("PRIOR DIAGNOSIS (do not type)   no prior_diagnosis_*.json yet (diagnose_135.sh writes them"
              " as jobs finish)" + ("; failed: " + " ".join(pfails) if pfails else "") + "\n")
        ptyped = ["pf " + " ".join(pfails)] if pfails else []
    print("==== TYPE THIS BACK ====")
    print(f"{tag} {NODE} {hours}h D{n_done}/{len(jobs)} F{len(fails)}")
    if floor:
        print(f"floor {x100(floor.get('p50'))} {x100(floor.get('p5'))} {x100(floor.get('p95'))}"
              + ("" if k_ref in (None, 128) else f" k{k_ref}"))
    if run:
        print("run " + " ".join(run))
    if inf:
        print("inf " + ranges(inf))
    if wait:
        print("wait " + ranges(wait))
    if stop:
        print("stop " + ranges(stop))
    for line in fails:
        print(line)
    for f in sorted(os.listdir(status_dir)) if os.path.isdir(status_dir) else ():
        if f.startswith("_abort_"):   # a second run (LANE) that stopped in its checks
            msg = (read_text(os.path.join(status_dir, f)) or "").splitlines()
            print(f"ABORT lane {f[len('_abort_'):]} {msg[0] if msg else ''}".rstrip())
    dfail = read_text(os.path.join(out_root, "_diag", "diag_fail.txt")) or ""
    dl =[f"{nums.get(p[1], p[1])}:{p[2]}" for p in (l.split() for l in dfail.splitlines()) if len(p) > 2]
    if dl:
        print("d " + " ".join(dl))
    for line in infl:
        print(line)
    if args.long:
        for _w, _n, line in sorted(ranked):
            print(line)
    for line in setting_lines(settings):
        print(line)
    for line in ptyped:
        print(line)
    print("==== END ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
