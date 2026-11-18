#!/usr/bin/env python3
"""Round 11: what is wrong with the prior, in English sentences, and the lines to type back.

    python configs/MeshGraphNets_Variational/hyperparameter_sweep/r11/prior_brief.py
    python .../r11/prior_brief.py --long     # the full verdict of every setting
    python .../r11/prior_brief.py --all      # one more typed line per other setting

Standard library only. It reads what prior_135.sh left per run
(prior_diagnosis_{ho,tr,ts}.json and, where the run kept a joint-phase
checkpoint, prior_diagnosis_jt.json, from methods/MeshGraphNets_Variational/
misc/prior_diagnosis.py), averages the seeds of a setting and prints

  * the verdict of the base setting in plain sentences: where the prior's gap
    over the posterior comes from (training fit / new geometries / outside the
    range), what kind of error it is (mean, width, shape, joint structure),
    when in the flow it happens, which latent brings it (z_g, z_l, their
    pairing), whether the sampler, the input, the target, the EMA weights or
    the kept epoch is to blame, and a ranked conclusion.
  * the CAUSES table: every candidate cause in a fixed order with YES (found),
    no (ruled out by a number) or ? (this run cannot tell, and why). No reading
    by anyone is needed.
  * the premise, the conclusion and the cause numbers of every other setting.
  * the block to type back (the last number of a line is its check digit):

    ==== TYPE THIS BACK ====
    R11P 135 a0 D20/20 F0        the setting of the lines, runs done / runs, failed parts
    ho 27 29 30 78 77 55 41 60 52 40 9
    tr ...
    ts ...
    ve 4 9 11 30 118 19 22 6 9
    in 110 243 71 85 730 99 87 60 20 5
    ex 30 80 45 70 33 79 31 36 52 70 77 100 104 31 85 3
    cs 1 5 9 10 12 14 u 34 36 1    causes found, `u`, causes that cannot be told
    ==== END ====

The global run is left out: its prior is another class (one vector per graph).
"""
import argparse
import os
import sys

sys.dont_write_bytecode = True   # importing the briefs must not leave a __pycache__ beside the configs
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brief as B            # noqa: E402  (ARMS, the config reader)
import unseen_brief as U     # noqa: E402  (the runs: R1-R5 and this round's jobs)
sys.path.insert(0, os.path.join(B.REPO, "methods", "MeshGraphNets_Variational", "misc"))
import prior_verdict as PV   # noqa: E402  (the summary, the sentences, the typed lines)

FIRST = ("base", "tfk8")     # diagnosed first; the block is the first of them that has a result
FAIL_FILE = os.path.join(U.OUT_ROOT, "_diag", "prior_fail.txt")
ALL_SCORES = ("P", "Q", "S", "R5", "Qc", "Qm")


def runs():
    """U.runs() without the global runs, the FIRST settings in front."""
    out = []
    for r in U.runs():
        try:
            kind = B.cfg(r["infer"]).get("latent_type", "global").lower()
        except OSError:
            kind = "global" if r["arm"] == "global" else "hybrid"
        if kind != "global":
            out.append(r)
    return sorted(out, key=lambda r: FIRST.index(r["arm"]) if r["arm"] in FIRST else len(FIRST))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--long", action="store_true", help="the full verdict of every setting")
    ap.add_argument("--all", action="store_true", help="one more typed line per other setting")
    ap.add_argument("--runs", action="store_true",
                    help="for prior_135.sh: label, infer config, diag dir, checkpoint, setting of every run")
    args = ap.parse_args()

    all_runs = runs()
    if args.runs:
        for r in all_runs:
            print(r["label"], U.rel(r["infer"]), U.rel(r["ddir"]), U.rel(r["model"]) if r["model"] else "-", r["arm"])
        return 0

    every = PV.PARTS + (PV.JOINT,)   # the joint-phase part is optional: it does not count as "done"
    summary, done = {}, 0
    for r in all_runs:
        have = 0
        for part in every:
            doc = B.load_json(os.path.join(r["ddir"], f"prior_diagnosis_{part}.json"))
            try:
                summary[r["label"], part] = PV.summarize(doc) if doc else None
            except (KeyError, TypeError, AttributeError):
                summary[r["label"], part] = None   # cut off while writing
            have += part in PV.PARTS and summary[r["label"], part] is not None
        done += have == len(PV.PARTS)

    arm_num = {arm: k for k, arm in enumerate(B.ARMS)}
    order = [a for a in FIRST if a in B.ARMS] + [a for a in B.ARMS if a not in FIRST]
    arms = {}
    for arm in order:
        rs = sorted((r for r in all_runs if r["arm"] == arm
                     and any(summary[r["label"], part] for part in PV.PARTS)), key=lambda r: r["seed"])
        parts = {part: PV.merge([summary[r["label"], part] for r in rs]) for part in every}
        if any(parts[part] for part in PV.PARTS):
            arms[arm] = (rs, {p: s for p, s in parts.items() if s})

    fails = []
    for row in (line.split() for line in (B.read_text(FAIL_FILE) or "").splitlines()):
        if len(row) > 2 and row[0] == "FAIL":
            fails.append(f"{row[1]}:{row[2]}")

    if not arms:
        print("no prior_diagnosis_*.json yet: bash " + U.rel(os.path.join(HERE, "prior_135.sh")))
        for f in fails:
            print("  failed " + f)
        return 0

    main_arm = next(iter(arms))
    for arm, (rs, parts) in arms.items():
        label = f"a{arm_num[arm]} {arm} (seed {'/'.join(str(r['seed']) for r in rs)})"
        for line in PV.verdict(parts, label, short=arm != main_arm and not args.long):
            print(line)
        print()

    print(PV.BLOCK_LEGEND)
    if args.all:
        print("  aN   (--all) another setting on the held-out geometries: " + " ".join(ALL_SCORES))
    print("SETTINGS (do not type)   " + "  ".join(f"a{arm_num[a]}={a}" for a in arms))
    if fails:
        print("FAILED (do not type)   " + " ".join(fails) + f"   reasons: {U.rel(FAIL_FILE)}")
    print()
    print("==== TYPE THIS BACK ====")
    print(f"{B.ROUND.upper()}P {B.NODE} a{arm_num[main_arm]} D{done}/{len(all_runs)} F{len(fails)}")
    for line in PV.block(arms[main_arm][1]):
        print(line)
    if args.all:
        for arm, (_, parts) in arms.items():
            s = parts.get("ho") or parts.get("ts") or parts.get("tr")
            if arm != main_arm and s:
                print(PV.checked(f"a{arm_num[arm]}", [PV.whole(s["sc"].get(k)) for k in ALL_SCORES]))
    print("==== END ====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
