#!/usr/bin/env python3
"""Promote one Phase-A compressor to the fixed path the Phase-B arms load.

    python configs/HI_MGNFlow/hyperparameter_sweep/promote_ae.py --half bot --arm c8

Phase B fans four prior arms out from ONE frozen compressor per board half, so
every Phase-B config names the same `ae_checkpoint`:

    ../../output/chi-mgnflow/saoi_sweep/<half>.ae.pth

This script copies <half>.ae_<arm>.pth onto that path -- but only after
checking that every Phase-B and Phase-C config of that half actually describes
the compressor being promoted.

WHY THE CHECK EXISTS. CHiMGNFlow._load_frozen_ae loads the checkpoint with
every `model.prior.*` key stripped and then raises on ANY remaining missing or
unexpected key, so a latent_ch or latent_dim that disagrees with the promoted
file is a hard RuntimeError several minutes into a run. The launcher will not
catch it first: _probe_checkpoints in cae_suite/preflight.py only probes fields
whose name ends in `modelpath`, so `ae_checkpoint` is existence-checked and
nothing more. This is the only place the comparison happens.

Nothing is copied when the check fails, and the exact edit is printed.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "output" / "chi-mgnflow" / "saoi_sweep"

# Keys a Phase-B config must reproduce from the promoted compressor.
#
# latent_ch / latent_dim shape the AE-side tensors directly (latent_head is
# Linear(latent_dim, 2*latent_ch), z_lift is Linear(latent_ch, latent_dim)), and
# the channel and hierarchy blocks decide the encoder's input width and V-cycle
# depth. Every one of them is outside `model.prior.*` and therefore part of the
# strict match. `prior_blocks` and `flow_time_freqs` are deliberately absent:
# they only shape model.prior, which is exactly what _load_frozen_ae discards.
LOCKED_KEYS = (
    "latent_dim", "latent_ch",
    "input_var", "output_var", "cond_var", "edge_var", "positional_features",
    "use_node_types", "use_world_edges",
    "use_multiscale", "multiscale_levels", "mp_per_level",
    "coarsening_type", "voronoi_clusters",
)

# Not a weight shape -- the AE takes no gradient in stage 2, so a mismatch here
# breaks nothing at load time. It is checked because build_model_config copies
# it into the Phase-B checkpoint's own model_config, and a wrong value there
# would make the shipped checkpoint misdescribe the compressor inside it.
PROVENANCE_KEYS = ("ae_kl_weight",)


def parse_config(path: Path) -> dict:
    """Read a native flat config, mirroring the launcher's parser quirks.

    `%` starts a full-line comment, `#` an inline one, key and value are split
    on the first run of whitespace (the files use a tab), and a comma-separated
    value is a list. Values are kept as raw text -- this script only ever
    compares them, so normalise() decides equality, not the parse.
    """
    out = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("%"):
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = re.split(r"\s+", line, maxsplit=1)
        if len(parts) != 2:
            continue
        out[parts[0].strip()] = parts[1].strip()
    return out


def normalise(value) -> str:
    """Canonical text for comparing a config value against a checkpoint value.

    The two sides arrive differently typed: the checkpoint stores whatever the
    native parser produced at training time (int 8, the list [4, 6, 8, 6, 4],
    the bool True, and the STRING '1e-6' -- `1e-6` has no '.' so it fails the
    int and float fast paths and never becomes a number). The config side is
    raw text. Rendering both to a lowercase comma-joined token string makes
    '4, 6, 8, 6, 4', [4, 6, 8, 6, 4] and '4,6,8,6,8'-style spacing differences
    compare correctly while still catching a real change.
    """
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif isinstance(value, str):
        items = [t for t in value.split(",")]
    else:
        items = [value]

    tokens = []
    for item in items:
        text = str(item).strip().lower()
        try:
            number = float(text)
        except (TypeError, ValueError):
            tokens.append(text)
            continue
        # 8 and 8.0 and 8e0 are the same budget; 1e-6 and 0.000001 likewise.
        tokens.append(repr(number))
    return ",".join(tokens)


def load_model_config(checkpoint_path: Path) -> dict:
    try:
        import torch
    except ImportError:
        sys.exit("FAIL: torch is not importable. Run this with the HI_MGNFlow "
                 "interpreter (see ai_cae4all.local.toml).")
    # weights_only=False: this is a checkpoint written by this repo's own
    # save_checkpoint, and model_config/normalization are plain dicts and numpy
    # arrays that the safe loader refuses.
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        sys.exit(f"FAIL: {checkpoint_path} has no model_config; it was not "
                 f"written by this suite's save_checkpoint.")
    stage = str(model_config.get("training_stage", "")).lower().strip()
    if stage != "train_ae":
        sys.exit(f"FAIL: {checkpoint_path.name} records training_stage="
                 f"'{stage}', not 'train_ae'. Promote a Phase-A checkpoint -- "
                 f"promoting a prior or combined run would freeze a compressor "
                 f"that was already fit alongside a prior.")
    if not isinstance(checkpoint.get("normalization"), dict):
        print(f"  NOTE: {checkpoint_path.name} carries no normalization dict. "
              f"That is fine for an ae_checkpoint (Phase B rebuilds its own "
              f"from its training split), but inference refuses to run without "
              f"one, so the Phase-B checkpoints must have it.")
    return model_config


def check_config(path: Path, model_config: dict, keys) -> list:
    cfg = parse_config(path)
    problems = []
    for key in keys:
        if key not in model_config:
            continue  # not recorded by build_model_config; nothing to compare
        if key not in cfg:
            continue  # absent from the config is absent from the run: defaults agree
        want, got = normalise(model_config[key]), normalise(cfg[key])
        if want != got:
            problems.append((key, cfg[key], model_config[key]))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half", required=True, choices=("bot", "top"))
    ap.add_argument("--arm", required=True,
                    help="Phase-A arm to promote, e.g. c8 (see ae_report.py)")
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--config-dir", default=str(SCRIPT_DIR))
    ap.add_argument("--force", action="store_true",
                    help="promote even though configs disagree -- this will "
                         "fail at load time; only useful to inspect a file")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    cfg_dir = Path(args.config_dir).resolve()
    source = out_root / f"{args.half}.ae_{args.arm}.pth"
    target = out_root / f"{args.half}.ae.pth"

    print(f"promote_ae: {args.half} arm {args.arm}")
    print(f"  source = {source}")
    print(f"  target = {target}")

    if not source.is_file():
        available = sorted(p.name for p in out_root.glob(f"{args.half}.ae_*.pth"))
        print(f"FAIL: no such checkpoint: {source}", file=sys.stderr)
        if available:
            print(f"      available for {args.half}: {', '.join(available)}", file=sys.stderr)
        else:
            print(f"      nothing matching {args.half}.ae_*.pth in {out_root} -- "
                  f"has Phase A finished?", file=sys.stderr)
        return 1

    model_config = load_model_config(source)
    print(f"  compressor: latent_ch={model_config.get('latent_ch')} "
          f"latent_dim={model_config.get('latent_dim')} "
          f"ae_kl_weight={model_config.get('ae_kl_weight')}")

    targets = sorted(cfg_dir.glob(f"config_train_prior_{args.half}_*.txt"))
    # Phase-C configs repeat latent_ch as documentation. rollout.py overrides
    # them from the checkpoint so a mismatch cannot change the model, but it
    # would leave the file describing a run that never happened.
    infer_targets = sorted(cfg_dir.glob(f"config_infer_{args.half}_*.txt"))
    if not targets:
        print(f"FAIL: no config_train_prior_{args.half}_*.txt in {cfg_dir}", file=sys.stderr)
        return 1

    failures = 0
    for path in targets:
        problems = check_config(path, model_config, LOCKED_KEYS + PROVENANCE_KEYS)
        if problems:
            failures += 1
            print(f"  MISMATCH {path.name}")
            for key, got, want in problems:
                print(f"      {key}: config says {got!r}, checkpoint says {want!r}")

    infer_failures = 0
    for path in infer_targets:
        problems = check_config(path, model_config, ("latent_ch", "latent_dim"))
        if problems:
            infer_failures += 1
            print(f"  MISMATCH {path.name}")
            for key, got, want in problems:
                print(f"      {key}: config says {got!r}, checkpoint says {want!r}")

    if failures or infer_failures:
        bad = failures + infer_failures
        print(f"\nFAIL: {bad} config(s) disagree with {source.name}. Nothing was "
              f"copied.", file=sys.stderr)
        lch = model_config.get("latent_ch")
        kl = model_config.get("ae_kl_weight")
        print(f"\nThe usual cause is that a different arm won Phase A than the "
              f"one these configs were written for. Fix every config of this "
              f"half in one go, from the repository root:\n", file=sys.stderr)
        print(f"    sed -i 's/^latent_ch\\t.*/latent_ch\\t{lch}/' \\\n"
              f"        configs/HI_MGNFlow/hyperparameter_sweep/config_train_prior_{args.half}_*.txt \\\n"
              f"        configs/HI_MGNFlow/hyperparameter_sweep/config_infer_{args.half}_*.txt\n"
              f"    sed -i 's/^ae_kl_weight\\t.*/ae_kl_weight\\t{kl}/' \\\n"
              f"        configs/HI_MGNFlow/hyperparameter_sweep/config_train_prior_{args.half}_*.txt\n",
              file=sys.stderr)
        print(f"then re-run this command. (The inline comments on those lines "
              f"are replaced too -- that is intended.)", file=sys.stderr)
        if not args.force:
            return 1
        print("--force given: copying anyway.", file=sys.stderr)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"\n  {len(targets)} prior config(s) and {len(infer_targets)} infer "
          f"config(s) agree with the compressor.")
    print(f"OK: promoted {source.name} -> {target.name}")
    print(f"    Phase B for '{args.half}' can now run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
