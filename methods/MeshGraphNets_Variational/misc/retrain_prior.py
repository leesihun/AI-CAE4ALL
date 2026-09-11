"""Retrain ONLY the conditional prior against a frozen encoder and decoder.

    cd methods/MeshGraphNets_Variational
    python misc/retrain_prior.py \
        --config ../../configs/MeshGraphNets_Variational/SAOI_sweep/config_train_8.txt \
        [--epochs 300] [--lr 3e-4] [--prior-hidden 512] [--prior-layers 8] \
        [--no-whiten] [--no-ema] [--out <modelpath stem>_priorfit.pth]

THE PROBLEM
    Training decodes z ~ q(z | y, g); deployment decodes z ~ p(z | g). When the
    encoder and decoder reproduce a part's variability but the prior path does
    not (misc/posterior_vs_prior.py), the prior's conditional for that part is
    too narrow or mis-shaped. Three mechanisms can do that, and this script
    removes all three at once because they share one intervention -- fitting
    the prior alone, against a posterior that has stopped moving.

    1. A MOVING TARGET. In joint training the encoder keeps receiving
       reconstruction gradients, so q drifts for the whole run and the prior
       chases it. Here the encoder is frozen and its (mu, logvar) cached, so
       the target is fixed.

    2. AN ILL-SCALED TARGET. MMD pins the AGGREGATE q(z) to N(0,I). With ~100
       realizations of each of a few parts, that aggregate is a mixture: total
       variance 1 is compatible with every PER-PART cloud -- the thing the
       prior must actually hit -- being far smaller and off-origin. A velocity
       MLP transporting N(0,I) onto a small off-centre cloud spends its
       capacity on the transport instead of the shape. `--whiten` (default)
       standardizes the target and installs the inverse as prior buffers, so
       sampling returns decoder units with no call-site change.

    3. AN UNDERSIZED TRUNK. `prior_mp_layers` and `prior_hidden_dim` are much
       smaller than the main network's. If the trunk cannot separate parts, the
       prior learns a conditional smeared across neighbours. `--prior-hidden` /
       `--prior-layers` rebuild the prior from scratch at a larger size; the
       encoder and decoder are untouched, and the saved model_config records
       the new shape so inference rebuilds it correctly.

    `prior_grad_to_encoder` needs no attention here: the encoder is frozen, so
    the FM objective cannot shrink the target distribution to lower its own
    loss.

COST
    The posterior encoder runs ONCE over the training set and its (mu, logvar)
    are cached -- 2 x num_z x D floats per sample. Each epoch is then the prior
    trunk over the graphs plus a tiny velocity MLP: hours, not days. Geometry
    augmentation is OFF so the cache stays paired with its graph, and the trunk
    sees the same unaugmented distribution inference sees.

OUTPUT
    A checkpoint the ordinary rollout loads unchanged: encoder/decoder weights
    byte-identical to the input, `normalization` carried over verbatim,
    `model_config` updated only where the prior's shape changed, and
    `ema_state_dict` rebuilt in the `module.`-prefixed form rollout.py strips.
    Point an inference config's `modelpath` at it and the existing pipeline
    runs end to end.

DECIDING WHETHER IT WORKED
    The per-epoch `aggregate prior/posterior ratio` is the [PriorDiag]
    statistic and is dominated by BETWEEN-geometry variance -- a coarse proxy
    only. The decision metric is the per-geometry diagnostic:

        python misc/posterior_vs_prior.py --config <infer config, modelpath = the new file>

    If the training set holds one realization per part (misc/count_realizations.py),
    none of this can help: the prior has no within-part spread to learn.
"""
import argparse
import copy
import math
import os
import sys
import time

os.environ.setdefault('HDF5_USE_FILE_LOCKING', 'FALSE')
import torch
from torch_geometric.loader import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from general_modules.load_config import load_config                  # noqa: E402
from general_modules.data_loader import load_data                    # noqa: E402
from model.conditional_prior import build_conditional_prior          # noqa: E402
from inference_profiles.rollout import _load_model_from_checkpoint   # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# pure helpers (unit-tested without a model)
# ─────────────────────────────────────────────────────────────────────────────

def ema_dict_from_state(state_dict):
    """The format rollout.py loads EMA weights from: AveragedModel keys."""
    out = {'module.' + k: v for k, v in state_dict.items()}
    out['n_averaged'] = torch.tensor(1, dtype=torch.long)
    return out


def spread_ratio(mu, logvar, prior_z):
    """Aggregate prior/posterior std ratio per slot -- the [PriorDiag] statistic.

    mu, logvar : [N, S, D] cached posterior params
    prior_z    : [M, S, D] prior draws over the same graphs
    Coarse proxy only: Var(mu) here is mostly BETWEEN-geometry variance.
    """
    post_var = mu.var(dim=0, unbiased=False) + logvar.exp().mean(dim=0)
    prior_var = prior_z.var(dim=0, unbiased=False)
    post_std = post_var.mean(dim=-1).sqrt()
    prior_std = prior_var.mean(dim=-1).sqrt()
    return (prior_std / post_std.clamp(min=1e-8)).tolist()


def whitening_from_posterior(mu, logvar, eps=1e-6):
    """(shift, scale) flattened, for the distribution of posterior SAMPLES.

    A sample is mu + sigma*eps, so across the training set its mean is E[mu]
    and its variance Var(mu) + E[sigma^2]. Standardizing by those is what puts
    the FM path's endpoint at unit scale.
    """
    flat_mu = mu.reshape(mu.shape[0], -1).double()
    flat_var = logvar.reshape(logvar.shape[0], -1).double().exp()
    shift = flat_mu.mean(dim=0)
    scale = (flat_mu.var(dim=0, unbiased=False) + flat_var.mean(dim=0)).sqrt()
    return shift.float(), scale.clamp(min=eps).float()


def cosine_lr(step, total, base, floor=1e-6):
    if total <= 1:
        return base
    return floor + 0.5 * (base - floor) * (1 + math.cos(math.pi * step / (total - 1)))


# ─────────────────────────────────────────────────────────────────────────────

def resolve_device(cfg, override=None):
    """The card to run on: --gpu if given, else the config's gpu_ids."""
    if not torch.cuda.is_available():
        return torch.device('cpu')
    gid = cfg.get('gpu_ids', 0) if override is None else override
    if isinstance(gid, list):
        gid = gid[0] if gid else 0
    try:
        gid = int(gid)
    except (TypeError, ValueError):
        gid = 0
    if not 0 <= gid < torch.cuda.device_count():
        gid = 0
    torch.cuda.set_device(gid)
    return torch.device(f'cuda:{gid}')


@torch.no_grad()
def cache_posterior(model, loader, device, label):
    """(mu, logvar) per sample id from the FROZEN posterior encoder.

    Mirrors MeshGraphNets._encode_vae exactly: raw normalized graph.y and
    graph.x (graph-aware) into vae_encoder, before the simulator's encoder.
    """
    inner = model.model
    mus, lvs = {}, {}
    t0 = time.time()
    for g in loader:
        g = g.to(device)
        batch = g.batch if getattr(g, 'batch', None) is not None else \
            torch.zeros(g.x.shape[0], dtype=torch.long, device=device)
        _, mu, lv = inner.vae_encoder(
            g.y, g.edge_index, g.edge_attr, batch,
            x=(g.x if inner.vae_graph_aware else None),
        )
        for i, sid in enumerate(g.sample_id.tolist()):
            mus[int(sid)] = mu[i].float().cpu()
            lvs[int(sid)] = lv[i].float().cpu()
    print(f"  cached {label} posterior: {len(mus)} samples in {time.time() - t0:.0f}s")
    return mus, lvs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='the TRAIN config of the arm')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--batch-size', type=int, default=0, help='0 = config batch_size')
    ap.add_argument('--prior-hidden', type=int, default=0,
                    help='rebuild the prior trunk at this width (0 = keep)')
    ap.add_argument('--prior-layers', type=int, default=0,
                    help='rebuild the prior trunk with this many MP layers (0 = keep)')
    ap.add_argument('--no-whiten', action='store_true',
                    help='do not standardize the latent target')
    ap.add_argument('--no-ema', action='store_true')
    ap.add_argument('--ema-decay', type=float, default=0.999)
    ap.add_argument('--val-draws', type=int, default=4,
                    help='prior draws per val graph for the spread-ratio proxy')
    ap.add_argument('--out', default=None)
    ap.add_argument('--gpu', type=int, default=None,
                    help="card to run on; default is the config's gpu_ids, "
                         "which is the card the arm TRAINED on and may still "
                         "be busy")
    a = ap.parse_args()

    cfg = load_config(a.config)
    cfg['num_timesteps'] = 1
    cfg['hierarchy_cache_keep'] = True
    dev = resolve_device(cfg, a.gpu)
    torch.manual_seed(1234)

    # ---- data: same split, same normalizers, NO augmentation --------------
    ds = load_data(cfg)
    tr, va, te = ds.split(0.8, 0.1, 0.1, seed=int(cfg['split_seed']))
    for s in (tr, va, te):
        s.augment_geometry = False
    bs = a.batch_size or int(cfg['batch_size'])
    nw = int(cfg.get('num_workers', 0))

    def mk(d, shuffle):
        return DataLoader(d, batch_size=bs, shuffle=shuffle, num_workers=nw,
                          persistent_workers=nw > 0,
                          prefetch_factor=4 if nw > 0 else None,
                          multiprocessing_context='spawn' if nw > 0 else None)

    train_loader, val_loader = mk(tr, True), mk(va, False)

    # ---- checkpoint --------------------------------------------------------
    src = cfg['modelpath']
    ck = torch.load(src, map_location=dev, weights_only=False)
    c2 = dict(cfg)
    for k, v in ck['model_config'].items():
        c2[k] = v
    c2['num_timesteps'] = 1
    model = _load_model_from_checkpoint(c2, ck, dev)
    if getattr(model, 'prior', None) is None:
        raise SystemExit('checkpoint has no joint conditional prior to retrain')

    # ---- optionally rebuild the trunk at a larger size --------------------
    rebuilt = {}
    if a.prior_hidden or a.prior_layers:
        pc = dict(c2)
        if a.prior_hidden:
            pc['prior_hidden_dim'] = a.prior_hidden
            rebuilt['prior_hidden_dim'] = a.prior_hidden
        if a.prior_layers:
            pc['prior_mp_layers'] = a.prior_layers
            rebuilt['prior_mp_layers'] = a.prior_layers
        old_n = sum(p.numel() for p in model.prior.parameters())
        model.prior = build_conditional_prior(pc).to(dev)
        new_n = sum(p.numel() for p in model.prior.parameters())
        c2.update(rebuilt)
        print(f"  prior trunk REBUILT from scratch: {old_n:,} -> {new_n:,} parameters "
              f"({rebuilt})")
    prior = model.prior

    # ---- freeze everything but the prior ----------------------------------
    for p_ in model.parameters():
        p_.requires_grad_(False)
    n_prior = 0
    for p_ in prior.parameters():
        p_.requires_grad_(True)
        n_prior += p_.numel()
    model.eval()
    prior.train()
    print(f"  training {n_prior:,} prior parameters; everything else frozen")

    # ---- cache the frozen posterior once -----------------------------------
    mu_tr, lv_tr = cache_posterior(model, train_loader, dev, 'train')
    mu_va, lv_va = cache_posterior(model, val_loader, dev, 'val')
    mu_stack = torch.stack([mu_tr[k] for k in sorted(mu_tr)])
    lv_stack = torch.stack([lv_tr[k] for k in sorted(lv_tr)])

    # ---- latent standardization -------------------------------------------
    if not a.no_whiten and hasattr(prior, 'z_shift'):
        shift, scale = whitening_from_posterior(mu_stack, lv_stack)
        prior.z_shift.copy_(shift.to(dev))
        prior.z_scale.copy_(scale.to(dev))
        print(f"  latent standardization ON: |shift| rms {shift.norm() / shift.numel() ** 0.5:.4f}, "
              f"scale min {scale.min():.4f} max {scale.max():.4f}")
        print(f"    (a scale far from 1 is exactly the mis-scaling this removes)")
    else:
        print("  latent standardization OFF")

    def target_for(ids):
        mu = torch.stack([mu_tr[int(s)] for s in ids]).to(dev)
        lv = torch.stack([lv_tr[int(s)] for s in ids]).to(dev)
        # fresh posterior SAMPLE every step -- the cloud the decoder was trained on
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    opt = torch.optim.Adam(prior.parameters(), lr=a.lr)
    ema = None if a.no_ema else copy.deepcopy(prior).eval()
    if ema is not None:
        for p_ in ema.parameters():
            p_.requires_grad_(False)

    steps_per_epoch = max(1, len(train_loader))
    total_steps = a.epochs * steps_per_epoch
    step = 0
    best = float('inf')
    out = a.out or (os.path.splitext(src)[0] + '_priorfit.pth')

    def save(tag):
        """Write a checkpoint the ordinary rollout loads with no changes."""
        keep = prior.state_dict()
        if ema is not None:                       # ship the averaged prior
            prior.load_state_dict(ema.state_dict())
        sd = model.state_dict()
        if ema is not None:
            prior.load_state_dict(keep)
        new = dict(ck)
        new['model_state_dict'] = sd
        new['ema_state_dict'] = ema_dict_from_state(sd)
        new.pop('optimizer_state_dict', None)
        new.pop('scheduler_state_dict', None)
        mc = dict(ck['model_config'])
        mc.update(rebuilt)                        # only the prior's shape moved
        new['model_config'] = mc
        new['prior_retrained_from'] = os.path.abspath(src)
        new['prior_retrain'] = {'epochs': a.epochs, 'lr': a.lr, 'tag': tag,
                                'whiten': not a.no_whiten, 'ema': not a.no_ema,
                                'rebuilt': rebuilt, 'augment_geometry': False}
        torch.save(new, out)

    print(f"  {a.epochs} epochs x {steps_per_epoch} steps, lr {a.lr:g} cosine -> {out}")
    for epoch in range(a.epochs):
        prior.train()
        tot, n, lr = 0.0, 0, a.lr
        for g in train_loader:
            g = g.to(dev)
            lr = cosine_lr(step, total_steps, a.lr)
            for grp in opt.param_groups:
                grp['lr'] = lr
            with torch.no_grad():
                z1 = target_for(g.sample_id.tolist())
            loss = prior.fm_loss(prior.condition(g), z1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
            opt.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pp in zip(ema.parameters(), prior.parameters()):
                        pe.mul_(a.ema_decay).add_(pp, alpha=1 - a.ema_decay)
                    for be, bp in zip(ema.buffers(), prior.buffers()):
                        be.copy_(bp)
            tot += float(loss) * g.num_graphs
            n += g.num_graphs
            step += 1
        train_fm = tot / max(n, 1)

        # ---- validation: FM loss + the aggregate spread-ratio proxy -------
        scorer = ema if ema is not None else prior
        scorer.eval()
        vt, vn, mus, lvs, zs = 0.0, 0, [], [], []
        with torch.no_grad():
            for g in val_loader:
                g = g.to(dev)
                ids = g.sample_id.tolist()
                mu = torch.stack([mu_va[int(s)] for s in ids]).to(dev)
                lv = torch.stack([lv_va[int(s)] for s in ids]).to(dev)
                z1 = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
                cond = scorer.condition(g)
                vt += float(scorer.fm_loss(cond, z1)) * g.num_graphs
                vn += g.num_graphs
                zp = scorer.sample_n_from_pooled(cond, a.val_draws)
                zs.append(zp.reshape(-1, zp.shape[-2], zp.shape[-1]).float().cpu())
                mus.append(mu.cpu())
                lvs.append(lv.cpu())
        val_fm = vt / max(vn, 1)
        ratio = spread_ratio(torch.cat(mus), torch.cat(lvs), torch.cat(zs))
        is_best = val_fm < best
        print(f"epoch {epoch + 1:>4}/{a.epochs}  lr {lr:.2e}  train fm {train_fm:.4e}  "
              f"val fm {val_fm:.4e}  aggregate prior/posterior ratio "
              + ' '.join(f"{r:.2f}" for r in ratio) + ('  <-- best' if is_best else ''))
        if is_best:
            best = val_fm
            save(f'best val fm {val_fm:.4e} @ epoch {epoch + 1}')

    print()
    print(f"  wrote {out}   (best val fm {best:.4e})")
    print("  Encoder/decoder weights are the input checkpoint's, byte for byte.")
    print("  Point an inference config's modelpath at this file and run")
    print("  misc/posterior_vs_prior.py -- that per-geometry sd_ratio is the decision")
    print("  metric; the aggregate ratio above is a between-geometry proxy.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
