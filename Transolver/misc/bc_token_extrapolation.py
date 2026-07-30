"""Does BC-token conditioning give Transolver boundary-condition EXTRAPOLATION?

Uses the real `Transolver` class from this repo, with a minimal prototype of the
proposed condition-token patch (concatenate K condition tokens to the M physics
slice tokens inside `_slice_attend`, deslice only the M).

Synthetic thermal-warpage-like problem, static (T=1) like ex1.h5:

    u(x) = g(dT) * shape(x; geometry)          (1 output channel)

`shape` depends on the sample geometry; `g` is the BC response law.
  regime "linear":    g(dT) = dT              (thermal expansion, small strain)
  regime "nonlinear": g(dT) = dT * (1 + 0.15 * dT)

Train on dT ~ U[1, 2]. Test on unseen geometries at dT = 1.5 (interpolation)
and dT = 4, 8 (extrapolation).

Variants (all with the same backbone):
  A  dT as a per-node INPUT CHANNEL   + dataset z-scored target   <- what the repo can do today
  B  dT as a CONDITION TOKEN         + dataset z-scored target    <- the "LLM-like token" proposal
  C  dT as a CONDITION TOKEN         + analytic amplitude factoring (predict u / dT)

Measured (2500 steps, 12 unseen geometries, relative L2 in physical units) --
see FOUNDATION_MODEL_DESIGN.md section 3 for the reading:

    regime = linear                     dT=1.5      dT=4      dT=8
      A  per-node channel + z-score       5.38 %   69.00 %   93.76 %
      B  condition token  + z-score       6.77 %   55.69 %   96.66 %
      C  condition token  + factoring     4.32 %   10.34 %   30.43 %

    regime = nonlinear                  dT=1.5      dT=4      dT=8
      A  per-node channel + z-score       6.94 %   64.23 %   93.56 %
      B  condition token  + z-score       5.98 %   95.58 %   99.42 %
      C  condition token  + factoring     5.27 %   75.20 %   99.97 %

Conclusions: (1) in-distribution the three tie, so condition tokens buy
EXTENSIBILITY, not accuracy; (2) tokens do not produce extrapolation -- the
z-scored output path is what breaks; (3) analytic amplitude factoring is the
only lever that moves it (6.7x at 2.7x BC); (4) even C degrades, because the
condition VALUE itself goes out of range and injects a spurious dependence the
target does not have -> hence condition_dropout / value clamping; (5) when the
true response law differs from the factored one, wrong conditioning is worse
than none (B 95.58 % vs A 64.23 %).
"""
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

TRANSOLVER = Path(__file__).resolve().parents[1]   # .../Transolver
sys.path.insert(0, str(TRANSOLVER))

from model.Transolver import Transolver                      # noqa: E402
from model.physics_attention import PhysicsAttentionIrregular  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)


# ----------------------------------------------------------------------------
# prototype patch: condition tokens inside slice attention
# ----------------------------------------------------------------------------
_orig_forward = PhysicsAttentionIrregular.forward
_orig_slice_attend = PhysicsAttentionIrregular._slice_attend


def _patched_forward(self, x, ptr, attention_kernel="naive", chunk_size=0,
                     use_checkpointing=False):
    """Same as the original packed loop, but publishes the current graph's
    condition tokens so `_slice_attend` can see them."""
    cond_list = getattr(self, "_cond_list", None)
    outs = []
    for i in range(ptr.shape[0] - 1):
        s, e = int(ptr[i].item()), int(ptr[i + 1].item())
        self._cond_current = None if cond_list is None else cond_list[i]
        x_g = x[s:e]
        if attention_kernel == "naive":
            out_g = self._forward_naive(x_g)
        elif attention_kernel == "slice_space":
            from model.physics_attention import make_tile_ranges
            out_g = self._forward_slice_space(
                x_g, make_tile_ranges(e - s, chunk_size), use_checkpointing)
        else:
            raise ValueError(attention_kernel)
        outs.append(out_g)
    self._cond_current = None
    return torch.cat(outs, dim=0)


def _patched_slice_attend(self, tokens):
    """tokens [H, M, D]; condition tokens [K, C] -> [H, M, D].

    The condition tokens join the token sequence for self-attention (so the
    physics states can read the BC), but are dropped before the deslice: the
    assignment matrix W is [H, N, M] over physics slices only, so node space
    never sees a condition token directly. This keeps every N-scaled tensor,
    tiling range and shard reduction in the kernel completely unchanged.
    """
    cond = getattr(self, "_cond_current", None)
    if cond is None:
        return _orig_slice_attend(self, tokens)

    H, M, D = tokens.shape
    ck = self.cond_proj(cond).view(-1, H, D).permute(1, 0, 2)   # [H, K, D]
    seq = torch.cat([tokens, ck], dim=1)                        # [H, M+K, D]
    q, k, v = self.to_q(seq), self.to_k(seq), self.to_v(seq)
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    attn = torch.softmax(dots.float(), dim=-1).to(v.dtype)
    attn = self.dropout(attn)
    return torch.matmul(attn, v)[:, :M]                         # physics slices only


PhysicsAttentionIrregular.forward = _patched_forward
PhysicsAttentionIrregular._slice_attend = _patched_slice_attend


class ConditionEncoder(nn.Module):
    """One BC -> one token. Value encoding is deliberately MONOTONE with linear
    tails (raw + log1p), never random Fourier features: a GELU MLP on such
    features continues linearly outside the training range, whereas Fourier
    features are periodic and alias wildly under extrapolation."""

    def __init__(self, latent_dim: int, num_cond: int = 1):
        super().__init__()
        self.type_emb = nn.Embedding(num_cond, latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2, latent_dim), nn.GELU(), nn.Linear(latent_dim, latent_dim))

    def forward(self, v_dimensionless: torch.Tensor) -> torch.Tensor:
        """v: [B, K] dimensionless condition values -> [B, K, latent_dim]"""
        feats = torch.stack([v_dimensionless,
                             torch.log1p(v_dimensionless.clamp(min=-0.999))], dim=-1)
        idx = torch.arange(v_dimensionless.shape[1], device=v_dimensionless.device)
        return self.mlp(feats) + self.type_emb(idx)[None, :, :]


def attach_conditioning(model: Transolver, latent_dim: int, num_cond: int = 1):
    model.cond_encoder = ConditionEncoder(latent_dim, num_cond).to(DEV)
    for m in model.modules():
        if isinstance(m, PhysicsAttentionIrregular):
            m.cond_proj = nn.Linear(latent_dim, m.heads * m.dim_head).to(DEV)
            nn.init.trunc_normal_(m.cond_proj.weight, std=0.02)
            nn.init.zeros_(m.cond_proj.bias)
    return model


def set_conditions(model: Transolver, cond_per_graph):
    """cond_per_graph: list of [K, latent_dim] (or None to disable)."""
    for m in model.modules():
        if isinstance(m, PhysicsAttentionIrregular):
            m._cond_list = cond_per_graph


# ----------------------------------------------------------------------------
# synthetic dataset
# ----------------------------------------------------------------------------
def response(dT, regime):
    return dT if regime == "linear" else dT * (1.0 + 0.15 * dT)


def make_sample(rng, dT, regime, n_nodes=None):
    n = n_nodes or int(rng.integers(1500, 2500))
    # Every latent geometry factor must be RECOVERABLE from the point cloud the
    # model sees (aspect from the x-extent, curv from z), otherwise the target
    # carries irreducible noise and nothing is learnable.
    aspect = rng.uniform(0.8, 1.2)
    curv = rng.uniform(-0.3, 0.3)

    x = rng.uniform(-1, 1, n) * aspect
    y = rng.uniform(-1, 1, n)
    z = curv * (x ** 2 + y ** 2)
    pos = np.stack([x, y, z], axis=1).astype(np.float32)

    shape = (np.sin(math.pi * x / aspect) * np.sin(math.pi * y)
             + 0.4 * (x ** 2 - y ** 2) * (1.0 + curv))
    u = response(dT, regime) * shape
    return pos, u.astype(np.float32)[:, None], float(dT)


def build_split(rng, n_samples, regime, dT_lo=1.0, dT_hi=2.0, dT_fixed=None):
    out = []
    for _ in range(n_samples):
        dT = dT_fixed if dT_fixed is not None else float(rng.uniform(dT_lo, dT_hi))
        out.append(make_sample(rng, dT, regime))
    return out


def normalize_positions(pos, position_scale):
    return (pos - pos.mean(axis=0)) / max(position_scale, 1e-8)


# ----------------------------------------------------------------------------
# training / evaluation
# ----------------------------------------------------------------------------
CFG = dict(
    output_var=1, input_var=1, latent_dim=64, num_layers=3, num_heads=4,
    slice_num=32, mlp_ratio=2, dropout=0.0, attention_kernel="slice_space",
    chunk_size=0, use_checkpointing=False, positional_features=0,
    use_node_types=False, num_timesteps=1, small_output_init=False, std_noise=0.0,
)

DT_REF = 1.5  # dimensionless reference (train-mean dT) -- the nondimensionalizer


def make_batch(samples, position_scale, y_mean, y_std, variant, model):
    xs, ps, ys, sizes, dts = [], [], [], [], []
    for pos, u, dT in samples:
        pn = normalize_positions(pos, position_scale)
        if variant == "A":
            xin = np.full((pos.shape[0], 1), dT / DT_REF, dtype=np.float32)
        else:
            xin = np.zeros((pos.shape[0], 1), dtype=np.float32)
        target = u / response(dT, "linear") if variant == "C" else u
        ys.append((target - y_mean) / y_std)
        xs.append(xin)
        ps.append(pn)
        sizes.append(pos.shape[0])
        dts.append(dT)

    ptr = torch.tensor([0] + list(np.cumsum(sizes)), dtype=torch.long, device=DEV)
    graph = SimpleNamespace(
        x=torch.from_numpy(np.concatenate(xs)).to(DEV),
        pos_normalized=torch.from_numpy(np.concatenate(ps).astype(np.float32)).to(DEV),
        y=torch.from_numpy(np.concatenate(ys).astype(np.float32)).to(DEV),
        ptr=ptr,
    )
    if variant in ("B", "C"):
        v = torch.tensor([[d / DT_REF] for d in dts], dtype=torch.float32, device=DEV)
        toks = model.cond_encoder(v)                      # [B, K=1, latent]
        set_conditions(model, [toks[i] for i in range(len(dts))])
    else:
        set_conditions(model, None)
    return graph, ptr, dts


def run_variant(variant, regime, steps=2500, batch=4, verbose=False):
    rng = np.random.default_rng(1234)
    train = build_split(rng, 160, regime)
    # train-split statistics, exactly as mesh_dataset.py fits them
    position_scale = float(np.sqrt(np.mean([
        np.mean(np.sum((p - p.mean(0)) ** 2, axis=1)) for p, _, _ in train])))
    if variant == "C":
        allv = np.concatenate([u / response(d, "linear") for _, u, d in train])
    else:
        allv = np.concatenate([u for _, u, _ in train])
    y_mean, y_std = float(allv.mean()), float(allv.std())

    model = Transolver(CFG, device=DEV)
    if variant in ("B", "C"):
        attach_conditioning(model, CFG["latent_dim"])
    n_par = sum(p.numel() for p in model.parameters())

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    model.train()
    t0 = time.time()
    for step in range(steps):
        idx = np.random.choice(len(train), batch, replace=False)
        graph, _, _ = make_batch([train[i] for i in idx], position_scale,
                                 y_mean, y_std, variant, model)
        pred, y = model(graph, add_noise=False)
        loss = torch.nn.functional.mse_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        sched.step()
        if verbose and step % 100 == 0:
            print(f"    step {step:4d} loss {loss.item():.5f}")

    # ---- evaluation on unseen geometries, in PHYSICAL units ----
    model.eval()
    results = {}
    for dT_test in (1.5, 4.0, 8.0):
        erng = np.random.default_rng(99)
        test = build_split(erng, 12, regime, dT_fixed=dT_test)
        num = den = 0.0
        with torch.no_grad():
            for s in test:
                graph, _, _ = make_batch([s], position_scale, y_mean, y_std,
                                         variant, model)
                pred, _ = model(graph, add_noise=False)
                phys = pred.cpu().numpy() * y_std + y_mean
                if variant == "C":
                    phys = phys * response(s[2], "linear")
                truth = s[1]
                num += float(np.sum((phys - truth) ** 2))
                den += float(np.sum(truth ** 2))
        results[dT_test] = math.sqrt(num / den)
    return results, n_par, time.time() - t0


if __name__ == "__main__":
    LABEL = {
        "A": "A  dT as per-node input channel, z-scored target  (repo today)",
        "B": "B  dT as CONDITION TOKEN, z-scored target",
        "C": "C  dT as CONDITION TOKEN + analytic amplitude factoring",
    }
    for regime in ("linear", "nonlinear"):
        print(f"\n{'='*78}\nregime: {regime}   "
              f"g(dT) = {'dT' if regime=='linear' else 'dT*(1+0.15*dT)'}   "
              f"train dT in [1,2]\n{'='*78}")
        print(f"{'variant':<52} {'dT=1.5':>9} {'dT=4':>9} {'dT=8':>9}")
        for variant in ("A", "B", "C"):
            res, n_par, dt = run_variant(variant, regime)
            print(f"{LABEL[variant]:<52} "
                  f"{res[1.5]*100:8.2f}% {res[4.0]*100:8.2f}% {res[8.0]*100:8.2f}%"
                  f"   ({n_par/1e3:.0f}k par, {dt:.0f}s)")
    print("\nmetric: relative L2 error in physical units on unseen geometries")
