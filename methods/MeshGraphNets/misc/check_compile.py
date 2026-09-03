"""Check that torch.compile captures the model without graph breaks.

MGN is memory-bound (LayerNorm/gather/add over [E, latent] dominate the GEMMs),
so the value of `use_compile True` is kernel fusion, and fusion only happens
inside a captured graph. One break in the processor loop costs most of it.

Run on the target GPU:

    python misc/check_compile.py                  # inductor (needs triton)
    python misc/check_compile.py --backend aot_eager   # frontend only

Expect "graphs: 1  breaks: 0" for every configuration. Anything else means
`use_compile True` is buying less than it looks like it is.

Known break sources, both fixed but easy to reintroduce:
  - passing a PyG Data across a torch.utils.checkpoint boundary (Dynamo aborts
    with "lift_tracked_freevar_to_input should not be called on root
    SubgraphTracer") -- keep checkpointed callables tensor-in/tensor-out
  - .item()/int() on a GPU tensor mid-forward -- also a CPU<->GPU sync
"""

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch._dynamo as dynamo
from torch_geometric.data import Data

from model.MeshGraphNets import MeshGraphNets

N, M, LATENT = 2000, 6000, 32
E = 2 * M


def base_config(**kw):
    cfg = dict(message_passing_num=3, edge_var=8, latent_dim=LATENT, input_var=4,
               output_var=4, positional_features=0, use_node_types=False,
               num_node_types=0, use_world_edges=False, use_multiscale=False,
               use_checkpointing=False, num_timesteps=50, std_noise=0.0)
    cfg.update(kw)
    return cfg


def make_graph(cfg, device, multiscale=False):
    mesh = torch.randint(0, N, (2, M), device=device)
    g = Data(x=torch.randn(N, cfg['input_var'], device=device),
             edge_attr=torch.randn(E, 8, device=device),
             edge_index=torch.cat([mesh, mesh.flip(0)], dim=1),
             y=torch.randn(N, cfg['output_var'], device=device),
             pos=torch.randn(N, 3, device=device))
    g.batch = torch.zeros(N, dtype=torch.long, device=device)
    if cfg['use_world_edges']:
        g.world_edge_index = torch.randint(0, N, (2, M // 2), device=device)
        g.world_edge_attr = torch.randn(M // 2, 8, device=device)
    if multiscale:
        n_prev = N
        for i in range(cfg['multiscale_levels']):
            n_c = max(n_prev // 8, 16)
            g[f'fine_to_coarse_{i}'] = torch.randint(0, n_c, (n_prev,), device=device)
            g[f'coarse_edge_index_{i}'] = torch.randint(0, n_c, (2, n_c * 4), device=device)
            g[f'coarse_edge_attr_{i}'] = torch.randn(n_c * 4, 8, device=device)
            g[f'num_coarse_{i}'] = torch.tensor([n_c], device=device)
            g[f'coarse_centroid_{i}'] = torch.randn(n_c, 3, device=device)
            g[f'unpool_edge_index_{i}'] = torch.stack([
                torch.randint(0, n_c, (n_prev * 2,), device=device),
                torch.randint(0, n_prev, (n_prev * 2,), device=device)])
            n_prev = n_c
    return g


def probe(label, cfg, device, backend, multiscale=False):
    print(f"\n--- {label} ---")
    torch.manual_seed(0)
    dynamo.reset()
    model = MeshGraphNets(cfg, device)
    model.train()
    graph = make_graph(cfg, device, multiscale)

    ok = True
    try:
        expl = dynamo.explain(model)(graph)
        status = 'OK' if expl.graph_break_count == 0 else 'BREAKS'
        print(f"  graphs: {expl.graph_count}  breaks: {expl.graph_break_count}  "
              f"ops: {expl.op_count}   [{status}]")
        for reason in expl.break_reasons:
            print(f"    BREAK: {getattr(reason, 'reason', reason)}")
            for frame in reversed(getattr(reason, 'user_stack', []) or []):
                if 'MeshGraphNets' in getattr(frame, 'filename', ''):
                    print(f"      at {frame.filename}:{frame.lineno} in {frame.name}")
                    break
        ok = expl.graph_break_count == 0
    except Exception:
        print("  explain() raised:")
        traceback.print_exc(limit=4)
        return False

    dynamo.reset()
    try:
        out, _ = torch.compile(model, dynamic=True, backend=backend)(graph)
        out.pow(2).sum().backward()
        print(f"  {backend} fwd+bwd: OK  out={tuple(out.shape)}")
    except Exception as e:
        print(f"  {backend} fwd+bwd: FAILED  {type(e).__name__}: "
              f"{str(e).splitlines()[0][:120]}")
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', default='inductor',
                    help="compile backend: inductor (default) or aot_eager")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device={device}  torch={torch.__version__}  backend={args.backend}")
    if device == 'cuda':
        print(f"gpu={torch.cuda.get_device_name(0)}  "
              f"cc={torch.cuda.get_device_capability(0)}  "
              f"arch_list={torch.cuda.get_arch_list()}")

    ms = dict(use_multiscale=True, multiscale_levels=2, mp_per_level=[2, 2, 2, 2, 2])
    cases = [
        ("flat MGN", base_config(), False),
        ("flat MGN + world edges", base_config(use_world_edges=True), False),
        ("flat MGN + checkpointing", base_config(use_checkpointing=True), False),
        ("HI-MGN", base_config(**ms), True),
        ("HI-MGN + world edges + checkpointing",
         base_config(use_world_edges=True, use_checkpointing=True, **ms), True),
    ]
    results = [(label, probe(label, cfg, device, args.backend, m))
               for label, cfg, m in cases]

    print("\n" + "=" * 60)
    failed = [label for label, ok in results if not ok]
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
