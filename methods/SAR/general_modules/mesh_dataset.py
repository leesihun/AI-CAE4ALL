"""Shared mesh HDF5 -> SAR graphs (the ex3 shell-buckling adapter).

SAR's unit of data is a *system* (one domain) with many *states* (draws of
the field on it). Here a system is one geometry -- every sample that shares
the `split_group_attr` metadata value (`shape_idx` on ex3) -- and its states
are that geometry's FEA draws. What the model is given, per node:

    pos     rows 0:3 at t=0, shifted by the train-set centre and divided by
            the train-set max |coordinate| (one isotropic factor)
    glob    the `cond_var` input-only rows (ex3: thickness, delta), each
            min/max-scaled to [-1, 1] over the train split (0 if constant)
    loc     node-type one-hot from the LAST nodal_data row (`use_node_types`)
    edge_attr  pos[dst] - pos[src] on the bidirectional mesh edges, divided
            by 2r (upstream ScaleEdgeAttr); r defaults to half the mean train
            edge length, so |edge_attr| is O(1)

and what it generates:

    target  rows 3:3+output_var at the final timestep, each channel
            min/max-scaled to [-1, 1] over the train split (upstream ScaleAttr)

Nothing else in the file is read. In particular the top-level `latent` group
and every metadata attribute except the grouping one (ex3 stores the hidden
imperfection parameters there) never reach the model: the one-to-many spread
has to come from SAR's own sampler, as it would for an unknown input.

The static part of a geometry (positions, edges, node types and, for the SAR
stage, the Guillard scale hierarchy) is identical across its draws, so it is
built once per run and cloned into every item; only the field rows are read
per item.
"""
import os
import time

import h5py
import numpy as np
import torch
from torch_geometric.utils import coalesce, to_undirected

from sar.graph import Graph
from sar.transforms import MeshCoarsening

from .grouped_split import grouped_split_ids


DEFAULT_GROUP_ATTR = 'shape_idx'
SPLIT_RATIOS = (0.8, 0.1, 0.1)   # MeshGraphNets_Variational training_profiles/setup.py


# ---------------------------------------------------------------------------
# Ids, groups, split
# ---------------------------------------------------------------------------

def list_sample_ids(h5_path):
    with h5py.File(h5_path, 'r') as f:
        return sorted(int(k) for k in f['data'].keys())


def group_ids(h5_path, sample_ids, attribute):
    """{group key: [sample ids in id order]}, keys in sorted order."""
    groups = {}
    with h5py.File(h5_path, 'r') as f:
        for sid in sample_ids:
            meta = f[f'data/{sid}'].get('metadata')
            if meta is None or attribute not in meta.attrs:
                raise ValueError(f'Sample {sid} lacks metadata attribute {attribute!r}')
            groups.setdefault(str(meta.attrs[attribute]), []).append(sid)
    return {k: groups[k] for k in sorted(groups)}


def split_ids(config):
    """(train, val, test) sample ids, exactly as MeshGraphNets_Variational splits.

    Same id list (sorted ints), same ratios (0.8/0.1/0.1), same seed key
    (`split_seed`, default 42) and the same grouped_split.py -- so with
    `split_group_attr shape_idx` every route trains on the same geometries.
    """
    h5_path = config['dataset_dir']
    ids = list_sample_ids(h5_path)
    seed = int(config.get('split_seed', 42))
    attr = config.get('split_group_attr')
    train_r, val_r, test_r = SPLIT_RATIOS
    if attr:
        return grouped_split_ids(h5_path, ids, attr, train_r, val_r, seed)
    rng = np.random.default_rng(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    n_train, n_val = int(len(ids) * train_r), int(len(ids) * val_r)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def group_attr(config):
    """The attribute that defines a SAR system (one geometry)."""
    return config.get('split_group_attr') or config.get('system_attr') or DEFAULT_GROUP_ATTR


# ---------------------------------------------------------------------------
# Row layout
# ---------------------------------------------------------------------------

def row_layout(config, num_rows):
    """Row indices of the target, the condition and the node-type rows."""
    out = int(config['output_var'])
    cond = int(config.get('cond_var', 0) or 0)
    target_rows = list(range(3, 3 + out))
    cond_rows = list(range(3 + out, 3 + out + cond))
    need = 3 + out + cond
    if num_rows < need:
        raise ValueError(f'nodal_data has {num_rows} rows but output_var {out} + cond_var {cond} '
                         f'need at least {need} (3 coordinate rows first)')
    type_row = None
    if bool(config.get('use_node_types', True)):
        if num_rows <= need:
            raise ValueError('use_node_types True needs a node-type row after the condition rows; '
                             f'nodal_data has only {num_rows} rows')
        type_row = num_rows - 1      # node type is the LAST row (DATASET_FORMAT.md)
    return target_rows, cond_rows, type_row


def _nodal(f, sid):
    return f[f'data/{sid}/nodal_data']


def _bidirectional_edges(mesh_edge, num_nodes):
    ei = torch.as_tensor(np.asarray(mesh_edge), dtype=torch.long)
    if ei.dim() != 2:
        raise ValueError(f'mesh_edge must be 2-D, got {tuple(ei.shape)}')
    if ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.t()
    ei = to_undirected(ei, num_nodes=num_nodes)
    ei = ei[:, ei[0] != ei[1]]
    # Guillard coarsening reads the senders of each node as contiguous runs of
    # the receiver-sorted edge list.
    return coalesce(ei, num_nodes=num_nodes, sort_by_row=False)


# ---------------------------------------------------------------------------
# Normalization (train split only)
# ---------------------------------------------------------------------------

def compute_normalization(config, train_groups, log=print):
    """Train-split statistics; stored in every checkpoint and reused verbatim."""
    h5_path = config['dataset_dir']
    t0 = time.time()
    with h5py.File(h5_path, 'r') as f:
        first = next(iter(train_groups.values()))[0]
        num_rows = _nodal(f, first).shape[0]
        target_rows, cond_rows, type_row = row_layout(config, num_rows)
        field_rows = target_rows + cond_rows

        # Geometry: one draw per system (identical across its draws).
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        types = set()
        for sids in train_groups.values():
            nodal = _nodal(f, sids[0])
            pos = nodal[0:3, 0, :]
            lo = np.minimum(lo, pos.min(axis=1))
            hi = np.maximum(hi, pos.max(axis=1))
            if type_row is not None:
                types.update(int(round(v)) for v in np.unique(nodal[type_row, 0, :]))
        center = 0.5 * (lo + hi)
        scale = float(np.max(np.abs(np.stack([lo - center, hi - center]))))
        if not np.isfinite(scale) or scale <= 0:
            scale = 1.0

        # Mean edge length in normalized units (for the 2r edge scaling).
        lengths = []
        for sids in train_groups.values():
            nodal = _nodal(f, sids[0])
            pos = (nodal[0:3, 0, :].T - center) / scale
            ei = np.asarray(f[f'data/{sids[0]}/mesh_edge'])
            if ei.shape[0] != 2 and ei.shape[1] == 2:
                ei = ei.T
            lengths.append(np.linalg.norm(pos[ei[1]] - pos[ei[0]], axis=1).mean())
        mean_edge = float(np.mean(lengths))

        # Field rows: every train draw (a min/max must see the extremes).
        f_lo = np.full(len(field_rows), np.inf)
        f_hi = np.full(len(field_rows), -np.inf)
        n_read = 0
        for sids in train_groups.values():
            for sid in sids:
                nodal = _nodal(f, sid)
                block = nodal[target_rows[0]:target_rows[-1] + 1, -1, :]
                if cond_rows:
                    # Conditions at t=0, as MGN-V reads them (input-only rows).
                    block = np.concatenate([block, nodal[cond_rows[0]:cond_rows[-1] + 1, 0, :]], axis=0)
                f_lo = np.minimum(f_lo, block.min(axis=1))
                f_hi = np.maximum(f_hi, block.max(axis=1))
                n_read += 1

    out = len(target_rows)
    r = config.get('edge_attr_r', 'auto')
    edge_r = 0.5 * mean_edge if str(r).lower() == 'auto' else float(r)
    stats = {
        'pos_center': [float(v) for v in center],
        'pos_scale': scale,
        'target_min': [float(v) for v in f_lo[:out]],
        'target_max': [float(v) for v in f_hi[:out]],
        'cond_min': [float(v) for v in f_lo[out:]],
        'cond_max': [float(v) for v in f_hi[out:]],
        'node_types': sorted(types),
        'edge_r': float(edge_r),
        'mean_edge_length': mean_edge,
        'output_var': out,
        'cond_var': len(cond_rows),
        'use_node_types': type_row is not None,
        'num_rows': int(num_rows),
    }
    log(f'Normalization from {len(train_groups)} train systems / {n_read} draws '
        f'({time.time() - t0:.1f}s): pos centre {np.round(center, 4).tolist()} scale {scale:.4g}; '
        f'mean edge {mean_edge:.4g} -> r {edge_r:.4g}; target min {np.round(f_lo[:out], 6).tolist()} '
        f'max {np.round(f_hi[:out], 6).tolist()}; node types {stats["node_types"]}')
    return stats


def cond_features(stats):
    return int(stats['cond_var']) + (len(stats['node_types']) if stats['use_node_types'] else 0)


def _minmax(values, vmin, vmax):
    """Upstream ScaleAttr per channel; a constant channel maps to 0."""
    vmin = np.asarray(vmin, dtype=np.float64)
    vmax = np.asarray(vmax, dtype=np.float64)
    span = vmax - vmin
    safe = np.where(span > 0, span, 1.0)
    out = 2.0 * (values - vmin) / safe - 1.0
    return np.where(span > 0, out, 0.0)


def normalize_target(values, stats):
    """[N, out] physical -> [-1, 1]."""
    return _minmax(values, stats['target_min'], stats['target_max'])


def denormalize_target(values, stats):
    """[..., out] in [-1, 1] -> physical."""
    vmin = np.asarray(stats['target_min'], dtype=np.float64)
    vmax = np.asarray(stats['target_max'], dtype=np.float64)
    return (np.asarray(values, dtype=np.float64) + 1.0) * 0.5 * (vmax - vmin) + vmin


# ---------------------------------------------------------------------------
# Static per-system graph
# ---------------------------------------------------------------------------

_INT_KEYS_PREFIX = ('edge_index', 'idx', 'batch', 'scale')


def build_static(f, sid, stats, num_scales, config):
    """The draw-independent part of one system's graph, as a dict of CPU tensors.

    num_scales >= 1 runs upstream MeshCoarsening(num_scales) once (Guillard
    levels, `scale`, `batch_k`); 0 skips it (a single-scale VAE needs none).
    Integer tensors are kept as int32 to halve the cache; `make_graph`
    restores int64.
    """
    nodal = _nodal(f, sid)
    num_rows = nodal.shape[0]
    _, _, type_row = row_layout(config, num_rows)
    raw = nodal[0:3, 0, :].T.astype(np.float64)
    num_nodes = raw.shape[0]
    pos = torch.as_tensor((raw - np.asarray(stats['pos_center'])) / stats['pos_scale'], dtype=torch.float32)
    edge_index = _bidirectional_edges(f[f'data/{sid}/mesh_edge'][()], num_nodes)
    deg = torch.bincount(edge_index[1], minlength=num_nodes)
    if int((deg == 0).sum()) > 0:
        raise ValueError(f'sample {sid}: {int((deg == 0).sum())} node(s) have no mesh edge; '
                         'Guillard coarsening needs a connected mesh')
    edge_attr = (pos[edge_index[1]] - pos[edge_index[0]]) / (2.0 * stats['edge_r'])
    g = Graph(pos=pos, edge_index=edge_index, edge_attr=edge_attr)
    if type_row is not None:
        types = np.rint(nodal[type_row, 0, :]).astype(np.int64)
        lut = {t: i for i, t in enumerate(stats['node_types'])}
        unknown = sorted(set(np.unique(types).tolist()) - set(lut))
        if unknown:
            raise ValueError(f'sample {sid}: node type(s) {unknown} not seen in the train split '
                             f'({stats["node_types"]})')
        idx = torch.as_tensor([lut[t] for t in types.tolist()], dtype=torch.long)
        g.loc = torch.nn.functional.one_hot(idx, num_classes=len(lut)).float()
    if num_scales >= 1:
        g = MeshCoarsening(num_scales)(g)
    static = {}
    for key in g.keys():
        value = g[key]
        if not isinstance(value, torch.Tensor):
            continue
        if value.dtype == torch.long:
            value = value.to(torch.int32)
        static[key] = value.contiguous()
    static['_num_nodes'] = num_nodes
    return static


def build_statics(config, groups, stats, num_scales, log=print, what='systems'):
    """{group key: static dict} for every system in `groups`."""
    statics = {}
    t0 = time.time()
    with h5py.File(config.get('_static_source') or config['dataset_dir'], 'r') as f:
        for i, (key, sids) in enumerate(groups.items()):
            statics[key] = build_static(f, sids[0], stats, num_scales, config)
            if (i + 1) % 10 == 0 or i + 1 == len(groups):
                log(f'  static graphs: {i + 1}/{len(groups)} {what} '
                    f'({time.time() - t0:.1f}s, scales={num_scales})')
    return statics


def build_statics_from(h5_path, config, groups, stats, num_scales, log=print, what='systems'):
    cfg = dict(config, _static_source=h5_path)
    return build_statics(cfg, groups, stats, num_scales, log=log, what=what)


def make_graph(static, target=None, glob=None):
    """A fresh Graph for one draw: static tensors cloned (the Collater edits
    edge_index_k / idx in place), int32 back to int64."""
    g = Graph()
    for key, value in static.items():
        if key.startswith('_'):
            continue
        value = value.clone()
        if value.dtype == torch.int32:
            value = value.long()
        g[key] = value
    if target is not None:
        g.target = target
    if glob is not None:
        g.glob = glob
    return g


def read_fields(f, sid, stats, config):
    """(target [N, out], glob [N, cond] or None), normalized.

    The target is the final timestep; the input-only condition rows are read
    at t=0, as MeshGraphNets_Variational reads them (identical on static data).
    """
    nodal = _nodal(f, sid)
    target_rows, cond_rows, _ = row_layout(config, nodal.shape[0])
    block = nodal[target_rows[0]:target_rows[-1] + 1, -1, :].T.astype(np.float64)
    target = torch.as_tensor(normalize_target(block, stats), dtype=torch.float32)
    glob = None
    if cond_rows:
        cond = nodal[cond_rows[0]:cond_rows[-1] + 1, 0, :].T.astype(np.float64)
        glob = torch.as_tensor(_minmax(cond, stats['cond_min'], stats['cond_max']),
                               dtype=torch.float32)
    return target, glob


def read_conditions_raw(f, sid, config):
    """Physical condition rows [cond, N] at t=0 (to check a system's draws share them)."""
    nodal = _nodal(f, sid)
    _, cond_rows, _ = row_layout(config, nodal.shape[0])
    if not cond_rows:
        return None
    return nodal[cond_rows[0]:cond_rows[-1] + 1, 0, :]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SystemDrawDataset(torch.utils.data.Dataset):
    """One item = one draw of one system.

    mode 'random': upstream semantics -- the epoch visits every system
    `draws_per_epoch` times and each visit picks one of its draws at random
    (torch RNG, so DataLoader workers get distinct seeded streams).
    mode 'fixed':  the first `draws_per_system` draws of every system, in
    order (validation, visualization).
    """

    def __init__(self, h5_path, groups, statics, stats, config, mode='random',
                 draws_per_epoch=1, draws_per_system=1):
        self.h5_path = h5_path
        self.keys = list(groups)
        self.groups = {k: list(v) for k, v in groups.items()}
        self.statics = statics
        self.stats = stats
        self.config = {k: config[k] for k in ('output_var', 'cond_var', 'use_node_types') if k in config}
        self.mode = mode
        self.draws_per_epoch = max(1, int(draws_per_epoch))
        if mode == 'fixed':
            self.items = [(k, s) for k in self.keys for s in self.groups[k][:max(1, int(draws_per_system))]]
        elif mode != 'random':
            raise ValueError(f'mode must be random or fixed, got {mode!r}')
        self._h5 = None
        self._pid = None

    def __len__(self):
        if self.mode == 'fixed':
            return len(self.items)
        return len(self.keys) * self.draws_per_epoch

    def _file(self):
        if self._h5 is None or self._pid != os.getpid():
            self._h5 = h5py.File(self.h5_path, 'r')
            self._pid = os.getpid()
        return self._h5

    def item_ids(self, index):
        if self.mode == 'fixed':
            return self.items[index]
        key = self.keys[index % len(self.keys)]
        sids = self.groups[key]
        return key, sids[int(torch.randint(len(sids), (1,)).item())]

    def __getitem__(self, index):
        key, sid = self.item_ids(index)
        target, glob = read_fields(self._file(), sid, self.stats, self.config)
        g = make_graph(self.statics[key], target=target, glob=glob)
        g.sample_id = torch.tensor([int(sid)])
        return g

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_h5'] = None
        state['_pid'] = None
        return state
