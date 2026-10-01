"""Single source of truth for the reviewed dataset/method baseline matrix.

--json emits path -> content for review/apply_patch; --check detects drift.
This script does not launch training or overwrite existing configs.
"""
from pathlib import Path
import argparse
import json
import math

ROOT = Path(__file__).resolve().parents[3]
DET = 'deterministic'
PROB = 'probabilistic'
DERIVED = 'derived/config_matrix'
METHODS = ['deeponet', 'point_deeponet', 'fno', 'transolver3', 'meshgraphnets', 'himgn']
REPOSITORIES = {**{m: 'Neural_Operator' for m in METHODS[:3]},
                'transolver3': 'Transolver', 'meshgraphnets': 'MeshGraphNets',
                'himgn': 'MeshGraphNets', 'himgn_v': 'MeshGraphNets_Variational',
                'chi_mgnflow': 'HI_MGNFlow', 'lsh_vae': 'SimulGenVAE', 'sdfflow': 'SDFFlow'}
# Lane assignment: one GPU per method group on an 8-card box, cheap routes
# sharing a lane and the few-example generative routes bundled two per lane.
# configs/run_all_135.sh and configs/run_all_136.sh drive these lanes. No
# generated arm sits on lane 2; only the hand-maintained geometry ex4 does.
LANE_GPU = {'deeponet': 0, 'fno': 0, 'point_deeponet': 1, 'transolver3': 3,
            'meshgraphnets': 4, 'himgn': 5, 'himgn_v': 6, 'lsh_vae': 6,
            'chi_mgnflow': 7, 'sdfflow': 7}
# SDFFlow geometry examples whose configs are written by hand, not by build();
# manifest.json lists them (generated: false) so run_matrix.py runs them.
HAND_GEOMETRY = ('ex2', 'ex3', 'ex4')
PAPERS = {
    'meshgraphnets': 'https://arxiv.org/abs/2010.03409',
    'himgn': 'https://arxiv.org/abs/2608.13827',
    'deeponet': 'https://doi.org/10.1038/s42256-021-00302-5 ; https://arxiv.org/abs/2111.05512',
    'point_deeponet': 'https://arxiv.org/abs/2412.18362',
    'fno': 'https://arxiv.org/abs/2010.08895',
    'transolver3': 'https://arxiv.org/abs/2602.04940',
    'himgn_v': 'https://arxiv.org/abs/1706.02262',
    'chi_mgnflow': 'https://arxiv.org/abs/2504.02843',
    'lsh_vae': 'https://doi.org/10.1007/s00366-023-01916-6',
    'sdfflow': 'https://arxiv.org/abs/2301.11445 ; https://arxiv.org/abs/2210.02747',
}


def case(key, file, fields, conditions=(), part=False, steps=1, dim=2,
         clusters=(1000, 100), geometry='fixed', indices=None, epochs=500, batch=2, **extra):
    category = extra.pop('category', DET)
    return dict(key=key, category=category, train=f'{category}/{file}.h5',
                infer=f'{category}/{file}_infer.h5', input=['x_coord', 'y_coord', 'z_coord'],
                output=list(fields), conditioner=list(conditions), partNo=['partNo'] if part else [],
                steps=steps, dim=dim, clusters=list(clusters), geometry=geometry,
                indices=indices, epochs=epochs, batch=batch, **extra)


# Budgets are judged by optimizer updates, ceil(n_train x windows / batch) x epochs,
# against the paper that used each dataset; every method in a slot gets at least
# that budget (seed-42 8:2 split, so n_train = 0.8 x samples):
#   ex1   80 x 1    5000 x b1 = 400k   HI-MGN "5000 epochs, a batch size of 1"
#   ex3   84 x 1    1250 x b1 = 105k   HI-MGN CRM 1000 epochs x 105 train samples
#   ex4   80 x 599   420 x b2 = 10.06M MGN CylinderFlow 10M steps x batch 2
#   ex5   80 x 399   627 x b2 = 10.01M MGN DeformingPlate 10M steps x batch 2
#   ex6   80 x 400   313 x b1 = 10.02M MGN FlagSimple 10M steps x batch 1
#   ex8  800 x 1     625 x b1 = 500k   Transolver elasticity 500 epochs x batch 1
#                                      x its 1000 train files, on our 800
#   ex10 952 x 1     500 x b1 = 476k   no published budget; batch 1 like ex7/ex8
# Transolver-3 always trains at batch 1 (its paper's recipe), so on ex4/ex5 it
# gets 2x the updates at equal sample presentations, and it keeps 500 epochs on
# ex6 (16.0M). ex2 keeps 500 epochs for every method but MGN/HI-MGN (100, their
# paper's budget, via HIMGN_PAPER_BUDGET). Those over-budgets are recorded in
# the README, not cut.
#
# std_noise is in node-normalized units (sigma_i = std_noise x node_std_i), one
# value per slot for every method that noises (MGN, HI-MGN, Transolver-3, the
# neural operators). A paper sigma sets it as sigma_paper / the largest train
# node std of the noised field, so no channel is noised above the paper value:
#   ex4 0.02 / 0.4812 (velocity_x)  = 0.0416   MGN CylinderFlow sigma 2e-2
#   ex5 0.003 / 0.02382 (uz)        = 0.126    MGN DeformingPlate sigma 3e-3
#   ex6 0.001 / 0.7437 (uy)         = 0.00134  MGN FlagSimple sigma 1e-3 (paper
#       Table 2; the official code's 3e-3 pairs with noise_gamma 0.1, ours is 1)
#   ex2, ex9 0.01                              HI-MGN's normalized 0.01
#
# ex1, ex2 and ex3 are the three HI-MGN paper cases (arXiv:2608.13827), which all
# use the [N, 5000, 100] hierarchy. The other cases are not in that paper; their
# clusters are scaled to their own (much smaller) meshes.
CASES = [
    # uz is identically zero here; keep the channel, drop its pull. The stress train
    # std is 3.274e-9, under every runtime's 1e-8 std floor, so normalized stress
    # has std 0.327 and (1e-8 / 3.274e-9)^2 = 9.33 restores its unit weight.
    case('ex1', 'ex1_static_thermoelastic', ['ux', 'uy', 'uz', 'stress'], part=True, epochs=5000, batch=1,
         clusters=(5000, 100), loss_weights=[1.0, 1.0, 0.001, 9.33]),
    case('ex2', 'ex2_dynamic_contact', ['ux', 'uy', 'uz', 'stress'], part=True,
         steps=49, dim=3, clusters=(5000, 100), geometry='displacement', indices=[0, 1, 2], batch=1,
         std_noise=0.01),
    case('ex3_full', 'unused', ['cp', 'cf_x', 'cf_y', 'cf_z'],
         ['mach', 'aoa_deg', 'aileron_inboard_deg', 'aileron_outboard_deg', 'elevator_deg',
          'htp_deg', 'normal_x', 'normal_y', 'normal_z', 'surface_area'],
         dim=3, clusters=(5000, 100), epochs=1250, batch=1, dense=True),
    case('ex3_mid', 'unused', ['cp', 'cf_x', 'cf_y', 'cf_z'],
         ['mach', 'aoa_deg', 'aileron_inboard_deg', 'aileron_outboard_deg', 'elevator_deg',
          'htp_deg', 'normal_x', 'normal_y', 'normal_z', 'surface_area'],
         dim=3, clusters=(5000, 100), epochs=1250, batch=1, dense=True),
    case('ex4', 'ex4_cylinder_flow', ['velocity_x', 'velocity_y', 'pressure'], part=True,
         steps=599, clusters=(256, 32), epochs=420, std_noise=0.0416),
    case('ex5', 'ex5_deforming_plate', ['ux', 'uy', 'uz', 'stress'], part=True,
         steps=399, dim=3, clusters=(256, 32), geometry='displacement', indices=[0, 1, 2], epochs=627,
         std_noise=0.126),
    case('ex6', 'ex6_flag_simple', ['ux', 'uy', 'uz'], part=True, steps=400,
         clusters=(256, 32), geometry='displacement', indices=[0, 1, 2], epochs=313, batch=1,
         transolver_epochs=500, std_noise=0.00134, dense=True),
    case('ex7', 'ex7_airfrans', ['velocity_x', 'velocity_y', 'pressure', 'nu_t'],
         ['u_inf_x', 'u_inf_y', 'sdf', 'normal_x', 'normal_y'], part=True,
         clusters=(4096, 256), batch=1),
    case('ex8', 'ex8_elasticity', ['sigma'], clusters=(128, 16), epochs=625, batch=1),
    case('ex9', 'ex9_plasticity', ['ux', 'uy'], ['uz_zero', 'die_profil'], steps=19,
         clusters=(512, 64), geometry='displacement', indices=[0, 1, -1], batch=8, std_noise=0.01,
         dense=True),
    case('ex10', 'ex10_deepjeb_mgn', ['stress', 'displacement_magnitude', 'z_disp'],
         ['lc_ver', 'lc_hor', 'lc_dia', 'lc_tor'], dim=3, clusters=(512, 64), batch=1,
         split_group_attr='bracket'),
    # spread=(output channel, plot label, statistic) drives the generated-vs-GT
    # distribution comparison the two probabilistic routes write at the end of
    # inference. Both were chosen by measuring the GT statistic over the eval
    # file, because a statistic that barely moves between scenes makes every
    # calibration score noise:
    #   ex1 density  max - min  cv=0.080 over 18 scenes (96 -> 130): the cold/hot
    #       contrast of the mixing layer. pressure varies more (cv=0.862) but is
    #       not the diagnostic this benchmark is about.
    #   ex2 damage   MEAN, not max - min. Damage saturates at 1, so max - min is
    #       1.000 with cv=0.000 on all 250 eval scenes -- it measures the
    #       saturation, not the crack. uy is the prescribed 0.02 load (cv=0.000)
    #       and ux cv=0.018. On a 0/1 field the mean is the damaged-node
    #       fraction, cv=0.047, which is what the stochastic crack path moves.
    case('ex1', 'ex1_turbulent_radiative_layer', ['density', 'pressure', 'velocity_x', 'velocity_y'],
         ['tcool'], steps=100, clusters=(4096, 256), category=PROB,
         periodic_box=[128.0 / 127.0, 0.0, 0.0], epochs=500,
         spread=(0, 'density', 'range')),
    case('ex2', 'ex2_crack_path', ['damage', 'ux', 'uy'], steps=19,
         clusters=(4096, 256), geometry='displacement', indices=[1, 2, -1], category=PROB, epochs=500,
         spread=(0, 'damage', 'mean')),
]
# Held-out scenes in each probabilistic infer file (len(data) of the *_infer.h5).
# The probabilistic infer configs draw this many samples per scene.
TEST_SCENES = {'ex1': 18, 'ex2': 250}
# Training samples of each slot under the seed-42 split (0.8 of the train file;
# ex10 splits by bracket, 952/116/124). Only for the manifest's `updates` field.
TRAIN_COUNTS = {(DET, 'ex1'): 80, (DET, 'ex2'): 40, (DET, 'ex3_full'): 84, (DET, 'ex3_mid'): 84,
                (DET, 'ex4'): 80, (DET, 'ex5'): 80, (DET, 'ex6'): 80, (DET, 'ex7'): 643,
                (DET, 'ex8'): 800, (DET, 'ex9'): 720, (DET, 'ex10'): 952,
                (PROB, 'ex1'): 57, (PROB, 'ex2'): 1400}
# (epochs, batch) of the three HI-MGN paper cases (arXiv:2608.13827), which trains
# HI-MGN and its MGN baseline on the same budget: 2D static thermoelastic 5000
# epochs, 3D dynamic contact 100, NASA-CRM 1000 x 105 train samples (1250 x our
# 84 is the same 105k updates), batch 1 throughout. Only those two routes take
# it; ex1 and ex3 now match the case budget above, so in practice it differs
# only on ex2, where the other methods keep 500 epochs.
HIMGN_PAPER_BUDGET = {'ex1': (5000, 1), 'ex2': (100, 1), 'ex3_full': (1250, 1), 'ex3_mid': (1250, 1)}
# Comment lines rendered into every config of a slot whose epoch count is set
# from a paper budget rather than a round number (see the budget table above).
EPOCH_NOTES = {
    'ex3': [
        'training_epochs 1250 is the HI-MGN NASA-CRM budget in updates: its 1000 epochs',
        '  x 105 train samples = 105k; our seed-42 split trains 84, and 1250 x 84 = 105k.',
    ],
    'ex4': [
        "training_epochs 420 (not the roster's 500) is the paper's CylinderFlow budget:",
        '  80 train trajectories x 599 windows = 47,920 items/epoch; 420 epochs = 20.1M',
        '  sample presentations = the 10M steps x batch 2 of arXiv:2010.03409. Batch- and',
        "  world-size-invariant, so it holds under ddp. Do not 'normalize' it to 500.",
    ],
    'ex5': [
        "training_epochs 627 (not the roster's 500) is the paper's DeformingPlate budget:",
        '  80 train trajectories x 399 windows = 31,920 items/epoch; 627 epochs = 20.0M',
        '  sample presentations = the 10M steps x batch 2 of arXiv:2010.03409.',
    ],
    'ex6': [
        "training_epochs 313 at batch 1 (not the roster's 500 x 2) is the paper's",
        '  FlagSimple budget: 80 train trajectories x 400 windows = 32,000 updates/epoch;',
        '  313 epochs = 10.0M = the 10M steps x batch 1 of arXiv:2010.03409.',
    ],
    'ex8': [
        "training_epochs 625 (not the roster's 500) is the Transolver elasticity budget in",
        '  updates: its 500 epochs x batch 1 x 1000 train files = 500k; our seed-42 split',
        '  trains 800, and 625 x 800 = 500k.',
    ],
}
EX6_TRANSOLVER_NOTE = [
    'training_epochs 500 at batch 1 = 16.0M updates, 1.6x the FlagSimple budget the',
    '  other ex6 mesh methods train on (313 epochs): only shortfalls were raised, so',
    '  this over-budget is kept and recorded in the campaign README.',
]
for _c in CASES:
    if _c['key'] == 'ex3_full':
        _c['train'] = 'deterministic/ex3_NASA_CRM_full.h5'
        _c['infer'] = 'deterministic/ex3_NASA_CRM_full_infer.h5'
    elif _c['key'] == 'ex3_mid':
        _c['train'] = f'{DERIVED}/ex3_NASA_CRM_mid_canonical.h5'
        _c['infer'] = f'{DERIVED}/ex3_NASA_CRM_mid_canonical_infer.h5'


def mesh_config(c, method):
    run = f"../../output/dataset_matrix/{c['category']}/{c['key']}/{method}"
    cfg = dict(model={'himgn': 'meshgraphnets', 'transolver3': 'transolver',
                      'himgn_v': 'meshgraphnets-v', 'chi_mgnflow': 'chi-mgnflow'}.get(method, method),
               mode='train', gpu_ids=LANE_GPU[method], parallel_mode='ddp', split_seed=42,
               dataset_dir='../../dataset/' + c['train'], infer_dataset='../../dataset/' + c['infer'],
               modelpath=run + '/model.pth', log_file_dir=run + '/train.log',
               inference_output_dir=run + '/infer', infer_timesteps=c['steps'],
               input_var=len(c['output']), output_var=len(c['output']), cond_var=len(c['conditioner']),
               feature_loss_weights=c.get('loss_weights') or [1.0] * len(c['output']),
               use_node_types=bool(c['partNo']),
               positional_features=4, time_integration='ar_ot',
               training_epochs=c.get('transolver_epochs', c['epochs']) if method == 'transolver3' else c['epochs'],
               batch_size=c['batch'], learningr=0.0001, weight_decay=0.0001, warmup_epochs=5,
               num_workers=2, prefetch_factor=2, grad_accum_steps=1, augment_geometry=False,
               std_noise=c.get('std_noise', 0.0), use_amp=True,
               use_checkpointing=True, use_ema=True, ema_decay=0.99, use_parallel_stats=False,
               val_interval=5, test_interval=50, test_batch_idx=[0, 1], plot_feature_idx=0,
               display_trainset=True, display_testset=True, write_preprocessing=False,
               use_world_edges=False, use_multiscale=False)
    if c.get('split_group_attr'):
        cfg['split_group_attr'] = c['split_group_attr']
    if method in ('meshgraphnets', 'himgn', 'himgn_v', 'chi_mgnflow'):
        cfg.update(edge_var=8, latent_dim=128, message_passing_num=15,
                   geometry_state_mode=c['geometry'])
        if method in ('meshgraphnets', 'himgn'):
            cfg['weight_decay'] = 0.0  # arXiv:2010.03409 trains with plain Adam
            if c['category'] == DET and c['key'] in HIMGN_PAPER_BUDGET:
                cfg['training_epochs'], cfg['batch_size'] = HIMGN_PAPER_BUDGET[c['key']]
        if c['indices'] is not None:
            cfg['displacement_state_indices'] = c['indices']
        if c['category'] == DET and c['key'] in ('ex2', 'ex5'):
            # Radius = multiplier x the MEDIAN mesh-edge length of the first ten
            # train samples (see _compute_world_edge_radius). ex2 takes 2x, a
            # couple of cells out (~26 world edges/node). On ex5 that median is
            # 0.017679 (seed-42 split), so 1.70x = 0.03005 is the MeshGraphNets
            # DeformingPlate r_W = 0.03 (2x would be 0.0354, 1.18x of it).
            cfg.update(use_world_edges=True,
                       world_radius_multiplier=1.7 if c['key'] == 'ex5' else 2.0,
                       world_max_num_neighbors=64, world_edge_backend='scipy_kdtree')
        if method != 'meshgraphnets':
            cfg.update(use_multiscale=True, coarsening_type='voronoi_seedmean',
                       multiscale_levels=2, voronoi_clusters=c['clusters'], mp_per_level=[2, 3, 5, 3, 2])
        if method == 'himgn':
            # [4,6,8,6,4] = 28 blocks is the HI-MGN paper's configuration for every
            # case (arXiv:2608.13827). The MGN baseline keeps its own paper's 15, so
            # the two do not share a block count here. message_passing_num is
            # ignored under use_multiscale; it is set to the total for readability.
            cfg.update(pool_type='mean', unpool_type='sum', learned_interpolation=True,
                       voronoi_branches=[1, 1], coarse_world_edges=False,
                       mp_per_level=[4, 6, 8, 6, 4], message_passing_num=28)
        if c['category'] == PROB:
            cfg.update(positional_features=0, hierarchy_variants=2, hierarchy_seed=42,
                       training_seed=42, best_by='crps', val_interval=10, num_vae_samples=16,
                       vae_batch_size=16, batch_size=16, std_noise=0.0)
            if c.get('periodic_box'):
                cfg['periodic_box'] = c['periodic_box']
        if method == 'himgn_v':
            # Global latent 32, the width the campaign README documents. This is
            # wider than SAOI_run and hyperparameter_sweep (16), the only MGN-V
            # settings trained so far, so their results do not transfer as-is;
            # batch 16 comes from those runs and is set for the whole
            # probabilistic category above. The MMD term
            # is a biased V-statistic over the batch dimension: at batch 2 / D=32 it
            # reads 2.42 even for a perfect posterior and separates a 3x-too-wide one
            # by only 1.6 sd, so it stops regularising the aggregate posterior at all.
            cfg.update(use_vae=True, vae_latent_dim=32, vae_mp_layers=3, vae_graph_aware=True,
                       recon_loss='mse', alpha_recon=1000.0, lambda_mmd=1.0,
                       mmd_bandwidth='median', posterior_min_std=0.05, beta_aux=10.0,
                       z_conditioning='adaln', use_conditional_prior=True, prior_type='gnn_e2e',
                       prior_family='fm', prior_mp_layers=3, prior_hidden_dim=128,
                       prior_nll_weight=1.0, prior_grad_to_encoder=0.0,
                       prior_fm_steps=30, prior_fm_solver='heun')
            # One stage, so it gets cHI-MGNflow's AE + flow epochs (500 + 500) to
            # train on the same number of updates; the prior freezes at 70%.
            cfg['training_epochs'] = 2 * c['epochs']
            cfg['prior_freeze_epoch'] = int(0.7 * cfg['training_epochs'])
        if method == 'chi_mgnflow':
            cfg.pop('std_noise')  # This runtime deliberately rejects the retired noise control.
            # val_flow_steps = flow_steps, so validation integrates the flow the
            # way inference does.
            cfg.update(latent_ch=4, ae_kl_weight=0.000001, ae_epochs=c['epochs'],
                       prior_blocks=4, flow_time_freqs=16, flow_t_sampling='uniform',
                       flow_loss_weighting='uniform', flow_steps=30, flow_solver='heun',
                       flow_predict='sample', val_flow_steps=30, val_num_samples=8)
    elif method == 'transolver3':
        crm = c['key'].startswith('ex3_')
        cfg.update(coordinate_normalization='centered_isotropic', latent_dim=256,
                   # mlp_ratio 2 is the official Transolver-3 MODEL_KWARGS; the paper
                   # does not state it.
                   num_layers=24 if crm else 8, num_heads=8, slice_num=64, mlp_ratio=2,
                   attention_kernel='slice_space', chunk_size=4096, dropout=0.0,
                   temperature_init=0.5, temperature_min=0.1, temperature_max=5.0,
                   # The paper trains NASA-CRM (454k nodes, our ex3) on the full mesh
                   # and subsamples only the multi-million-node benchmarks, so every
                   # case here (ex2 200k, ex7 185k included) is full mesh.
                   amortized_training=False, amortized_cache_nodes=0, amortized_query_nodes=0,
                   # decoupled builds its token cache from the same mesh it decodes,
                   # so it matches direct at twice the embedding cost, and it has no
                   # temporal rollout (ex2/4/5/6/9 would raise NotImplementedError).
                   infer_mode='direct', infer_chunk_size=8192, write_test_predictions=True,
                   # Transolver-3 (arXiv:2602.04940) uses one recipe everywhere:
                   # lr 1e-3, weight decay 0.05, batch 1, warmup 5% of the epochs.
                   max_grad_norm=1.0, learningr=0.001, weight_decay=0.05,
                   warmup_epochs=cfg['training_epochs'] // 20, batch_size=1)
        if c['key'] in ('ex7', 'ex3_full', 'ex3_mid'):
            # fp32 training (bf16 autocast is the runtime default): bf16 rounds
            # the input rows so 24-25% of ex7's and 3.0% of ex3's nodes become
            # identical rows, the reason the neural operators run fp32 too.
            # Inference is fp32 on every route either way.
            cfg['use_amp'] = False
    else:
        dim = c['dim']
        # The grid spans each axis of the train bbox, so its shape follows the
        # domain's aspect at ~4k (2D) / ~16k (3D) points: x:y ex4 3.9, ex9 3.3,
        # ex7 2.0, ex6 1.5, ex1/ex8 square; x:y:z ex2 1:0.96:1, ex3 1:0.92:0.1
        # (z floored at 8), ex5 0.74:1:0.6, ex10 0.59:1:0.35. Modes and width are
        # the FNO paper's (arXiv:2010.08895): 2D 12 x 12 at width 32 (its Darcy /
        # 2D Navier-Stokes models, 2.4M parameters), 3D 8 per axis at width 20
        # (its 3D Navier-Stokes model, 6.6M), capped at 3 / 7 / 6 on the thin
        # final axes of ex3 / ex5 / ex10 (the earlier 0.375 x grid). Every entry
        # is inside its grid's Nyquist limit (res//2, res//2 + 1 on the last axis).
        resolution, modes = {
            'ex4': ([128, 32], [12, 12]), 'ex9': ([128, 32], [12, 12]),
            'ex7': ([96, 48], [12, 12]), 'ex6': ([78, 52], [12, 12]),
            'ex2': ([26, 26, 26], [8, 8, 8]),
            'ex3_full': ([48, 44, 8], [8, 8, 3]), 'ex3_mid': ([48, 44, 8], [8, 8, 3]),
            'ex5': ([24, 32, 20], [8, 8, 7]), 'ex10': ([26, 44, 16], [8, 8, 6]),
        }.get(c['key'], ([64, 64], [12, 12]))
        assert len(resolution) == dim, (c['key'], resolution)
        cfg.update(coordinate_normalization='centered_isotropic', operator_dim=dim,
                   dimension_tolerance=0.0001, grid_padding=0.1, out_of_bounds_policy='error',
                   sdf_source='none', global_condition_features='none', integration_weight_source='none',
                   write_test_predictions=True, max_grad_norm=1.0, checkpoint_interval=50,
                   learningr=0.001 if c['steps'] == 1 else 0.0001,
                   # fp32 throughout: bf16 rounds the trunk's input rows (coords +
                   # conditions + positional + one-hot) so 3.0% of ex3's and 24-25% of
                   # ex7's nodes become identical rows (coords alone: 37% / 40%). FNO's
                   # spectral path needs fp32 regardless.
                   use_amp=False)
        if method == 'deeponet':
            # ReLU is one of the DeepONet papers' activations (ReLU/tanh/ELU; GELU
            # is not). The sensor grid spans the train bbox like the FNO grid, so
            # the square ex1/ex8 domains take 32 x 32; 32 x 16 there would sample
            # one axis at half the resolution of the other.
            square = c['key'] in ('ex1', 'ex8')
            cfg.update(deeponet_branch_source='fixed_sensors',
                       deeponet_sensor_resolution=([32, 32] if square else [32, 16]) if dim == 2 else [16, 16, 8],
                       deeponet_hidden_channels=256, deeponet_branch_depth=4,
                       deeponet_trunk_depth=4, deeponet_basis_dim=128,
                       deeponet_activation='relu', deeponet_multi_output='split_both',
                       deeponet_max_branch_params=100000000, infer_query_chunk_size=16384)
        elif method == 'point_deeponet':
            # 5000 resampled sensors is the paper's recipe, but it samples with
            # replacement below 5000 nodes: ex4/5/6/8/9 meshes (727-3131 nodes)
            # see 80-99% of their nodes, and the 35% of ex10 samples just under
            # 5000 see ~63% while the rest see all. Those slots take every node
            # (point_sensor_count 0, the runtime's all-points path; no mesh there
            # exceeds 5014 nodes). omega0 10 and weight decay 1e-5 are the official
            # Point-DeepONet code's values; the paper states neither.
            full = c['key'] in ('ex4', 'ex5', 'ex6', 'ex8', 'ex9', 'ex10')
            cfg.update(point_variant='mesh_state', point_sensor_count=0 if full else 5000,
                       point_sampling='random',
                       point_resample_each_epoch=True, point_hidden_channels=128,
                       point_feature_dim=128, pointnet_depth=3, pointnet_activation='relu',
                       pointnet_norm='batch', point_branch_merge='sum', point_trunk_depth=3,
                       point_refiner_depth=2, point_siren_omega0=10.0, point_output_activation='identity',
                       infer_query_chunk_size=16384, weight_decay=0.00001)
        elif method == 'fno':
            cfg.update(fno_variant='mesh', fno_grid_resolution=resolution,
                       fno_modes=modes,
                       fno_hidden_channels=32 if dim == 2 else 20, fno_layers=4,
                       fno_use_channel_mlp=False,  # arXiv:2010.08895 layer is sigma(W v + K v), no MLP
                       fno_norm='none')
    return cfg


def dense_config(c):
    run = f"../../output/dataset_matrix/deterministic/{c['key']}/lsh_vae"
    crm = c['key'].startswith('ex3_')
    cfg = dict(model='simulgenvae', mode='train', gpu_ids=LANE_GPU['lsh_vae'], parallel_mode='single', split_seed=42,
               dataset_dir='../../dataset/' + c['train'], num_var=len(c['output']), field_start_row=3,
               cond_var=6 if crm else len(c['conditioner']), node_start=0, node_end=0, timesteps_reduced=0,
               latent_dim=8, latent_dim_end=32, num_filter_enc=[64, 64, 48, 48, 32, 24, 16],
               # alpha / beta_target ~ 1e6 (LSH-VAE Eq. 13); beta_target defaults to 1.0
               network_size='small', loss_type=1, alpha=1000000.0, init_beta_divisor=4,
               lc_data_type='hdf5' if crm else 'csv', lc_filter=[128, 128, 128], lc_dropout=0.1,
               # 5000 epochs is LSH-VAE Table 1. Its Eq. 14 holds beta at 1e-4 x
               # beta_target until 0.3 n and then raises it to beta_target by n; the
               # runtime ramps linearly, so the ramp starts at 0.3 n (the default
               # start fraction) and runs 0.7 n = 3500 epochs to the end. The
               # runtime default (0) would end the ramp at 0.8 n instead.
               kl_warmup_start_frac=0.3, kl_warmup_epochs=3500,
               vae_training_epochs=5000, vae_batch_size=2 if crm else 8,
               vae_learningr=0.001, vae_weight_decay=0.0, vae_warmup_epochs=20,
               vae_use_amp=False, vae_use_ema=False, vae_num_workers=0,
               lc_training_epochs=5000, lc_batch_size=16, lc_learningr=0.001,
               lc_weight_decay=0.00001, lc_warmup_epochs=20, lc_num_workers=0,
               vae_modelpath=run + '/vae.pth', lc_modelpath=run + '/lc.pth',
               pipeline_log_file=run + '/pipeline.log', vae_log_file_dir=run + '/vae.log',
               lc_log_file_dir=run + '/lc.log', output_dir=run + '/infer',
               skip_completed_stages=True, test_interval=50, display_testset=True, num_test_samples=2)
    if not crm:
        name = Path(c['train']).stem
        cfg['param_dir'] = f'../../dataset/{DERIVED}/{name}_conditions.csv'
    return cfg


# Comment lines rendered just above one key of the SDFFlow train config.
SDF_KEY_NOTES = {
    'latent_tokens': (
        '128 (was 512) from the 2026-09 token-count study (paper/sdfflow sec:tsize): 512 reconstructs',
        '1.5x better than 128 but its flow stage makes one-body shapes only 31-38% of the time at',
        '1,713 training shapes; 128 + the flow-stage prescription below matched or beat 32 tokens.'),
    'kl_weight': (
        '1e-8, the value the token-count study trained every arm with. The 2026-09 KL sweep',
        '(epoch 500) found 1e-4 collapsed the posterior (SNR 0); 1e-6 / 1e-8 / 1e-10 all kept it.'),
    'fm_hidden': (
        'Flow-stage prescription from the token-count study: a larger DiT (512 x 12 blocks),',
        'logit-normal time sampling shifted by -ln(sqrt(tokens/32)) = -0.693 for 128 tokens,',
        '8 posterior draws per training shape, and 250 epochs (8 draws = 8x the steps per epoch).')}


def sdf_config():
    run = '../../output/dataset_matrix/geometry_generation/ex1/sdfflow'
    return dict(model='sdfflow', mode='train', gpu_ids=LANE_GPU['sdfflow'], parallel_mode='single', seed=42, split_seed=42,
                dataset_dir='../../dataset/geometry_generation/ex1_deepjeb.h5', split_by_parent=True,
                num_encoder_points=6144, num_query_points=8192,
                # 3DShape2VecSet uses M=512 x C0=32; 128 is the token-count study's pick
                # (SDF_KEY_NOTES). FPS needs latent_tokens <= num_encoder_points (6144).
                latent_tokens=128, latent_dim=32,
                encoder_query_type='fps', encoder_dim=256, encoder_heads=8, encoder_blocks=4,
                encoder_self_attention=True, decoder_type='attention', decoder_hidden=256,
                decoder_layers=4, decoder_heads=8, fourier_bands=8, kl_weight=0.00000001,
                clamp_dist=0.1, deterministic_warmup_epochs=50, posterior_noise_warmup_epochs=100,
                posterior_noise_max_scale=1.0, kl_warmup_epochs=200, posterior_min_std_rel=0.05,
                surface_weight=1.0, normal_weight=0.1, eikonal_weight=0.1, hybrid_grad_points=1024,
                use_conditions=True, condition_names=['volume', 'area'], min_condition_std=0.00001,
                condition_clip=5.0, cond_dropout=0.2, cond_dropout_mode='all',
                fm_arch='dit', fm_hidden=512, fm_blocks=12, fm_heads=8, fm_cond_hidden=128,
                fm_time_sampling='logit_normal', fm_time_logit_mean=-0.693, fm_time_logit_std=1.0,
                fm_latent_draws=8, encode_batch_size=8, ode_steps=50,
                vae_training_epochs=1500, vae_batch_size=8, vae_learningr=0.0001,
                vae_weight_decay=0.0001, vae_warmup_epochs=20, vae_num_workers=2,
                vae_use_amp=False, vae_use_ema=True, vae_ema_decay=0.99,
                vae_val_interval=5, vae_test_interval=100, vae_num_test_shapes=2, vae_mc_resolution_test=64,
                fm_training_epochs=250, fm_batch_size=64, fm_learningr=0.0001,
                fm_weight_decay=0.0001, fm_warmup_epochs=10, fm_num_workers=0,
                fm_use_amp=True, fm_use_ema=True, fm_ema_decay=0.99,
                fm_val_interval=5, fm_test_interval=50, fm_num_test_shapes=2, fm_mc_resolution_test=64,
                display_testset=True, skip_completed_stages=True, output_dir=run + '/samples',
                vae_modelpath=run + '/vae.pth', fm_modelpath=run + '/fm.pth',
                fm_best_modelpath=run + '/fm_best.pth', pipeline_log_file=run + '/pipeline.log',
                vae_log_file_dir=run + '/vae.log', fm_log_file_dir=run + '/fm.log')


def short(n):
    for div, unit in ((1e6, 'M'), (1e3, 'k')):
        if n >= div:
            return f'{n / div:.4g}{unit}'
    return str(n)


def updates(c, method, cfg):
    """(optimizer updates on one card, note) for one train config: per stage,
    ceil(n_train x windows / batch) x epochs, windows = steps (1 when static).
    The note splits a two-stage arm; it is '' for one stage."""
    n = TRAIN_COUNTS[(c['category'], c['key'])]
    if method == 'lsh_vae':
        # One item per trajectory. The LC loader drops its last partial batch.
        vae = math.ceil(n / cfg['vae_batch_size']) * cfg['vae_training_epochs']
        lc = n // cfg['lc_batch_size'] * cfg['lc_training_epochs']
        return vae + lc, f'VAE {short(vae)} + LC {short(lc)}'
    per_epoch = math.ceil(n * c['steps'] / cfg['batch_size'])
    if method == 'chi_mgnflow':
        ae, flow = per_epoch * cfg['ae_epochs'], per_epoch * cfg['training_epochs']
        return ae + flow, f'AE {short(ae)} + flow {short(flow)}'
    return per_epoch * cfg['training_epochs'], ''


def render(c, method, config, notes=(), trailer=(), key_notes=None):
    """notes go just before the closing 'No source HDF5 writes' line, trailer after it;
    key_notes maps a key to comment lines written directly above that key."""
    comments = [f"{c['category']}/{c['key']} - {method}; paper-informed native adaptation, not an exact reproduction.",
                'Field storage order (0-based): input -> output -> conditioner -> partNo.',
                'input [0:3]: ' + ', '.join(c['input']),
                'output/state [3:]: ' + ', '.join(c['output']),
                'conditioner: ' + (', '.join(c['conditioner']) or 'none'),
                'partNo: ' + (', '.join(c['partNo']) or 'absent'),
                'Static state input is zero; temporal input is current state and target is next-state delta.',
                'Reference: ' + PAPERS[method],
                'See configs/campaigns/dataset_matrix/README.md for deviations and verification boundaries.',
                'No source HDF5 writes. Checkpoints are created by the paired training config.']
    if method == 'sdfflow':
        comments[1:7] = [
            'Logical order: input -> output -> conditioner -> partNo (named HDF5 arrays, not nodal rows).',
            'Train input: surface_points, surface_normals, query_xyz; generation input: latent noise and query_xyz.',
            'Output: signed_distance (zero level set is the generated surface).',
            'Conditioner order: volume, area; partNo: absent.',
            'The default infer config samples the unconditional branch; add named cond_values for conditional generation.',
        ]
    if method == 'lsh_vae':
        comments[6] = 'Full-trajectory LSH-VAE + latent conditioner. Prediction uses conditions only, not target fields.'
        comments.append('LC: six global CRM parameters in rows 7:13.' if c['key'].startswith('ex3_') else
                        'LC CSV order is declared in the accompanying *_conditions.json sidecar.')
    closing = comments.index('No source HDF5 writes. Checkpoints are created by the paired training config.')
    comments[closing:closing] = list(notes)
    comments.extend(trailer)
    def value(v):
        if isinstance(v, (list, tuple)):
            return ', '.join(value(x) for x in v)
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            if v == 0:
                return '0.0'
            # Fixed-point (the native parsers read 1e-4 as a string), but the
            # shortest round-trip form when it has no exponent: 9.33, not
            # 9.3300000000000001.
            fixed = format(v, '.16f').rstrip('0').rstrip('.')
            short = repr(v).rstrip('0').rstrip('.')
            return short if 'e' not in short and len(short) < len(fixed) else fixed
        return str(v)
    key_notes = key_notes or {}
    return '\n'.join('% ' + s for s in comments) + '\n\n' + ''.join(
        ''.join(f'% {n}\n' for n in key_notes.get(k, ())) + f'{k:<32} {value(v)}\n'
        for k, v in config.items())


def build():
    files, pairs = {}, []
    for c in CASES:
        methods = METHODS if c['category'] == DET else ['himgn_v', 'chi_mgnflow']
        if c.get('dense'):
            methods = methods + ['lsh_vae']
        for method in methods:
            train = dense_config(c) if method == 'lsh_vae' else mesh_config(c, method)
            infer = dict(train)
            infer['mode'] = 'reconstruct' if method == 'lsh_vae' else 'inference'
            if c.get('spread') and method in ('himgn_v', 'chi_mgnflow'):
                # Ground truth for the distribution comparison is the same
                # held-out file the rollout runs on, so each generated ensemble
                # is scored against the one simulation that scene actually
                # produced -- pooling the draws instead measures the marginal,
                # in which a model that ignores the geometry scores perfectly.
                channel, label, stat = c['spread']
                infer.update(eval_dataset='../../dataset/' + c['infer'],
                             spread_channel=channel, spread_label=label,
                             spread_stat=stat, make_histogram=True,
                             show_histogram=False, histogram_bins=30,
                             save_rollouts=False)
                # save_rollouts False: one HDF5 per (scene, draw) is
                # 18 x 16 x 79 MB = 23 GB on ex1 and 250 x 16 x 16 MB = 63 GB on
                # ex2, per route. spread_values.npz plus spread_metrics.json are
                # the surviving record, which is what this study is measuring.
            if method == 'lsh_vae':
                infer['dataset_dir'] = '../../dataset/' + c['infer']
                infer['batch_size'] = 2
                if 'param_dir' in infer:
                    infer['param_dir'] = infer['param_dir'].replace('_conditions.csv', '_infer_conditions.csv')
            trailers = ((), ())
            if c['category'] == PROB:
                scenes = TEST_SCENES[c['key']]
                infer['num_vae_samples'] = scenes
                trailers = ((), (f"num_vae_samples is draws PER held-out scene. Set to {scenes} = the "
                                 "number of test cases in infer_dataset/eval_dataset "
                                 f"({Path(c['infer']).name}), so each scene's ensemble is as large "
                                 "as the held-out set it is scored against.",))
            notes = ()
            if c['category'] == DET and method != 'lsh_vae':
                slot = c['key'].split('_')[0]
                notes = (EX6_TRANSOLVER_NOTE if slot == 'ex6' and method == 'transolver3'
                         else EPOCH_NOTES.get(slot, ()))
            base = f"configs/{REPOSITORIES[method]}/{c['category']}/{c['key']}/baseline"
            paths = [f'{base}/config_{mode}_{method}.txt' for mode in ('train', 'infer')]
            for path, cfg, trailer in zip(paths, (train, infer), trailers):
                files[path] = render(c, method, cfg, notes, trailer)
            total, split = updates(c, method, train)
            pairs.append(dict(category=c['category'], example=c['key'], method=method, train=paths[0], infer=paths[1],
                              updates=total, **({'updates_note': split} if split else {})))
    sdf = sdf_config()
    c = dict(category='geometry_generation', key='ex1', input=['surface_points', 'surface_normals', 'query_xyz'],
             output=['signed_distance'], conditioner=['volume', 'area'], partNo=[])
    base = 'configs/SDFFlow/geometry_generation/ex1/baseline'
    infer = dict(model='sdfflow', mode='sample', gpu_ids=LANE_GPU['sdfflow'], parallel_mode='single', seed=42,
                 vae_modelpath=sdf['vae_modelpath'], fm_modelpath=sdf['fm_best_modelpath'],
                 # 209 = the seed-42 parent-split test count, so the generated set can
                 # be scored against the held-out reference set at equal size.
                 output_dir=sdf['output_dir'], num_samples=209, mc_resolution=128, ode_steps=50,
                 cfg_scale=1.0, condition_ood_policy='error', max_condition_z=3.0)
    paths = [f'{base}/config_{mode}_sdfflow.txt' for mode in ('train', 'infer')]
    for path, cfg, key_notes in zip(paths, (sdf, infer), (SDF_KEY_NOTES, None)):
        files[path] = render(c, 'sdfflow', cfg, key_notes=key_notes)
    pairs.append(dict(category=c['category'], example=c['key'], method='sdfflow', train=paths[0], infer=paths[1]))
    assert len(pairs) == 75 and len(files) == 150
    # Hand-maintained SDFFlow geometry configs: listed so the campaign runner
    # runs them, but not rendered here. Their gpu_ids (0/1/2) and output paths
    # are their own, so `generated: false` exempts them from the one-lane-per-
    # method rule and from --check.
    for key in HAND_GEOMETRY:
        base = f'configs/SDFFlow/geometry_generation/{key}/baseline'
        pairs.append(dict(category='geometry_generation', example=key, method='sdfflow',
                          train=f'{base}/config_train_sdfflow.txt',
                          infer=f'{base}/config_infer_sdfflow.txt', generated=False))
    manifest = dict(version=1, excluded=['shell_buckling', 'grainpaint'], cases=CASES, pairs=pairs)
    files['configs/campaigns/dataset_matrix/manifest.json'] = json.dumps(manifest, indent=2) + '\n'
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--prefix', default='', help='Filter --json output paths for bounded review batches')
    args = parser.parse_args()
    files = build()
    if args.json:
        print(json.dumps({p: v for p, v in files.items() if p.startswith(args.prefix)}))
    else:
        different = [p for p, content in files.items()
                     if not (ROOT / p).exists() or (ROOT / p).read_text(encoding='utf-8') != content]
        if different:
            raise SystemExit('Missing or different generated files:\n' + '\n'.join(different))
        print(f'75 train/infer pairs: generated files match source of truth '
              f'(manifest also lists {len(HAND_GEOMETRY)} hand-maintained SDFFlow pairs).')


if __name__ == '__main__':
    main()
