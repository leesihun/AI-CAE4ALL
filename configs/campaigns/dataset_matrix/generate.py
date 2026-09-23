"""Single source of truth for the reviewed dataset/method baseline matrix.

--json emits path -> content for review/apply_patch; --check detects drift.
This script does not launch training or overwrite existing configs.
"""
from pathlib import Path
import argparse
import json

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
# configs/run_all_135.sh and configs/run_all_136.sh drive these lanes. Lane 2
# has no home arms; its worker goes straight to stealing (run_matrix.sh).
LANE_GPU = {'deeponet': 0, 'fno': 0, 'point_deeponet': 1, 'transolver3': 3,
            'meshgraphnets': 4, 'himgn': 5, 'himgn_v': 6, 'lsh_vae': 6,
            'chi_mgnflow': 7, 'sdfflow': 7}
PAPERS = {
    'meshgraphnets': 'https://arxiv.org/abs/2010.03409',
    'himgn': 'https://arxiv.org/abs/2608.13827',
    'deeponet': 'https://doi.org/10.1038/s42256-021-00302-5',
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


CASES = [
    case('ex1', 'ex1_static_thermoelastic', ['ux', 'uy', 'uz', 'stress'], part=True, epochs=2000, batch=2,
         loss_weights=[1.0, 1.0, 0.001, 1.0]),  # uz is identically zero here; keep the channel, drop its pull.
    case('ex2', 'ex2_dynamic_contact', ['ux', 'uy', 'uz', 'stress'], part=True,
         steps=49, dim=3, clusters=(5000, 250), geometry='displacement', indices=[0, 1, 2], batch=1),
    case('ex3_full', 'unused', ['cp', 'cf_x', 'cf_y', 'cf_z'],
         ['mach', 'aoa_deg', 'aileron_inboard_deg', 'aileron_outboard_deg', 'elevator_deg',
          'htp_deg', 'normal_x', 'normal_y', 'normal_z', 'surface_area'],
         dim=3, clusters=(5000, 250), batch=1, dense=True),
    case('ex3_mid', 'unused', ['cp', 'cf_x', 'cf_y', 'cf_z'],
         ['mach', 'aoa_deg', 'aileron_inboard_deg', 'aileron_outboard_deg', 'elevator_deg',
          'htp_deg', 'normal_x', 'normal_y', 'normal_z', 'surface_area'],
         dim=3, clusters=(4096, 256), batch=1, dense=True),
    case('ex4', 'ex4_cylinder_flow', ['velocity_x', 'velocity_y', 'pressure'], part=True,
         steps=599, clusters=(256, 32), epochs=420),
    case('ex5', 'ex5_deforming_plate', ['ux', 'uy', 'uz', 'stress'], part=True,
         steps=399, dim=3, clusters=(256, 32), geometry='displacement', indices=[0, 1, 2]),
    case('ex6', 'ex6_flag_simple', ['ux', 'uy', 'uz'], part=True, steps=400,
         clusters=(256, 32), geometry='displacement', indices=[0, 1, 2], dense=True),
    case('ex7', 'ex7_airfrans', ['velocity_x', 'velocity_y', 'pressure', 'nu_t'],
         ['u_inf_x', 'u_inf_y', 'sdf', 'normal_x', 'normal_y'], part=True,
         clusters=(4096, 256), batch=1),
    case('ex8', 'ex8_elasticity', ['sigma'], clusters=(128, 16), batch=16),
    case('ex9', 'ex9_plasticity', ['ux', 'uy'], ['uz_zero', 'die_profil'], steps=19,
         clusters=(512, 64), geometry='displacement', indices=[0, 1, -1], batch=8, dense=True),
    case('ex10', 'ex10_deepjeb_mgn', ['stress', 'displacement_magnitude'],
         ['lc_ver', 'lc_hor', 'lc_dia', 'lc_tor'], dim=3, clusters=(512, 64), batch=4,
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
               positional_features=4, time_integration='ar_ot', training_epochs=c['epochs'],
               batch_size=c['batch'], learningr=0.0001, weight_decay=0.0001, warmup_epochs=5,
               num_workers=2, prefetch_factor=2, grad_accum_steps=1, augment_geometry=False,
               std_noise=0.0 if c['steps'] == 1 else 0.01, use_amp=True,
               use_checkpointing=True, use_ema=True, ema_decay=0.99, use_parallel_stats=False,
               val_interval=5, test_interval=50, test_batch_idx=[0, 1], plot_feature_idx=0,
               display_trainset=True, display_testset=True, write_preprocessing=False,
               use_world_edges=False, use_multiscale=False)
    if c.get('split_group_attr'):
        cfg['split_group_attr'] = c['split_group_attr']
    if method in ('meshgraphnets', 'himgn', 'himgn_v', 'chi_mgnflow'):
        cfg.update(edge_var=8, latent_dim=128, message_passing_num=15,
                   geometry_state_mode=c['geometry'])
        if c['indices'] is not None:
            cfg['displacement_state_indices'] = c['indices']
        if c['category'] == DET and c['key'] in ('ex2', 'ex5'):
            # Radius = multiplier x the MEDIAN mesh-edge length (see
            # _compute_world_edge_radius). 2x reaches a couple of cells out:
            # ex5 26 -> 42 world edges/node, ex2 ~26/node. The MeshGraphNets
            # DeformingPlate r_W = 0.03 is 1.58x this dataset's median.
            cfg.update(use_world_edges=True, world_radius_multiplier=2.0,
                       world_max_num_neighbors=64, world_edge_backend='scipy_kdtree')
        if method != 'meshgraphnets':
            cfg.update(use_multiscale=True, coarsening_type='voronoi_seedmean',
                       multiscale_levels=2, voronoi_clusters=c['clusters'], mp_per_level=[2, 3, 5, 3, 2])
        if method == 'himgn':
            cfg.update(pool_type='mean', unpool_type='sum', learned_interpolation=True,
                       voronoi_branches=[1, 1], coarse_world_edges=False)
        if c['category'] == PROB:
            cfg.update(positional_features=0, hierarchy_variants=2, hierarchy_seed=42,
                       training_seed=42, best_by='crps', val_interval=10, num_vae_samples=16,
                       vae_batch_size=16, batch_size=16, std_noise=0.0)
            if c.get('periodic_box'):
                cfg['periodic_box'] = c['periodic_box']
        if method == 'himgn_v':
            # latent 16 matches SAOI_run and hyperparameter_sweep, the only MGN-V
            # settings that have actually been trained; batch 16 comes from the same
            # runs and is set for the whole probabilistic category above. The MMD term
            # is a biased V-statistic over the batch dimension: at batch 2 / D=32 it
            # reads 2.42 even for a perfect posterior and separates a 3x-too-wide one
            # by only 1.6 sd, so it stops regularising the aggregate posterior at all.
            cfg.update(use_vae=True, vae_latent_dim=16, vae_mp_layers=3, vae_graph_aware=True,
                       recon_loss='mse', alpha_recon=1000.0, lambda_mmd=1.0,
                       mmd_bandwidth='median', posterior_min_std=0.05, beta_aux=10.0,
                       z_conditioning='adaln', use_conditional_prior=True, prior_type='gnn_e2e',
                       prior_family='fm', prior_mp_layers=3, prior_hidden_dim=128,
                       prior_nll_weight=1.0, prior_grad_to_encoder=0.0,
                       prior_fm_steps=30, prior_fm_solver='heun', prior_freeze_epoch=350)
        if method == 'chi_mgnflow':
            cfg.pop('std_noise')  # This runtime deliberately rejects the retired noise control.
            cfg.update(latent_ch=4, ae_kl_weight=0.000001, ae_epochs=500,
                       prior_blocks=4, flow_time_freqs=16, flow_t_sampling='uniform',
                       flow_loss_weighting='uniform', flow_steps=30, flow_solver='heun',
                       flow_predict='sample', val_flow_steps=20, val_num_samples=8)
    elif method == 'transolver3':
        crm = c['key'].startswith('ex3_')
        cfg.update(coordinate_normalization='centered_isotropic', latent_dim=256,
                   num_layers=24 if crm else 8, num_heads=8, slice_num=64, mlp_ratio=4,
                   attention_kernel='slice_space', chunk_size=4096, dropout=0.0,
                   temperature_init=0.5, temperature_min=0.1, temperature_max=5.0,
                   amortized_training=c['key'] in ('ex2', 'ex7'),
                   amortized_cache_nodes=32768 if c['key'] in ('ex2', 'ex7') else 0,
                   amortized_query_nodes=16384 if c['key'] in ('ex2', 'ex7') else 0,
                   infer_mode='decoupled', infer_chunk_size=8192, write_test_predictions=True,
                   max_grad_norm=1.0, learningr=0.001 if crm else 0.0002,
                   weight_decay=0.05 if crm else 0.0001,
                   warmup_epochs=25 if crm else 5, batch_size=1 if crm else c['batch'])
    else:
        dim = c['dim']
        resolution = [64, 64] if dim == 2 else [32, 32, 16]
        if c['key'] in ('ex1', 'ex4', 'ex7', 'ex9') and dim == 2:
            resolution = [128, 32]
        cfg.update(coordinate_normalization='centered_isotropic', operator_dim=dim,
                   dimension_tolerance=0.0001, grid_padding=0.1, out_of_bounds_policy='error',
                   sdf_source='none', global_condition_features='none', integration_weight_source='none',
                   write_test_predictions=True, max_grad_norm=1.0, checkpoint_interval=50,
                   learningr=0.0005 if c['steps'] == 1 else 0.0001)
        if method == 'deeponet':
            cfg.update(deeponet_branch_source='fixed_sensors',
                       deeponet_sensor_resolution=[32, 16] if dim == 2 else [16, 16, 8],
                       deeponet_hidden_channels=256, deeponet_branch_depth=4,
                       deeponet_trunk_depth=4, deeponet_basis_dim=128,
                       deeponet_activation='gelu', deeponet_multi_output='split_both',
                       deeponet_max_branch_params=100000000, infer_query_chunk_size=16384)
        elif method == 'point_deeponet':
            cfg.update(point_variant='mesh_state', point_sensor_count=5000, point_sampling='random',
                       point_resample_each_epoch=True, point_hidden_channels=128,
                       point_feature_dim=128, pointnet_depth=3, pointnet_activation='relu',
                       pointnet_norm='batch', point_branch_merge='sum', point_trunk_depth=3,
                       point_refiner_depth=2, point_siren_omega0=30.0, point_output_activation='identity',
                       infer_query_chunk_size=16384)
        elif method == 'fno':
            cfg.update(fno_variant='mesh', fno_grid_resolution=resolution,
                       fno_modes=[16, 12] if dim == 2 else [12, 12, 6],
                       fno_hidden_channels=64, fno_layers=4, fno_use_channel_mlp=True,
                       fno_norm='none', use_amp=False)
    return cfg


def dense_config(c):
    run = f"../../output/dataset_matrix/deterministic/{c['key']}/lsh_vae"
    crm = c['key'].startswith('ex3_')
    cfg = dict(model='simulgenvae', mode='train', gpu_ids=LANE_GPU['lsh_vae'], parallel_mode='single', split_seed=42,
               dataset_dir='../../dataset/' + c['train'], num_var=len(c['output']), field_start_row=3,
               cond_var=6 if crm else len(c['conditioner']), node_start=0, node_end=0, timesteps_reduced=0,
               latent_dim=8, latent_dim_end=32, num_filter_enc=[64, 64, 48, 48, 32, 24, 16],
               network_size='small', loss_type=1, alpha=1000.0, init_beta_divisor=4,
               lc_data_type='hdf5' if crm else 'csv', lc_filter=[128, 128, 128], lc_dropout=0.1,
               vae_training_epochs=2000, vae_batch_size=2 if crm else 8,
               vae_learningr=0.001, vae_weight_decay=0.0, vae_warmup_epochs=20,
               vae_use_amp=False, vae_use_ema=False, vae_num_workers=0,
               lc_training_epochs=2000, lc_batch_size=16, lc_learningr=0.001,
               lc_weight_decay=0.00001, lc_warmup_epochs=20, lc_num_workers=0,
               vae_modelpath=run + '/vae.pth', lc_modelpath=run + '/lc.pth',
               pipeline_log_file=run + '/pipeline.log', vae_log_file_dir=run + '/vae.log',
               lc_log_file_dir=run + '/lc.log', output_dir=run + '/infer',
               skip_completed_stages=True, test_interval=50, display_testset=True, num_test_samples=2)
    if not crm:
        name = Path(c['train']).stem
        cfg['param_dir'] = f'../../dataset/{DERIVED}/{name}_conditions.csv'
    return cfg


def sdf_config():
    run = '../../output/dataset_matrix/geometry_generation/ex1/sdfflow'
    return dict(model='sdfflow', mode='train', gpu_ids=LANE_GPU['sdfflow'], parallel_mode='single', seed=42, split_seed=42,
                dataset_dir='../../dataset/geometry_generation/ex1_deepjeb.h5', split_by_parent=True,
                num_encoder_points=6144, num_query_points=8192, latent_tokens=32, latent_dim=32,
                encoder_query_type='fps', encoder_dim=256, encoder_heads=8, encoder_blocks=4,
                encoder_self_attention=True, decoder_type='attention', decoder_hidden=256,
                decoder_layers=4, decoder_heads=8, fourier_bands=8, kl_weight=0.0001,
                clamp_dist=0.1, deterministic_warmup_epochs=50, posterior_noise_warmup_epochs=100,
                posterior_noise_max_scale=1.0, kl_warmup_epochs=200, posterior_min_std_rel=0.05,
                surface_weight=1.0, normal_weight=0.1, eikonal_weight=0.1, hybrid_grad_points=1024,
                use_conditions=True, condition_names=['volume', 'area'], min_condition_std=0.00001,
                condition_clip=5.0, cond_dropout=0.2, cond_dropout_mode='all',
                fm_arch='dit', fm_hidden=256, fm_blocks=8, fm_heads=8, fm_cond_hidden=128,
                fm_time_sampling='uniform', encode_batch_size=8, ode_steps=50,
                vae_training_epochs=1500, vae_batch_size=8, vae_learningr=0.0001,
                vae_weight_decay=0.0001, vae_warmup_epochs=20, vae_num_workers=2,
                vae_use_amp=False, vae_use_ema=True, vae_ema_decay=0.99,
                vae_val_interval=5, vae_test_interval=100, vae_num_test_shapes=2, vae_mc_resolution_test=64,
                fm_training_epochs=500, fm_batch_size=64, fm_learningr=0.0001,
                fm_weight_decay=0.0001, fm_warmup_epochs=10, fm_num_workers=0,
                fm_use_amp=True, fm_use_ema=True, fm_ema_decay=0.99,
                fm_val_interval=5, fm_test_interval=50, fm_num_test_shapes=2, fm_mc_resolution_test=64,
                display_testset=True, skip_completed_stages=True, output_dir=run + '/samples',
                vae_modelpath=run + '/vae.pth', fm_modelpath=run + '/fm.pth',
                fm_best_modelpath=run + '/fm_best.pth', pipeline_log_file=run + '/pipeline.log',
                vae_log_file_dir=run + '/vae.log', fm_log_file_dir=run + '/fm.log')


def render(c, method, config):
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
    def value(v):
        if isinstance(v, (list, tuple)):
            return ', '.join(value(x) for x in v)
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, float):
            return format(v, '.16f').rstrip('0').rstrip('.') if v != 0 else '0.0'
        return str(v)
    return '\n'.join('% ' + s for s in comments) + '\n\n' + ''.join(f'{k:<32} {value(v)}\n' for k, v in config.items())


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
            base = f"configs/{REPOSITORIES[method]}/{c['category']}/{c['key']}/baseline"
            paths = [f'{base}/config_{mode}_{method}.txt' for mode in ('train', 'infer')]
            for path, cfg in zip(paths, (train, infer)):
                files[path] = render(c, method, cfg)
            pairs.append(dict(category=c['category'], example=c['key'], method=method, train=paths[0], infer=paths[1]))
    sdf = sdf_config()
    c = dict(category='geometry_generation', key='ex1', input=['surface_points', 'surface_normals', 'query_xyz'],
             output=['signed_distance'], conditioner=['volume', 'area'], partNo=[])
    base = 'configs/SDFFlow/geometry_generation/ex1/baseline'
    infer = dict(model='sdfflow', mode='sample', gpu_ids=LANE_GPU['sdfflow'], parallel_mode='single', seed=42,
                 vae_modelpath=sdf['vae_modelpath'], fm_modelpath=sdf['fm_best_modelpath'],
                 output_dir=sdf['output_dir'], num_samples=16, mc_resolution=128, ode_steps=50,
                 cfg_scale=1.0, condition_ood_policy='error', max_condition_z=3.0)
    paths = [f'{base}/config_{mode}_sdfflow.txt' for mode in ('train', 'infer')]
    for path, cfg in zip(paths, (sdf, infer)):
        files[path] = render(c, 'sdfflow', cfg)
    pairs.append(dict(category=c['category'], example=c['key'], method='sdfflow', train=paths[0], infer=paths[1]))
    assert len(pairs) == 75 and len(files) == 150
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
        print('75 train/infer pairs: generated files match source of truth.')


if __name__ == '__main__':
    main()
