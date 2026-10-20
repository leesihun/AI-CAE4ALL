"""The port's changes to upstream dgn4cfd code: weights_only checkpoints that
round-trip, an LDGN rebuilt from its own checkpoint alone (the VGAE inside
it), the `learnable_variance=False` constructor fix, and a validation pass
that leaves the importance step sampler untouched. (`sample_n` returning
exactly n -- 5 with batch 2 -- is checked by test_end_to_end.)"""
import torch

from dgn4cfd.loader import Collater
from dgn4cfd.nn import VGAE, LatentDiffusionGraphNet
from dgn4cfd.nn.diffusion import DiffusionProcess
from dgn4cfd.nn.model import decode_arch
from general_modules import mesh_dataset as md
from general_modules.load_config import load_config
from training_profiles.train import ldgn_arch, num_scales_for, vae_arch


def _batch(case, num_scales):
    config = load_config(case.write_config())
    train_ids, _, _ = md.split_ids(config)
    train = md.group_ids(config['dataset_dir'], train_ids, md.group_attr(config))
    stats = md.compute_normalization(config, train, log=lambda *_: None)
    statics = md.build_statics(config, train, stats, num_scales, log=lambda *_: None)
    ds = md.SystemDrawDataset(config['dataset_dir'], train, statics, stats, config, mode='fixed',
                              draws_per_system=1)
    return config, stats, Collater().collate([ds[0], ds[1]]), ds


def _vgae(config, stats):
    torch.manual_seed(0)
    return VGAE(arch=vae_arch(config, stats))


def test_vgae_checkpoint_roundtrip(case, tmp_path):
    config, stats, batch, _ = _batch(case, 2)
    vae = _vgae(config, stats)
    path = tmp_path / 'vae.pth'
    torch.save(vae.checkpoint_dict(stage='vae'), path)
    ckpt = torch.load(path, weights_only=True)
    back = VGAE(checkpoint=ckpt)
    assert decode_arch(ckpt['arch'])['activation'] is torch.nn.SELU
    for (k, a), (_, b) in zip(vae.state_dict().items(), back.state_dict().items()):
        assert torch.equal(a, b), k
    vae.eval(), back.eval()
    batch.field = batch.target
    outs = []
    for m in (vae, back):
        torch.manual_seed(3)          # the encoder samples its latent
        outs.append(m(batch.clone())[0])
    torch.testing.assert_close(outs[0], outs[1])


def _ldgn(config, vae, learnable_variance=True):
    vae_ckpt = vae.checkpoint_dict(stage='vae')
    arch = ldgn_arch(config, decode_arch(vae_ckpt['arch']))
    torch.manual_seed(1)
    return LatentDiffusionGraphNet(autoencoder_checkpoint=vae_ckpt, arch=arch,
                                   diffusion_process=DiffusionProcess(50, 'linear'),
                                   learnable_variance=learnable_variance)


def test_ldgn_rebuilt_from_its_checkpoint_alone(case, tmp_path):
    config, stats, _, _ = _batch(case, 2)
    vae = _vgae(config, stats)
    model = _ldgn(config, vae)
    L = num_scales_for(vae.arch['depths'], model.arch['depths'])
    _, _, batch, _ = _batch(case, L)
    path = tmp_path / 'model.pth'
    torch.save(model.checkpoint_dict(stage='ldgn', autoencoder_arch=vae.checkpoint_dict()['arch']), path)
    ckpt = torch.load(path, weights_only=True)
    back = LatentDiffusionGraphNet(autoencoder_checkpoint={'arch': ckpt['autoencoder_arch']},
                                   checkpoint=ckpt)
    assert back.diffusion_process.num_steps == 50 and back.learnable_variance
    outs = []
    for m in (model, back):
        torch.manual_seed(7)
        outs.append(m.sample(batch.clone(), steps=[0, 10, 25, 49]))
    torch.testing.assert_close(outs[0], outs[1])


def test_fixed_variance_constructs():
    """Upstream read `self.out_node_features` before defining it when
    learnable_variance=False."""
    vae = VGAE(arch={'in_node_features': 3, 'cond_node_features': 1, 'cond_edge_features': 3,
                     'latent_node_features': 2, 'depths': [1, 1], 'fnns_width': 8, 'dim': 3})
    model = _ldgn({'ldgn_fnns_width': 8, 'ldgn_depths': [1, 1]}, vae, learnable_variance=False)
    assert model.num_fields == 2 and model.node_decoder.out_features == 2


def test_validation_pass_leaves_step_sampler_alone(case):
    from training_profiles.train import _settings, _validation_losses
    config, stats, _, _ = _batch(case, 2)
    model = _ldgn(config, _vgae(config, stats))
    L = num_scales_for(model.autoencoder.arch['depths'], model.arch['depths'])
    config, stats, _, ds = _batch(case, L)
    model.prepare_fit(_settings(config, 'ldgn', None))
    before = {k: v.clone() for k, v in model.fit_state().items()}
    losses = _validation_losses(model, ds, 2, torch.device('cpu'))
    assert 'train_loss' in losses
    after = model.fit_state()
    for k in before:
        assert torch.equal(before[k], after[k]), k
