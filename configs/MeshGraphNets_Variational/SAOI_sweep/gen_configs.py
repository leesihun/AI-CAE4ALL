"""Generate the MeshGraphNets-V SAOI sweep: 8 arms, one per card 0-7.

    python configs/MeshGraphNets_Variational/SAOI_sweep/gen_configs.py

THE DESIGN: a 2^3 FULL factorial over the PRIOR -- section x prior fit x trunk.

    Eight arms is exactly the full grid of three two-level factors, so nothing
    is confounded with anything.

      section      bot | top          the board's two halves, independent models
      prior fit    joint | tail       prior_freeze_epoch 0  |  700 of 1000
      prior trunk  small | large      hidden 256 / 5 MP    |  512 / 8 MP

WHY THE PRIOR, AND NOT THE DECODER
    Posterior reconstruction is excellent while the generated distribution is
    ~2x too narrow (sd_ratio 0.45-0.53 on every arm of the previous grid).
    Training decodes z ~ q(z | y, g); deployment decodes z ~ p(z | g). The
    decoder is common to both paths, so if the posterior path reproduces a
    part's spread and the prior path does not, the prior's conditional for
    that part is what is wrong. Three mechanisms can make it so, and the two
    factors here address all three:

      1. a MOVING target -- the encoder keeps receiving reconstruction
         gradients, so q drifts for the whole run and the prior chases it.
         The `tail` level freezes the simulator at epoch 700 and fits the prior
         alone for the last 300, against a posterior that has stopped moving,
         on a fresh cosine over the prior's own parameters;
      2. an ILL-SCALED target -- MMD pins the AGGREGATE q(z) to N(0,I), and
         with ~100 realizations of each of a few parts that aggregate is a
         mixture: unit total variance is compatible with every per-part cloud
         being far smaller and off-origin. The `tail` level also fits latent
         standardization (z_shift / z_scale) from the frozen posterior, so the
         velocity net regresses a unit-scale target;
      3. an UNDERSIZED trunk -- prior_hidden_dim 256 / prior_mp_layers 5 is
         much smaller than the main network. If the trunk cannot separate
         parts, the prior learns a conditional smeared across neighbours.

    prior_grad_to_encoder is closed on every arm: open, the FM objective can
    lower itself by shrinking the target distribution (integrated CFM loss is
    pi*a/2 for target scale a), and nothing here should be moving the target.

WHAT IS HELD FIXED
    beta_aux 10 on every arm -- the peak-to-valley term on the decoded field at
    the balance point estimated from realistic recon/aux magnitudes. It is the
    I(z;y) floor that keeps z informative (the old z-readout head is gone), so
    it stays ON, and it stays CONSTANT so it is not a factor. Its own
    dose-response is a different sweep.

    The previous 2^(4-1) grid moved W1/sd by 0.001-0.054 across
    z_conditioning, prior_grad_to_encoder, capacity and lambda_mmd, against a
    0.10 arm-to-arm range. None of those is spent again.

BUDGET
    1000 epochs at a measured ~346 s/epoch. The `tail` arms are CHEAPER, not
    dearer: their last 300 epochs run the posterior encoder under no_grad and
    the small prior only -- no decoder forward, no decoder backward.
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / 'campaigns'))

from sweep_common import write_sweep      # noqa: E402

OUT_ROOT = '../../output/meshgraphnets-v/saoi_sweep'
EPOCHS = '1000'
FREEZE_AT = '700'

SMALL = {'prior_hidden_dim': ('256', 'prior trunk: small (production)'),
         'prior_mp_layers': ('5', 'prior trunk: small (production)')}
LARGE = {'prior_hidden_dim': ('512', 'prior trunk: LARGE -- can it separate parts?'),
         'prior_mp_layers': ('8', 'prior trunk: LARGE')}
JOINT = {'prior_freeze_epoch': ('0', 'prior fit: joint to the end (the baseline recipe)')}
TAIL = {'prior_freeze_epoch': (FREEZE_AT,
                               f'prior fit: TAIL -- simulator frozen at {FREEZE_AT}, '
                               f'latent standardization fitted, prior alone on a fresh '
                               f'cosine for the remaining {1000 - int(FREEZE_AT)} epochs')}

ARMS = []
_n = 0
for half in ('bot', 'top'):
    for fit, fit_tag in ((JOINT, 'joint'), (TAIL, 'tail')):
        for trunk, trunk_tag in ((SMALL, 'small'), (LARGE, 'large')):
            _n += 1
            over = dict(fit)
            over.update(trunk)
            ARMS.append((str(_n), half, over, f'{half} {fit_tag} {trunk_tag}'))

BANNER = """%   ============================================================
%   SAOI warpage, MeshGraphNets-V -- arm {arm}   ({note})
%
%   section   {half}
{axis}
%   gpu {gpu} -- one arm per card, nothing shared
%
%   A 2^3 FULL factorial over the PRIOR: section x prior fit (joint | frozen
%   tail with latent standardization) x prior trunk (small | large). The
%   decoder is common to the posterior and prior paths; the posterior path
%   reproduces a part's spread and the prior path does not, so the prior is
%   what this sweep varies. beta_aux is held at 10 on every arm.
%
%   GENERATED by gen_configs.py from ../SAOI_all_input/config_train_{half}.txt.
%   Do not hand-edit -- regenerate.
%   ============================================================
"""

INFER_BANNER = """%   ============================================================
%   SAOI inference -- arm {arm} ({note}), eval set {tag}.
%
%   `eval_dataset` is the `_compare_` file: the 125 realizations of the ONE
%   part in `infer_dataset`. That pairing is what makes sd_ratio's target 1 --
%   the truth here IS the conditional distribution, not a single outcome.
%
%   Architecture keys mirror the train config. rollout.py overrides them from
%   checkpoint['model_config'] at load time, so the checkpoint always wins;
%   they are kept in sync so this file documents what runs. The latent
%   standardization a `tail` arm fitted rides inside the checkpoint as prior
%   buffers and needs nothing here.
%
%   gpu {gpu}. GENERATED by gen_configs.py -- do not hand-edit.
%   ============================================================
"""

SPEC = dict(
    here=HERE,
    prod_dir=HERE.parent / 'SAOI_all_input',
    out_root=OUT_ROOT,
    slug='meshgraphnets-v',
    title='MeshGraphNets-V SAOI sweep -- 2^3 over the prior (section x fit x trunk), one card each:',
    banner=BANNER,
    infer_banner=INFER_BANNER,
    arms=ARMS,
    gpu_of=lambda arm: int(arm) - 1,          # arms 1..8 -> cards 0..7
    infer_sources={
        's26fe_main': 'config_infer_s26fe_main_{half}.txt',
        's26fe_sec':  'config_infer_s26fe_sec_{half}.txt',
        'sm_l345u':   'config_infer_sm_l345u_main_{half}.txt',
    },
    # Keys whose value must be identical in the train and inference configs,
    # or the inference file documents a model the checkpoint is not.
    arch_keys={'prior_hidden_dim', 'prior_mp_layers'},
    train_fixed={
        'Training_epochs': (EPOCHS,
                            'no resume exists and cosine_T0 = epochs - warmup, so '
                            'this is a COMPLETE run at its own schedule; '
                            '~346 s/epoch measured on one card'),
        'Batch_size': ('16', 'the value the previous grid measured as its winner'),
        'val_interval': ('30', 'CRPS is the selection metric'),
        'test_interval': ('200', 'periodic plots; 5 over the run'),
        'best_by': ('crps', 'select on the GENERATIVE metric, not posterior recon'),
        'hierarchy_cache_keep': ('True',
                                 'REQUIRED: every arm shares one cache; a finishing '
                                 'arm must not delete it'),
        'recon_loss': ('mse',
                       'MSE penalizes extreme-node error far more than Huber, which '
                       'caps it -- and peak-to-valley IS an extreme-value statistic'),
        'alpha_recon': ('1000', 'reconstruction multiplier'),
        'beta_aux': ('10',
                     'HELD CONSTANT: peak-to-valley MSE on the decoded field at the '
                     'balance point estimated from recon/aux magnitudes. It is the '
                     'I(z;y) floor that keeps z informative now the z-readout head is '
                     'gone, so it stays on; it stays fixed so it is not a factor'),
        'num_vae_samples': ('10000', 'draws per scene; unused in train mode'),
        'prior_grad_to_encoder': (
            '0.0',
            'CLOSED on every arm. Open, the FM objective reaches the encoder and '
            'can lower itself by SHRINKING the target distribution (integrated CFM '
            'loss is pi*a/2 for target scale a). This sweep is about fitting the '
            'prior to the posterior, so nothing may be moving the posterior toward '
            'the prior. The previous grid measured the flag at delta 0.001'),
    },
    infer_fixed={
        'num_vae_samples': ('2000',
                            'draws per scene. THE DOMINANT COST of this stage: '
                            'arms x eval sets x draws forwards'),
        'save_rollouts': ('False',
                          'one HDF5 per (scene, draw) would be tens of thousands of '
                          'files; the spread values still reach spread_values.npz'),
        'make_histogram': ('True', 'GT vs generated z_disp peak-to-valley'),
        'show_histogram': ('False', 'headless box: save the PNG, do not open a viewer'),
        'hierarchy_seed': ('0, 1, 2, 3',
                           'build one partition per seed and rotate across draw '
                           'batches, so one unlucky partition cannot poison a scene'),
    },
    # The deterministic control, written beside every stochastic inference
    # config as config_infer_<arm>_<tag>_det.txt and dumped under infer/det/.
    det_fixed={
        'num_vae_samples': ('1', 'DETERMINISTIC CONTROL: one draw per scene'),
        'prior_temperature': ('1e-9',
                              '~0 kills the prior sampling noise: z is the ODE image '
                              'of the origin, the conditional centre. Its spread '
                              'against the 125 truths is pure BIAS -- what no width '
                              'fix can repair'),
    },
)

if __name__ == '__main__':
    write_sweep(SPEC)
