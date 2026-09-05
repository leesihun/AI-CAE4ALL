# Conditional generation for SDFFlow: FEA-label conditions, partial requests, sample-time accuracy (2026-09)

**Status: design + label analysis + implementation, not yet measured on a trained
model.** No `ex5` checkpoint exists (`output/geometry_generation/` is absent in
this checkout), and the user asked for no training. Everything numeric in this
note comes from one of three sources, each labelled: (a) the DeepJEB label CSV
and `dataset/deepjeb.h5` (exact, re-runnable), (b) the ex1 guidance pilot of
`GUIDANCE_MECHANISMS_SOTA_AND_PLAN_2026-08.md` section 2 (measured on an old
checkpoint, method-selection evidence only), (c) the literature (cited by URL in
section 9). Section 6 lists what stays unverified until `config_train_v3_fea.txt`
has trained. This note is design context; the code and the configs win on
current behaviour (`methods/SDFFlow/CLAUDE.md` is the contract).

Files this design touches: `methods/SDFFlow/add_fea_conditions.py`,
`general_modules/{condition_names, sdf_dataset, descriptor_proxy,
descriptor_calibration, descriptor_refinement, descriptor_guidance}.py`,
`model/velocity_net.py`, `inference_profiles/{sample, interpolate, evaluate}.py`,
`design_loop/problem.py`, `cae_suite/specs/sdfflow.py`, `studio/src/constants.js`,
and `configs/SDFFlow/{config_train_v3_fea, config_calibrate_descriptors,
config_sample_conditional, config_cond_sweep, config_evaluate_conditional}.txt`.

---

## 1. Why conditional generation replaces SLERP for this user

The v3 model (`config_train_v3.txt`, ex4) is conditioned on `volume,area`.
`GEOMETRY_UPGRADE_MESHING_SEMANTIC_2026-08.md` section 1.3 measured that this
condition vector has about **1.5 effective degrees of freedom** (`bbox_y` std 0,
`bbox_x` CV 0.45%, volume-area r 0.63; z-scored singular values
`[1, .706, .553, .422]`), so the generator is "an unconditional sampler with a
volume knob". The only controllable morph it offered was `interpolate`: a slerp
in noise space between two *samples the model happened to produce*, which a
designer cannot steer -- there is no way to say "the same bracket, but lighter
and stiffer under the vertical load".

What a bracket designer specifies is a small set of performance numbers: a mass
budget, an allowable peak stress under the certification load cases, a minimum
first mode. DeepJEB carries exactly those labels for all 2138 shapes
(`bracket_labels.csv`: mass, per-load-case max von Mises stress and max
displacement, first two eigenfrequencies; section 1.6 of the GEOMETRY_UPGRADE
note flagged them as the unused Level-2 conditions). The design therefore
replaces "interpolate between two accidents" with three things:

1. **Condition the FM on FEA labels** (section 3) so a request is expressed in
   engineering quantities, with `cond_dropout_mode per_dim` so a request may
   specify *some* of them and leave the rest `nan`.
2. **`interpolation_space cond_sweep`** -- the controllable morph: one fixed
   source noise integrated under a straight-line sweep of conditions from
   `cond_values_a` to `cond_values_b`. The noise fixes "which bracket", the
   conditions move "how heavy / how stiff". `slerp_noise` and `lerp_latent`
   remain for comparison and for unconditional checkpoints.
3. **Sample-time accuracy tools** (section 4) so the geometric part of the
   request (volume, area) is *hit* rather than approximately followed, and an
   audit that reports how far the decoded mesh is from every requested number.

DeepJEB's own generation was unconditional (a DeepSDF auto-decoder on 263
curated SimJEB seeds, filtered by mesh quality and an IQR outlier rule; the FEA
labels are post hoc), so there is no published conditional baseline on this
data to compare against.

---

## 2. Label analysis

Scripts and raw outputs: `scratchpad/cond_analysis/{01_join.py, 02_analysis.py,
03_followup.py, 04_final_set.py, joined.csv, tables*.md, results*.json}` (session
scratch; the numbers below are the ones that decided the design). Join:
`basename(source) == item_name`, 2138/2138 shapes matched, 0 unmatched on either
side, 0 NaN in the 46 columns, 263 parents (1-21 variants, median 8). Pandas
reads purely numeric item names as `int` -- the builder compares as `str`.

### 2.1 Dispersion, skew, transform

| name (registry) | csv column | CV raw | skew raw | skew log | min | median | max | dup rows | transform |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `mass_kg` | `mass(kg)` | 0.306 | +0.73 | +0.06 | 0.537 | 1.183 | 2.407 | 2.25% | identity (registry); log would remove the skew |
| `log_max_ver_stress_mpa` | `max_ver_stress(MPa)` | 0.263 | +0.80 | +0.21 | 608 | 1041 | 2278 | 0.09% | log |
| `log_max_hor_stress_mpa` | `max_hor_stress(MPa)` | 0.278 | +1.15 | +0.59 | 389 | 567 | 1426 | 0.09% | log |
| `log_max_dia_stress_mpa` | `max_dia_stress(MPa)` | 0.225 | +0.74 | +0.17 | 287 | 529 | 1089 | 0.19% | log |
| `log_max_tor_stress_mpa` | `max_tor_stress(MPa)` | 0.240 | +1.06 | +0.59 | 236 | 323 | 744 | 0.28% | log |
| `log_max_ver_magdisp_mm` | `abs_max_ver_magdisp(mm)` | 0.386 | +0.81 | -0.01 | 0.179 | 0.415 | 1.154 | 0.09% | log |
| `log_max_hor_magdisp_mm` | `abs_max_hor_magdisp(mm)` | 0.490 | +1.26 | +0.32 | 0.137 | 0.303 | 1.094 | 0.09% | log |
| `log_max_dia_magdisp_mm` | `abs_max_dia_magdisp(mm)` | 0.319 | +0.77 | +0.01 | 0.082 | 0.177 | 0.422 | 0.09% | log |
| `log_max_tor_magdisp_mm` | `abs_max_tor_magdisp(mm)` | 0.249 | +1.26 | +0.77 | 0.079 | 0.124 | 0.261 | 0.09% | log |
| `log_first_mode_freq_hz` | `1st_mode_freq(Hz)` | 0.435 | +0.55 | -0.22 | 752 | 3006 | 6936 | 0.65% | log |
| `log_second_mode_freq_hz` | `2nd_mode_freq(Hz)` | 0.359 | +0.38 | -0.38 | 954 | 3904 | 7603 | 1.68% | log (tie; consistency with 1st mode) |
| `surface_area_mm2` | `surface_area(mm2)` | 0.156 | +0.63 | +0.17 | 29040 | 45730 | 76070 | 1.54% | identity (registry); duplicate of `area` |
| `volume_mm3` | `volume(mm3)` | 0.306 | +0.73 | +0.06 | 120200 | 264600 | 538600 | 1.78% | identity (registry); duplicate of `mass_kg` |
| (not in registry) `CG_x/y/z(mm)` | | 0.284 / 0.014 / 0.129 | +0.12 / -0.29 / -0.05 | | | | | | identity; CG_y constant |
| (not in registry) `I_1/I_2/I_3` | | 0.295 / 0.286 / 0.351 | +0.57 / +0.47 / +0.97 | -0.10 / -0.17 / +0.20 | | | | | log; mass x length^2 |
| (not in registry) `min_ver/tor_stress` | | 0.247 / 0.246 | -1.10 / -1.26 | n/a (negative) | | | | | compressive principal peak; r 0.90 / 0.73 with the max |

- `mass(kg)/volume(mm3)` = 4470 kg/m3, CV 0.0002% (constant Ti-6Al-4V density).
  Because the physical extent is constant (183.8 mm, CV 0.79%),
  `mass_kg = 4.793 x stored normalized volume` (CV 2.3%) and
  `surface_area_mm2 = 10468 x stored area` (CV 1.6%): the three absolute-unit
  names carry no information beyond the geometric `volume`/`area` already in v3
  (GroupKFold GBM R^2 from the five geometric descriptors: mass 0.998,
  surface_area 0.997).
- corr(log stress, log volume): ver -0.755, hor -0.734, dia -0.549, tor -0.174;
  corr(log disp, log volume): ver -0.846, hor -0.776, dia -0.761, tor -0.735.
- The log transform brings every stress/displacement skew from +0.7..+1.3 to
  <= 0.6 (ver/dia displacement and mass to ~0.0). `mass_kg` stays identity in
  the registry (skew +0.73) because it is not recommended as an FM condition; if
  it is ever used as one, `max_condition_z` will be asymmetric in identity space.

### 2.2 Correlation matrix (Pearson r, stored space; `bbox_y` constant, omitted)

| | bbox_x | bbox_z | volume | area | mass | ver_S | hor_S | dia_S | tor_S | ver_D | hor_D | dia_D | tor_D | f1 | f2 | area_mm2 | vol_mm3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **bbox_x** | +1.00 | +0.29 | -0.33 | -0.41 | -0.37 | +0.23 | +0.23 | +0.14 | -0.01 | +0.21 | +0.19 | +0.20 | +0.20 | -0.26 | -0.25 | -0.46 | -0.37 |
| **bbox_z** | +0.29 | +1.00 | -0.10 | -0.14 | -0.15 | +0.08 | +0.09 | +0.04 | -0.02 | +0.07 | +0.07 | +0.07 | +0.09 | -0.06 | -0.06 | -0.14 | -0.15 |
| **volume** | -0.33 | -0.10 | +1.00 | +0.63 | +1.00 | -0.76 | -0.72 | -0.57 | -0.17 | -0.85 | -0.77 | -0.78 | -0.70 | +0.82 | +0.83 | +0.65 | +1.00 |
| **area** | -0.41 | -0.14 | +0.63 | +1.00 | +0.64 | -0.58 | -0.53 | -0.38 | -0.08 | -0.58 | -0.63 | -0.32 | -0.61 | +0.56 | +0.51 | +1.00 | +0.64 |
| **mass** | -0.37 | -0.15 | +1.00 | +0.64 | +1.00 | -0.75 | -0.72 | -0.56 | -0.17 | -0.84 | -0.76 | -0.78 | -0.70 | +0.81 | +0.82 | +0.66 | +1.00 |
| **ver_S** | +0.23 | +0.08 | -0.76 | -0.58 | -0.75 | +1.00 | +0.84 | +0.82 | +0.25 | +0.91 | +0.93 | +0.60 | +0.76 | -0.55 | -0.58 | -0.58 | -0.75 |
| **hor_S** | +0.23 | +0.09 | -0.72 | -0.53 | -0.72 | +0.84 | +1.00 | +0.68 | +0.24 | +0.78 | +0.82 | +0.51 | +0.65 | -0.53 | -0.58 | -0.54 | -0.72 |
| **dia_S** | +0.14 | +0.04 | -0.57 | -0.38 | -0.56 | +0.82 | +0.68 | +1.00 | +0.33 | +0.75 | +0.75 | +0.54 | +0.63 | -0.41 | -0.44 | -0.38 | -0.56 |
| **tor_S** | -0.01 | -0.02 | -0.17 | -0.08 | -0.17 | +0.25 | +0.24 | +0.33 | +1.00 | +0.24 | +0.29 | +0.12 | +0.28 | -0.07 | -0.07 | -0.08 | -0.17 |
| **ver_D** | +0.21 | +0.07 | -0.85 | -0.58 | -0.84 | +0.91 | +0.78 | +0.75 | +0.24 | +1.00 | +0.94 | +0.80 | +0.82 | -0.68 | -0.72 | -0.58 | -0.84 |
| **hor_D** | +0.19 | +0.07 | -0.77 | -0.63 | -0.76 | +0.93 | +0.82 | +0.75 | +0.29 | +0.94 | +1.00 | +0.57 | +0.86 | -0.54 | -0.57 | -0.63 | -0.76 |
| **dia_D** | +0.20 | +0.07 | -0.78 | -0.32 | -0.78 | +0.60 | +0.51 | +0.54 | +0.12 | +0.80 | +0.57 | +1.00 | +0.56 | -0.74 | -0.80 | -0.33 | -0.78 |
| **tor_D** | +0.20 | +0.09 | -0.70 | -0.61 | -0.70 | +0.76 | +0.65 | +0.63 | +0.28 | +0.82 | +0.86 | +0.56 | +1.00 | -0.54 | -0.56 | -0.61 | -0.70 |
| **f1** | -0.26 | -0.06 | +0.82 | +0.56 | +0.81 | -0.55 | -0.53 | -0.41 | -0.07 | -0.68 | -0.54 | -0.74 | -0.54 | +1.00 | +0.95 | +0.56 | +0.81 |
| **f2** | -0.25 | -0.06 | +0.83 | +0.51 | +0.82 | -0.58 | -0.58 | -0.44 | -0.07 | -0.72 | -0.57 | -0.80 | -0.56 | +0.95 | +1.00 | +0.51 | +0.82 |
| **area_mm2** | -0.46 | -0.14 | +0.65 | +1.00 | +0.66 | -0.58 | -0.54 | -0.38 | -0.08 | -0.58 | -0.63 | -0.33 | -0.61 | +0.56 | +0.51 | +1.00 | +0.66 |
| **vol_mm3** | -0.37 | -0.15 | +1.00 | +0.64 | +1.00 | -0.75 | -0.72 | -0.56 | -0.17 | -0.84 | -0.76 | -0.78 | -0.70 | +0.81 | +0.82 | +0.66 | +1.00 |

(S = log max stress, D = log max magnitude displacement, f1/f2 = log mode
frequencies.)

### 2.3 Effective dimensionality (PCA on z-scored columns)

| block | k cols | participation ratio | #PC 90% | #PC 95% | #PC 99% | top eigen-fractions (%) |
|---|---:|---:|---:|---:|---:|---|
| geometric, 5 stored (bbox_y dropped) | 4 | 2.86 | 3 | 4 | 4 | 50.4, 25.1, 15.4, 9.0 |
| geometric, v3 `volume,area` | 2 | **1.43** | 2 | 2 | 2 | 81.6, 18.4 |
| FEA registry, 13 | 13 | 2.15 | 5 | 6 | 9 | 66.7, 11.1, 6.6, 5.4, 2.9, 2.3 |
| FEA minus volume_mm3 / surface_area_mm2 | 11 | 2.10 | 4 | 6 | 8 | 67.2, 12.6, 7.3, 3.8, 3.4, 2.2 |
| geometric + FEA, 17 non-constant | 17 | 2.62 | 6 | 8 | 11 | 59.8, 9.6, 8.2, 5.8, 4.7, 3.7 |

The PR 2.86 of the 4-column geometric block is an artefact of z-scoring the
near-constant `bbox_x`/`bbox_z`; the honest v3 number is 1.43. The first PC of
the FEA block (67%) is the size/compliance axis aligned with volume.

| condition set | k | participation ratio | #PC 90% | #PC 95% | eigen-fractions (%) | test/train NN-dist ratio (z-space, v3 split) |
|---|---:|---:|---:|---:|---|---:|
| v3 (volume, area) | 2 | 1.43 | 2 | 2 | 81.6, 18.4 | 1.10 |
| **REC**: volume, area + ver_S, dia_S, tor_S, f1 | 6 | **2.53** | 4 | 4 | 58.6, 18.1, 10.5, 8.1, 3.0, 1.7 | 1.10 |
| ALT-A: volume, area + ver_D, dia_S, tor_S, f1 | 6 | 2.46 | 4 | 4 | 59.8, 17.9, 9.7, 7.9, 3.2, 1.5 | 1.02 |
| ALT-B (greedy): volume, area + hor_D, tor_S, f1, dia_S | 6 | 2.54 | 4 | 5 | 58.6, 18.1, 9.7, 8.3, 3.6, 1.7 | 1.02 |
| brief example: volume, area + mass, ver_S, tor_S, ver_D, f1 | 7 | 2.04 | 4 | 4 | 67.7, 14.3, 7.3, 7.0, 2.7, 1.0, 0.0 | 1.03 |
| REC FEA-only (no geometric) | 4 | 2.41 | 3 | 3 | 58.1, 23.8, 14.1, 4.0 | 1.05 |

### 2.4 Predictability of each label from geometry (5-fold CV R^2)

`geo` = the five stored descriptors; `geo+mass` adds `mass_kg`; KF = shuffled
KFold (siblings leak), GKF = parent-grouped GroupKFold. Random and grouped folds
agree to +-0.01, so the descriptor -> label map generalises across parents and
the residual is genuinely shape-specific. Residual = `1 - R^2(GBM, geo+mass,
GKF)` = the shape-specific signal a conditional FM has to carry (an upper bound
on the learnable signal -- it includes label noise from max-of-field quantities).

| label | ridge geo KF | GBM geo KF | ridge geo GKF | GBM geo GKF | GBM geo+mass GKF | **residual var** |
|---|---:|---:|---:|---:|---:|---:|
| `log_max_ver_stress_mpa` | 0.565 | 0.597 | 0.565 | 0.591 | 0.592 | **0.41** |
| `log_max_hor_stress_mpa` | 0.518 | 0.534 | 0.514 | 0.519 | 0.519 | **0.48** |
| `log_max_dia_stress_mpa` | 0.255 | 0.296 | 0.252 | 0.292 | 0.293 | **0.71** |
| `log_max_tor_stress_mpa` | 0.032 | -0.032 | 0.024 | -0.037 | -0.042 | **1.00** |
| `log_max_ver_magdisp_mm` | 0.640 | 0.761 | 0.654 | 0.753 | 0.755 | **0.25** |
| `log_max_hor_magdisp_mm` | 0.625 | 0.653 | 0.622 | 0.637 | 0.642 | **0.36** |
| `log_max_dia_magdisp_mm` | 0.622 | 0.699 | 0.628 | 0.690 | 0.695 | **0.31** |
| `log_max_tor_magdisp_mm` | 0.546 | 0.601 | 0.541 | 0.583 | 0.583 | **0.42** |
| `log_first_mode_freq_hz` | 0.423 | 0.763 | 0.446 | 0.761 | 0.760 | **0.24** |
| `log_second_mode_freq_hz` | 0.341 | 0.807 | 0.351 | 0.804 | 0.803 | **0.20** |
| `mass_kg` | 0.983 | 0.998 | 0.983 | 0.998 | - | **0.00** |
| `surface_area_mm2` | 0.922 | 0.997 | 0.919 | 0.997 | 0.997 | **0.00** |

Ridge is 0.3-0.4 below GBM for the mode frequencies: the volume -> frequency
relation is nonlinear in stored space. Adding log physical extent changes
nothing (+< 0.01), confirming the absolute labels are coherent with the
normalized geometry.

### 2.5 Redundancy between labels and the candidate sets

Stress <-> displacement within a load case (parent-grouped): vertical r +0.911
(GBM R^2 0.82 either way), horizontal 0.824 (0.66), diagonal 0.545 (0.30),
torsion 0.284 (0.03-0.09). From the two vertical labels alone: hor stress 0.70,
dia stress 0.65, **tor stress 0.00**, hor disp 0.92, dia disp 0.77, tor disp
0.70, f1 0.47 (0.80 with geometry), f2 0.56 (0.83).

Greedy forward selection (mean ridge GKF R^2 with which the selected names +
`volume,area` predict all 11 non-duplicate labels): `volume,area` alone 0.578
(worst: tor stress 0.03) -> +`hor_magdisp` 0.723 -> +`tor_stress` 0.807 ->
+`second_mode` 0.863 -> +`dia_stress` 0.907 -> +`hor_stress` 0.937 ->
+`dia_magdisp` 0.962.

| set | FEA names | span mean | span min (label) | mean unique | least unique member | max pair abs r | max abs r vs volume |
|---|---|---:|---|---:|---|---:|---:|
| **REC** | ver_S, tor_S, dia_S, f1 | 0.889 | 0.644 (`tor_magdisp`) | 0.396 | `ver_S` (0.153) | 0.82 | 0.82 |
| ALT-A | ver_D, tor_S, dia_S, f1 | 0.897 | 0.654 (`hor_stress`) | 0.415 | `ver_D` (0.123) | 0.75 | 0.85 |
| ALT-B (greedy, f1) | hor_D, tor_S, f1, dia_S | 0.903 | 0.701 (`hor_stress`) | 0.411 | `hor_D` (0.162) | 0.75 | 0.82 |
| brief example | mass, ver_S, tor_S, ver_D, f1 | 0.894 | 0.702 (`dia_stress`) | 0.295 | `mass_kg` (0.005) | 0.91 | 1.00 |
| S5 four stresses + f1 | ver_S, hor_S, dia_S, tor_S, f1 | 0.917 | 0.644 (`tor_magdisp`) | 0.346 | `ver_S` (0.108) | 0.84 | 0.82 |
| N3 no torsion | ver_S, dia_S, f1 | 0.807 | 0.114 (`tor_stress`) | 0.241 | `ver_S` (0.153) | 0.82 | 0.82 |

Unique information of the recommended members (1 - GBM GKF R^2 from the other
members + `volume,area`): `log_max_ver_stress_mpa` 0.153 (R^2 from volume,area
alone 0.560), `log_max_dia_stress_mpa` 0.295 (0.264), `log_max_tor_stress_mpa`
0.909 (-0.016), `log_first_mode_freq_hz` 0.228 (0.728).

REC / ALT-A / ALT-B differ by 0.01-0.02 in span and 0.05-0.10 in participation
ratio -- inside GBM cross-validation noise. The tie was broken by engineering
meaning: the GE-challenge stress constraints, `design_loop/problem.py`'s
default `vertical,diagonal` load cases, and the first mode as the challenge's
modal quantity. The brief's example set is dominated: `mass_kg` is a pure
duplicate of volume (unique 0.005) and `ver_magdisp` is r 0.91 with `ver_stress`.

### 2.6 Torsion stress: the only orthogonal label, and the riskiest

R^2 -0.04 from geometry, 0.10 from all other labels, ICC(1) 0.06 (siblings
differ as much as strangers), r 0.28 with its own displacement, log-skew +0.59.
It is not a mesh-discretization artefact by the available test: adding
DeepJEB's `num_tets` / `num_nodes` / `min_jac` raises its R^2 by +0.002 (vs
+0.03 / +0.04 for ver / hor stress; mesh density `num_tets/volume` has CV 19%
and is 0.98 predictable from geometry), and its tensile / compressive principal
peaks track each other (r 0.73, ratio 1.02). Without it the best 3-name set
leaves torsion stress at R^2 0.11. It is physically plausible as a hyper-local
hot-spot quantity the VAE latent may not resolve; expect the FM's conditional
accuracy on it to be the lowest of the set, and audit it only relatively (the
in-repo tet4 solver is documented optimistic and reports the 99.5th-percentile
von Mises, not the CSV's max).

### 2.7 Units and geometry sanity

| quantity | value |
|---|---|
| physical longest extent 1.8/scale (y axis) | mean 183.8 mm, CV 1.37% with all shapes; **CV 0.79% (175.2-186.4 mm) excluding shape 2099** |
| `design_loop/problem.py` length_scale 0.19/1.8 m per unit | 190 mm extent, +3.3% vs dataset mean |
| the same constant used for the CONDITION AUDIT | +8.71% median error on `mass_kg`, +9.69% on `volume_mm3`, +6.44% on `surface_area_mm2` over all 2137 usable shapes, on a *perfect* decode (L^3 and L^2 scaling of a 3.37% length offset). `inference_profiles/sample.py` therefore defaults the audit to `0.1838 / 1.8` m per unit and `4470` kg/m3 and records `length_scale_source` / `density_source` in `fea_audit`; `opt_length_scale` / `opt_material_rho` still override. The design loop keeps 0.19 m / 4430 kg/m3, where only relative comparisons matter |
| solver density vs label density | `design_loop/fea.py`'s Ti-6Al-4V default is **4430** kg/m3 while the labels imply **4470** (0.9%). The solve is linear-static, so density enters nothing the audit scores except `mass_kg`; the audit uses 4470 and the loop keeps 4430 |
| csv volume(mm3) x scale^3 vs stored `volume` | median abs dev **0.039%**, p95 0.077%, p99 0.093%, max 0.122% (excl. 2099; 2099: +428.7%) |
| csv surface_area(mm2) x scale^2 vs stored `area` | median 0.034%, p95 0.078%, max 0.241% (excl. 2099; 2099: +423.3%) |
| mass/volume density | 4470 kg/m3, CV 0.0002% |
| mass_kg per normalized volume unit | 4.793 kg (CV 2.3%); surface_area_mm2 per normalized area unit 10468 mm^2 (CV 1.6%) |
| shape h5_idx 2099 (`131_561`, parent 131, 11 variants; v3 **test** split) | stored bbox -> x 53.5 / y 89.4 / z 64.7 mm vs siblings 108.8 / 183.3 / 65.0 mm: **partial STL, full-bracket CSV row** (mass 1.071 kg, siblings 0.95-1.45; f1 3652 Hz, siblings 1363-3245). It is the single shape behind the "heavy-tailed bbox_z" (1.303 vs 0.620-0.672, CV 0.81% without it). |

The absolute FEA labels are coherent with the normalized SDF to well under 1%,
so a mass target can be converted to a normalized volume target
(`mass_kg / 4.793`, carrying the 2.3% extent floor) and mass can be audited from
the decoded mesh without FEA. Shape 2099's labels do not describe its geometry;
it is now dropped by the `eval_exclude_shapes 2099` line in the ex5 evaluate
configs rather than by prose (`evaluate.py::excluded_shape_ids` /
`allowed_positions` remove it from the pool BEFORE the seeded subset is drawn,
and print what they dropped). `add_fea_conditions.py` still does not self-check
`|volume_mm3 * scale^3 / volume - 1|`
(not in the shared contract), so the exclusion is the evaluation's job; drop it
from every conditional benchmark on the test split.

### 2.8 Parent lineage and the v3 split

ICC(1) is 0.31-0.50 for the recommended labels (ver stress 0.45, dia stress
0.31, f1 0.49; torsion 0.06); within-parent share of variance 44-61%; the median
within-parent raw range / median is 53% (ver stress), 52% (dia), 63% (tor),
85% (f1), 59% (volume) against between-parent spreads of 80-164%. Sibling
variants are large morphs, not perturbations.

Under the reproduced v3 split (`split_seed 42`, `split_by_parent True`: train
1713 shapes / 209 parents, val 216 / 26, test 209 / 28) test conditions lie
inside the train range (<= 0.5% outside per name) and the test -> train
nearest-neighbour distance equals the train's own spacing (ratio 0.96-1.10 in
1-d and in the 6-d joint space). **A parent-grouped conditional evaluation is
in-distribution in condition space and out-of-family in shape space**: it tests
whether seen condition values can be realised with unseen geometry families,
not condition extrapolation. Genuine extrapolation needs explicitly out-of-range
targets. Test targets still reach `|z|` 3.82 (area), 3.21 (tor stress), 3.06
(volume), 3.02 (f2) under train statistics, so the conditional benchmark needs
`max_condition_z >= 4`.

Stored-space train mean / std for the recommended FEA names: log ver stress
6.966 / 0.257, log dia stress 6.281 / 0.223, log tor stress 5.818 / 0.226, log f1
7.970 / 0.455 -- all far above `min_condition_std`.

---

## 3. Condition set, transforms, and why per-dimension dropout

### 3.1 The set (`config_train_v3_fea.txt`, ex5)

```text
condition_names  volume,area,log_max_ver_stress_mpa,log_max_dia_stress_mpa,log_max_tor_stress_mpa,log_first_mode_freq_hz
```

| name | in / out | why |
|---|---|---|
| `volume` | IN | the size DOF (CV 29.6%), proxy-guidable (C2/E2), exactly measurable from the decoded mesh; carries `mass_kg` exactly |
| `area` | IN | the second geometric DOF (r 0.63 with volume), proxy-guidable; keeps ex4 vs ex5 comparable |
| `log_max_ver_stress_mpa` | IN | primary GE load case and the cleanest label; stands in for ver displacement (r 0.91) and most of the horizontal case; 41% of its variance is shape-specific |
| `log_max_dia_stress_mpa` | IN | design_loop's second default load case; only 29% predictable from geometry; r 0.82 with ver stress accepted because the remaining 30% is shape-specific |
| `log_max_tor_stress_mpa` | IN | the one orthogonal label (unique info 0.91); flagged riskiest (ICC 0.06) |
| `log_first_mode_freq_hz` | IN | the modal DOF; r 0.95 with the 2nd mode, so exactly one; the GE-challenge quantity |
| `mass_kg`, `volume_mm3` | OUT | identical to `volume` (density CV 0.0002%, extent CV 0.8%); keep in the sidecar for audit / report |
| `surface_area_mm2` | OUT | `10468 x area` |
| displacements (4) | OUT | 0.58-0.85 predictable from geometry, r 0.91 with their stress; ALT-A (ver disp for ver stress) is numerically equivalent if displacement is the specification |
| `log_max_hor_stress_mpa`, `log_max_hor_magdisp_mm` | OUT | recovered from the vertical case (R^2 0.70 / 0.92); not a default design_loop case |
| `log_second_mode_freq_hz` | OUT | r 0.95 with the 1st |
| `bbox_x`, `bbox_y`, `bbox_z` | OUT | CV 0.45% / 0% / 0.81% -- constants |
| CG, inertias, min stresses | OUT (not in registry) | CG_y constant, CG position not performance, inertias ride on volume, min stress is the compressive peak of the same field |

Participation ratio 2.53 and 4 PCs for 90% variance: roughly 2.5-4 usable DOF
versus 1.4 today.

### 3.2 Transforms and where values live

`general_modules/condition_names.py` stores stress, displacement and frequency
as natural logs and mass / absolute volume / absolute area as identity. Every
condition the FM sees, every `cond_values` entry, and every sidecar row is in
**stored** space; `to_stored` / `from_stored` convert; evaluation reports
relative errors in **raw** units (MPa, mm, Hz) after `from_stored`. The sidecar
(`cond_extra` + attrs) is append-only and backward compatible, and the dataset
merges it after the five geometric names so every existing config keeps its
`cond_names` order.

### 3.3 Why `cond_dropout_mode per_dim`

A designer specifies two or three numbers, not six. Under the legacy dropout
(`all`: one Bernoulli mask per sample, one learned null embedding) the network
has never seen a partially observed condition vector, so "leave torsion stress
unspecified" cannot be expressed. `per_dim` draws an independent mask per entry
with probability `cond_dropout`, replaces masked entries by a learned
`null_values[cond_dim]`, concatenates the mask itself to the input (width
`2 * cond_dim`), and uses the all-masked row as the unconditional branch. This
is the union of two established patterns rather than an invention: per-attribute
dropout with a learned placeholder (Composer: independent p = 0.5 per condition;
CoLay: p_cond = 0.5 "uniformly samples different combinations"; DiT / FreeGress:
learned null tokens beat fixed zeros) and feeding the observed-mask as an input
(the imputation convention of CSDI / VAEAC).

Two documented consequences shape the config and the reading of results:

- **No explicit drop-all term.** With independent Bernoulli(p) masks the
  all-masked (unconditional) row appears with probability p^k -- 6e-5 for k = 6
  at p = 0.2 -- so the CFG unconditional branch is effectively untrained.
  Composer and CoLay add an explicit drop-all probability (0.1) for this reason;
  the shared contract's `per_dim` does not. This is harmless while `cfg_scale`
  stays 1.0 (the branch is never evaluated) and is why a CFG curve must not be
  re-measured on this checkpoint without first adding a drop-all term.
  `config_train_v3_fea.txt` uses `cond_dropout 0.2` (v3: 0.1) so that any given
  two-of-six-unspecified request pattern is 1.6% of training rows (about 28k
  examples over 27k updates x 64) while 74% of entries stay observed; CCDM shows
  label accuracy degrading as p_drop rises past ~0.2, so the value is not pushed
  toward Composer's 0.5.
- **Mask leakage.** Conditioning on a subset yields p(x | observed), which
  implies the unobserved attributes through their correlations (stress and
  displacement are about -0.7 with volume; volume-area 0.63). "Unspecified" will
  not mean "free"; the conditional evaluation should report the spread of masked
  dimensions to show it. No source names this for engineering attributes.

---

## 4. Sampling-time accuracy plan

The ex1 pilot (GUIDANCE_MECHANISMS section 2; 6 held-out shapes x 4 samples,
res-96 Marching Cubes measurement, method-selection evidence with the leakage
and unpaired-seed caveats listed there):

| method | valid | volume med / p95 | area med / p95 | FM NFE / output |
|---|---:|---:|---:|---:|
| A. plain conditional | 23/24 | 7.60% / 15.12% | 5.21% / 12.12% | 50 |
| A-rej. best-4-of-16 | 24/24 | 6.55% / 9.91% | 2.58% / 5.85% | 200 |
| C. uncalibrated guidance | 24/24 | 23.07% / 27.09% | 3.05% / 13.11% | 85 |
| **C2. calibrated endpoint guidance** | 24/24 | **1.70% / 4.12%** | 3.24% / 15.33% | 85 |
| D2. calibrated source-space optimisation | 24/24 | 2.06% / 5.93% | 3.07% / 13.77% | 1050 + backward |
| **E2 on A (3 Newton rounds)** | 24/24 | **0.284% / 4.16%** | **0.713% / 4.21%** | 50 + decoder/MC |
| **C2+E2** | 24/24 | **0.077% / 2.55%** | **0.255% / 5.34%** | 85 + decoder/MC |

Decisions:

1. **Calibration is the load-bearing piece** (the "2" in C2/E2). The soft proxy
   `volume = sum(sigmoid(-sdf/tau)) h^3`, `area = sum(||grad occ||) h^3` on a
   48^3 grid is biased (+168% volume on ex1) but almost perfectly linear in the
   true measurement (R^2 0.98; area 0.60). Uncalibrated guidance made volume
   *worse* (7.6% -> 23%). The implementation uses **cell-centre quadrature**
   (the pilot multiplied endpoint nodes by a uniform `h^3`), verified on analytic
   sphere / box SDFs to < 0.01% (soft volume) and ~0.3% (soft area) of the
   analytically smeared values; the calibration artifact records the VAE and FM
   SHA-256s, resolution, tau, measure resolution, names and split, and refuses a
   mismatch. It is fitted on the **val** split (`config_calibrate_descriptors.txt`)
   and never reused across checkpoints (`ex1` coefficients are not `ex5`
   coefficients).
2. **E2 first, C2 second, both opt-in and off by default.** E2 (`newton_rounds`)
   is a post-processor on the retained latents: proxy Jacobian for direction and
   scale, the real res-96 measurement for acceptance, RMS cap 0.12 sqrt(D) and
   `dz, dz/2, dz/4` backtracking, invalid meshes rejected. It had the best
   single-method joint accuracy in the pilot and needs no change to the sampler.
   C2 (`guidance_enabled`) runs inside the ODE as a `guidance_fn` callback so
   `velocity_net.py` stays free of decoder / calibration / Marching Cubes
   concerns; the window `t_start <= t < 1` follows the literature's finding that
   guidance is harmful at high noise and unnecessary at the end, and the endpoint
   (x1_hat) target is the form that stays on-manifold (>93% vs <1% for an epsilon
   target). C2's area p95 was worse than E2's in the pilot; it is a median-best
   candidate, not a default.
3. **`guidance_step_mode velocity_dt` is the default; `per_step_jump` reproduces
   the pilot.** FM-native guidance (OC-Flow, FMPS, "On the Guidance of Flow
   Matching") adds a control to the *velocity* and integrates it over dt, so the
   total strength is NFE-invariant; the diffusion-side recipes (DPS, FreeDoM,
   MPGD, TFG) apply per-step state jumps at a fixed step count. The pilot was a
   per-step jump at 50 steps. `velocity_dt` scales the jump by `dt x 50` so it
   coincides with the pilot at 50 steps and stays constant at 25 / 100; the
   25/50/100 NFE-invariance check is still required (no paper publishes one).
4. **CFG stays at 1.0.** The in-repo measurement (36 samples) is 8.5% -> 21.2%
   volume error from cfg 1.0 to 3.0. Theory explains why overshoot *can* happen
   for a point continuous target (guided sampling concentrates on the boundary of
   the conditional support and shifts Theta(sqrt(w)) off the conditional mean;
   neither DDPM nor DDIM with CFG samples the tilted distribution), but the
   empirical literature is mixed -- CCDM, FreeGress and Guided Flows report label
   accuracy *improving* with scale up to ~2 in regimes with one well-covered
   scalar. The defensible position: keep 1.0 as the default, get accuracy from
   C2/E2 and per-dim conditioning, and re-measure the CFG curve on the FEA
   checkpoint (after adding a drop-all term; section 3.3), where low gamma
   (1.5-2) might start to help.
5. **Rejection (`candidate_multiplier`) is retained.** It is the shipped
   mechanism, needs nothing but decoder calls, and is the baseline every other
   method is scored against. `config_sample_conditional.txt` stacks
   best-8-of-32 + 3 Newton rounds.
6. **Guidance and Newton act on `volume` / `area` only.** FEA-named targets have
   no soft proxy; they are honoured by the conditional FM and measured afterwards
   by `condition_audit`.

Regime caveat (literature): no published result covers ~1.7k 3D shapes
conditioned on 5-10 continuous scalars. The nearest anchors are 2D
field-conditioned topology optimisation (900-4,500 instances; a break-even near
100 designs with *dense* image conditioning) and a 1.6k-airfoil CFG DDPM with a
best-fit slope of achieved vs requested Cl of 0.29. Expect condition *following*
to be the bottleneck before guidance is; TopoDiff's ablation is the relevant
precedent -- its geometric constraint (volume fraction) was met at ~1.8% by
conditioning alone and guidance only improved the performance target.

---

## 5. Evaluation protocol (`config_evaluate_conditional.txt`)

- **Paired base noise.** For each of `eval_num_shapes` seeded-random shapes of
  `eval_split`, every method in `eval_methods` (`plain`, `rejection`, `c2`, `e2`,
  `c2e2`) starts from the same `z0` (seeded by `eval_seed` and the shape index),
  so method differences are not noise differences -- the pilot's unpaired seeds
  (A `1000+s`, C2 `2000+s`) were its largest protocol flaw.
- **Targets** are the shape's own TRUE stored conditions restricted to the
  checkpoint's `cond_names`; errors are `|actual - target| / |target|` in **raw**
  units (`from_stored`) as median and p95, plus valid / watertight rate, latent
  RMS drift from the plain sample, NFE and wall time, to `eval_conditional.json`
  / `.csv`.
- **Calibration on val, scoring on test, test touched once.** All method
  hyperparameters (eta, rounds, methods) are chosen on val.
- **FEA audit is relative-only.** `condition_audit fea` gmsh-meshes each decoded
  mesh and solves the load cases the FEA-named conditions need with
  `design_loop` (`Bracket` at `opt_length_scale`, SI loads since the correction
  of section 8); `surrogate` uses the HI-MGN bridge. Both fall back to
  `geometric` with one printed message when unavailable and record the backend
  in the metadata. Their numbers are comparable *between designs at equal
  discretization*, not to the CSV: tet4 reaches only 0.40-0.90 of the Timoshenko
  deflection depending on refinement, the solver reports the 99.5th-percentile
  von Mises rather than the max, DeepJEB used second-order tets, and the default
  length scale is 3.3% above the dataset's mean extent. Under `geometric`, the
  FEA-named conditions are reported as "not measurable geometrically".
- **What this benchmark is.** In-distribution in condition space, out-of-family
  in shape space (section 2.8). It answers "can the model realise seen
  performance values with brackets it has never seen", not "can it extrapolate".
  Extrapolation targets must be constructed explicitly, and `max_condition_z`
  must be >= 4 even for in-range test targets.
- **Exclude shape h5 index 2099** (`131_561`): its labels belong to a different
  geometry, so it would look like a gross conditional failure.
- **Success criteria** carry over from GUIDANCE_MECHANISMS section 3.3: a paired,
  multi-seed improvement over `rejection` in median *and* p95 with no loss in
  validity, no zero-crossing failures, bounded latent drift, and a visual /
  topology audit, reported with wall time and peak VRAM -- not FM NFE alone.

---

## 6. What is unverified until ex5 trains

Verified without a checkpoint (deterministic CPU tests on analytic SDFs and a
mock decoder, `methods/SDFFlow/tests/test_conditional_tools.py`): soft
volume/area against closed-form sphere / box values and their convergence in
tau; the affine fit recovering known coefficients; calibration save / load /
compatibility refusal; Newton driving a mock sphere to a target volume, joint
volume-area targets, the RMS cap, never accepting a worse step, skipping an
invalid start; the guidance callback being zero outside its window and signed
inside; `velocity_dt` and `per_step_jump` agreeing at the reference step count
and `velocity_dt` scaling with dt; `newton_rounds 0` and out-of-window guidance
being bitwise no-ops; the launcher accepting the five new configs with only the
missing-checkpoint error and every new diagnostic code firing on synthetic
configs. The sidecar builder's join (2138/2138) and the label statistics are
verified on the real CSV and HDF5.

Not verified, and not to be claimed until measured on ex5:

1. That the FM *follows* four FEA conditions at all -- the residual variances of
   section 2.4 are an upper bound on the learnable signal, not a floor on FM
   error; torsion stress may be unlearnable from the latent.
2. That `per_dim` conditioning reaches the accuracy of `all` on fully specified
   requests (Composer / CoLay report a single-condition cost for joint training).
3. Any C2 / E2 number on this architecture: the pilot ran on ex1 (1 x 256 MLP
   decoder); the 48^3 query-to-32-token gradient graph of the attention decoder
   is a separate memory / time benchmark, and the calibration slopes and R^2
   will differ.
4. The `velocity_dt` NFE-invariance claim (25 / 50 / 100 steps).
5. The CFG curve on the FEA checkpoint (requires a drop-all term first).
6. That `cond_sweep` strips are continuous in the condition -- expect
   discontinuities where the sweep crosses a topology decision (DeepJEB genus
   6-9); `body_count_raw` per panel is the detector, and 3+ seeds are needed
   before reading a condition effect.
7. The FEA audit's *relative* agreement with the CSV ranking of stresses across
   shapes (never its absolute agreement).
8. Whether `cond_dropout 0.2` is the right per-entry probability; it was set by
   the pattern-coverage argument of section 3.3, not measured.

---

## 7. Runbook

All commands from the `AI-CAE4ALL` root unless noted; every path inside a
config is `../../...` because the native process runs in `methods/SDFFlow`.

```bash
# 0. Append the FEA labels to the dataset once (dry run first; refuses unmatched shapes
#    and an existing sidecar; --list_names prints the registry).
cd methods/SDFFlow
python add_fea_conditions.py --h5 ../../dataset/deepjeb.h5 \
    --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv --dry_run
python add_fea_conditions.py --h5 ../../dataset/deepjeb.h5 \
    --csv D:/CAE_datasets_raw/deepjeb/Scalar/bracket_labels.csv
cd ../..

# 1. Train the FEA-conditioned pair (ex5; VAE stage identical to v3, so a finished ex4 VAE
#    copied to ex5/sdfflow_vae.pth is reused by skip_completed_stages).
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3_fea.txt --check
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_train_v3_fea.txt

# 2. Calibrate the soft proxy on the val split (writes ex5/eval/descriptor_calibration.pth).
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_calibrate_descriptors.txt

# 3. Sample a partial request (two of six conditions unspecified), best-8-of-32 + 3 Newton rounds.
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_sample_conditional.txt

# 4. Sweep one noise row from a light to a heavy, stiff bracket (5 panels + strip PNG).
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_cond_sweep.txt

# 5. Benchmark plain / rejection / e2 on the test split with paired noise (test touched once).
python AI_CAE4ALL_main.py --config configs/SDFFlow/config_evaluate_conditional.txt
```

Turning C2 on: set `guidance_enabled True` in `config_sample_conditional.txt`
(the `guidance_*` values are in its header) or add `c2,c2e2` to `eval_methods`
in the benchmark, after the calibration of step 2 exists. `condition_audit fea`
needs `gmsh` and `pyamg` in the SDFFlow venv; `surrogate` needs the HI-MGN
checkpoint and config of `config_optimize_surrogate.txt`.

---

## 8. Load-unit correction in `design_loop/problem.py` (2026-09)

While checking load-case parity for the FEA audit, the literature pass found
that DeepJEB's FEA used the GE bracket-challenge loads **in SI** -- vertical
35.6 kN (+z), horizontal 37.8 kN, diagonal 42.3 kN at 42 deg, torsion 565 N*m
(the challenge's 8,000 / 8,500 / 9,500 lbf and 5,000 lbf*in converted) -- while
`methods/SDFFlow/design_loop/problem.py::Bracket.LOAD_CASES` held the imperial
numerals `8000 / 8500 / 9500 / 5000` and `fea.py` read them as N and N*m next to
SI material constants (E = 113.8 GPa, rho = 4430 kg/m3, metres). The three
forces were therefore **4.448x too small** and the torsion moment **8.85x too
large**. A concurrent change (agent B6) converted the table to SI with explicit
`LBF_TO_N` / `LBIN_TO_NM` factors; the docstring in `problem.py` now records the
history.

What this does and does not invalidate:

- **Relative rankings from earlier `optimize` runs stand.** Linear statics makes
  every stress and displacement scale exactly with a common force factor
  (compliance by its square), and the loop's allowables are the median of a
  population analyzed under the *same* factor, so with the default
  `vertical,diagonal` cases -- both forces, same factor -- the objective and the
  ranking of designs are unchanged. `output/geometry_generation/ex1/optimization*/`
  results keep their ranking.
- **Absolute numbers from those runs must not be quoted.** Their stresses and
  deflections were understated about 4.4x, and any run that mixed the torsion
  case with a force case weighted torsion 8.85 x 4.45 = about 39x too heavily
  against it, so no recorded absolute stress / deflection and no
  force-vs-torsion trade-off from before the correction may be cited.
- **Even SI-correct audit values are relative-only.** tet4 is stiff (0.40-0.90 of
  the Timoshenko tip deflection over the refinement range tested), the solver
  reports the 99.5th-percentile von Mises rather than DeepJEB's max, DeepJEB used
  second-order tets, and the default `opt_length_scale` (190 mm extent) is 3.3%
  above the dataset's 183.8 mm mean. Wiring a tet10 element remains the change
  to make before quoting absolute stresses.
- **Frame conventions are flagged, not rotated.** The repo's horizontal case is
  +y (the long axis) and its diagonal is decomposed 42 deg from the horizontal;
  the DeepJEB paper labels its horizontal +x and torsion about -z in *its* frame,
  and the challenge text reads "42 deg from vertical". `problem.py` documents
  these as deliberate no-rotation choices. Any `condition_audit fea` comparison
  with the stored labels therefore reports ratios / rankings, and
  `sample.py::load_cases_used()` reads the live table at runtime for the
  metadata so the numbers are never duplicated.

---

## 9. Sources

Engineering conditional generators and their constraint accuracy: TopoDiff
(https://ar5iv.labs.arxiv.org/html/2208.09591), Diffusing the Optimal Topology
(https://ar5iv.labs.arxiv.org/html/2303.09760), latent-space TO
(https://arxiv.org/html/2508.05624v1), Optimize Any Topology
(https://arxiv.org/html/2510.23667), cCDM (https://arxiv.org/html/2408.08526),
trajectory-aware FM for TO (https://arxiv.org/html/2607.14652), conditional
airfoil DDPM (https://arxiv.org/html/2408.15898v1), DDOM
(https://arxiv.org/html/2306.07180), Guided Flows
(https://arxiv.org/html/2311.13443). DeepJEB (https://arxiv.org/abs/2406.09047,
https://arxiv.org/html/2406.09047v1) and DeepJEB++
(https://arxiv.org/html/2606.12994).

Per-attribute dropout and null representations: Composer
(https://ar5iv.labs.arxiv.org/html/2302.09778), CoLay
(https://arxiv.org/html/2405.13045), Ho & Salimans
(https://ar5iv.labs.arxiv.org/html/2207.12598), DiT
(https://github.com/facebookresearch/DiT/blob/main/models.py), FreeGress
(https://arxiv.org/html/2312.17397), learnable proxy tokens
(https://arxiv.org/pdf/2501.17823), CSDI (https://arxiv.org/abs/2107.03502),
VAEAC (https://arxiv.org/abs/1806.02382), CCDM (https://arxiv.org/html/2405.03546).

Guidance mechanisms: TFG taxonomy (https://arxiv.org/html/2409.15761v2), On the
Guidance of Flow Matching (https://arxiv.org/html/2502.02150), FMPS
(https://arxiv.org/html/2411.07625v3), prediction-target manifold study
(https://arxiv.org/html/2607.00647), OC-Flow (https://arxiv.org/html/2410.18070),
D-Flow (https://arxiv.org/abs/2402.14017), FlowGrad (CVPR 2023), guidance
interval (https://arxiv.org/html/2404.07724).

CFG theory and continuous targets: Bradley & Nakkiran
(https://arxiv.org/abs/2408.09000), Chidambaram et al.
(https://arxiv.org/html/2409.13074), Karras et al. autoguidance
(https://arxiv.org/html/2406.02507v1), CFG++ (https://arxiv.org/html/2406.08070).

Fixed-noise sweeps and their failure modes: reproducibility of diffusion models
(https://arxiv.org/abs/2310.05264), Diffusion Autoencoders
(https://arxiv.org/abs/2111.15640), CcGAN (https://ar5iv.labs.arxiv.org/html/2011.07466),
PcDGAN (https://ar5iv.labs.arxiv.org/html/2106.03620), Prompt-to-Prompt
(https://arxiv.org/abs/2208.01626), spontaneous symmetry breaking
(https://arxiv.org/abs/2305.19693), critical windows (https://arxiv.org/abs/2403.01633),
pitchfork bifurcation (https://arxiv.org/html/2603.20092v3), winning noise
tickets (https://arxiv.org/pdf/2607.06843), Determinism of Randomness
(https://arxiv.org/pdf/2511.07756).

In-repo: `GUIDANCE_MECHANISMS_SOTA_AND_PLAN_2026-08.md` (sections 1.B, 2, 3),
`GEOMETRY_UPGRADE_MESHING_SEMANTIC_2026-08.md` (sections 1.3, 1.4, 1.6, 4.6),
`SOTA_CONDITIONAL_GEOMETRY_SURVEY_2026-07.md`.


## 6. Corrections applied after review (2026-09-06)

Recorded here because several of them change numbers this note quotes.

1. **The export-path "truth" no longer falls back to a convex hull.**
   `sdf_sampling.mesh_descriptors` substitutes `mesh.convex_hull.volume` for a
   non-watertight mesh (2-4x the solid volume on a holed bracket), while
   `mesh_extraction.mesh_report` -- what the audit REPORTS -- returns `None` for
   the same mesh. `descriptor_calibration.true_descriptors` now measures
   locally and reports NaN volume there, so a torn row leaves the volume fit
   (its area pair stays) and E2's line search rejects a candidate that tears
   open instead of scoring it against a hull. Simulated on the pilot's own
   relation, a 2 / 5 / 10% open-row rate had dragged the fitted slope from
   0.858 to 0.346 / 0.279 / 0.142. `calibrate` prints the watertight rate and
   `calibration_min_r2` (0.5) refuses a fit too weak to steer with.
2. **One Marching Cubes grid.** `newton_measure_resolution` now defaults to
   `mc_resolution` and `DescriptorCalibration.check_compatible` pins the
   measurement grid, so the fit, the E2 acceptance test and the reported audit
   are the same operator. The three shipped ex5 configs used to mix res 96 and
   128, a 0.02-0.10% systematic volume offset -- 10-35% of the accuracy C2/E2
   are quoted at, and indistinguishable from model error in the output.
3. **The pilot percentages are an expectation, not a reproduction.** This
   implementation differs from the pilot code in two deliberate ways:
   the C2 loss denominator is the PROXY target (`a * true + b`), which shifts
   the volume:area weight ratio by ~11x at the pilot's coefficients, and
   `descriptor_proxy` integrates on cell centres (`h = 2/R`) where the pilot
   summed node-centred samples (`h = 2/(R-1)`, over-integrating by 6.5%), so
   the pilot's `a = 0.86` does not transfer. Re-measure on ex5.
4. **`condition_score` is comparable across candidates.** It was an RMS over
   only the dimensions a given mesh could provide, so a torn decode (no
   volume) was scored on `area` alone and out-ranked a watertight candidate
   with a worse error. It is now an RMS over the fixed geometric support of the
   request with a 3-sigma penalty per unmeasurable dimension; the value is
   unchanged when nothing is missing. On the ex5 set only 2 of 6 conditions are
   geometric and `volume` is the one that disappears, so the old form was worst
   exactly here.
5. **The benchmark prints its coverage.** `eval_conditional` now shows the
   per-condition `n` and a PAIRED block over the shapes every method could be
   measured on, plus a SCOPE line naming what C2/E2 actually optimise
   (`volume`, `area`) against what is scored. An unpaired median is a
   method-selected subsample, and a method that breaks the meshes it cannot
   correct looks better for it.
6. **`max_condition_z` / `condition_ood_policy` are read by the evaluate FM
   tasks**, where the targets used to be clamped to the checkpoint's
   `condition_clip` for sampling and scored unclamped -- a permanent,
   method-independent error floor with no visible signal. The evaluate default
   is `warn` (`sample` keeps `error`).
7. **`cond_dropout_all_prob` (0.1) trains the CFG branch under `per_dim`.**
   Without it the all-masked row -- the branch `sample_latents` evaluates under
   CFG -- appears with probability `cond_dropout ** cond_dim` = 6.4e-5 at the
   ex5 settings, about 111 rows in the whole run.
8. **NaN conditions cannot reach the optimizer.** `train_fm` checks finiteness
   before the near-zero-variance guard (`nan < min_condition_std` is False), so
   an `add_fea_conditions.py --allow_missing` sidecar no longer trains a whole
   run on a NaN loss.
9. **The sidecar prerequisite is enforced at preflight.** The dataset probe
   returns the merged condition vocabulary and `SDF-COND-FEA-003` (ERROR) fires
   with the builder command; `train_pipeline.py` repeats the check before stage
   1, so a direct native run fails in seconds instead of after the VAE.
10. **E2 stays inside `latent_clip`.** The corrected latent is clamped before it
    is measured, so E2 is not credited with accuracy from a latent region the
    other arms were forbidden to reach, and the metadata records
    `post_newton_clipped_fraction`.
11. **`opt_*` values are validated wherever they are consumed.** The condition
    audit made them a `sample` / `evaluate` key set; `opt_material_nu 0.9`
    (a singular stiffness) used to pass `--check --strict` there.
