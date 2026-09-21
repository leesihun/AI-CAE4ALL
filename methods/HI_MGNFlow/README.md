# cHI-MGNflow

LDGN 스타일 2단계 모델 (Lino, Pfaff & Thuerey, *Learning Distributions of Complex
Fluid Simulations with Diffusion Graph Networks*, ICLR 2025,
[arXiv:2504.02843](https://arxiv.org/abs/2504.02843); "LDGN"은 그 논문이 자기
압축-잠재 변형에 붙인 이름이다). 백본은 논문의
자체 멀티스케일 그래프 대신 이 리포의 HI-MGN V-cycle을 그대로 쓴다 — **논문의
MGN+flow 설계를 채택하되 MGN만 HI-MGN으로 바꾼 것**이 이 방법의 전체 아이디어다.

```text
Stage 1 (train_ae)     y --[HierarchyEncoder, y-aware]--> mu, logvar --[reparam]--> z1  (coarsest 레벨)
                        y --[HierarchyEncoder, y-blind]--> skip levels + prior 조건 g
                        z1, skips --[MultiscaleDecoder]--> y_hat
                        loss = recon(y_hat, y) + ae_kl_weight * KL(q(z|y) || N(0,I))     [KL은 아주 작게]

Stage 2 (train_prior, AE는 frozen)
                        z1 ~ q(z|y)                        (frozen encoder, no grad)
                        z0 ~ N(0,I)                         z1과 같은 크기 (coarsest 레벨 x latent_ch)
                        t ~ [0,1),  y_t = (1-s*t)*z0 + t*z1,  s = 1 - 1e-4
                        loss = || velocity(y_t, t, g) - (z1 - s*z0) ||^2                 [coarsest 레벨에서만]

추론 (generate)         g = HierarchyEncoder_blind(조건만, y 없음)
                        z0 ~ N(0,I)  -->  ODE로 z0 -> z1 적분 (K스텝, coarsest 레벨에서만, 저렴)
                        y_hat = MultiscaleDecoder(z1, skips)         풀메시 forward는 encode 1회 + decode 1회뿐
```

## 왜 이렇게 바뀌었나

flow matching을 쓰는 이유 자체는 바뀌지 않았다: MeshGraphNets-variational은 학습
때 posterior `q(z|y,g)`에서 잠재를 뽑고 추론 때는 별도로 학습한 근사 prior
`p(z|g)`에서 뽑는데, 이 둘이 정확히 같을 이유가 없다 (aggregate posterior
drift). flow matching은 학습과 추론이 항상 같은 `N(0,I)`에서 시작해서 동일한
ODE로 목표 분포에 도달하므로 이 불일치 자체가 없다.

이 리포의 cHI-MGNflow는 처음에 이 flow matching을 **필드 공간에서 그대로**
(압축 없이, `[N, output_var]` 전체에 대해) 적용해서 만들어졌다. 그런데
reconstruction test부터 결과가 맞지 않았고, 참고 논문(Lino, Pfaff, Thuerey)을
다시 확인해보니 그 논문은 필드 공간이 아니라 **압축된 coarse 잠재 위에서**
flow matching을 하고 있었다. 그래서 이 논문의 설계를 그대로 채택하기로
결정했고, 이번 재작성으로 이전 단일 단계(필드 공간) 아키텍처는 코드베이스에서
전부 교체됐다 — 남아있는 이전 버전은 없다. (flow matching의 수학 자체 —
경로 구성, ODE 적분기, 시간 임베딩 — 는 `model/flow.py`에서 거의 그대로
재사용된다. 바뀐 것은 그 수학이 적용되는 대상이 필드에서 coarse 잠재로
바뀌었다는 것과, 그 앞에 압축기(compressor) 단계가 새로 생겼다는 것이다.)

`model/autoencoder.py`의 모듈 docstring이 이 설계 선택의 근거를 논문 자체의
ablation을 인용해서 직접 담고 있다: 논문에서 **같은 멀티스케일 구조를 쓴
VGAE**(더 무거운 KL, 더 coarse한 latent)로 만든 baseline도 여전히 collapse
한다 — 즉 계층 구조(hierarchy) 자체가 LDGN이 이기는 이유가 아니라는 뜻이다.
결정적인 lever는 "`N(0,I)`를 직접 샘플링하는 대신 **prior를 명시적으로
학습시키는 것**"이며, 필드 공간 flow matching(=이전 cHI-MGNflow)은 논문의
표현으로 "정확히 그 ablation이 지는 쪽의 접근 방식"이다.

| | MeshGraphNets-V | 현재 cHI-MGNflow |
|---|---|---|
| 학습 시 잠재의 출처 | posterior `q(z\|y,g)` | posterior `q(z\|y,g)` (stage 1 압축기) |
| 추론 시 잠재의 출처 | 별도로 학습한 근사 prior `p(z\|g)` — **posterior와 불일치할 수 있음** | `N(0,I)` → 학습된 flow ODE로 변환 — **posterior에 도달하도록 명시적으로 학습됨** |
| 무작위성이 사는 공간 | 풀링된(pooled) 잠재 벡터 | coarsest 메시 레벨의 per-node 잠재, `[n_coarse, latent_ch]` |

## 구조

**두 단계, 하나의 모델.** `CHiMGNFlow`(얇은 래퍼)가 `LatentDiffusionGraphNet`
(압축기 + prior를 합친 실제 모델)을 감싼다.

- **`CHiMGNFlow`** (`model/CHiMGNFlow.py`) — 생성자에서 `LatentDiffusionGraphNet`을
  만들고 `self.model.apply(init_weights)`(전역 kaiming 초기화)를 먼저 실행한
  다음 `self.model.reset_zero_init_heads()`(디코더 마지막 층, prior의
  `out_head`, 모든 `AdaLNZero` 헤드를 다시 0 근처로 되돌림)를 실행한다 — **이
  순서가 중요하다.** 반대로 하면 kaiming 초기화가 zero-init을 덮어써서 필요한
  near-identity 스케일링이 깨진다. `_load_frozen_ae(ae_checkpoint)`는
  `train_prior` 단독 모드에서 완료된 stage 1 체크포인트를 로드하고 즉시
  freeze한다. `ae_parameters()` / `prior_parameters()` / `freeze_ae()`는
  옵티마이저를 단계별로 분리하기 위한 델리게이션이다. `forward(task, graph,
  z_t=None, t=None)`이 **DDP에 안전한 유일한 진입점**이다 (`task='ae'` →
  `forward_ae`, `task='prior'` → `forward_prior_step`) — 두 단계 중 정확히
  하나만 그래디언트를 받는 시점이 항상 존재하므로, DDP는
  `find_unused_parameters=True`로 돈다 (`training_profiles/distributed_training.py`).
  backward가 걸리는 모든 호출은 반드시 `nn.Module.__call__`(즉 이
  `forward` 디스패치)를 통해야 한다.
- **`LatentDiffusionGraphNet`** (`model/CHiMGNFlow.py`) — 실제 합성 모델.
  생성자에서 `edge_var`가 `EDGE_FEATURE_DIM`(=8, `general_modules/edge_features.py`)과
  다르면 즉시 에러를 내고, `use_multiscale`이 참이 아니면 즉시 `ValueError`를
  낸다 — coarse 잠재를 놓을 coarse 레벨 자체가 없기 때문이다(아래 "반드시
  알아야 할 것들" 참고). `hierarchy_encoder`(`HierarchyEncoder`),
  `decoder_ascend`(`MultiscaleDecoder`), `prior`(`LatentFlowPrior`)를 갖고
  있다. `encode_both(graph)`가 **두 번 통과하는 shared encoder**를 구현한다:
  같은 가중치로 (1) `y`를 포함한 입력으로 한 번 통과시켜 `mu`/`logvar`(posterior)를
  얻고, (2) `y`를 0으로 채운 입력으로 한 번 더 통과시켜 디코더 skip과 prior
  조건을 얻는다. **두 번째(y-blind) 패스만이 디코더/prior로 가는 유일한
  통로**이므로, 학습 때도 추론 때와 똑같이 "y를 모르는" 정보만 skip에 실린다
  — 그렇지 않으면 디코더가 잠재를 거치지 않고 skip을 통해 y를 그대로
  새어나가게(leak) 학습해버릴 수 있다. `generate(graph, flow_cfg)`가
  `@torch.no_grad()` 추론 경로 전체다: 조건을 1회 encode, coarse 잠재 ODE를
  K스텝 적분, 1회 decode. `flow_predict='ensemble_mean'`을 넘기면 즉시
  `ValueError`를 내며 — "`generate()`는 호출당 정확히 샘플 1개만 뽑는다;
  잠재를 평균하지 말고 디코딩된 필드를 여러 번 호출로 평균하라"는 메시지로
  명시적으로 막아놓았다. 실제 평균은 `training_profiles/training_loop.py`의
  `_generate_fields`가 한 단계 위에서 수행한다.
- **`model/autoencoder.py`** — `HierarchyEncoder`(V-cycle 하강 팔, 위에서
  설명한 두 번의 forward 모두 이 클래스가 담당), `MultiscaleDecoder`(V-cycle
  상승 팔, y-blind skip 레벨만 읽음), `LatentFlowPrior`(**coarsest 레벨
  하나에만** 얹힌 작은 AdaLN-Zero GnBlock 스택 — 이게 K-step ODE 적분을 거의
  공짜로 만드는 지점이다. 여기만 K번 돌고, 나머지 V-cycle 전체는 encode/decode
  에서 딱 1번만 돈다).
- **`model/flow.py`** — 범용 flow-matching 수학(`TimeEmbedding`, `draw_times`,
  `loss_weight`, `sample_path`, `predict_x0`, `integrate`, `predict_mean`,
  `resolve_flow_config`). 텐서가 필드인지 잠재인지 상관하지 않는 순수 함수들
  이며, 이전 단일 단계 cHI-MGNflow에서 거의 그대로 이식되어 지금은 coarse
  잠재에 적용된다.

| 파일 | 역할 |
| --- | --- |
| `CHiMGNFlow_main.py` | 엔트리포인트. 배너 출력, stdout/stderr UTF-8 재설정, argparse → training/inference 프로필로 위임 |
| `model/CHiMGNFlow.py` | `CHiMGNFlow`(래퍼) + `LatentDiffusionGraphNet`(압축기+prior 합성 모델) |
| `model/autoencoder.py` | `HierarchyEncoder`, `MultiscaleDecoder`, `LatentFlowPrior` |
| `model/flow.py` | 범용 flow-matching 수학 (경로 구성, ODE 적분기, 시간 임베딩) |
| `model/blocks.py` | 공유 GN 블록 / `AdaLNZero` 등 HI-MGN 계열 전체가 쓰는 저수준 빌딩 블록 |
| `training_profiles/training_loop.py` | `ae_loss`, `prior_loss`, 학습/평가 루프, `_generate_fields`, `_crps`, `[FlowDiag]` 로그 |
| `training_profiles/single_training.py` | 에폭 드라이버, 체크포인트 저장(및 `best_by` 메시지) |
| `training_profiles/distributed_training.py` | DDP 설정 (`find_unused_parameters=True`) |
| `training_profiles/setup.py` | 데이터셋 분할, 체크포인트 저장/로드, 로그 파일 초기화 |
| `inference_profiles/rollout.py` | 오토레그레시브 롤아웃, `SAMPLING_TIME_KEYS` config-wins 오버라이드 |
| `general_modules/config_validation.py` | 네이티브 런타임 가드 (`parallel_mode` 등) |
| `general_modules/edge_features.py` | `EDGE_FEATURE_DIM = 8` |
| `general_modules/mesh_dataset.py` | HDF5 로딩, coarsening 계층 캐시 (MGN-V와 공유) |
| `tests/` | `test_flow_smoke.py` 외 3개 — "검증 현황" 참고 |

## 모드

| mode | 무엇을 하나 | 필수 키 (`_BASE_TRAIN_REQUIRED` 외 추가분) |
| --- | --- | --- |
| `train_ae` | Stage 1만: `HierarchyEncoder`+`MultiscaleDecoder`를 recon+KL로 학습 | 없음 |
| `train_prior` | Stage 2만: `ae_checkpoint`를 로드해 즉시 freeze, `LatentFlowPrior`만 flow-matching으로 학습 | `ae_checkpoint` |
| `train` | 결합: `ae_epochs`만큼 stage 1 → 메모리 내에서 `freeze_ae()` → `training_epochs`만큼 stage 2. 체크포인트 왕복 없음 | `ae_epochs` |
| `inference` | `generate()`로 오토레그레시브 롤아웃 | (별도 셋) `modelpath`, `infer_dataset`, `input_var`, `output_var`, `edge_var` |

`_BASE_TRAIN_REQUIRED` = `dataset_dir, modelpath, input_var, output_var, edge_var, latent_dim, training_epochs, batch_size, learningr`
— `train_prior`조차 이걸 전부 요구하는 이유는, 그 모드도 (곧 덮어써질) 전체
`LatentDiffusionGraphNet`(압축기+prior)을 일단 생성해야 하기 때문이다.

## 예측 모드 (`flow_predict`)

| 값 | 동작 | 샘플당 비용 |
| --- | --- | --- |
| `mean` | encode 1회 → `predict_mean(velocity, z0) = z0 + velocity(z0, 0.0)` (coarse 잠재, 1 forward) → decode 1회 | encode + prior 1회 forward + decode |
| `sample` | encode 1회 → coarse 잠재 ODE를 `flow_steps`만큼 적분(heun=2K, euler=K forward) → decode 1회 | encode + prior K(또는 2K)회 forward + decode |
| `ensemble_mean` | **`generate()` 내부에서는 금지**(`ValueError`) — `training_loop.py::_generate_fields`가 `generate()`(=`sample`)를 `num_vae_samples`번 호출하고, **디코딩된 필드**를 평균 | M × (encode + prior K회 forward + decode) |

`mean`의 도출 (`model/flow.py::predict_mean` docstring 그대로): 경로는
`y_t=(1-s*t)*z0+t*y1`, `u=y1-s*z0`, `s=1-`sigma_min`(=1e-4)`이고 t=0에서는
`y_0=z0`이 이미 알려져 있으므로 회귀 최적점은
`v*(z0,0,g) = E[y1|g] - s*z0`이다. 여기서 Euler 스텝 한 번(`dt=1`)을 밟으면
`z0 + v*(z0,0,g) = E[y1|g] + sigma_min*z0` — 즉 `sigma_min`(1e-4) 스케일의
잔차 하나만 남는 조건부 평균이다. 같은 체크포인트가 분포도 샘플링하고
결정론적 예측(1 forward, 적분 없음)도 서빙한다는 뜻이다. 코드 docstring이
직접 언급하는 두 가지 주의점: (1) 학습 예산은 모든 t에 퍼져 있는데 이 모드는
t=0 한 점의 정확도에 전부 의존한다, (2) t=0 슬라이스가 덜 학습됐다면 M*K배
비싼 `sample` 앙상블 평균이 더 나을 수 있다. `misc/eval_prediction_modes.py`가
이 둘을 재는 스크립트로 존재하지만, 이번 재작성에서 새 API 대비 최신 상태인지
확인하지 않았다.

## 실행

```bash
cd methods/HI_MGNFlow

# Stage 1만 (압축기)
python CHiMGNFlow_main.py --config ../../configs/HI_MGNFlow/<run>/config_train_ae.txt

# Stage 2만 (frozen AE 위에 prior) -- ae_checkpoint 필요
python CHiMGNFlow_main.py --config ../../configs/HI_MGNFlow/<run>/config_train_prior.txt

# 결합 (stage1 -> stage2, 체크포인트 왕복 없음)
python CHiMGNFlow_main.py --config ../../configs/HI_MGNFlow/<run>/config_train.txt

# 추론 (오토레그레시브 롤아웃)
python CHiMGNFlow_main.py --config ../../configs/HI_MGNFlow/<run>/config_infer.txt
```

런처 경유(리포 루트에서, `model chi-mgnflow`):

```bash
python AI_CAE4ALL_main.py --config configs/HI_MGNFlow/SAOI_all_input/config_train_bot.txt --check --strict
python AI_CAE4ALL_main.py --config configs/HI_MGNFlow/SAOI_all_input/config_train_bot.txt
```

(`config_train_bot.txt`는 `mode train`, 즉 결합 모드다.)

## 반드시 알아야 할 것들

1. **샘플당 풀메시 forward는 정확히 2번뿐이다 — encode 1회, decode 1회.**
   이전(필드 공간) 설계는 ODE 스텝마다 풀메시 V-cycle 전체를 다시 돌려야 했다
   (K스텝이면 K번). 지금은 K스텝 ODE 적분이 **coarsest 레벨 하나**에서만
   일어나고, 값비싼 V-cycle 전체는 샘플당 딱 두 번만 돈다. 이게 이 재작성의
   핵심 이득이다.
2. **`flow_steps`(K)는 학습 비용을 바꾸지 않는다.** 학습(`prior_loss`)은
   매 스텝마다 그래프당 임의의 `t` 하나만 뽑아서 학습하며 K와 무관하다. K는
   순수한 추론 시점(sampling-time) 선택이고, 같은 체크포인트가 재학습 없이
   어떤 K로도 적분된다(스모크 테스트가 K=2/4/6/12를 같은 체크포인트로 확인).
3. **coarsening 계층은 encode, decode, 모든 ODE 스텝, 앙상블의 모든 draw에서
   반드시 동일해야 한다.** `model/flow.py::integrate`의 docstring이 이걸
   명시적으로 경고한다 — 계층이 스텝마다 바뀌면 명목상 "같은 잠재 공간"이
   실제로는 스텝마다 달라진다. `hierarchy_seed` + `hierarchy_variants`가
   이걸 지키면서도 rollout 호출마다는 다른 계층을 쓸 수 있게 해주고,
   `inference_profiles/rollout.py`의 `SAMPLING_TIME_KEYS`가 이 둘(`flow_steps`,
   `flow_solver`)만은 체크포인트 값이 아니라 실행 중인 config 값을 쓰도록
   보장한다(그 외 모든 `model_config` 키는 체크포인트가 이긴다).
4. **`flow_predict='ensemble_mean'`은 잠재가 아니라 디코딩된 필드를
   평균한다.** `generate()` 내부에서 잠재를 평균하려고 하면 즉시 에러가
   난다 — 위 "예측 모드" 표 참고.

## Config 키

두 단계 아키텍처가 추가한 키 (`AE_PRIOR_KEYS`):

| 키 | 기본값 | 의미 |
| --- | --- | --- |
| `latent_ch` | 4 | coarse 메시 노드 1개당 압축 잠재 채널 수 (풀링된 벡터 1개가 아님). **아키텍처를 결정** — 체크포인트는 학습 당시 값에서만 로드됨 |
| `ae_kl_weight` | 1e-6 | stage 1의 `KL(q(z\|y)‖N(0,I))` 가중치. 논문 기준 ~1e-6으로 아주 작게 — near-lossless 압축기가 목적이며 생성적 병목이 아니다 |
| `prior_blocks` | 4 | stage 2 `LatentFlowPrior` 트렁크의 AdaLN-Zero GnBlock 개수 |
| `ae_checkpoint` | — | `train_prior` 전용. 완료된 `train_ae` 체크포인트 경로 |
| `ae_epochs` | — | `train`(결합) 전용. stage 1 에폭 수 (stage 2는 `training_epochs`만큼) |

flow-matching ODE 표면 (`FLOW_ONLY_KEYS`, 이전 필드 공간 설계와 그대로 공유):

| 키 | 기본값 | 의미 |
| --- | --- | --- |
| `flow_steps` | 30 | 추론 시 ODE 스텝 수 (sampling-time, 재학습 불필요) |
| `flow_solver` | `heun` | `heun`(2차, 스텝당 2회 forward) 또는 `euler` |
| `flow_time_freqs` | 16 | 시간 임베딩의 Fourier octave 수. **아키텍처를 결정**(AdaLN 입력 폭) |
| `flow_t_sampling` | `uniform` | `uniform` 또는 `logitnormal`(경로 중간에 학습 예산 집중) |
| `flow_t_logit_scale` | 1.0 | `logitnormal` 사용 시 스케일 |
| `flow_loss_weighting` | `uniform` | `uniform`(velocity 예측) 또는 `x0`((1-s·t)² 재가중 — velocity head는 그대로 두고 data-prediction 효과만 얻음) |
| `flow_det_prob` | 0.0 | 학습 그래프 중 `t=0`으로 고정할 비율 — deterministic 모드를 실제로 학습시킴 |
| `flow_predict` | `sample` | `sample` / `mean` / `ensemble_mean` (위 "예측 모드" 참고) |
| `val_flow_steps`, `val_num_samples` | —, 8 | 검증용 ODE 스텝/앙상블 크기 (`val_interval`마다 돌아서 추론보다 싸게 잡음) |
| `best_by` | `crps` | 체크포인트 선택 지표: `recon` / `crps` / `det` |

제거된 키 (전부 known, 즉 `CFG-UNKNOWN`이 아니라 각각 정확한 진단으로 안내됨):

| 키 그룹 | 진단 코드 | 의미 |
| --- | --- | --- |
| `use_vae`, `num_z`, `mmd_bandwidth`, `prior_type`, `prior_fm_*`, `latent_inflation`, `es_*` 등 variational 전용 키 전체 | `FLOW-REMOVED` (경고) | posterior/학습된 prior가 없으므로 조용히 무시됨 |
| `flow_head`, `flow_head_eps` | `FLOW-HEAD-REMOVED` (경고) | 이전 단일 단계 설계의 출력 파라미터화 잔재. 지금은 velocity head 하나뿐이라 나눗셈을 막을 대상이 없음 |
| `pipeline_microbatches`, `noise_gamma`, `noise_std_ratio` | `FLOW-RUNTIME-REMOVED` (에러) | 복사됐지만 구현 안 된 model-split/노이즈 경로 |
| `std_noise` (0이 아닌 값) | `FLOW-RUNTIME-REMOVED` (에러) | 위와 동일 |
| `std_noise 0` | `FLOW-LEGACY-NOISE` (알림) | 구버전 config 호환용 no-op으로만 허용 |

그 외 가드레일: `use_multiscale`이 참이 아니면 `FLOW-FLAT` **에러**(스펙과
`LatentDiffusionGraphNet.__init__` 양쪽에서 거의 같은 문구로 중복 검증됨).
`ae_kl_weight > 1e-3`이면 `FLOW-AEKL-LARGE` 경고. `message_passing_num`을
써도 `FLOW-MPNUM-INERT` 알림만 뜨고 조용히 무시됨(V-cycle 깊이는
`mp_per_level`만 읽음). `parallel_mode`가 `ddp`가 아니면(`model_split` 등)
`FLOW-PARALLEL` 에러 — cHI-MGNflow는 `ddp`만 지원한다.

## 학습 중 읽어야 할 로그

`log_training_config`가 시작 시 V-cycle 구조, coarse 잠재 크기, 단계별 설명을
찍고, `train`(결합) 모드면 이 줄도 찍는다:

```text
  Combined run: {ae_epochs} AE epochs, then {training_epochs} prior epochs on the frozen AE
```

Stage 2 검증마다(`evaluate_prior_sampling_epoch`) 찍히는 줄 — `det(1fwd)`가
`mean` 모드(1 forward), `crps`/`spread`가 `sample` 모드(앙상블)를 잰다:

```text
  [FlowDiag] crps=... det(1fwd) mse=... 1-draw mse=... spread/gt=...  (steps=..., S=...)
```

`spread/gt`가 0.02 미만이면 추가로 경고가 찍힌다:

```text
  [FlowDiag] WARNING: ensemble spread is ~0 -- the model is ignoring the noise channel.
```

## 검증 현황

**확인됨 — 이번 재작성 세션에서 실제로 실행해서 확인**

- CPU 스모크 테스트(`tests/test_flow_smoke.py`, 단독 실행은
  `python tests/test_flow_smoke.py`): 합성 6×6 메시로 stage 1 gradient flow,
  stage 2 gradient isolation(compressor freeze 후 prior에만 grad가 붙고
  compressor에는 grad가 없는지 둘 다 assert), 3-draw 생성에서 0이 아닌
  spread, 같은 체크포인트로 K=2/4/6/12 전부 finite — 전부 PASS.
- `methods/HI_MGNFlow/tests/` 전체 스위트(4개 테스트) — PASS.
- 런처 preflight 계약 검증: 실제 체크인된 config
  (`configs/HI_MGNFlow/SAOI_all_input/config_train_bot.txt`, `mode train`)에
  대해 `--check --strict --skip-filesystem-check --skip-environment-check` →
  0 errors / 0 warnings / 5 notices. filesystem 체크를 켜면 나는 유일한
  에러는 로컬에 없는 데이터셋 파일이며(학습/데이터셋이 원격 GPU 박스로
  옮겨갔기 때문에 예상된 결과), 그 외에는 깨끗하다.
- 루트 계약 테스트(`tests/test_config_key_contracts.py`,
  `tests/test_native_config_consumption_parity.py`)의 chi-mgnflow 관련
  케이스 전부 PASS.

**아직 안 됨 — 솔직하게 남겨둠**

- 이 두 단계 아키텍처로 실제 GPU 학습을 돌린 적은 **아직 없다.** recon/KL
  수렴 곡선, CRPS 수치, 이전 아키텍처와의 정량 비교 전부 미확보 상태다.
  이전 문서에 있던 Wave 0 / Wave A 실측 표는 **삭제된 이전 단일 단계(필드
  공간) 아키텍처**를 RTX 2080 SUPER에서 측정한 것이라 지금 아키텍처에는
  적용되지 않으므로 이 문서에서 뺐다 — 새 아키텍처로 다시 측정해야 한다.
- `misc/eval_prediction_modes.py`, `misc/wave_a_sweep.py`, `misc/wave0_report.py`,
  `misc/eval_det_baseline.py`, `misc/field_intrinsic_dim.py`,
  `misc/score_rollouts.py`, `misc/plot_loss*.py`, `misc/plot_arm_curves.py`는
  이전 단일 단계 API를 직접 참조했을 가능성이 있고, 새 `CHiMGNFlow`/
  `LatentDiffusionGraphNet` API에 맞게 동작하는지 이번 세션에서 확인하지
  않았다.

**여전히 유효한 것으로 재확인된 harness 버그 수정 3건** (아키텍처와 무관하게
지금 코드에도 남아있음 — 이번 세션에 grep으로 재확인)

| 증상 | 원인 | 수정 위치 |
| --- | --- | --- |
| Windows 콘솔(cp949 등)에서 배너 출력 시 `UnicodeEncodeError` | 박스 드로잉 글리프를 담은 첫 print가 런처가 파이프한 스트림의 ANSI 코드페이지로 인코딩됨 | `CHiMGNFlow_main.py:9-17` — stdout/stderr를 `encoding='utf-8', errors='replace'`로 재설정 |
| 체크포인트를 다른 `flow_steps`로 추론했는데 조용히 체크포인트에 저장된 값으로 되돌아감 | `model_config` 복원이 모든 키를 체크포인트 값으로 덮어씀 | `inference_profiles/rollout.py:568-578` — `SAMPLING_TIME_KEYS=('flow_steps','flow_solver')`만 config가 이기고, 나머지는 체크포인트가 이김 |
| 체크포인트 저장 로그가 항상 "new best recon"이라 실제 `best_by=crps`인 실행에서도 오해를 줌 | 저장 메시지가 지표 이름을 하드코딩 | `training_profiles/single_training.py:204` — 실제 `best_by` 값을 메시지에 반영 |
