# cHI-MGNflow — Config Reference

전체 키 카탈로그. 런처 스펙 [`cae_suite/specs/chi_mgnflow.py`](../../cae_suite/specs/chi_mgnflow.py)가
검증의 단일 출처이고, 이 문서는 각 키가 **왜** 그 값이어야 하는지를 설명한다.

파일 형식은 모노레포 공통 플랫 텍스트다. `key<TAB>value`, `%`는 주석, `#`는 줄 끝 주석.

```bash
python AI_CAE4ALL_main.py --config <path> --check       # 검증만
python AI_CAE4ALL_main.py --config <path> --explain-config
python AI_CAE4ALL_main.py --config <path>               # 검증 통과 시 실행
```

---

## 0. 값이 파싱되는 방식 (모노레포 공통 함정)

| 쓴 값 | 파싱 결과 | 주의 |
|---|---|---|
| `100` | `int` | |
| `1e-4` | **`str`** | `.`이 없어 `int()`/`float()` fast-path를 못 타고 문자열로 남는다. 소비 측에서 `float()`로 변환한다 |
| `0.0001` | `float` | 안전하게 쓰려면 이 형식 |
| `0` (단일) | **스칼라** `0` | 리스트가 아니다. `test_batch_idx 0`은 `for x in 0` → TypeError. **항상 2개 이상** |
| `0, 1` | `list` | 쉼표 또는 공백 구분 |
| `True` / `False` | `bool` | |
| 경로 키 | 대소문자 보존 | 그 외 문자열 값은 전부 소문자화 |

**BOM은 하드 에러**다. 중복 키도 에러(네이티브 파서는 조용히 마지막 값을 취함).

---

## 1. 라우팅 · 경로

| 키 | 필수 | 설명 |
|---|---|---|
| `model` | ✅ | `chi-mgnflow` |
| `mode` | ✅ | `train` \| `inference` |
| `gpu_ids` | ✅ | **물리** GPU 인덱스. `CUDA_VISIBLE_DEVICES`를 쓰지 않으므로 `gpu_ids 6`은 진짜 6번 카드다 |
| `parallel_mode` | | `ddp` (기본) \| `model_split` |
| `log_file_dir` | | 로그 경로. 트레이너가 `outputs/` 를 앞에 붙이므로 SAOI 계열 설정의 `../..` 접두는 그것을 상쇄하는 것이다 |
| `modelpath` | ✅ | 체크포인트. **best와 last가 이 한 파일을 공유**한다 |
| `dataset_dir` | train ✅ | 학습 HDF5 |
| `infer_dataset` | infer ✅ | 추론 HDF5 |
| `inference_output_dir` | | 기본 `outputs/rollout` |
| `split_seed` | 권장 | 결정론적 80/10/10 분할 시드 |

---

## 2. 데이터 계약

`nodal_data`는 `[num_features, num_timesteps, num_nodes]`, 행 `0:3`은 참조 좌표.

```
행 [3 : 3+input_var]              state      : 입력이자 출력
행 [3+input_var : ... +cond_var]  conditions : 입력 전용 (모델이 읽되 예측하지 않음)
```

| 키 | 설명 |
|---|---|
| `input_var` ✅ | state 블록 폭. **T=1이면 이 블록은 항등적으로 0** — 그래도 선언해야 한다. T>1에서 직전 물리 상태를 나르는 자리이고, 정적/동적을 같은 코드로 돌리는 근거다 |
| `output_var` ✅ | flow가 생성하는 채널 수. 노드 특징에 `y_t`로 다시 들어간다 |
| `cond_var` | 입력 전용 행 수 (두께, part no, 유량 등). 기본 0 |
| `edge_var` ✅ | **반드시 8** (deformed dx/dy/dz/dist + reference 4개). 다르면 즉시 raise |
| `positional_features` | 회전 불변 노드 특징 수 (centroid_dist, mean_edge_len 등). 보통 4 |
| `use_node_types` / `num_node_types` | 노드 타입 one-hot. 타입 행은 **마지막 행**이다 |
| `feature_loss_weights` | 채널별 가중치. 합이 1로 정규화된다 |

**최종 노드 특징 레이아웃** (여기가 변분 트리와 다른 유일한 지점):

```
[ state (input_var) | y_t (output_var) | conditions | positional | node-type one-hot ]
                      ^^^^^^^^^^^^^^^^
                      ODE 현재 상태 — 모델이 내부에서 끼워 넣는다
```

---

## 3. 네트워크 · 학습

| 키 | 기본 | 설명 |
|---|---|---|
| `latent_dim` ✅ | | Encoder/Processor/Decoder 폭 |
| `message_passing_num` | | **`use_multiscale True`면 무시된다** |
| `training_epochs` ✅ | | §7 참조 — FM은 결정론 회귀보다 많이 필요하고, 배수는 아직 미측정 |
| `batch_size` ✅ | | DDP에서는 **rank당** 값 |
| `learningr` ✅ | | `0.0001` 형식으로 (`1e-4`는 문자열로 파싱됨) |
| `num_workers` | | 동시 실행 arm 수 × 이 값이 프로세스 총량 |
| `grad_accum_steps` | 1 | 0 = epoch당 1스텝 |
| `std_noise` | | **정적 데이터에서는 0을 권장.** MGN 입력 노이즈는 AR 롤아웃 안정화용인데 T=1 + `ar_ot`에는 롤아웃이 없다. 게다가 T=1에서 state 블록은 0이라 노이즈가 죽은 채널에 들어가고, 동시에 `edge_attr`(유일한 실제 신호인 기하)에 얹힌다. flow는 이미 잡음 섞인 입력으로 학습한다 |
| `augment_geometry` | | 학습 전용 랜덤 Z-회전 + 반사 |
| `weight_decay`, `warmup_epochs` | | 옵티마이저 |
| `use_amp` | True | bfloat16 |
| `use_checkpointing` | | gradient checkpointing (AdaLN 변조도 체크포인트 구간 안에서 재계산) |
| `use_ema` / `ema_decay` | | 검증·추론은 EMA 가중치를 쓴다 |
| `use_compile` | False | |

---

## 4. 멀티스케일 (사실상 필수)

| 키 | 설명 |
|---|---|
| `use_multiscale` | **True 권장.** 끄면 런처가 `FLOW-FLAT` 경고 |
| `coarsening_type` | `voronoi_seedmean` (프로덕션) |
| `voronoi_clusters` | 레벨별 클러스터 수, 예 `1000, 100` |
| `multiscale_levels` | 조대화 레벨 수 |
| `mp_per_level` | `2*levels+1` 개. 예 `4, 6, 8, 6, 4` = 하강arm, 최저, 상승arm |
| `hierarchy_variants` | 캐시할 독립 시드 분할 수. epoch마다 회전시켜 분할 불변성을 학습 |
| `hierarchy_seed` | **추론 시 분할 고정** (§6 참조) |
| `hierarchy_cache_dir` / `_keep` / `_build_workers` | 캐시 관리. 여러 arm이 캐시를 공유하면 `_keep True` 필수 |

> **왜 flow에서 더 중요한가**: `t≈0`에서 네트워크 입력은 백색잡음 + 기하뿐이다.
> 여기서 전역 구조를 복원하려면 denoiser가 부품 전체를 봐야 하고, flat 스택은
> 수용영역 안에서 그걸 못 한다. 최저 레벨(예: 100 클러스터)이 그 역할을 한다.

---

## 5. Flow matching — 이 방법 고유의 키

### 5.1 샘플링 시점 선택 (재학습 불필요)

| 키 | 기본 | 설명 |
|---|---|---|
| `flow_steps` | 30 | ODE 적분 스텝 K |
| `flow_solver` | `heun` | `heun`(2차, 스텝당 2 eval) \| `euler`(1차, 1 eval) |

**이 둘은 학습이 끝난 뒤 자유롭게 바꿀 수 있다.** 학습이 만드는 것은 연속 함수
`v(y_t, t, g)`이고 K는 그 함수의 수치 적분 해상도일 뿐이다. 모델 오차는 고정 구간
`t∈[0,1]`에서 적분되므로 `∫₀¹ ε dt ≈ ε` — **K배로 쌓이지 않는다.** K에 의존하는 것은
이산화 오차 `O(1/K)`(Heun이면 `O(1/K²)`)뿐이다.

체크포인트가 이 두 값을 기록하지만 **추론에서는 config가 이긴다**
(`SAMPLING_TIME_KEYS`, `rollout.py`). 로그에 이렇게 찍힌다:

```
flow_steps: 12 (config wins; checkpoint recorded 20)
```

### 5.2 아키텍처 결정 (체크포인트가 이김)

| 키 | 기본 | 설명 |
|---|---|---|
| `flow_time_freqs` | 16 | 시간 임베딩의 Fourier 옥타브 수. **AdaLN 입력 폭 = 2×이 값** → 체크포인트가 이 값에 묶인다. 바꾸면 `load_state_dict` 실패 |

### 5.3 학습 시점 선택

| 키 | 기본 | 설명 |
|---|---|---|
| `flow_t_sampling` | `uniform` | `t` 스케줄. `logitnormal`은 경로 중간에 예산을 집중 — 거기가 속도장이 가장 어렵다 (t=0은 거의 순수 잡음, t=1은 거의 정답이라 양 끝이 상대적으로 쉽다). SD3가 rectified flow에 채택한 방식. **최적점을 바꾸지 않고 수렴 속도만 바꾸므로 두 설정은 직접 비교 가능** |
| `flow_t_logit_scale` | 1.0 | `logitnormal`의 폭. 클수록 균등에 가까워진다 |

### 5.4 검증

| 키 | 기본 | 설명 |
|---|---|---|
| `val_flow_steps` | `flow_steps` | 검증용 저해상도 적분. 검증은 `val_interval`마다 ODE를 돌리므로 여기를 낮추면 직접 비용이 준다 |
| `val_num_samples` | 8 | 검증 CRPS 앙상블 크기. **추정량은 어떤 S에서도 불편(unbiased)** — S는 분산만 줄인다. 잡음 하한은 검증 **그래프 수**가 정하지 S가 정하지 않는다 |
| `best_by` | `crps` | `recon`(1스텝 속도 회귀 loss) \| `crps`(샘플링 앙상블 점수). **추론이 하는 일과 같은 것을 재는 쪽은 `crps`다** |

### 5.5 추론

| 키 | 설명 |
|---|---|
| `num_vae_samples` | scene당 draw 수. **비용 = draw × K × (heun이면 2)**. 20,000 forward를 넘으면 런처가 `FLOW-COST` 경고 |
| `vae_batch_size` | 한 배치에 묶는 draw 수. `auto`면 여유 VRAM 기준 자동 |
| `save_rollouts` | False면 HDF5를 안 쓰고 스프레드 히스토그램만. 다중 draw 분포 연구를 디스크 예산 안에서 하게 해준다 |
| `make_histogram` / `histogram_bins` / `histogram_clip_quantile` | 스프레드 히스토그램 |
| `infer_timesteps` | 롤아웃 스텝 수. T=1 학습 체크포인트는 1로 클램프된다 |

---

## 6. FM에서만 생기는 제약 — 계층 고정

**ODE 적분 내내, 그리고 같은 지오메트리의 draw 사이에서 조대화 분할을 고정해야 한다.**

스텝마다 Voronoi 분할을 새로 만들면 각 스텝이 서로 다른 벡터장이 되어, ODE가
불연속 객체를 적분하게 되고 결과가 무의미해진다. draw끼리 분할이 다르면 분할
차이가 만든 변동이 물리적 스프레드로 잘못 섞인다.

`rollout.py`가 그래프를 적분 진입 전 한 번만 만들고, `hierarchy_seed`가 분할을
재현 가능하게 고정한다. 로그에서 확인:

```
Coarsening hierarchy: 1 partition(s), seeds=[1234] (fixed)
```

forward가 1회뿐이던 모델에서는 존재할 수 없던 위험이라, 기존 트리의 경험이
여기서는 통하지 않는다.

---

## 7. 학습 예산 — 유일하게 미측정인 값

FM의 회귀 목표 `y − z₀`에는 환원 불가능한 잡음이 섞여 있고, 그 크기는 정확히
**진짜 조건부 불확실성 `Var(y|g)`** 다. gradient SNR이 낮아진 만큼 step이 더 든다.

> 추가 학습량은 **분포가 얼마나 넓은가**에 비례하지, **적분을 몇 번 하는가**에
> 비례하지 않는다.

문헌 기준 추정은 결정론 회귀 대비 **3~10배 step**이지만, 이 데이터에서 측정된 적이
없다. `training_epochs`를 크게 잡기 전에 §8의 Wave 0을 돌려라.

스텝당 비용은 반대로 싸다 (128²블록 등가):

```
결정론 MGN     8
MeshGraphNets-V 33   = 시뮬레이터 8 + posterior encoder 5 + prior trunk 20
cHI-MGNflow     8    ← 보조 네트워크가 없다
```

---

## 8. 제거된 키

아래는 **코드에 존재하지 않는다.** config에 남아 있으면 런처가 `FLOW-REMOVED`
경고(`--strict`에서는 에러)를 낸다.

```
use_vae  vae_latent_dim  vae_mp_layers  vae_graph_aware  posterior_min_std
num_z  z_conditioning  mmd_bandwidth  mmd_gather_ranks  lambda_mmd  beta_aux
alpha_recon  recon_loss  vae_valid_prior_samples
prior_type  use_conditional_prior  prior_family  prior_nll_weight  prior_fm_steps
prior_fm_solver  prior_mp_layers  prior_hidden_dim  prior_temperature
prior_kl_reg_weight  prior_cov_rank  prior_min_std  prior_mixture_components
prior_grad_to_encoder
gamma_es  es_samples  es_steps  es_noise_source  es_start_epoch
```

**MeshGraphNets-V 설정을 복사해 올 때 반드시 이 블록을 지워라.** 지우지 않으면
경고만 뜨고 조용히 무시되므로, 의도한 설정으로 돌고 있다고 착각하기 쉽다.

---

## 9. 로그 읽는 법

```
Epoch 120/6000 LR: 1.00e-04 | Train fm=3.21e-01 | Valid fm=3.44e-01 | CRPS 8.7e-02 spread 0.31
  [FlowDiag] crps=8.7e-02  1-draw mse=1.9e-01  spread/gt=0.31  (steps=12, S=8)
```

| 항목 | 의미 | 병리 신호 |
|---|---|---|
| `Train fm` / `Valid fm` | 1스텝 속도 회귀 MSE | 학습이 되고 있는지만 알려준다. 샘플 품질과는 별개 |
| `CRPS` | 앙상블 전체 점수. `best_by crps`가 선택하는 값 | 이것이 제품 지표다 |
| `spread/gt` | 멤버 간 표준편차 ÷ 정답 표준편차 | **→ 0이면 노이즈 채널 무시(붕괴).** `[FlowDiag] WARNING` 발생. 1.0 근처가 건강 |
| `1-draw mse` | 샘플 **한 장**의 오차 | 결정론 회귀보다 나쁜 게 **정상**. 보정된 앙상블 멤버는 평균 위에 앉으라고 만든 게 아니다 |

실측 예 (ex1, 4 epoch, 25k 노드):

```
Epoch 0: Train fm=3.84  Valid fm=3.05  CRPS 7.63e-01  spread 1.550
Epoch 2: Train fm=1.63  Valid fm=1.77  CRPS 4.96e-01  spread 1.189
Epoch 3: Train fm=1.34  Valid fm=1.45  CRPS 4.73e-01  spread 1.030
```

spread가 1.0으로 수렴하는 것이 보정이 잡혀가는 신호다.

---

## 10. 진단 코드

| 코드 | 심각도 | 뜻 |
|---|---|---|
| `FLOW-REMOVED` | WARNING (strict: ERROR) | 변분 트리의 잠재/prior 키가 남아 있다 |
| `FLOW-POSITIVE` | ERROR | `flow_steps` / `val_flow_steps` / `val_num_samples` / `flow_time_freqs`가 1 미만 |
| `FLOW-SOLVER` | ERROR | `flow_solver`가 `heun`/`euler`가 아님 |
| `FLOW-TSAMPLING` | ERROR | `flow_t_sampling`이 `uniform`/`logitnormal`이 아님 |
| `FLOW-TSCALE` | ERROR | `flow_t_logit_scale` ≤ 0 |
| `FLOW-BESTBY` | ERROR | `best_by`가 `recon`/`crps`가 아님 |
| `FLOW-COST` | WARNING | 추론 forward 총량이 20,000 초과 |
| `FLOW-FLAT` | WARNING | `use_multiscale`이 꺼져 있다 |

공통 코드(`PATH-*`, `ENV-*`, `NATIVE-CHECK-*`, `CFG-*`)는 모노레포 규약을 따른다.
