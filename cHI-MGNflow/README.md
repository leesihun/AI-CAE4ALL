# cHI-MGNflow

Conditional flow matching on mesh fields. The HI-MGN V-cycle, re-read as a
velocity field instead of a decoder.

```
학습:   z0 ~ N(0,I)                       노이즈 필드 [N, output_var]
        y_t = (1-t)*z0 + t*y              경로 위 한 점 (닫힌 식, 순간이동)
        loss = MSE( v(y_t, t, g), y-z0 )  항 하나. forward 1회.

추론:   y ~ N(0,I) 에서 시작해 ODE를 K스텝 적분 -> 필드 한 장
        새 노이즈 = 새 샘플
```

---

## 왜 갈라져 나왔나

`MeshGraphNets - variational`의 생성 경로에는 구조적 결함이 있다: 학습은 정답을
요약한 posterior latent `q(z|y,g)`를 디코더에 먹이고, 추론은 기하만 보고 추측한
`p(z|g)`를 먹인다. **디코더는 non-posterior z를 추론에서 처음 만난다.** objective의
네 항(recon / MMD / aux / fm_loss) 중 어느 것도 prior에서 뽑은 z를 필드 공간에서
채점하지 않으므로, 이 간극은 학습 중에 관측조차 되지 않는다.

여기서는 학습과 추론이 **같은 분포**에서 출발한다: 양쪽 다 `N(0,I)`. 간극이 존재할
수 없다.

| | MeshGraphNets-V | cHI-MGNflow |
|---|---|---|
| 무작위성의 출처 | posterior(학습) / prior(추론) — **불일치** | `N(0,I)` 양쪽 동일 |
| 무작위성의 차원 | `num_z x vae_latent_dim` (예: 48) | `[N, output_var]` (출력과 동일) |
| 표현 가능한 샘플 | 48차원 다양체 위 | **제약 없음** (국소 변형 포함) |
| loss 항 수 | 4 | **1** |
| 노드별 감독 | 집계·대리 지표 | **정확한 회귀 목표, 분산 0** |
| 보조 네트워크 | posterior encoder + FM prior trunk | 없음 |
| 스텝당 forward 비용 | 33 (128²블록 등가) | **8** |
| 샘플 1개 추론 비용 | 8 | `K x 2` (Heun) — 유일한 실질 비용 |

---

## 구조

백본은 `MeshGraphNets - variational`과 **동일**하다. 두 가지만 다르다.

1. **노드 특징에 ODE 상태가 실린다.**
   `[ state | y_t | conditions | positional | node-type one-hot ]`
   `y_t`는 `output_var` 채널. `state` 블록은 T=1에서 항등적으로 0이고, T>1에서는
   직전 물리 상태를 나른다 — 정적/동적이 분기 없이 같은 코드로 돌아간다.

2. **AdaLN-Zero의 입력이 잠재 벡터가 아니라 시간 임베딩이다.**
   `t -> Fourier(flow_time_freqs) -> [2*freqs] -> 28개 AdaLN 헤드`
   모듈(`model/blocks.py::AdaLNZero`), 적용 방식, zero-init 전부 그대로.

Encoder, 28개 GnBlock, pool / unpool / skip merge, Decoder, coarsening 캐시는
한 줄도 바뀌지 않았다.

### 파일

| 경로 | 역할 |
|---|---|
| `model/flow.py` | 경로 구성, 시간 임베딩, ODE 적분기, config 해석 |
| `model/CHiMGNFlow.py` | 속도 네트워크 `v(y_t, t, g)` — V-cycle + 시간 조건화 |
| `training_profiles/training_loop.py` | FM loss, 샘플링 검증(CRPS/spread), 주기적 시각화 |
| `inference_profiles/rollout.py` | ODE 적분 기반 샘플링 |
| `tests/test_flow_smoke.py` | 합성 데이터 end-to-end 스모크 테스트 (CPU, 수십 초) |

`model/vae.py`와 `model/conditional_prior.py`는 **존재하지 않는다.**

---

## 예측 모드 — 하나의 체크포인트, 세 가지 사용법

이 방법은 확률적 샘플러만이 아니다. **같은 체크포인트에서 결정론 예측이 공짜로 나온다.**

| 모드 | 계산 | forward | 결과 |
|---|---|---|---|
| **mean-1step** | `z₀ + v(z₀, 0)` | **1** | `E[y\|g]` — **해석적으로 정확** |
| mean-ens-M | M개 draw의 평균 | `M·K·2` | `E[y\|g]` — 추정 |
| draw-1 | 1개 draw | `K·2` | **샘플** 하나 |

`mean-1step`이 왜 정확한가: `t=0`에서 경로 위 점은 노이즈 필드 그 자체(`y₀ = z₀`)이므로,
그것으로 조건화하면 `z₀`가 알려진 값이 되고 회귀 최적해는

```
v*(z₀, 0, g) = E[y₁ − s·z₀ | z₀, g] = E[y₁|g] − s·z₀        (y₁ ⊥ z₀)
```

`dt=1` Euler 한 스텝을 밟으면

```
z₀ + v*(z₀, 0, g) = E[y₁|g] + (1−s)·z₀ = E[y₁|g] + σ_min·z₀     σ_min = 1e-4
```

**즉 forward 1회로 조건부 평균이 나온다 — 결정론 모델과 정확히 같은 추론 비용.**

단서 두 가지: ⑴ 이 모드는 `t=0` 슬라이스의 정확도에만 의존하는데, 학습 예산은 모든 `t`에
분산돼 있었다 (결정론 모델은 그 한 질의에 전부를 쓴다). ⑵ 앙상블 평균은 같은 양을 다르게
추정하며, `t=0`이 덜 학습됐으면 그쪽이 이길 수 있다. 둘 다
`misc/eval_prediction_modes.py`가 실측한다.

---

## 실행

```bash
# 런처 경유 (권장) — 설정 검증 후 자동 실행
python AI_CAE4ALL_main.py --config configs/cHI-MGNflow/SAOI_all_input/config_train_bot.txt --check
python AI_CAE4ALL_main.py --config configs/cHI-MGNflow/SAOI_all_input/config_train_bot.txt

# 직접 실행
cd cHI-MGNflow && python CHiMGNFlow_main.py --config <path>

# 스모크 테스트
cd cHI-MGNflow && python tests/test_flow_smoke.py
```

### 분석 도구

| 스크립트 | 용도 |
|---|---|
| `misc/eval_prediction_modes.py` | 세 예측 모드를 한 split에서 비교 (MSE / R² / corr / 비용) |
| `misc/wave_a_sweep.py` | K × solver 스윕. **학습 비용 0** — 체크포인트 하나 위에서 |
| `misc/wave0_report.py` | 결정론 vs flow 학습 곡선 → `training_epochs` 배수 |
| `misc/field_intrinsic_dim.py` | 필드의 내재 차원 (다항식 R², PCA rank) |

---

## 반드시 알아야 할 세 가지

### ① `flow_steps`(K)는 학습 비용에 곱해지지 않는다

학습은 ODE를 적분하지 않는다. 경로가 닫힌 식이라 무작위 `t` 한 점으로 순간이동해
그 지점의 속도만 묻는다 — forward 1회, 연쇄 없음.

그리고 **K는 학습이 끝난 뒤에 정한다.** 같은 체크포인트로 K=8이든 K=100이든
샘플링할 수 있다. 학습이 만드는 것은 연속 함수 `v(y_t, t, g)`이고, K는 그 함수를
나중에 몇 점에서 평가할지 정하는 수치 적분 해상도일 뿐이다.

모델 오차도 K배로 쌓이지 않는다. 적분 구간이 `t∈[0,1]`로 고정이라 오차는
`∫₀¹ ε dt ≈ ε`로 누적된다. K에 의존하는 것은 이산화 오차 `O(1/K)`
(Heun이면 `O(1/K²)`) 뿐이고, 그건 적분기 정확도이지 모델 정확도가 아니다.

### ② 학습 step은 더 필요하다 — 하지만 그 배수는 K가 아니다

회귀 목표 `y - z0`에 환원 불가능한 잡음이 섞여 있고, 그 크기는 정확히
**진짜 조건부 불확실성 `Var(y|g)`** 다. gradient SNR이 낮아진 만큼 step이 더 든다.

> 추가 학습량은 **분포가 얼마나 넓은가**에 비례하지, 적분을 몇 번 하는가에
> 비례하지 않는다. 스프레드 모델링의 정직한 가격표다.

배수는 아직 측정된 적이 없다. **착수 전에 재라**: 학습 데이터의 20%로 결정론
회귀와 FM을 각각 돌려 loss가 평탄해지는 epoch을 비교하면 반나절에 나온다.

### ③ 계층은 적분 내내, 그리고 샘플 간에 고정해야 한다

이 리포에 없던 새 제약이다. ODE 스텝마다 Voronoi 분할을 새로 만들면 각 스텝이
**서로 다른 벡터장**이 되어, 적분 결과가 무의미해진다. 같은 지오메트리의 서로 다른
draw끼리도 계층을 같게 해야 한다 — 안 그러면 분할 차이가 만든 변동이 물리적
스프레드로 잘못 섞인다.

`inference_profiles/rollout.py`는 그래프를 스텝 진입 전에 한 번만 만들고,
`hierarchy_seed`가 분할을 재현 가능하게 고정한다. forward가 1회뿐이던 모델에서는
이 위험이 존재하지 않았다.

---

## Config 키

| 키 | 기본값 | 성격 |
|---|---|---|
| `flow_steps` | 30 | **샘플링 시점 선택.** 재학습 없이 변경 가능 |
| `flow_solver` | `heun` | `heun`(2차, 2 eval/step) 또는 `euler` |
| `flow_time_freqs` | 16 | **아키텍처 결정.** AdaLN 입력 폭 → 체크포인트가 이 값에 묶인다 |
| `val_flow_steps` | `flow_steps` | 검증용 저해상도 적분 |
| `val_num_samples` | 8 | 검증 CRPS 앙상블 크기. 추정량은 어떤 S에서도 불편(unbiased); S는 분산만 줄인다 |
| `best_by` | `crps` | `recon`(1스텝 속도 회귀) 또는 `crps`(추론을 모사하는 샘플링 지표) |
| `num_vae_samples` | 1 | 추론 시 scene당 draw 수. **비용 = draw × K × 2** |

`use_vae` / `vae_*` / `lambda_mmd` / `beta_aux` / `alpha_recon` / `posterior_min_std` /
`num_z` / `z_conditioning` / `prior_*` / `recon_loss` 는 **전부 제거되었다.** config에
남아 있으면 런처가 `FLOW-REMOVED` 경고를 낸다 (`--strict`에서는 에러).

---

## 학습 중 읽어야 할 로그

```
Epoch 120/6000 LR: 1.00e-04 | Train fm=3.21e-01 | Valid fm=3.44e-01 | CRPS 8.7e-02 spread 0.31
  [FlowDiag] crps=8.7e-02  1-draw mse=1.9e-01  spread/gt=0.31  (steps=12, S=8)
```

- `spread/gt → 0` : 모델이 노이즈 채널을 무시하는 중 (붕괴). `[FlowDiag] WARNING`이 뜬다
- `1-draw mse` : 샘플 **한 장**의 오차. 결정론 회귀보다 나쁜 게 정상이다 —
  보정된 앙상블 멤버는 평균 위에 앉으라고 만든 것이 아니다
- `CRPS` : `best_by crps`가 선택하는 값. 앙상블 전체의 품질

---

## 검증 현황

**실제 GPU 학습 + 추론까지 돌려서 확인했다** (RTX 2080 SUPER, `dataset/ex1.h5`,
100 샘플 × 25k 노드, T=1 정적 — SAOI와 같은 데이터 계약).

```
학습 (4 epoch, V-cycle 2레벨, AMP, EMA, gradient checkpointing, 계층 캐시)
  Epoch 0: Train fm=3.84  Valid fm=3.05  CRPS 7.63e-01  spread 1.550
  Epoch 2: Train fm=1.63  Valid fm=1.77  CRPS 4.96e-01  spread 1.189
  Epoch 3: Train fm=1.34  Valid fm=1.45  CRPS 4.73e-01  spread 1.030
  VRAM peak 1.55GB / reserved 2.16GB       best_by crps 로 체크포인트 선택됨

추론 (42k 노드, 6 draw, K=12 Heun)
  Coarsening hierarchy: 1 partition(s), seeds=[1234] (fixed)   <- FM 필수 조건 충족
  flow_steps: 12 (config wins; checkpoint recorded 20)         <- K는 샘플링 시점 선택
  6개 draw가 실제로 서로 다름: draw간 표준편차 / 필드 스케일 = 0.57
```

`spread/gt`가 1.55 → 1.03으로 수렴하는 것이 보정이 잡혀가는 신호다.

- ✅ 실제 GPU 학습 · 추론 · 롤아웃 저장
- ✅ `tests/test_flow_smoke.py` — 경로 구성, AdaLN zero-init(가중치 0 / gate 1),
  실제 dataloader 배치 학습, 비영 앙상블 스프레드, 같은 체크포인트의 K=2/4/6/12 적분
- ✅ 런처 등록 — `--list-models`, `--describe chi-mgnflow`
- ✅ preflight `--check --strict` **0 errors / 0 warnings / 0 notices**
  (네이티브 프로브 + 데이터셋 프로브 포함), Wave-B 스윕 arm 포함
- ⚠️ **DDP 멀티 GPU 경로는 미검증** — 이 머신에 GPU가 1장뿐이다.
  단일 GPU 경로만 실행으로 확인됐다

### Wave 0 실측 — HI-MGN 대비 (ex1, 100샘플, 동일 백본·데이터·예산 24 epoch)

두 체크포인트를 **동일한 지표 코드로, 동일한 val split(10 그래프)에서** 채점했다
(`misc/eval_det_baseline.py` + `misc/eval_prediction_modes.py`).

| 모드 | forward | MSE | R² | corr | params |
|---|---|---|---|---|---|
| **HI-MGN (결정론)** | 1 | 1.0667e+00 | −0.790 | **0.2095** | 878,724 |
| flow `mean-1step` | **1** | **1.0232e+00** | **−0.699** | 0.0612 | 1,005,700 |
| flow `mean-ens-8` | 320 | **1.0035e+00** | **−0.637** | 0.1390 | |
| flow `draw-1` | 40 | 1.1020e+00 | −0.840 | 0.1162 | |

**읽는 법 — 판정은 "동등"이지 "우월"이 아니다.**

- **MSE / R²**: flow의 1-forward 결정론 모드가 HI-MGN을 4% 앞선다. **같은 추론 비용**에서다
- **corr**: HI-MGN이 3배 앞선다. MSE와 상관계수는 다른 것을 재는데, 평균 쪽으로 더
  수축한 예측기는 MSE가 낮으면서 상관은 낮아진다 — 미학습 조건부 평균 추정기의
  전형적 서명이다
- **둘 다 R² < 0** = 두 모델 모두 split 평균보다 못하다. **24 epoch은 턱없이 부족하다.**
  이 표는 "동일 예산에서 flow의 결정론 모드가 HI-MGN에 밀리지 않는다"까지만 말한다

### Wave 0 — epoch 배수

| 자기 개선량의 X%에 도달한 epoch | det | flow | 비율 |
|---|---|---|---|
| 50% | 2 | 2 | 1.00× |
| 80% | 2 | 2 | 1.00× |
| 90% | 6 | 4 | **0.67×** |

**flow가 결정론보다 더 오래 걸리지 않았다** — 예상(3~10배)과 반대다. 다만 24 epoch에서
어느 쪽도 평탄해지지 않았으므로 이건 "초반 속도"를 잰 것이지 "필요 시간"이 아니다.
**3~5배 예산으로 재측정한 뒤에 Wave B를 확정할 것.**

### Wave A 실측 — K가 클수록 나빠졌다

| solver | K | fwd/draw | CRPS | spread |
|---|---|---|---|---|
| heun | **4** | 8 | **4.703e-01** | 0.362 |
| heun | 8 | 16 | 4.763e-01 | 0.378 |
| heun | 20 | 40 | 5.251e-01 | 0.362 |
| euler | **4** | **4** | 4.780e-01 | 0.287 |
| euler | 20 | 20 | 5.109e-01 | 0.341 |

**적분을 더 정밀하게 할수록 CRPS가 나빠진다** — 순진한 기대와 정반대다.
속도장이 아직 제대로 학습되지 않았기 때문이다: K가 작으면 성긴 스텝이 정칙화처럼
작동하고, K가 크면 나쁜 벡터장을 충실히 적분해 그 오차를 정확히 누적한다.

> **이 순서는 모델이 제대로 학습되면 뒤집힐 것이다.** 이산화 오차 논변은 점근적으로
> 큰 K가 최소한 동등하다고 말한다. 그래서 Wave A는 **체크포인트마다 다시** 돌려야
> 하고, 학습을 소모하지 않으므로 그래도 된다.

### 실행 중에 잡은 실제 버그 3개

| | 증상 | 수정 |
|---|---|---|
| 배너 인코딩 | 런처가 stdout을 파이프하면 첫 `print`에서 `UnicodeEncodeError` (cp949) — 학습 시작 전에 죽는다 | 진입점에서 stdout/stderr를 UTF-8로 강제 |
| `flow_steps` 무시 | 체크포인트 `model_config`가 config를 덮어써서, K를 12로 요청해도 20으로 돌았다. **"K는 샘플링 시점 선택"이라는 설계와 정면 충돌** | `SAMPLING_TIME_KEYS`는 config가 이기도록. 로그에 `(config wins; checkpoint recorded 20)` |
| 체크포인트 저장 메시지 | `best_by crps`로 선택하면서 메시지는 `valid_loss`를 찍었다 | 선택에 쓴 지표를 찍도록 (`new best crps=4.73e-01`) |
