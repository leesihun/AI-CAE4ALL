# cHI-MGNflow — SAOI 파라메트릭 스윕 설계

MeshGraphNets-V의 SAOI wave-3(2⁴ 16 arm, GPU당 2개, `gen_sweep_configs.py`)를
참고 골격으로 삼되, **한 가지가 구조적으로 다르다.**

---

## 0. 왜 wave-3를 그대로 옮기면 안 되는가

wave-3의 네 축은 `z_conditioning` / `prior_grad_to_encoder` / `vae_latent_dim` /
`lambda_mmd` — **전부 posterior·prior 서브시스템 안쪽**이고, 이 방법에는 그
서브시스템이 존재하지 않는다. 옮길 축이 하나도 없다.

그리고 더 중요한 차이:

> **wave-3는 모든 축이 full training run을 소모했다.
> 여기서는 축의 절반이 학습을 전혀 소모하지 않는다.**

`flow_steps`(K)와 `flow_solver`는 **샘플링 시점 선택**이다. 학습이 만드는 것은
연속 함수 `v(y_t, t, g)`이고, K는 그것을 나중에 몇 점에서 평가할지 정할 뿐이다.
같은 체크포인트로 K=4든 K=100이든 돌아간다.

**이 두 축을 학습 슬롯에 넣으면 안 된다.** 16 arm 예산이 통째로 낭비된다.

---

## 1. 세 파도

```
Wave 0   배수 측정        2 arm      ~1일       ← 나머지 전부의 크기를 정한다
Wave A   무료 축 (K·solver)  0 arm    ~1시간     ← Wave 0 체크포인트 위에서 추론만
Wave B   학습 축          16 arm     예산에 따라
```

### Wave 0 — `training_epochs`를 얼마로 잡을 것인가 (가장 먼저)

```
configs/HI_MGNFlow/wave0/config_wave0_det.txt     결정론 HI-MGN
configs/HI_MGNFlow/wave0/config_wave0_flow.txt    cHI-MGNflow
                                                   ← 백본·데이터·예산 동일

python AI_CAE4ALL_main.py --config .../config_wave0_det.txt   > det.log
python AI_CAE4ALL_main.py --config .../config_wave0_flow.txt  > flow.log
python methods/HI_MGNFlow/misc/wave0_report.py --det det.log --flow flow.log
python methods/HI_MGNFlow/misc/eval_prediction_modes.py --config .../config_wave0_flow.txt
```

FM은 결정론 회귀보다 step이 더 든다. 목표 `y − z₀`에 환원 불가능한 잡음이 섞여
있고, 그 크기가 정확히 `Var(y|g)`이기 때문이다. **배수는 3~10배로 추정되지만
이 데이터에서 측정된 적이 없다.**

```
arm  det   :  MeshGraphNets (결정론) — 같은 백본, 같은 데이터, 같은 batch/lr
arm  flow  :  cHI-MGNflow          — 위와 동일 + flow matching
```

학습 데이터의 20%만 쓰고, 두 loss 곡선이 평탄해지는 epoch을 비교한다.
비율이 Wave B의 `training_epochs`를 정한다.

> 이걸 건너뛰고 wave-3처럼 `training_epochs 2000`을 그대로 쓰면, 모든 arm이
> 미수렴 상태에서 비교되어 **순위가 수렴 속도의 함수가 되어버린다.**

부수 산출물: `det` 체크포인트는 Wave B의 **사전학습 초기값**으로 재사용할 수 있다
(AdaLN-Zero가 초기에 정확히 항등이므로, 결정론 가중치에서 이어받아도 첫 스텝의
성능 손실이 0이다). 예상 수렴 가속 중 가장 큰 항목.

### Wave A — 학습을 전혀 소모하지 않는 축

Wave 0의 `flow` 체크포인트 하나 위에서:

```
flow_steps   ∈ {4, 8, 12, 20, 30, 50}
flow_solver  ∈ {heun, euler}
                                        = 12개 조합, 전부 추론만
```

각 조합으로 검증 split을 샘플링해 CRPS를 잰다. 얻는 것:

- **프로덕션 K의 하한.** CRPS가 평탄해지는 지점 아래로는 이산화 오차가 지배한다
- Heun이 Euler 대비 몇 스텝을 절약하는지 (스텝당 2배 비용을 상쇄하는가)
- Wave B의 `val_flow_steps`를 정할 근거 — 검증이 매 `val_interval`마다 ODE를
  돌리므로 여기가 곧 Wave B의 벽시계에 직접 들어간다

**소요: 12 × (검증 그래프 수 × K × 2) forward.** 몇십 분.

### Wave B — 학습을 소모하는 축 (2⁴ = 16 arm)

| 축 | 수준 | 왜 이 축인가 |
|---|---|---|
| `batch_size` | 16 \| 32 | **gradient 분산이 FM의 실제 비용 동인**이다. 스텝당 forward 비용이 33 → 8로 떨어졌으므로 같은 VRAM에서 더 큰 배치가 들어간다. 그 여유를 분산 감소에 쓰는 것이 가장 직접적인 대응 |
| `flow_t_sampling` | uniform \| logitnormal | 예산을 경로 중간(속도장이 가장 어려운 구간)에 집중. **최적점을 바꾸지 않고 수렴 속도만 바꾸므로 두 수준이 직접 비교 가능하다.** SD3가 rectified flow에 채택 |
| `voronoi_clusters` | 1000,100 \| 2000,250 | **FM 고유의 가설.** `t≈0`에서 입력은 백색잡음 + 기하뿐이고, 전역 구조 복원은 최저 레벨이 담당한다. 결정론 회귀보다 이 축이 더 중요할 것이라는 예측을 검증 |
| `learningr` | 0.0001 \| 0.0003 | 목표가 더 시끄러워졌으므로 최적 LR이 이동했을 수 있다. batch와 짝지어 봐야 하는 고전적 쌍 |

**의미 있는 2-way 상호작용**(full factorial이 one-factor-at-a-time 대비 사주는 것):

- `batch × t_sampling` — 둘 다 gradient 분산을 다른 각도에서 친다. 더해지나 겹치나?
- `batch × lr` — 고전적 결합
- `clusters × t_sampling` — logitnormal이 중간 t에 집중하는데, 전역 구조가 형성되는
  구간이 바로 거기다. 최저 레벨의 중요도가 커져야 한다
- `clusters × batch` — VRAM 예산의 두 소비처

**GPU 배치**: arm 인덱스 `i = 8·B + 4·T + 2·C + L`을 `min(i, 15-i)`번 GPU에.
**비트 반전 짝**이 한 GPU를 공유하므로 어느 GPU도 특정 수준에 치우치지 않고,
각 카드가 batch 16 하나 + 32 하나를 받아 VRAM이 균형 잡힌다 (wave-3와 동일 논리).

> ⚠️ 첫 arm의 `VRAM peak=` 줄을 반드시 확인할 것. batch 32 + `voronoi 2000,250`이
> 최악 조합이다. 안 맞으면 batch 수준을 {8, 16}으로 낮추고 재생성하면 축은 전부 산다.

---

## 2. 스윕에 넣지 말아야 할 것

| | 이유 |
|---|---|
| `flow_steps`, `flow_solver` | **Wave A에서 공짜로 얻는다.** 학습 슬롯 낭비 |
| `flow_time_freqs` | 아키텍처 결정이라 재학습이 필요하지만, 효과가 거의 없을 것으로 예상. 3순위 |
| `best_by` | 축이 아니라 선택 규칙. 전 arm `crps` 고정 |
| `val_num_samples` | CRPS 추정량은 어떤 S에서도 불편. S는 분산만 줄인다. 전 arm 동일하게 고정해야 arm 간 비교가 성립 |
| `use_multiscale` | 끄면 `t≈0`에서 전역 구조를 못 만든다. 축이 아니라 요구사항 |

---

## 3. 채점

wave-3의 `score_sweep.py`가 겪은 함정을 그대로 물려받지 않도록:

| 함정 | 대응 |
|---|---|
| `log_file_dir`이 트레이너에서 `outputs/` 접두를 받는다 | 스코어러가 **양쪽 경로를 다 본다** |
| `[FlowDiag]`가 `tqdm.write` → **stdout에만** 찍히고 로그 파일에는 안 남는다 | `run_sweep.sh`의 per-arm transcript도 `--run-logs`로 함께 읽는다 |
| epoch 줄에 CRPS가 없을 때 정규식 `m.group(1)`이 AttributeError | `m is None`을 먼저 검사 |
| best/last가 한 `modelpath`를 공유 | 이미 수정됨 — best만 유지, 저장 메시지가 선택 지표를 찍는다 |

**arm당 읽을 값**: `best CRPS`, 그때의 `spread/gt`, `1-draw mse`, `Valid fm`,
`도달 epoch`, `VRAM peak`, 벽시계.

**순위는 `CRPS`로 매기되 `spread/gt`를 함께 본다.** CRPS가 좋은데 spread가 0에
가까우면 붕괴한 것이고, 그 arm은 이겨도 쓸 수 없다.

---

## 4. 순서 요약

```
1.  Wave 0   det vs flow, 데이터 20%          → training_epochs 배수 확정
                                              → det 체크포인트 = 사전학습 초기값
2.  Wave A   K × solver, 체크포인트 1개        → 프로덕션 K, val_flow_steps 확정
3.  Wave B   2⁴ 16 arm, 확정된 예산으로        → batch / t_sampling / clusters / lr
4.  결선     승자 셀 + Wave A의 K로 프로덕션 재학습 (전체 데이터, 4 GPU DDP)
```

Wave 0과 A를 먼저 하지 않으면 Wave B의 16 arm이 **잘못된 epoch 예산**과
**임의의 K**로 돌게 되고, 그 결과 순위는 재현되지 않는다.
