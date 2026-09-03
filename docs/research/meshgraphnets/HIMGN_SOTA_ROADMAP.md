# HI-MGN → SOTA 로드맵 (Transolver / AB-UPT 급으로)

작성 2026-08-30. 근거: 이 저장소의 as-built 코드 + 실측 기록 + 2025–2026 문헌 조사.

논문은 이미 나갔다. 이 문서는 **다음 논문**을 위한 것이고, 목표는
"HI-MGN을 조금 더 좋게"가 아니라 **HI-MGN이 이길 수 있는 트랙을 고르고 그 트랙에서
1등을 만드는 것**이다.

---

## 0. 요약 — 결론부터

**HI-MGN이 Transolver/AB-UPT에 밀리는 이유는 "그래프라서"가 아니다.** 다섯 가지
구체적 메커니즘이 빠져 있고, 다섯 개 모두 지금 코드 구조 안에서 구현 가능하다.

| # | 빠진 메커니즘 | 증거 | 예상 효과 | 난이도 |
| --- | --- | --- | --- | --- |
| 1 | **coarse level의 attention** | 전체 FLOP의 96%가 level 0에 있음(§3.1). 전역 결합 담당 층이 compute-starved | 대 (M4GN +32~56%, MGN-T pi-beam 10×) | 중 |
| 2 | **rollout-consistent 목적함수** | 자체 ex9 실측: 1-step val 2.3× 개선 ↔ 19-step rollout 2× 악화 (R² −0.07→−3.59) | 대 — 현재 R²<0을 R²>0.9로 | 하 |
| 3 | **query/mesh 분리 디코더 (neural field)** | AB-UPT ablation: large decoder + anchor attention이 vorticity 37.1%→6.8% | 대 (정확도 + 메모리 + mesh-free 추론) | 중 |
| 4 | **물리 기반 계층** | FPS는 geometry-only. 물리 residual 기반이 rollout RMSE 8.0e-3→6.5e-3 | 중 | 중 |
| 5 | **스케일링 레시피 / 사전학습** | Garnier: 같은 FLOP에서 N∝C^0.75, 마스킹 사전학습으로 MGN 대비 rollout −52% | 중~대 | 하 |

**전략적 결론(§5):** DrivAerNet++ 표면압력(CarBench)에서 AB-UPT를 이기려 하지 마라.
정상상태 geometry→field는 implicit field/transformer의 귀납 편향이 지배하는 판이고
HI-MGN에 구조적 우위가 없다. **과도 응답 + 접촉 + Lagrangian**(crash, deforming plate,
plasticity, EAGLE) 트랙으로 가라. 거기서는 계층적 GNN이 여전히 최전선이고, 경쟁 상대는
M4GN / MGN-T / EvoMesh이며 **셋 중 누구도 anchor attention + 학습형 계층 + rollout
학습을 동시에 갖고 있지 않다.**

---

## 1. 현재 HI-MGN의 정확한 좌표 (as-built)

`model/MeshGraphNets.py::_forward_multiscale` 기준.

```text
Encoder(node MLP, edge MLP, world-edge MLP)
  → [level 0]  GnBlock × 4      (N≈200k, E_dir/N=7.63)   ← skip 0
  → pool (mean / attention)
  → [level 1]  GnBlock × 6      (N=5,000)                ← skip 1
  → pool
  → [coarsest] GnBlock × 8      (N=100)
  → UnpoolBlock → Linear(2d→d) merge skip 1 → GnBlock × 6
  → UnpoolBlock → Linear(2d→d) merge skip 0 → GnBlock × 4
  → Decoder: 노드별 MLP(d → output_var)
```

- **GnBlock** = edge MLP(3d→d) + sum-aggregation node MLP(2d→d) + residual. 표준 MGN.
- 학습형 요소: `AttentionPoolBlock`, `UnpoolBlock`, `voronoi_branches` 멀티 파티션 —
  전부 **transfer operator**(층 사이)이지 **processor**(층 안)가 아니다.
- 손실: `F.mse_loss` (정규화된 Δstate). `time_integration ar_ot`(teacher forcing) 기본.
- 계층: FPS + multi-source BFS Voronoi, **오프라인·고정·물리 무관**.

즉 HI-MGN은 **"U-Net 모양으로 배치한 MeshGraphNets"** 이고, 2024년 기준 좋은 설계였다.
2026년 기준으로는 processor가 여전히 순수 message passing이라는 점이 병목이다.

---

## 2. SOTA가 실제로 이긴 이유 (수치 근거)

### 2.1 정상상태 공력 (CarBench / DrivAerNet++, 8,150 샘플, 표면압력)

| model | params | Rel L2 | R² | latency |
| --- | --- | --- | --- | --- |
| **AB-UPT** | 6.01M | **0.1358** | 0.9675 | 30.7 ms |
| TransolverLarge | 7.58M | 0.1457 | 0.9595 | 28.4 ms |
| Transolver | 2.47M | 0.1503 | 0.9577 | 29.8 ms |
| Transolver++ | 1.81M | 0.1573 | 0.9543 | 28.5 ms |
| PointTransformer | 3.05M | 0.1909 | 0.9359 | 95.7 ms |
| RegDGCNN (graph) | 1.44M | 0.2006 | 0.9327 | **232 ms / 27 GB** |

CarBench의 결론: *"transformer-based and tokenization-aware architectures consistently
outperform point-based and graph-based networks."* 그래프 계열은 **정확도가 아니라
메모리·지연시간에서 죽는다**(27 GB, 232 ms). Transolver-3도 같은 얘기를 한다 —
GNN(Graph U-Net, GINO)은 NASA-CRM 급에서는 괜찮은데 DrivAerML 급에서 무너진다.

> **이 표에서 배울 것:** 다음 논문에서 latency/memory/params를 반드시 보고해야 한다.
> 그래프 계열이 지는 축이 정확히 거기다. 이기지 못하면 최소한 설명해야 한다.

### 2.2 과도 응답 / 접촉 (여기선 얘기가 완전히 다르다)

| 방법 | 데이터셋 | 결과 |
| --- | --- | --- |
| **M4GN** (micro GNN + macro transformer) | DeformingBeam | RMSE-all 1.87 vs EAGLE 4.22 (**−56%**), 추론 22% 빠름 |
| M4GN | CylinderFlow / DeformingPlate | −36% / −32% |
| **MGN-T** (MPNN×2 → physics-attention → MPNN×2) | pi-beam | 8.65 mm vs MGN 86.73 mm (**10×**), params 0.5M vs 2.0M |
| MGN-T | DeformingPlate | RMSE-1 0.10e-3 (1위), RMSE-all 3.2e-3 (2위, M4GN 2.6e-3) |
| **Physics-informed coarsening** | DeformingPlate | rollout RMSE 6.50e-3 vs FPS 8.0e-3, BSMS 16.6e-3, MGN 12.75e-3 |
| **Garnier transformer** (adjacency-masked attention) | 3D-Aneurysm | 1-step 395.8 vs MGN 1795, BSMS-GNN 719 |
| **EvoMesh** (학습형 계층 + anisotropic MP) | 5개 벤치마크 | 고정 계층 대비 "by large margins" |

**정상상태에서는 transformer가 이기고, 과도 응답에서는 GNN+transformer 하이브리드가
이긴다.** 그리고 그 하이브리드들이 전부 HI-MGN의 이웃이다. 이게 §5 전략의 근거다.

### 2.3 attention 쪽에서 나온 두 개의 반전

- **Transolver++**: Physics-Attention의 물리 상태가 깊이가 깊어지면 **average pooling으로
  퇴화**한다. eidetic state(local-adaptive + slice reparameterization)로 고침, +13%.
  → HI-MGN의 `AttentionPoolBlock`도 score head를 **정확히 0으로 초기화**해 시작한다
  (`MeshGraphNets.py:36-46`). 설계상 mean pooling에서 출발한다는 뜻이고, 학습이 거기서
  탈출하는지 **측정한 적이 없다** (§6.1 진단 실험).
- **LinearNO (AAAI'26)**: Physics-Attention은 linear attention의 특수 케이스이고, slice
  간 attention은 오히려 해가 될 수 있으며 효과의 대부분은 **slice/deslice(= pool/unpool)
  자체**에서 나온다. → HI-MGN의 pool/unpool이 이미 그 역할이다. **좋은 소식:** 우리는
  이미 절반을 갖고 있다. 없는 건 slice 공간에서의 전역 상호작용, 즉 §4.A.

---

## 3. 진단 — HI-MGN에 없는 것 (코드 근거 포함)

### 3.1 계층은 있는데 compute가 계층을 따라가지 않는다 ★가장 중요★

GnBlock 1개 비용 ≈ `(6 + 3·E_dir/N)·N·d²`. ex2 설정(N=200k, E_dir/N=7.63, d=128,
`mp_per_level 4,6,8,6,4`, `voronoi_clusters 5000,100`):

| level | 노드 수 | 블록 수 | 비용 | 전체 대비 |
| --- | --- | --- | --- | --- |
| 0 (fine) | 200,000 | 8 | ~758 GFLOP | **96.4 %** |
| 1 | 5,000 | 12 | ~28 GFLOP | 3.6 % |
| coarsest | 100 | 8 | ~0.4 GFLOP | **0.05 %** |

**전역 결합을 담당하라고 만든 층이 전체 연산의 0.05%를 쓴다.** 게다가 거기서 하는 일은
100노드 그래프 위 message passing — 지름 5짜리 그래프에서 8홉을 도는 것이라 정보 전파는
이미 포화됐고 부족한 건 표현력이다.

같은 자리에 **attention**을 넣으면:

| 배치 | 레이어당 비용 | 전체 예산 대비 증가 |
| --- | --- | --- |
| coarsest(100 토큰) full attn, 6층 | 0.01 GFLOP | **+0.008 %** |
| level 1(5,000 토큰) full attn, 6층 | 7.4 GFLOP | +5.6 % |
| level 1 **anchor attention** M=512, 6층 | 0.72 GFLOP | **+0.55 %** |

**사실상 공짜다.** M4GN(segment transformer), MGN-T(physics-attention processor),
AB-UPT(anchor attention)가 전부 같은 자리를 공격한 이유가 이거다. HI-MGN은 그 자리에
계층 인프라(pool/unpool/캐시/배칭)를 전부 만들어 놓고 **transformer만 안 꽂았다.**

### 3.2 목적함수가 rollout을 학습하지 않는다

자체 측정(`studio-ex9-train-and-test-roster`, ex9 소성, 87 held-out, 19-step):

| model | 1-step valid (2ep → 20/40ep) | rollout relL2 | rollout R² |
| --- | --- | --- | --- |
| MGN-flat | 2.21e-1 → **9.50e-2** | 0.970 → **1.985** | −0.074 → **−3.587** |
| HI-MGN (2ep) | 1.75e-1 | 1.107 | −0.409 |
| DeepONet | 2.30e-1 → 1.25e-1 | 0.540 → **0.206** | 0.606 → **0.945** |

**1-step 손실을 2.3배 개선하면 19-step rollout이 2배 나빠진다.** 예산 문제가 아니라
목적함수 문제다(exposure bias). `time_integration ar_rt`가 구현돼 있지만 전체 궤적을
backprop해서 비싸고, ex9급 이상에서 현실적이지 않다. **중간 지대가 없다.**

문헌의 중간 지대: **pushforward**(1-step 손실 + k−1 스텝은 no-grad로 상태만 전진),
**temporal bundling**, **noise 스케줄**, 그리고 2026년의 **JAWS(Jacobian 정규화) + PF(5)**
조합 — long-term RMSE 최저, 학습시간 −7.8%, 메모리 −20%.

### 3.3 디코더가 메시 노드에 못 박혀 있다

`Decoder`는 fine 노드마다 MLP 하나다. 결과:

1. 학습 시 **모든 노드에서 손실을 계산해야 한다** → 200k 노드 그래프가 그대로 메모리에.
   AB-UPT는 스텝당 **16k 점만 샘플링해도 정확도가 포화**함을 실측했다(Fig. 5):
   *"neural surrogates do not require million-scale meshes to train accurate models."*
   ex2에서 200k → 16k면 fine level 비용이 **12배** 줄고, 그 예산을 깊이/폭으로 돌릴 수 있다.
2. 메시 밖 임의 지점 질의 불가 → CAD만으로 추론 불가, 다해상도 평가 불가.
3. **하드 물리 제약을 걸 자리가 없다.** AB-UPT는 디코더가 좌표의 함수라서
   `ω̂ = ∇ × û`를 유한차분으로 구성해 `∇·ω̂ = 0`을 **구조적으로** 보장한다.

AB-UPT ablation이 이 항목의 중요도를 그대로 보여준다:

| | p_s | u | ω |
| --- | --- | --- | --- |
| UPT baseline | 4.38% | 3.20% | 37.13% |
| + large decoder | 4.25% | 2.73% | **15.03%** |
| + anchor attention | 3.41% | 2.09% | **6.76%** |
| AB-UPT 전체 | 3.01% | 1.90% | 6.52% |

vorticity(미분량 = 국소 고주파)에서 37% → 6.8%. **미분량·응력 같은 채널이 디코더에 가장
민감하다.** HI-MGN 출력에 stress가 들어 있다는 걸 생각하면 남의 얘기가 아니다.

### 3.4 계층이 물리를 모른다

FPS + BFS Voronoi는 **순수 기하**다. 응력 집중부, 접촉면, 경계층, 소성 힌지가 넓은 평면
영역과 똑같이 취급된다. 물리 residual 기반 선택(`s_i = ‖ρü − ∇·σ − ρb‖`)의 TopK가
DeformingPlate rollout RMSE **6.50e-3**, FPS 8.0e-3, 학습형 attention 샘플링 8.1e-3,
BSMS 16.6e-3. **기하 < 학습형 < 물리.**

여기에 자체 기록의 두 함정이 얽혀 있다:

- `mgnv-coarsening-seed-train-infer-mismatch`: 추론이 계층을 **시드 없이** 재구성해서
  학습 때와 ~50% 다른 분할이 나왔다. 물리/특징 기반 결정론적 선택이면 이 실패 모드가 사라진다.
- `bistride-is-2x-not-4x`: bi-stride는 레벨당 정확히 2배라 k=100에 도달하지 못한다.

### 3.5 비용이 edge에 묶여 있다

측정된 스케일링: MGN `∝ E`, Transolver `∝ N`. ex2에서 HI-MGN ~758 N·d² vs Transolver
~100 N·d². **같은 FLOP 예산이면 HI-MGN은 항상 더 얕다.** 커널 프로파일(fp16):
GEMM 22%, `LayerNorm[E,128]` **26%**, gather 19%, add 20%, SiLU 9% — 연산이 아니라
**edge 텐서 위의 메모리 이동이 지배**한다. CarBench가 그래프 계열을 232 ms/27 GB로 찍은
것과 같은 현상이다.

### 3.6 학습 레시피가 2022년식

현재: `LearningR 1e-4`, `Batch_size 4`, `Training_epochs 100`, MSE, AdamW wd 1e-4.
Garnier et al.(2508.18051)이 60개 모델로 뽑은 레시피: **RMSNorm(post-residual)**,
AdamW β=(0.9, **0.95**), **cosine + warmup**(큰 모델에서 필수), 크기별 LR(S/M/L 1e-3,
XL 1e-4), 그리고 **N ∝ C^0.75** 스케일링 법칙. 그 위에 **마스킹 그래프 사전학습**(Cloze).
결과: 500k 파라미터로 MGN 동급 + 7배 빠름, 51M으로 평균 +38.8%, rollout RMSE −52%.

사전학습 데이터는 이미 있다 — **ex4~ex9 여섯 개 데이터셋**.

---

## 4. 개선안 (우선순위)

설계 규칙 두 가지를 지킨다. 저장소가 이미 그렇게 하고 있다:

1. **초기화 시 기존 baseline을 비트 단위로 재현**할 것 (`ATTENTION_TRANSFER_DESIGN.md` §4의
   zero-init 관례). 그러면 새 arm이 baseline을 못 이길 때 그건 학습 문제지 구현 문제가 아니다.
2. DDP가 `find_unused_parameters=False`로 돌기 때문에 **만들어놓고 안 부르는 모듈은
   backward를 죽인다.** 옵션 모듈은 조건부로 *생성*해야 한다.

### Tier 1 — 먼저, 싸고, 확실한 것

#### A. Coarse level latent transformer (`coarse_processor`) ★1순위★

`coarsest_blocks`(그리고 선택적으로 level ≥ 1의 pre/post)를 transformer 블록으로 교체
또는 병렬 추가.

```text
coarse_processor    gn | attn | hybrid     # 기본 gn = 현재 동작
attn_layers         6
attn_heads          8
attn_anchors        512                    # 0 = full attention
attn_levels         1, 2                   # 어느 레벨에 적용할지
```

- **`hybrid`**: `h ← h + α · TransformerBlock(h)`, α는 **0으로 초기화**된 스칼라(LayerScale).
  step 0에서 현재 HI-MGN과 수치적으로 동일 → 규칙 1 만족.
- **anchor attention**(AB-UPT): 코어스 토큰 N_c 중 M개를 균등 샘플링해 anchor로 쓰고,
  anchor끼리는 full self-attention(M²), 나머지는 anchor에 cross-attention만((N_c−M)·M).
  O(N_c²) → O(M·N_c). 5,000 토큰 레벨에서 **레이어당 0.72 GFLOP, 전체 예산의 +0.55%**.
- **위치 인코딩**: 코어스 centroid 좌표의 Fourier feature. `_pool_one`이 이미
  `fine_pos`/centroid를 넘기고 있으므로 새 데이터 경로가 필요 없다. M4GN은 여기에
  random-walk structural encoding을 얹어 이득을 봤다.
- **attention 레벨에서 edge feature를 버릴 수 있다** → §3.5의 `LayerNorm[E,d]` 26% 비용이
  그 레벨에서 사라진다. 즉 **정확도와 속도를 동시에** 가져가는 유일한 항목.

근거: M4GN(−32~56%), MGN-T(pi-beam 10×, 파라미터 1/4), AB-UPT ablation(anchor attention이
단일 최대 기여), LinearNO(pool/unpool은 이미 있고 slice 공간 상호작용만 없다).

#### B. Rollout-consistent 학습 (`time_integration ar_pf`)

`ar_ot`(teacher forcing)와 `ar_rt`(전 궤적 backprop) 사이의 빈칸을 채운다.

```text
time_integration    ar_pf
pushforward_steps   2        # k−1 스텝은 no-grad, 마지막 1스텝만 backprop
pushforward_warmup  10       # 처음 N epoch은 ar_ot (커리큘럼)
```

`general_modules/time_integration.py`에 세 번째 스킴을 추가하고 `resolve_rollout_window`가
`pushforward_steps`를 반환하게 하면 데이터로더 쪽은 그대로다. 비용은 AR-OT의 k배 forward +
1배 backward — `ar_rt`(49스텝 전부 backprop)보다 훨씬 싸다.

**이게 §3.2의 R² −3.59를 직접 겨냥한다.** ex9에서 DeepONet이 0.945를 찍는데 HI-MGN이
음수인 건 아키텍처 문제가 아니라 이 항목 하나다. 논문 전에 반드시 닫아야 하는 구멍.

같이 볼 것: `std_noise`를 고정값(0.01)에서 **스케줄**로, 그리고 JAWS류 Jacobian 정규화
(rollout 방향의 국소 확대율 억제).

#### C. 손실 함수

현재 `F.mse_loss(정규화된 Δ)`. 문제:

- 채널 스케일이 정규화로만 맞춰져 있어 **저분산 채널이 학습에서 사라진다.**
  (자체 기록 `ex9-cond-var-degeneracy`가 정확히 이 부류의 사고였다.)
- 문헌 표준은 **relative L2**(scale-invariant). Transolver-3, CarBench, AB-UPT 전부 이걸 쓴다.
  같은 지표로 학습하지 않으면 비교표에서 손해를 본다.

```text
loss_type       mse | rel_l2 | huber
loss_spectral   0.0        # 고주파 성분 가중 (선택)
```

`feature_loss_weights` 키가 이미 있으니 배선은 그대로 두고 감축 방식만 바꾸면 된다.

### Tier 2 — 아키텍처 변경

#### D. Anchored neural-field decoder

`Decoder`(노드별 MLP)를 **좌표 질의 디코더**로 교체:

```text
h_query(x) = CrossAttention(q = Fourier(x) + h_skip(x),  k,v = anchor tokens)
y(x)       = MLP(h_query(x))
```

얻는 것:

1. **학습 시 질의점 서브샘플링** — `decoder_query_points 16000`. ex2 200k → 16k면 디코더·손실
   경로가 12배 싸지고 그 예산을 §4.A의 깊이로 돌린다. AB-UPT 실측상 16k에서 포화한다.
   동시에 증강 효과("같은 입력 조합을 두 번 보지 않는다").
2. **mesh-free 추론** — CAD/포인트클라우드만으로 예측. `Geometry_generation` +
   `geometry_ingest` 파이프라인과 바로 연결된다(저장소 전체를 관통하는 스토리가 생긴다).
3. **하드 제약** — 유체 채널에 `∇×`로 divergence-free를 구조적으로 강제. 구조 문제라면
   대칭성/평형 잔차를 같은 방식으로.
4. **다해상도 평가** — 학습 메시보다 고운 메시에서 평가 가능(OOD 일반화 지표로 강력).

주의: 디코더가 **커야** 효과가 난다(AB-UPT ablation의 "large decoder"가 vorticity를
37%→15%로 만든 항목). 작은 MLP로 바꾸면 오히려 손해다.

#### E. 물리/특징 기반 coarsening (`coarsening_type voronoi_physics`)

FPS의 **시드 선택만** 바꾼다 — 나머지(BFS Voronoi, 코어스 엣지, 캐시, 배칭)는 그대로 재사용.

```text
seed_score   uniform | curvature | residual | field_grad
seed_mix     0.5      # score-weighted FPS와 순수 FPS의 혼합 비율
```

- `curvature` / `field_grad`: 오프라인 계산 가능 → 캐시 시그니처만 확장하면 끝.
- `residual`: 첫 epoch 이후 모델 자체 잔차로 갱신(문헌의 TopK 방식).
- 혼합을 두는 이유: 순수 점수 기반은 커버리지를 잃는다. 문헌도 50% 다운샘플링에서 최적을 봤다.

`COARSENING_ABLATION_DESIGN.md` §5.2의 "새 키 하나에 6군데 수정" 경고를 그대로 따를 것.

#### F. 학습형 적응 계층 (EvoMesh 방향)

노드 선택 확률을 물리 상태로부터 **학습**하고, 방향 의존(anisotropic) 메시지 전달을 쓴다.
가장 큰 변경이고 가장 큰 상방이지만 D/E 이후가 맞다. 부수 효과로
`mgnv-coarsening-seed-train-infer-mismatch`(추론 시 무시드 재구성) 문제가 원리적으로 사라진다.

### Tier 3 — 스케일 & 논문 마감용

#### G. Edge 비용 재조정

`LayerNorm[E,d]` 26% + gather 19%가 실측 병목. 선택지:
(a) attention 레벨에서 edge feature 폐기(§4.A에서 공짜로 따라옴),
(b) 블록 간 edge MLP 공유(파라미터·메모리 동시 절감, MGN-T가 0.5M로 간 이유),
(c) edge 갱신을 저랭크로.

#### H. 스케일링 레시피

RMSNorm(post-residual), AdamW β2=0.95, cosine+warmup, 모델 크기별 LR, bf16(B300 네이티브).
FLOP 예산 대비 파라미터를 **N ∝ C^0.75**로 배치. 지금의 `LearningR 1e-4 / batch 4 / 100 epoch`은
탐색된 값이 아니라 물려받은 값이다.

#### I. 다중 데이터셋 사전학습

ex4~ex9로 마스킹 그래프 사전학습 후 파인튜닝. Garnier 기준 rollout에서 가장 크게 남는 항목
중 하나였다. 데이터가 이미 있으므로 **추가 시뮬레이션 비용 0**.

#### J. 불확실성

`MeshGraphNets - variational`의 energy-score 작업(`mgnv-aggregate-posterior-drift`)을 접붙이면
"결정론적 SOTA + 보정된 불확실성"이라는, transformer 계열이 아직 잘 안 하는 축이 생긴다.
단, 그 메모의 경고(기하가 비정보적이면 `es_noise_source=prior`가 파괴적)를 유지할 것.

---

## 5. 논문 전략 — 어디서 싸울 것인가

### 하지 말 것

**DrivAerNet++ 표면압력에서 AB-UPT 정면 승부.** 정상상태 geometry→field는
(a) 시간 축이 없어 GNN의 강점(국소 인과, 접촉, Lagrangian 갱신)이 전부 무의미하고,
(b) implicit field가 구조적으로 유리하며,
(c) 그래프는 latency/memory에서 이미 10배 진다(232 ms vs 30 ms).
여기서 이기려면 HI-MGN을 AB-UPT로 개조해야 하는데, 그러면 그건 HI-MGN이 아니다.

### 할 것

**과도 응답 · 접촉 · 대변형 트랙.** crash(자체 AR-RT 계보, arXiv:2510.15201),
deforming plate, plasticity(ex9), EAGLE, CylinderFlow, CarCrashNet.

포지셔닝 문장:

> **HI-MGN2** — *anchored latent attention과 rollout-consistent 학습을 갖춘 계층적 메시 GNN.*
> 그래프 분기는 접촉·world edge·Lagrangian 기하를 유지하고(트랜스포머가 잘 못 다루는 것),
> anchor attention 분기가 전역 수용영역을 산다(메시지 패싱이 못 사는 것).

이 조합이 비어 있다는 게 핵심이다:

| | 계층 | coarse attention | rollout 학습 | neural field 디코더 |
| --- | --- | --- | --- | --- |
| MGN | ✗ | ✗ | ✗ | ✗ |
| BSMS-GNN | ✓ 고정 | ✗ | ✗ | ✗ |
| M4GN | ✓ 고정(segment) | ✓ | ✗ | ✗ |
| MGN-T | ✗ | ✓ | ✗ | ✗ |
| EvoMesh | ✓ 학습형 | ✗ | ✗ | ✗ |
| AB-UPT | ✗ (branch) | ✓ anchor | ✗ (정상상태) | ✓ |
| Transolver++ | ✗ | ✓ slice | ✗ | ✗ |
| **HI-MGN2 (제안)** | ✓ | ✓ anchor | ✓ | ✓ |

### 반드시 넣을 baseline

MGN, BSMS-GNN, EvoMesh, M4GN, MGN-T, Transolver(++), AB-UPT. 앞의 다섯은 코드가 공개돼
있고 뒤의 둘은 이 저장소에 이미 있다 — **다른 어느 그룹보다 유리한 위치다.**

### 반드시 넣을 지표

1-step RMSE, **full-rollout RMSE**, **2× 지평 rollout**(외삽), params, peak memory, latency,
학습 GPU-hour. CarBench가 그래프 계열을 죽인 축을 선제적으로 보고해야 한다.

---

## 6. 실험 순서 (싼 것부터)

### 6.1 진단 (코드 변경 0, 1일)

1. **attention pool이 실제로 mean에서 탈출했는가** — 학습된 체크포인트에서
   `AttentionPoolBlock`의 클러스터별 softmax 엔트로피 측정. 균등에 가까우면
   Transolver++가 지적한 퇴화가 여기서도 일어난 것이고, 그 자체로 논문 그림 하나다.
2. **레벨별 FLOP/시간 프로파일 실측** — §3.1의 96 / 3.6 / 0.05 계산을
   `parallelism/profile.py`로 확인.
3. **coarsest 그래프 통계** — `voronoi_clusters 5000,100`에서 코어스 지름·차수 분포.
   지름이 5 이하면 8개 블록 중 3개는 이미 죽은 연산이다.

### 6.2 Tier 1 (2~3주)

| 실험 | arm | 지표 | 성공 기준 |
| --- | --- | --- | --- |
| E1 | `coarse_processor hybrid` (coarsest만) | 1-step + rollout | rollout relL2 −10% 이상 |
| E2 | E1 + `attn_levels 1,2` + anchor 512 | 위 + 시간/메모리 | 시간 +10% 이내에서 −20% |
| E3 | `time_integration ar_pf`, k = 2, 4 | 19-step rollout R² | **R² > 0** (현재 −0.4) |
| E4 | `loss_type rel_l2` | 채널별 오차 | 저분산 채널 오차 −30% |

E3가 가장 중요하다. E3 없이 E1/E2를 평가하면 rollout 지표가 전부 노이즈에 묻힌다.
**순서: E3 → E4 → E1 → E2.**

### 6.3 Tier 2 (1~2개월)

- E5 neural-field 디코더 + 질의점 서브샘플링 → 절약한 예산으로 깊이 증가
- E6 물리 기반 시드 → E3 기준선 위에서 rollout 비교
- E7 ex4~ex9 전체 그리드 + baseline 7종

---

## 7. 이 저장소에서의 함정 (전부 자체 기록에서 나온 것)

1. **새 config 키는 6군데를 고쳐야 한다** — 네이티브 `general_modules/load_config.py`,
   `cae_suite/specs/meshgraphnets.py`의 `known_keys` + validator, 캐시 시그니처(`_coarse_params`),
   config 파일들, spec의 required/default, 그리고 `--audit-configs` 재실행.
   빠뜨리면 `CFG-UNKNOWN-001` 경고로 조용히 통과한다.
2. **DDP `find_unused_parameters=False`** — 옵션 모듈은 만들었으면 반드시 호출돼야 한다.
   `learned_interpolation False`에서 `UnpoolBlock`을 아예 생성하지 않는 이유가 그거다.
3. **zero-init 관례를 지켜라** — 새 attention 경로는 step 0에서 baseline과 비트 단위로
   같아야 한다. 그러면 "구현 버그 vs 학습 실패" 구분에 드는 시간이 0이 된다.
4. **계층 캐시 시그니처** — 새 coarsening 키가 시그니처에 안 들어가면 **다른 실험이 같은
   캐시를 먹는다.** 조용한 오염이고 발견이 매우 어렵다.
5. **학습/추론 계층 불일치** — 추론이 계층을 무시드로 재구성하면 분할이 ~50% 달라진다.
   `hierarchy_seed`를 반드시 확인할 것.
6. **잘린 계층이 조용히 돈다** — `len(hierarchy) < multiscale_levels`면 할당됐지만 호출되지
   않는 블록이 생기고 `mp_per_level`이 거짓말을 한다. 경고를 넣어야 한다.
7. **rollout 채점 시 파일 개수를 확인** — 학습셋에 추론을 돌려놓고 채점한 사고가 실제로 있었다.
   결과를 인용하기 전에 rollout 파일 수 == held-out 샘플 수인지 assert할 것.
8. **로컬 2080S의 bf16은 5.5배 느리다**(MAGMA fallback). 로컬 반복은 fp16+GradScaler,
   실측은 B300에서.

---

## 참고 문헌

- AB-UPT — arXiv:2502.09692 (TMLR) / 자동차·항공 후속 arXiv:2510.15808
- Transolver++ — arXiv:2502.02414 · Transolver-3 — arXiv:2602.04940
- LinearNO ("Transolver is a Linear Transformer") — arXiv:2511.06294 (AAAI'26)
- CarBench — arXiv:2512.07847
- M4GN — arXiv:2509.10659
- MeshGraphNet-Transformer (MGN-T) — arXiv:2601.23177
- EvoMesh — arXiv:2410.03779 (ICML'25)
- Erwin (ball-tree 계층 트랜스포머) — arXiv:2502.17019 (ICML'25)
- Training Transformers for Mesh-Based Simulations — arXiv:2508.18051
- Physics-Informed Coarsening for Multigrid Graph Neural Surrogates — arXiv:2605.31013
- UPT — arXiv:2402.12365 · GAOT — arXiv:2505.18781
- Message Passing Neural PDE Solvers (pushforward / temporal bundling) — arXiv:2202.03376
- PDE-Refiner — NeurIPS'23 · JAWS — arXiv:2603.05538
- Structure-Preserving Learning / Geometry Generalization — arXiv:2602.02788
