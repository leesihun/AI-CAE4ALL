# 조건 정확도 SOTA 메커니즘 해부 + 실측 실험 + 구현 계획

**날짜:** 2026-08-25
**선행 문서:** `SOTA_CONDITIONAL_GEOMETRY_SURVEY_2026-07.md`, `GEOMETRY_UPGRADE_MESHING_SEMANTIC_2026-08.md`
**성격:** 이 문서는 네 부분으로 구성된다. **1부**는 관련 기법을 "무엇을 어떻게 하는가" 수준까지 해부하고,
**2부**는 `ex1` 체크포인트의 pilot 실험(C2/E2 포함), **3부**는 그 결과로 세운 잠정 구현·검증 계획,
**4부**는 출처다. 1부 없이 2부만 읽어도 실험 코드는 이해할 수 있지만, "왜 이 계열을 시험했는지"의 근거는
1부에 있다. 논문이 보인 결과와 이 저장소에 대한 공학적 추론은 구분해 적는다.

---

## 1부 — 리서치한 기법들의 아키텍처

세 그룹으로 나눈다: **(A) 3D 형상 latent 백본**, **(B) 조건을 정확히 맞추는 sampling-time 메커니즘**,
**(C) 백본+메커니즘을 엔지니어링 역설계로 통합한 시스템**. SDFFlow는 (A)의 global implicit-latent
계열에 있고 다중-token VecSet 경로도 코드에 있으나, 학습된 `ex1`은 A.0에서 보듯 아직 단일 global latent다.
이번 pilot의 실제 타깃은 (B)이고, (C)는 여러 조각을 실제 시스템으로 합칠 때 추가로 필요한 것을 보여준다.

### 1.A 3D 형상 latent 백본

#### A.0 현재 저장소의 `SDFVAE` — 정확히 무엇이며 어디까지 하는가

**한 문장 정의:** 이 저장소의 `SDFVAE`는 정규화된 형상의 **표면 점군과 법선**을 받아 대각 Gaussian
latent posterior `q_phi(z | P, N)`로 압축하고, 임의의 3D 쿼리점 `x`에서 연속 SDF
`f_psi(x, z)`를 평가하는 **amortized implicit-shape VAE**다. voxel/mesh를 직접 인코딩하는 VAE가 아니고,
조건 `cond`를 직접 받는 conditional VAE도 아니다. 데이터셋의 5차원 `cond`
(`bbox_x,bbox_y,bbox_z,volume,area`)는 뒤의 FM 단계에서만 쓰일 수 있고, 현재 `ex1` FM은 config의
`condition_names=bbox_x,bbox_z,volume,area`만 골라 **4차원 조건**으로 학습했으므로 `bbox_y`는 제외된다.
따라서 전체 SDFFlow에서 역할은 다음처럼 분리된다.

```text
재구성: surface points + normals -> SDFVAE encoder -> mu -> SDF decoder -> 0-level MC mesh

생성:   N(0,I) noise + requested cond -> conditional FM ODE -> normalized latent
        -> train-latent mean/std로 역정규화 -> frozen SDFVAE decoder -> 0-level MC mesh
```

즉 `SDFVAE`는 **형상 표현/복원기**, `VelocityNet`은 그 표현 공간의 **조건부 생성 prior**다. 이름에는 VAE가
들어가지만 production sampling은 VAE prior `N(0,I)`에서 latent를 바로 뽑지 않는다. 작은 KL로 정규화한
encoder mean들의 경험 분포를 별도의 rectified flow가 다시 학습한다. 운용 관점에서는
**약하게 정규화된 shape autoencoder + learned latent flow**라고 보는 것이 가장 정확하다.

##### A.0.1 데이터가 모델에 들어가는 방식

`dataset/deepjeb.h5`의 각 형상은 `surface_points`, `surface_normals`, `sdf_points`, `sdf_values`, `cond`를
가진다. 현재 파일은 2138형상이고, 형상당 표면점 8192개와 SDF 쿼리 10240개(near-surface 8192 + uniform
2048)를 저장한다. 한 VAE step에서는 이 중 표면점/법선 4096개와 SDF 쿼리/값 4096개를 무작위로 다시
뽑는다.

| 입력 | 실제 `ex1` batch shape | 쓰는 곳 | 의미 |
|---|---:|---|---|
| `surface_points` | `[B,4096,3]` | encoder | `[-1,1]^3` 안으로 정규화된 표면 좌표 |
| `surface_normals` | `[B,4096,3]` | encoder | 각 표면점의 방향 정보 |
| `query_points` | `[B,4096,3]` | decoder supervision | near-surface + 공간 균일 쿼리 |
| `query_sdf` | `[B,4096]` | reconstruction loss | 음수=내부, 양수=외부 |
| `cond` | `[B,5]` | **VAE에서는 미사용** | FM이 subset을 고르는 원본 5 descriptor; `ex1`은 `bbox_y` 제외 |

split은 seed 42의 80/10/10(1710/214/214)으로 고정된다. 다만 `SDFShapeDataset.__getitem__`은
`np.random.default_rng()`를 인자 없이 새로 만들고 객체의 `seed` 필드를 사용하지 않으므로, **split은
재현되지만 매 epoch의 점 subsampling과 FM용 1회 latent encoding은 bitwise 재현되지 않는다.** 정확한 실험
재현이 필요하면 worker/shape/epoch에서 파생한 RNG seed를 명시해야 한다.

##### A.0.2 Encoder — 점 집합을 learned query token으로 읽는 Perceiver형 집약기

좌표 `x=(x,y,z)`는 먼저 8-band Fourier feature로 바뀐다.

```text
gamma(x) = [x, sin(2^i*pi*x), cos(2^i*pi*x)]_(i=0..7)   # 3*(1+2*8)=51차원
point feature = Linear([gamma(x), normal])               # 54 -> encoder_dim
```

`ex1`에서 `encoder_dim=256`, head 4개, block 2개다. 학습 가능한 query
`Q in R^(latent_tokens x 256)`를 batch에 복제한 다음, 각 block에서

```text
Q <- Q + MHA(LN(Q), LN(point_features), LN(point_features))
Q <- Q + FFN(LN(Q))              # 256 -> 1024 -> 256, SiLU
```

를 수행한다. 점들 사이의 self-attention은 없고, 같은 point context를 두 번 읽으면서 **latent query만**
갱신한다. 입력점 순서를 섞어도 attention의 key/value 집합이 같으므로 출력이 변하지 않는 set encoder다.
마지막 `LayerNorm -> Linear(256, 2*latent_dim)`이 각 token의 `mu, logvar`를 만든다.

`config_train_v2.txt`에서만 `latent_tokens=32`, `encoder_self_attention=true`라서 각 cross-attention 뒤에
token-token self-attention이 추가된다. 이는 여러 token이 정보를 교환하게 하지만, 현재 학습된 `ex1`은
`latent_tokens=1`이므로 **하나의 learned query가 전체 형상을 하나의 전역 벡터로 집약**한다.

##### A.0.3 Latent posterior와 warmup

posterior는 token/채널별 대각 Gaussian이다.

```text
z = mu + noise_scale * eps * exp(0.5*logvar),  eps ~ N(0,I)
KL = mean_B sum_(token,channel) KL(q_phi(z|P,N) || N(0,I))
```

`ex1`의 latent는 `[B,1,256]`이고 flat dimension도 256이다. 처음 20 epoch는 `z=mu`, `KL weight=0`인
결정론적 autoencoder로 시작한다. 이후 posterior noise는 80 epoch에 걸쳐 **0.1까지만** 증가하고, KL weight는
100 epoch에 걸쳐 `1e-5`까지 증가한다. 표준 VAE처럼 noise scale 1.0으로 끝나지 않는다. validation,
reconstruction, FM latent 추출은 모두 stochastic sample이 아니라 `mu`를 사용한다. 이 때문에 posterior를
두기는 하지만, 실제 downstream 표현은 안정적인 encoder mean이다.

##### A.0.4 두 decoder 경로

`SDFVAE`는 config에 따라 서로 다른 두 neural-field decoder를 만든다.

**현재 `ex1`: `SDFDecoderMLP`.** 각 쿼리점마다 51차원 Fourier 좌표와 flat latent 256차원을 붙여 307차원
입력을 만든다. 512-wide SiLU MLP 8층을 통과시키며 5번째 linear 앞에서 원래 307차원 입력을 한 번 다시
concatenate하는 DeepSDF식 skip을 쓴다. 마지막 scalar head가 SDF 하나를 낸다. 쿼리점끼리는 상호작용하지
않으므로 임의 개수의 점을 독립적으로 평가할 수 있다. 다만 Fourier feature, SiLU, 초기화 방식까지 포함하면
원 논문 DeepSDF의 복제라기보다 **DeepSDF-style pointwise implicit MLP**다.

**미학습 `v2`: `SDFDecoderAttention`.** 쿼리 Fourier feature를 `d_model=512`로, 각 64차원 latent token을
512차원으로 projection한 뒤, 각 쿼리가 32개 token에 4개의 cross-attention block으로 질의한다. 마지막
`LayerNorm -> Linear(512,1)`이 SDF를 낸다. 이 경로가 실제 **VecSet-style query-to-token decoder**다.
query-query self-attention은 없어서 연속장 평가의 pointwise 성질은 유지된다.

| 체크포인트/설정 | latent | decoder | encoder params | decoder params | SDFVAE 총 params | 상태 |
|---|---:|---|---:|---:|---:|---|
| `ex1` checkpoint | `1x256` | MLP, 512x8 | 1,726,976 | 2,153,985 | **3,880,961** | VAE epoch 499 학습 완료 |
| `config_train_v2.txt` | `32x64`(flat 2048) | attention, 512x4 | 25,334,400 | 12,675,073 | **38,009,473** | checkpoint 없음 |

따라서 `v2`는 token 수만 바꾼 작은 변형이 아니다. flat latent가 8배, SDFVAE 파라미터가 약 9.8배가 되고,
decoder의 query-token attention 비용도 생기는 별도 아키텍처다. `ex1`에서 맞춘 guidance grid/batch와
step-size가 `v2`에서도 그대로 메모리와 정확도를 보장하지 않는다.

##### A.0.5 실제 VAE loss

기본 reconstruction은 prediction까지 clamp하는 손실이 아니다.

```text
L_recon = mean |f_psi(x,z) - clamp(SDF_GT(x), -0.1, 0.1)|
L_ex1   = L_recon + beta_KL * KL
```

GT만 truncation하고 prediction은 clamp하지 않아, 초기 출력이 band 밖에 있어도 안쪽으로 당기는 gradient가
남는다. `ex1`에는 `surface_weight`, `normal_weight`, `eikonal_weight`가 없어서 세 값이 모두 0이고, 아래
hybrid 경로는 실행되지 않았다.

```text
L_surface = mean |f(x_surface,z)|
L_normal  = mean [1 - cos(grad_x f(x_surface,z), n_GT)]
L_eikonal = mean (||grad_x f(x_query,z)|| - 1)^2
```

양의 weight가 하나라도 있으면 surface/query 좌표에 대해 `create_graph=True`로 미분하고, decoder parameter까지
2차 미분이 흘러가도록 fp32 math-attention 경로를 강제한다. 구현은 TripoSG의 아이디어를 따른
**TripoSG-style 변형**이지 논문의 TSDF `L1+L2` 목적함수를 그대로 복제한 것은 아니다.

##### A.0.6 FM으로 넘어가는 지점과 추론

VAE가 끝나면 `train_fm.py::_encode_split`이 EMA VAE로 각 형상을 한 번씩 `mu`에 인코딩하고
`[tokens,latent_dim]`을 flat vector로 만든다. train split에서 dimension별 `latent_mean/std`를 구해 표준화하고,
FM은 `noise -> standardized mu`의 rectified-flow velocity를 학습한다. 선택된 raw condition도 train 통계로
표준화되어 이 단계에만 들어간다. 생성 때는 Euler ODE 결과를

```text
z_phys = z_normalized * latent_std + latent_mean
```

으로 되돌린 뒤 frozen VAE decoder를 호출한다. reconstruction은 입력 mesh를 정규화하고 표면점/법선을 뽑아
곧바로 `mu`를 decode한다. 두 경로 모두 dense grid에서 SDF를 평가하고 zero-level Marching Cubes를 수행하며,
여러 connected component가 생기면 `mesh_extraction.py`가 가장 큰 component만 남긴다.

##### A.0.7 이 구조가 이번 guidance 실험에 주는 해석

1. **`ex1`은 현재 VecSet이 아니다.** encoder는 attention set aggregator지만 표현은 단일 global vector이고,
   decoder도 query-to-token attention이 아닌 global-latent MLP다. `v2`부터 VecSet-style이라고 부르는 것이
   코드와 일치한다.
2. **조건 오차는 VAE와 FM 두 층에서 생긴다.** FM이 정확한 latent를 골라도 decoder/MC가 요청 descriptor를
   보존한다는 보장은 없다. C2/E2가 frozen decoder까지 미분하거나 실측하는 이유가 이것이다.
3. **`ex1` SDF는 metric SDF라고 보장되지 않는다.** hybrid normal/eikonal loss를 학습하지 않았으므로 SDF값과
   zero level set은 쓸 수 있어도 `|grad SDF|=1`은 보장되지 않는다. 고정 `tau`의 sigmoid occupancy가 실제
   공간 두께를 일정하게 뜻하지 않아 soft-volume/area calibration이 checkpoint별로 필요한 구조적 이유다.
4. **global latent 보정에는 locality/topology 안전장치가 없다.** descriptor gradient가 256차원 전체를 움직이므로
   volume/area는 맞으면서 국소 형상이나 제조 가능성이 나빠질 수 있다. watertight와 scalar error만으로는
   learned shape distribution 보존을 입증하지 못한다.
5. **calibration은 VAE checkpoint와 수치 설정에 종속된다.** decoder weight뿐 아니라 soft grid resolution,
   `tau`, MC resolution, latent normalization이 바뀌면 affine 계수를 다시 맞춰야 한다.

#### A.1 3DShape2VecSet / TripoSG / Hunyuan3D — VecSet 계열 (SDFFlow가 서 있는 자리)

**공통 골격:** 표면 점군(원조 3DShape2VecSet은 좌표만, TripoSG/Hunyuan은 법선도 사용) → transformer
인코더 → **N개의 latent vector 집합** → 쿼리점이 이 집합에 cross-attention해 occupancy/SDF를 낸다. 생성
단계는 latent set 전체에 diffusion/flow를 건다. SDFFlow의 **미학습 `v2` 경로**가 이 골격의 간소화된
VecSet-style 구현이다. 반면 학습된 `ex1`은 attention encoder 뒤에 token 하나와 MLP decoder를 쓰는
global-latent neural field이므로, "토큰 하나짜리 VecSet"은 계보를 설명하는 비유일 뿐 정확한 구조명은 아니다.

**TripoSG(2502.06608)가 더한 것 — hybrid VAE loss.** scalar SDF값만 맞추면 decoder의 공간 미분 방향은
충분히 감독되지 않는다. TripoSG는 TSDF `L1+L2`, surface-normal cosine, eikonal, KL 항을 함께 쓴다.

- **SDF loss** — TSDF 회귀. SDFFlow의 기본 손실은 GT만 clamp한 L1이라 정확히 같은 식은 아니다.
- **Surface normal loss** — 표면점에서 정규화한 `grad SDF`를 GT 법선과 cosine loss로 맞춘다. derivative
  정보를 표면 위에서 직접 감독해 qualitative edge/detail을 돕는다.
- **Eikonal loss** — 임의의 점에서 `‖∇SDF‖ ≈ 1`을 강제. 이게 없으면 디코더가 "SDF값은 맞지만 기울기가
  이상한" 퇴화해를 배울 수 있어서, 부호 판정 경계(제로 레벨셋)가 흔들린다.
- 저자 ablation의 정량 차이는 작다: SDF-only → normal → normal+eikonal에서 Chamfer `4.60→4.56→4.57`,
  normal consistency `0.955→0.956→0.957`, F-score는 모두 `0.999`다. 따라서 "확연한 정량 개선"보다는
  **작고 일관된 normal-consistency 개선과 qualitative detail 개선**이 근거에 맞다.
  `model/sdf_vae.py::hybrid_geometry_losses`에는 같은 계열의 surface/normal/eikonal 항이 있지만 TripoSG 식의
  복제는 아니며, `ex1`에서는 weight가 0이라 학습되지 않았다.

**Hunyuan3D 2.0(2501.12202)이 더한 것 — 스케일과 variable token length.** 같은 ShapeVAE + flow-DiT
계열에서 released VAE의 token 길이는 최대 3072다. 다만 multi-resolution은 저해상도→고해상도 curriculum이
아니라, 학습 중 미리 정한 token-length 집합에서 길이를 무작위로 뽑는 방식이다. edge/corner importance
sampling도 함께 쓴다. 2138개 브래킷에서 3072 token이 필요한지는 논문으로 결정할 수 없다. 계산량·표본효율
위험이 크므로 local ablation 전에는 **우선순위가 낮다**고 판단하는 것이지, "과적합만 난다"가 입증된 것은 아니다.

#### A.2 TRELLIS — Structured Latent (SLat), sparse-voxel 계열

VecSet과의 결정적 차이: latent를 **하나의 순서없는 집합**이 아니라 **공간에 고정된 sparse voxel 격자**
`{(z_i, p_i)}`로 둔다. `p_i`는 표면과 교차하는 활성 voxel의 3D 위치(기본 64³ 격자에 평균 ~20K개 활성),
`z_i`는 그 위치의 로컬 latent 벡터. 인코딩은 멀티뷰 렌더링 → DINOv2 특징을 각 활성 voxel에 projection해서
모으고, shifted-window attention sparse VAE로 압축한다. 생성은 **2단계**다. ① binary active grid를 별도
3D-conv VAE가 연속 16³ structure latent로 압축하고, dense rectified-flow transformer가 이 latent를 생성한 뒤
active grid로 복원한다. ② 그 active 위치의 local latent는 sparse-conv down/up sampler로 2³ token에
pack/unpack하고, time-modulated transformer가 핵심 denoising을 수행한다. 디코더는 3D Gaussian / radiance
field / FlexiCubes mesh 등 여러 출력 포맷을 지원한다.

**VecSet 대비 트레이드오프:** SLat은 좌표가 locality anchor라서 국소 편집에 유리한 inductive bias를 갖는다.
TRELLIS의 실제 편집은 단순히 voxel 하나를 바꾸는 연산이 아니라, bounding box 밖을 고정한 RePaint식 flow
sampling을 structure/local-latent 두 단계에 적용한다. 더 좋은 국소 디테일과 더 많은 데이터 필요성은 matched
VecSet ablation으로 입증된 보편 법칙이 아니라 설계상 기대와 프로젝트 비용 판단이다. 이 저장소에서는 선행 문서가
지목한 encoder/data 병목을 먼저 실험하고, 2138개로 SLat의 이득을 따로 입증하기 전까지 후순위로 둔다.

#### A.3 SPAGHETTI / SALAD — 파트 수준 Gaussian 분해 (구조적 편집)

**SPAGHETTI(SIGGRAPH 2022)**가 기초를 놓았다: occupancy supervision은 쓰지만 explicit part label 없이 GMM
clustering으로 self-supervised part assignment를 만든다. 각 파트는 두 종류의 파라미터로 표현된다 —
**외재(extrinsic)**: center + covariance + mixture weight `pi`의 Gaussian component, **내재(intrinsic)**:
세부 형태 embedding. Gaussian SDF들을 겹쳐 합산하는 방식은 아니다. GMM은 coarse spatial support와 part
assignment/regularization을 제공하고, transformer occupancy network가 part embedding들을 조합해 최종
occupancy를 예측한다. 학습 뒤에는 part를 이동·크기조정·교환하는 편집을 재학습 없이 할 수 있다.

**SALAD(ICCV 2023)**는 이 표현 위에 **cascaded diffusion**을 얹는다: 먼저 **저차원 외재 파라미터**(Gaussian
위치·공분산)에 대해 diffusion을 학습하고, 그 결과를 조건으로 **고차원 내재 임베딩**에 대해 두 번째 diffusion을
학습한다. 이렇게 나누는 이유는 고차원 latent를 통째로 diffusion하면 학습이 비효율적이기 때문 — 저차원
"뼈대"를 먼저 확정하고 "살"을 그 위에 입힌다.

**SDFFlow와의 접점:** VecSet token에 위치·공분산을 예측하게 하는 것은 SALAD와 TRELLIS 사이의
Gaussian-anchored VecSet이라는 유망한 설계 가설이다. 그러나 좌표를 덧붙이는 것만으로 token이 일관된 part를
담거나 변환에 equivariant해지지는 않는다. SPAGHETTI가 쓰는 GMM likelihood, affine-equivariance training,
cluster 기반 part-disentanglement 같은 objective가 별도로 필요하다. 따라서 **한 architecture project로 함께
노릴 수는 있지만 part editing이 자동으로 따라오는 것은 아니다.** 데이터 확대 뒤 검증할 Tier 3.4 가설이다.

### 1.B Sampling-time 조건 정확도 메커니즘 — pilot 방법들의 계보

아래 방법들은 frozen generator를 sampling time에 제어한다는 공통점이 있지만 요구조건은 다르다. D-Flow/OC-Flow/
local guidance는 미분가능 terminal objective가 필요하고, Lagrangian Dual Flows는 매 RHS evaluation에서 제약
Jacobian VJP가 필요하며, Dflow-SUR는 별도의 미분가능 surrogate가 있어야 한다. 따라서 "재학습 없이"는
generator weight에 관한 말일 뿐, surrogate/calibration 없이 전부 지금 즉시 적용된다는 뜻은 아니다.

#### B.1 D-Flow / FlowGrad — source(노이즈) 공간 최적화

**D-Flow**의 아이디어는 단순하다: rectified flow의 샘플링은 `z0(노이즈) → ODE 적분 → z1(데이터)`인
**결정론적 함수**다. 그러니 `z0`을 학습 가능한 변수로 두고, ODE 적분 전체를 통해 `z1`에 대한 목표 손실을
**backprop**하면 된다. 의사코드:

```text
z0 = randn()  # requires_grad
for outer_iter in range(N):
    z = z0
    for t in ode_steps:               # 이 루프 전체가 autograd 그래프에 남는다
        z = z + velocity_net(z, t) * dt
    loss = target_loss(decode(z))
    loss.backward()                   # z0까지 그래디언트가 흘러간다
    optimizer.step()
```

**FlowGrad**는 frozen velocity-net의 weight를 학습하는 것이 아니라, 각 이산 시간의 additive control
`u_k`를 샘플별 test-time 변수로 최적화한다. 원 논문은 terminal loss와 `lambda * integral ||u(t)||^2 dt`를
함께 쓰고 VJP 재귀 및 straightness 기반 step skipping으로 비용을 낮춘다. D-Flow는 시작점 하나, FlowGrad는
시간별 trajectory control을 움직인다는 직관은 맞다.

**비용 구조:** ODE 전체를 매 outer iteration마다 다시 적분하고 backprop하므로 비싸다. D-Flow 원 실험은
midpoint solver(보통 6 NFE), gradient checkpointing, L-BFGS+line search를 썼다. 오늘 D/D2는 Euler 25-step,
Adam, source norm penalty를 쓴 **D-Flow 계열 변형**이며 원 논문 구현을 그대로 재현한 것은 아니다.

#### B.2 OC-Flow — 위 둘을 포괄하는 최적제어 프레임

**OC-Flow(ICLR 2025, arXiv 2410.18070)**는 pretrained flow에 샘플별 open-loop additive control
`theta_t`를 더하고 terminal reward와 quadratic running cost를 함께 최적화한다. practical algorithm은
state trajectory를 다시 풀고 terminal co-state gradient로 persistent time-indexed controls를 outer-loop에서
반복 갱신한다. OC-Flow 저자들은 D-Flow를 one-control asynchronous case로, 단순화된 FlowGrad update를
`gamma -> infinity, dt -> 0` 극한으로 해석한다. 원 FlowGrad에도 control penalty가 있으므로 "수식상 완전히
동일한 특수해"라기보다 **OC-Flow 논문이 제시한 통합적 해석**이라고 쓰는 편이 정확하다.

Euclidean Theorem 2와 SO(3) Theorem 5의 monotonic-improvement/convergence 결과는 reward/model derivative의
Lipschitz·boundedness, 충분히 큰 `gamma` 등의 가정과 연속시간 해석 아래 성립하고 Euler에는 `O(dt)` 오차가
있다. 실험도 "항상 D-Flow/FlowGrad보다 낮은 MAE"로 요약할 수 없다. QM9에서는 FlowGrad보다 6/6, D-Flow보다
5/6 property MAE가 낮았지만, image 지표는 혼재하고 peptide는 두 방법과 직접 비교하지 않았다.

**중요한 분류 교정:** 오늘 C/C2는 OC-Flow의 경량판이 아니다. persistent control, outer optimization,
co-state/adjoint, running cost가 하나도 없기 때문이다. 직접 대응하는 것은 `On the Guidance of Flow Matching`과
FMPS-gradient가 쓰는 **one-step endpoint-prediction/localized guidance**다.

```text
x1_hat = z_t + (1-t) * v_theta(z_t,t)
guidance direction = grad_(z_t) J(x1_hat)
```

`x1_hat`은 학습된 ODE 전체가 실제 직선이라는 보장이 아니라 affine probability path에서 유도한 conditional
endpoint estimate/한 번의 Euler 예측이다. C2는 이미 conditional FM 위에 descriptor loss를 얹고,
gradient RMS normalization, `(1-t)` schedule, affine target mapping을 자체 사용하므로 FMPS와도 동일하지 않다.
정확한 이름은 **calibrated one-step endpoint-prediction guidance (FMPS-gradient/localized-guidance 계열)**다.

#### B.3 Lagrangian Dual Flows — hard constraint (등식/부등식)

**arXiv 2607.04513(2026-07)**은 differentiable 제약 위반을 연속시간 극한에서 0으로 보내는 primal-dual
flow를 제안한다. "유한 step에서도 정확히 만족"한다고 읽으면 안 된다.
primal 상태 `x`뿐 아니라 **dual 변수(Lagrange multiplier) `λ`도 같이 시간에 따라 적분**한다:

```text
ẋ_t = v_θ(x_t, t) − J_g(x_t)ᵀ λ_t − c·J_g(x_t)ᵀ g(x_t)     (primal: velocity + 제약 보정)
λ̇_t = g(x_t) / (1−t)^p                                    (dual: 위반량을 누적, t→1에서 가중치 폭증)
```

`g(x)=0`이 등식 제약이다. 부등식은 slack만 추가하는 데서 끝나지 않고 ReLU soft-projected slack dynamics로
`s>=0`를 보존한다. 핵심 성질과 한계는 다음과 같다.

1. projection subproblem/pseudoinverse 없이 **ODE RHS evaluation마다 VJP 한 번**을 쓴다. MNIST 100-step에서
   PCFM 1919.91ms, LDF 183.31ms로 약 10.47배 빠르지만, 같은 표의 violation은 각각 `1.0e-7`과 `4.9e-2`라
   같은 tolerance의 속도 비교는 아니다.
2. Theorem 1/3의 `||g(x_t)||=O((1-t)^alpha)`는 `p=2`, smoothness, uniformly full-rank/well-conditioned
   constraint Jacobian, bounded trajectory, small-variation 등의 가정과 exact continuous integration에서
   `t -> 1-`일 때의 점근 결과다. finite-step solver는 tolerance만 제공한다.
3. 보장은 입력한 `g`에만 해당한다. `g`가 biased soft-volume이면 res96 MC/trimesh volume은 보장하지 않으며,
   constraint satisfaction과 learned data distribution 보존도 별개다.

**SDFFlow에서의 자리:** differentiable proxy 제약에 대한 feasibility 수단으로 rejection 빈도를 낮출 수는 있지만,
`max_condition_z` OOD guard와 true-mesh 최종 검증을 원리적으로 즉시 대체하지 않는다. 둘은 다른 실패를 막는다.
Tier 2의 응력/벽두께 상하한에 쓸 때도 surrogate/proxy calibration과 finite-step 실측 gate가 필요하다.

#### B.4 Dflow-SUR — source-space 최적화 + surrogate, 항공 역설계 특화

**arXiv 2512.08336(v1 2025-12-09)**은 B.1(D-Flow)의 아이디어를 **surrogate 모델과 묶어** 항공역학
역설계에
적용한 것이다. 메커니즘은 D-Flow와 동일(노이즈/source 변수를 ODE 전체로 backprop해서 최적화)이지만,
목표 손실이 "형상 자체의 목표"가 아니라 **"학습된 surrogate가 예측한 양력/항력 비"**다. 즉 flow matching
생성기와 물성 예측 surrogate를 하나의 미분가능 파이프라인으로 이어붙인 것 — "생성→surrogate 채점"을
gradient로 닫은 버전이다. 저자 표의 airfoil loss는 `4.80e-8` 대 strongest energy baseline `4.80e-4`, 시간은
801s 대 3136.75s(74.47% 감소)다. abstract는 wing L/D 11.8% 개선을 주장하지만 표의 mean
`21.1845 vs 18.3998`로 직접 계산하면 LHS 대비 약 15.1%라 내부 불일치가 있다. 더구나 이는 surrogate
평가이고 high-fidelity CFD 검증이 아니다. SDFFlow에서 같은 형태를 만들려면 GINO/Transolver를 단지 "연결"하는
것 외에 end-to-end differentiability, geometry/mesh 표현 연결, surrogate OOD 검증이 필요하다.

### 1.C 엔지니어링 역설계 통합 시스템 — (A)+(B)를 실제로 조립하면

#### C.1 PhysGen(CVPR 2026, arXiv 2512.00422) — 이 저장소의 "북극성"

**SP-VAE(Shape-and-Physics VAE):** 표면 균일점 + salient edge 점을 뽑아 **bidirectional cross-attention**으로
섞은 뒤 **하나의 공유 latent**로 압축한다(SDFFlow의 `PointCloudEncoder`와 같은 자리, 다만 입력에 edge 점이
추가됨). 이 하나의 latent에서 **세 개의 디코더**가 갈라진다:

- **Shape decoder** — SDF 회귀(SDFFlow의 `SDFDecoderMLP/Attention`과 동일한 역할).
- **Pressure decoder** — 표면 압력장을 self-attention + channel reweighting + MLP 3-branch로 예측, 임의
  3D 쿼리점에서 값을 낸다.
- **Drag decoder** — 같은 특징에서 전역 항력계수(스칼라 하나)를 3-layer MLP로 낸다.

학습은 2단계다: ① shape/physics를 **따로** 사전학습(안정성), ② 500 epoch **공동 미세조정**
(`λ_shape=10, λ_physics=0.1, λ_drag=10`).

**Physics-guided rectified flow (샘플링 단계):** 최초 flow는 100 step이고, 이후 두 phase와 re-noise를
`K=20` cycle 반복한다. 두 phase가 모두 20 step인 것은 아니다.

1. **Drag-guided velocity phase** — 각 flow step 자체에 drag decoder terminal-target gradient를 넣는다.
   최초 100-step 뒤, 각 re-noise cycle에서는 `t=0.75`부터 남은 25 flow step을 수행한다.
2. **Pressure-force physical refinement** — 20 step. 여기서는 drag decoder가 아니라 pressure decoder의
   압력과 표면 법선·면적으로 `F_x,F_y,F_z`를 만들고 `L_x,L_y,L_z`를 latent까지 backprop한다.
3. 그 다음 **latent를 다시 t=0.75까지 재노이즈(re-noise)해서** 다음 사이클을 시작한다 — 순수 최적화로
   너무 멀리 가서 "그럴듯한 형상" 분포를 벗어나는 것을 막는 장치다.

이것과 C2+E2의 공통점은 **global generation 뒤 refinement를 둔다**는 넓은 pattern뿐이다. PhysGen은 learned
drag/pressure gradient와 stochastic re-noise를 반복하지만, E2는 true mesh descriptor를 line search에 쓰고
re-noise/alternating cycle이 없다. 구조적으로 같다고 부르면 과장이다.

**결과(DrivAerNet++, 약 8000대 중 5819 train/1147 test):** 형상 재구성 IoU 91.89%(Dora 88.61% 대비 우위),
항력계수 예측 MSE 4.0×10⁻⁵(TripNet 9.1×10⁻⁵ 대비), **guided 생성이 unguided 대비 F-score +21.09%,
Chamfer Distance −22.68%**다. OpenFOAM 20-sample 평균은 ShapeNet unconditional −22.70%, DrivAerNet++
unconditional −15.47%, DrivAerNet++ conditional −6.53%다. `15.47~22.70%`는 unconditional 두 설정에만 해당한다.

**SDFFlow와의 대응은 roadmap analogy다.** shape decoder 자리는 비슷하지만 PhysGen 재현에는 physics label을
공유 latent로 공동학습하는 SP-VAE, pressure/drag head, drag-regularized flow, pressure-force refinement,
re-noise alternation이 모두 필요하다. 별도 GINO/Transolver를 붙이는 것만으로 같은 latent/gradient 경로가 생기지
않는다. C2/E2는 그중 "sampling-time differentiable objective와 최종 검증"을 시험한 인프라 조각이다.

#### C.2 3DID(arXiv 2512.08987) — triplane latent + 2단계(guidance 후 위상보존 정제)

**PG-VAE(Physics-Geometry VAE):** geometry/physics branch가 각각 learnable token과 cross/self-attention을
수행하고, 두 token latent를 concatenate+MLP로 통합한 뒤 reshape/upsample해 **triplane latent**를 만든다.
디코더는 SDF가 아니라 occupancy와 pressure/velocity field를 이 triplane에서 복원한다.

**2단계 최적화:**

1. **Gradient-guided diffusion sampling** — diffusion 샘플링 중 Bayes 규칙으로 "unconditional score를
   conditional score로 대체"(= score에 목표 그래디언트를 더함). 이는 sampling-time gradient guidance의
   diffusion 버전(rectified flow가 아니라 DDPM)이다.
2. **Topology-preserving refinement** — guidance로 나온 형상을 **free-form deformation(FFD) 제어점**으로
   다시 최적화하되, **미분가능 GNN surrogate(MeshGraphNet)**가 손실을 제공하고 smoothness/부피보존 페널티로
   메시가 망가지지 않게 막는다. 즉 1단계는 "latent에서 크게 움직이기", 2단계는 "메시 표면에서 미세 조정"으로
   역할을 나눈다. E2와는 global search 뒤 local refinement라는 **pipeline slot**만 비슷하다. 3DID는 surrogate
   driven FFD이고 true CFD/mesh descriptor accept-reject Newton loop가 아니다.

**결과:** broad CEM/GP/backprop 비교인 Table 1에서는 simulation drag 0.3536 대 최선 baseline 0.4097로
13.6% 개선이다. `0.4066→0.3536`(13.0%)은 representation ablation Table 2의 TripNet 비교이므로 두 표를
섞으면 안 된다. novelty 1.1709, predicted drag 0.2607은 맞다.

#### C.3 LAMP(arXiv 2510.22491) — same-topology parameter 외삽의 별도 접근

gradient guidance는 학습 분포 밖에서 prior/목적함수의 충돌이 커질 수 있다. LAMP는 모든 외삽의 일반 해법은
아니지만, topology와 parameterization이 정렬된 exemplar bank에서는 발상이 완전히 다르다:

1. mean design에 먼저 맞춘 **shared initialization에서 시작해**, 각 exemplar 하나에 별도 SDF decoder를
   overfit시킨다(DeepSDF
   auto-decoder 학습과 비슷하되, 형상마다 별도 가중치 세트를 만든다는 점이 다르다).
2. 같은 초기화에서 출발했으므로 이 가중치들은 **정렬된(aligned) weight space**를 이룬다 — 여기서
   "가중치 공간에서의 선형 결합"이 "형상 공간에서의 선형 보간"과 (국소적으로) 대응한다는 게 이론적 근거
   (Assumption A2의 local error는 `O(max_i ||w_i-w_0||^2)`). 이론 전개의 convex coefficient 조건과 달리
   실제 extrapolation은 음의 `alpha`도 허용하므로 큰 외삽은 empirical filter에 더 의존한다.
3. 목표 설계 파라미터가 주어지면, 그 파라미터를 만족하는 **혼합계수 α를 최소자승으로 직접 푼다** — gradient
   descent가 아니라 **닫힌 형태에 가까운 선형 대수 풀이**(10ms 미만).
4. **안전장치:** 혼합 디코더 출력과 개별 출력의 alpha-weighted average가 어긋나면(linearity mismatch)
   거부한다. ROC AUC 0.989는 diagnostic sweep의 **사람이 표시한 visible mesh collapse/왜곡** 판별값이지
   target-parameter 정확도나 물리 유효성 AUC가 아니다.

**결과:** 단일 파라미터 ±100% 외삽 R²=0.902(DNI 0.143), 4파라미터 동시 ±50% R²=0.867(DNI −5.768)은
논문 수치와 맞다. 다만 car length 외 대부분은 직접 기하/CFD 실측이 아니라 mesh surrogate가 예측한 parameter
fidelity다. **한계:** A1은 parameter→control-point 선형성이고, common topology는 별도의 추가 제한이다.
DeepJEB처럼 genus 6~9가 섞인 전체 bank에는 적용할 수 없고, topology를 나눈 뒤에도 parameter label,
선형 parameterization, aligned decoder bank가 필요하다. 이를 SDFFlow 후보 주변 local search로 쓰는 것은
가능한 연구 제안이지 확인된 drop-in 사용법이 아니다. LAMP 자체에는 true measure→Newton 반복도 없다.

### 1.D 요약 지도 — 어떤 상황에 어떤 기법인가

| 상황 | 쓸 기법 | SDFFlow 대응 |
|---|---|---|
| latent가 이미 학습 분포 안, 스칼라 타깃에 **가깝게** | endpoint-prediction/local guidance | 오늘의 C/C2 pilot |
| 더 정확해야 하고 시간 여유 있음 | D-Flow류 source 최적화 (B.1) | 오늘의 D/D2 |
| proxy 제약 위반을 연속시간 극한에서 0으로 | Lagrangian Dual Flows (B.3) | 미구현; true-mesh gate는 별도 |
| surrogate 예측치를 목표로 | Dflow-SUR (B.4) | differentiable 연결+OOD 검증 후 |
| 근접한 scalar 오차를 실측하며 줄이기 | proxy-Jacobian + true-measure line search | 오늘의 E2 pilot; LAMP와 무관 |
| 학습 범위 밖의 **parameterized same-topology bank** | LAMP (C.3) | 미구현 연구안 |
| 형상+물리를 한 latent로, 생성+검증 번갈아 | PhysGen (C.1) | 북극성 — Tier 2 이후 |
| 파트 단위 편집 | SPAGHETTI/SALAD + part objectives (A.3) | Tier 3.4, 데이터 확대 후 |

---

## 2부 — 오늘의 pilot: C2 / E2가 정확히 무엇인가

체크포인트 `output/geometry_generation/ex1`(VAE epoch 499 + FM epoch 299, latent `1x256`, MLP decoder)로
validation split의 첫 6형상(446, 1308, 781, 1750, 375, 1123), 형상당 4샘플, `ode_steps=50`을 측정했다.
각 candidate의 평가값은 res96 SDF decode → zero-level Marching Cubes → trimesh `volume/area`에서 얻었다.
이는 soft proxy가 아니라 **실제 export 경로의 수치 측정**이라는 뜻이지 연속 표면의 exact ground truth라는 뜻은
아니다. res96 discretization, Marching Cubes, watertight 판정, largest-component 선택의 수치 오차가 남는다.

### 2.0 먼저 고정해야 할 증거 수준

원본 `guidance_experiment.py`, `guidance_round2.py`, 두 JSON은 session scratchpad에 남아 있어 아래 집계와
코드 경로는 대조했다. 그러나 이 실험은 **방법 선택용 pilot**이지 독립 test benchmark가 아니다.

- calibration의 24점과 최종 평가는 같은 6개 validation target을 사용한다. C2/D2는 다른 noise seed지만 같은
  target이고, E2-on-A는 calibration에 쓴 24개 A sample을 그대로 재사용한다. 따라서 calibration/test leakage가
  있으며 결과는 낙관적일 수 있다.
- A는 seed `1000+s`, C2는 `2000+s`, D2는 `3000+s`라 method 간 source noise가 paired되지 않았다. 작은
  `n=24`에서 차이를 방법 효과만으로 돌릴 수 없다.
- 1라운드 C/D는 `soft_res=40,tau=0.02`, 2라운드 C2/D2는 `soft_res=48,tau=0.032`다. 같은 proxy 설정에서
  calibration on/off를 교차하지 않았으므로 `23.1%→1.7/2.1%`를 calibration **단독 효과**로 식별할 수 없다.
  E→E2도 proxy 설정, residual scaling, step cap, line search가 동시에 바뀌었다.
- 24개 sample은 6개 형상 안에 4개씩 묶인 clustered sample이다. 형상 일반화의 실질 표본 수는 6이고 p95는
  약 두 번째 worst observation이라 불확실성이 크다. confidence interval도 없다.
- aggregate JSON에는 sample별 raw measurement/latent/mesh가 없어 percentile을 원자료에서 재집계하거나
  shape plausibility를 시각 감사할 수 없다. `valid/watertight` 외 Chamfer, topology, diversity, latent drift,
  self-intersection/manufacturability를 측정하지 않았다.
- A baseline은 23/24 valid이고 나머지는 24/24다. 기존 표는 invalid를 percentile에서 제외하므로 validity를
  함께 읽어야 한다.

따라서 아래 수치는 **현재 checkpoint와 고정된 pilot protocol에서 관찰된 결과**로는 맞지만, "SOTA 확정",
"production 기본값", "다른 architecture에서도 99배"를 입증하지 않는다. 독립 calibration/test split과 paired
multi-seed 재현이 끝날 때까지 구현 후보로 취급한다.

### 2.1 미분가능 목적함수 — 모든 guidance/보정의 공통 재료

SDF 격자(48³)에서 volume/area를 미분가능하게 근사:

```text
occ(x) = sigmoid(-SDF(x) / τ)                    # 부드러운 내부/외부 지시함수, τ=0.032
volume = Σ occ(x) · h³                           # 복셀 부피의 가중합
area   = Σ ‖∇occ(x)‖ · h³                        # occupancy 경계의 "두께"를 면적 근사로
```

`vae.decode_flat`이 미분가능하므로 이 값들은 latent `z`에 대해 grad를 낼 수 있다. **이게 B.1~B.4 모든 문헌이
전제하는 "미분가능 목적함수"의 SDFFlow판 구현**이다. 다만 grid는 endpoint를 포함한 48개 node인데 모든 node에
동일 `h^3`, `h=2/(R-1)`를 곱하므로 엄밀한 cell-center/trapezoid quadrature가 아니다. 또한 A.0에서 보았듯
`ex1`은 eikonal-trained metric SDF가 아니어서 fixed `tau`가 일정한 공간 두께를 뜻하지 않는다. 이 두 수치
설정과 decoder bias까지 합친 proxy를 empirical calibration한 것이 다음 절이다.

### 2.2 C2 = calibrated endpoint-prediction guidance (FMPS-gradient-like)

```text
for i in range(ode_steps):
    v = velocity_net(z, t, cond)
    z_next = z + v * dt
    t_next = t + dt
    if t_next >= 0.3 and t_next < 1.0:
        v_next = velocity_net(z_next, t_next, cond)
        x1_hat = z_next + (1 - t_next) * v_next
        loss = ((soft_volume(x1_hat) - calibrated_target_volume) / target)^2
             + ((soft_area(x1_hat)   - calibrated_target_area)   / target)^2
        g = grad(loss, z_next)
        g = g / sqrt(mean(g^2) + eps)       # sample별 RMS normalization
        z = z_next - eta * (1 - t_next) * g
    else:
        z = z_next
```

**"C2"의 2는 calibration을 뜻한다.** 1라운드(C, 보정 없음)에서 이 방식은 volume 오차를 **7.6%→23.1%로
악화**시켰다 — 이유를 진단해보니 `soft_volume`이 실제 volume 대비 **median +168% 편향**(체계적으로 과대추정)
돼 있었다. 그런데 회귀해보니 R²=0.98로 **거의 완벽하게 선형**이었다(`soft ≈ 0.86·true + 0.40`). 그래서
true target을 proxy 좌표로 forward-map(`proxy_target = a·true_target + b`)한 뒤 같은 guidance를 돌린 게 C2다.
`(soft-b)/a`를 쓰는 inverse calibration은 아니다. area는 `soft≈0.357·true+1.653`, R²=0.598,
median bias −25.4%였다.

결과: volume median은 baseline 7.6% 대 C2 1.7%였고, area median은 5.2% 대 3.2%였다. 다만 2.0의
calibration/test leakage, unpaired seed, `res/tau` 동시 변경 때문에 "calibration만으로 4.5배"라고 인과
해석할 수는 없다. 동일 proxy/동일 `z0`의 factorial ablation이 필요하다.

**이산화 주의:** 실제 experiment의 correction은 `dt`를 곱한 velocity correction이 아니라 매 step의 state
jump다. 따라서 `ode_steps`를 25/50/100으로 바꾸면 총 guidance strength가 달라진다. 이 결과는 50 step에만
유효하며, production 구현 전에는 correction에 `dt`를 포함할지 또는 `eta`를 기준 step 수로 rescale할지 정하고
NFE-invariance ablation을 해야 한다.

### 2.3 D2 = 보정된 source-space 최적화 (§1.B.1, D-Flow 계열)

```text
z0 = randn(); z0.requires_grad = True
for outer_iter in range(40):
    z = z0
    for i in range(25):                   # 저해상도(25-step) sub-ODE — 비용 절감
        z = z + velocity_net(z, t, cond) * dt
    loss = descriptor_loss(z, calibrated_target)   # C2와 같은 손실
    loss.backward()                       # 25스텝 전체를 통해 z0까지 backprop
    adam_step(z0)
z_final = full_ode(z0, steps=50)          # 최적화된 z0에서 최종 고해상도 적분
```

D2의 volume median은 2.1%였다. 한 candidate trajectory당 `40*25=1000` 최적화 NFE에 최종 50-step ODE가
더해져 **1050 forward NFE + 전체 backward**다. C2는 50 base + 후반 35 lookahead = 85 NFE라 약 12.4배
적다. 그러나 C2가 D2를 strict하게 dominate하지는 않는다. C2가 volume median/p95는 낫지만 D2가 area
median/p95(`3.07/13.77%` 대 `3.24/15.33%`)는 조금 낫고, source seed도 paired되지 않았다. 현재 증거는
"비슷한 descriptor accuracy에서 C2가 훨씬 싸다"까지다. surrogate 전체를 terminal objective로 통과할 때 D2가
유리할지는 별도 실험 가설이다.

### 2.4 E2 = true-measure/proxy-Jacobian 보정 (norm cap + backtracking)

`GEOMETRY_UPGRADE...md` §4.9(e)의 "목표 이동 → 디코딩 → 실측 → 오차만큼 재이동" 반복을, **단일 축 수동
조정에서 다중 descriptor 동시 최소노름 linearized step으로 일반화**한 것이다. true residual과 proxy Jacobian을
섞으므로 하나의 동일 residual을 미분하는 고전적 Newton/Gauss-Newton보다는 **hybrid quasi-Newton**이 정확하다.

```text
for round in range(3):
    true_vol, true_area = measure_res96(z)
    Jv, Ja = grad(soft_volume, z_n), grad(soft_area, z_n)   # normalized flat latent에 대한 2xD Jacobian
    residual_proxy_units = [a_v*(target_vol-true_vol),
                            a_a*(target_area-true_area)]
    dz = J.T @ solve(J @ J.T + 1e-6*I, residual_proxy_units)
    dz = clip_norm(dz, cap=0.12*sqrt(D_flat))               # coordinate RMS step <= 0.12
    for step in [dz, dz/2, dz/4]:                           # line search
        z_try = z + step
        if sqrt((dv_true/tv)^2 + (da_true/ta)^2) decreases:
            z = z_try; break
```

핵심은 **방향/scale은 differentiable proxy Jacobian에서 얻고, step acceptance는 res96 실제 export 경로의
relative volume-area norm으로 판단**하는 것이다. 한 descriptor가 악화돼도 combined norm이 줄면 채택하며,
형상 품질 자체는 acceptance criterion이 아니다.

1라운드의 raw step은 A 위에서 volume median 2.44%였지만 p95 31.97%였고, 다른 start에서는 invalid/collapse와
최대 약 98% p95가 나왔다. 2라운드 E2는 A 위에서 volume median/p95 `0.284/4.158%`, area
`0.713/4.211%`였다. 이 개선은 line search만의 ablation이 아니라 cap `0.35→0.12`, calibration slope scaling,
`soft_res/tau` 변경이 함께 들어간 결과다. 그래도 **해당 pilot에서 단독 방법 중 가장 좋은 joint scalar accuracy**를
보인 것은 맞다.

### 2.5 C2+E2 — 최종 조합

C2 뒤에 E2를 적용한 pilot은 **volume median 0.077%, area median 0.255%**였다. baseline median 대비 약
99배/20배다. 실험용 best-4-of-16 rejection proxy(6.55%/2.58%) 대비 약 **85배/10배**다. 다만 checked-in
`config_sample_extrapolation.txt`는 `num_samples=32,candidate_multiplier=4`, 즉 best-32-of-128이므로 이
4-of-16 결과를 "현재 production의 실측값"이라 부를 수 없다. rejection fraction은 같아도 별도 측정이 필요하다.

또한 조합이 모든 tail metric을 지배하지는 않는다. C2+E2의 volume p95 2.55%는 E2의 4.16%보다 좋지만,
area p95 5.34%는 E2의 4.21%보다 나쁘다. median 중심 후보로는 강하지만 최종 채택안은 아니다.

### 2.6 전체 비교표

FM 비용은 **retained output 1개당 trajectory-equivalent forward NFE**로 단위를 맞췄다. 실제 GPU wall time은
batch size와 backward/decoder 비용에 따라 다르다.

| 방법 | valid | volume med / p95 | area med / p95 | FM NFE/output | 해석 |
|---|---:|---:|---:|---:|---|
| A. plain conditional | 23/24 | 7.60% / 15.12% | 5.21% / 12.12% | 50 | baseline |
| A-rej. 실험용 best-4-of-16 | 24/24 | 6.55% / 9.91% | 2.58% / 5.85% | 200 | 16*50/4; production mechanism proxy |
| C. uncalibrated local guidance | 24/24 | 23.07% / 27.09% | 3.05% / 13.11% | 85 | `res40,tau=.02` |
| D. uncalibrated source-opt | 24/24 | 23.11% / 27.40% | 5.96% / 11.92% | 1050 + BP | `res40,tau=.02` |
| **C2. calibrated local guidance** | 24/24 | **1.70% / 4.12%** | 3.24% / 15.33% | 85 | `res48,tau=.032` |
| D2. calibrated source-opt | 24/24 | 2.06% / 5.93% | 3.07% / 13.77% | 1050 + BP | `res48,tau=.032` |
| **E2 on A** | 24/24 | **0.284% / 4.16%** | **0.713% / 4.21%** | 50 + decoder/MC | 3 correction rounds |
| **C2+E2** | 24/24 | **0.077% / 2.55%** | **0.255% / 5.34%** | 85 + decoder/MC | median-best pilot |

calibration 24 sample 생성은 `+24 calls`가 아니라 총 **1200 trajectory-NFE**(6 target batch로 실행한 실제
FM forward invocation은 300)인 one-time offline cost다. D/D2의 1050은 `40*25` optimization forward와 최종
50-step ODE이고, backward 비용은 별도다. E2는 sample마다 초기 res96 측정 1회와 round당 최대 3개 candidate
측정, 그리고 soft-Jacobian decode가 든다.

2라운드 전체 wall time 23,427.9초(6.51시간)는 calibration+C2+D2+E2+C2+E2를 모두 직렬 수행한 값이다.
batch/chunk 최적화로 줄일 여지는 크지만, "수십 배 단축"은 아직 측정하지 않았다. 특히 attention decoder인
`v2`에서는 48³ query-to-32-token gradient graph가 훨씬 비싸므로 별도 memory/time benchmark가 필요하다.

---

## 3부 — 앞으로 할 일 (구현 계획)

### 3.1 순서와 이유

1. **먼저 pilot을 재현 가능한 benchmark로 승격한다.** scratch script를 checked-in evaluation tool로 옮기고
   checkpoint/script hash, git commit, package/GPU, method별 동일 `z0`, sample별 raw measurement, validity,
   latent drift와 mesh path를 저장한다. calibration은 validation split, 최종 평가는 손대지 않은 test split으로
   분리한다. 이것 없이는 다음 구현의 acceptance 기준이 없다.
2. **descriptor proxy와 calibration을 독립 module로 만든다.** `general_modules/descriptor_calibration.py`에
   cell-center 또는 trapezoid quadrature, soft volume/area, affine fit `(a,b,R²)`, residual plot/stat를 넣는다.
   calibration artifact에는 VAE/FM SHA256, soft resolution/tau, MC resolution, descriptor names, split indices를
   함께 기록한다. 기존 checkpoint를 제자리 수정하기보다 versioned sidecar 또는 새 bundle로 저장해 provenance를
   보존한다. sphere/box analytic SDF로 proxy bias와 gradient부터 검증한다.
3. **E2는 opt-in 후처리로 먼저 구현한다.** `sample.py` 또는 별도 `general_modules/descriptor_refinement.py`에
   2×D solve, normalized-flat-latent RMS cap, backtracking, per-step true report를 넣는다. 현재 rejection과
   `max_condition_z`를 대체하지 않고 함께 두며, line search objective에 descriptor뿐 아니라 invalid mesh,
   latent drift/shape-quality gate를 추가한다.
4. **C2는 VAE-aware 상위 sampler로 구현한다.** `model/velocity_net.py::sample_latents`에 SDFVAE와 grid logic을
   직접 넣으면 model layer가 decoder/calibration/MC에 결합된다. `sample.py`가 generic guidance callback을
   넘기거나 `general_modules/descriptor_guidance.py`가 ODE를 감싸게 한다. correction을 `dt`가 포함된 velocity로
   정의할지 per-step jump로 정의할지 먼저 고정하고 25/50/100 NFE invariance를 시험한다.
5. **config와 spec을 동시에 갱신한다.** native parser, `cae_suite/specs/sdfflow.py::known_keys`, sample-mode
   active fields, validator를 한 change로 묶는다. calibration metadata가 현재 checkpoint/config와 맞지 않으면
   silent reuse하지 말고 hard error로 막는다.
6. **CI는 결정론적 unit/integration test로 나눈다.** analytic sphere/box 또는 mock decoder로 (a) soft
   descriptor gradient 방향, (b) affine mapping, (c) damped solve shape, (d) backtracking accepted step의 true
   residual non-increase, (e) disabled path bitwise equivalence를 검사한다. "tiny VAE+FM을 몇 step 학습하면 stochastic
   baseline보다 좋아진다"는 assertion은 flaky하므로 CI 기준으로 쓰지 않는다. 실제 ex1 res96 benchmark는
   느린 local/GPU regression으로 둔다.
7. **area proxy는 별도 ablation한다.** resolution/tau/quadrature를 먼저 교차하고, 그래도 R²가 낮으면
   DMTet/FlexiCubes류 differentiable surface extraction을 검토한다. 현재 6-shape 결과만으로 교체 우선순위를
   확정하지 않는다.
8. **Lagrangian Dual Flows와 Dflow-SUR는 Tier 2 이후 별도 연구다.** LDF는 soft proxy의 finite-step tolerance와
   true-mesh gate를 같이 평가하고, surrogate guidance는 geometry→solver-input 변환까지 end-to-end gradient와
   surrogate OOD/CFD 검증이 준비된 뒤 진행한다.

### 3.2 새 config 키 (초안)

```text
% guided sampling (C2)
guidance_enabled          false          % 독립 test 재현 전 default-off
guidance_t_start          0.3
guidance_eta              0.1
guidance_step_mode        velocity_dt    % 또는 per_step_jump; eta 의미를 명시
guidance_targets          volume,area
soft_descriptor_resolution 48
soft_descriptor_tau       0.032
descriptor_calibration_path ../output/.../descriptor_calibration.pth

% Newton 보정 (E2)
newton_rounds              0              % 0=off; 검증 config에서 3
newton_step_cap_rms        0.12           % cap=값*sqrt(latent_flat_dim)
newton_line_search_tries   3
newton_measure_resolution  96

% calibration
calibration_num_shapes         64          % batch size와 총 sample 수를 분리
calibration_samples_per_shape  4
calibration_batch_size         2           % decoder memory tuning
```

`sample_latents`/`sample.py`의 기존 `cond_values`, `candidate_multiplier`, `max_condition_z` 키와 공존—
guidance/Newton은 `cond_values`가 주어졌을 때만 의미가 있으므로 미지정 시(무조건부 생성) 자동 비활성.

### 3.3 무엇을 성공 기준으로 볼 것인가

6개 validation 형상 재측정만으로 default를 바꾸지 않는다. 최소 기준은 다음과 같다.

1. validation에서만 calibration/hyperparameter를 고정하고, 독립 test target에서 모든 방법이 같은 initial
   `z0`를 쓰는 paired multi-seed 평가를 한다. 형상 단위 bootstrap confidence interval을 함께 낸다.
2. rejection baseline 대비 volume/area median 개선뿐 아니라 p95, valid/watertight rate, no-zero-crossing,
   latent RMS drift, topology/Chamfer 또는 최소한 visual audit가 악화되지 않아야 한다. 특히 C2+E2 area p95가
   E2보다 나빴던 tail을 다시 본다.
3. wall time, peak VRAM, decoder calls를 함께 보고한다. FM NFE만으로 production 비용을 판단하지 않는다.
4. 이 조건에서 **volume 10배, area 5배** 개선의 paired estimate와 lower confidence bound가 유지될 때 opt-in
   production profile로 승격한다. 한 release 동안 rejection/true-measure fallback을 유지한 뒤 default를 검토한다.
5. `ex2`가 실제 학습되면 그 checkpoint에 대해 calibration을 새로 하고 같은 protocol을 반복한다. `ex1`
   계수를 재사용하지 않는다.

---

## 4부 — 출처

**Guidance/제약 메커니즘**

- D-Flow (ICML 2024) — https://arxiv.org/abs/2402.14017 / https://proceedings.mlr.press/v235/ben-hamu24a.html
- FlowGrad (CVPR 2023) — https://openaccess.thecvf.com/content/CVPR2023/html/Liu_FlowGrad_Controlling_the_Output_of_Generative_ODEs_With_Gradients_CVPR_2023_paper.html
- OC-Flow (Training-Free Guided Flow Matching with Optimal Control, ICLR 2025) — https://arxiv.org/abs/2410.18070
- On the Guidance of Flow Matching — https://arxiv.org/html/2502.02150
- Constrained Flow Matching via Lagrangian Dual Flows — https://arxiv.org/html/2607.04513v1
- Dflow-SUR — https://arxiv.org/html/2512.08336
- Flow Matching Posterior Sampling — https://arxiv.org/html/2411.07625v3

**백본**

- 3DShape2VecSet — https://arxiv.org/abs/2301.11445
- TripoSG — https://arxiv.org/abs/2502.06608
- Hunyuan3D 2.0 — https://arxiv.org/pdf/2501.12202
- TRELLIS (Structured 3D Latents) — https://arxiv.org/html/2412.01506
- SPAGHETTI — https://dl.acm.org/doi/abs/10.1145/3528223.3530084
- SALAD — https://arxiv.org/abs/2303.12236

**엔지니어링 역설계 통합 시스템**

- PhysGen (CVPR 2026) — https://arxiv.org/pdf/2512.00422 / https://kasvii.github.io/PhysGen/
- 3DID — https://arxiv.org/html/2512.08987
- LAMP — https://arxiv.org/html/2510.22491v3

**데이터**

- DeepJEB — https://arxiv.org/abs/2406.09047
- DrivAerNet++ — https://github.com/Mohamedelrefaie/DrivAerNet

**오늘의 실측 실험 원본 스크립트/결과** (세션 스크래치패드, 참고용이며 checked-in 아님):
`guidance_experiment.py`(1라운드), `guidance_round2.py`(2라운드, calibration+line-search),
`guidance_results.json`, `guidance_round2_results.json`. 최종 benchmark로 쓰기 전에 §3.1처럼 원자료·환경·hash를
포함한 checked-in evaluation artifact로 승격해야 한다.
