# MGN-V global / local / hybrid 검토

검토일: 2026-10-19 (Asia/Seoul). 현재 소스·설정·HDF5 메타데이터와 1차 논문을 대조했다. 실제 ex3 학습, FEA 추가 실행, 새로운 성능 비교는 수행하지 않았다. 아래에서 구현 사실, 데이터 사실, 문헌 사실, 검증할 가설을 구분한다.

## 1. 권하는 연구의 중심

**같은 계층적 메시 모델에서 확률상태의 공간적 배치를 바꾸었을 때, 어떤 조건에서 국소 변동·장거리 의존성·구조 응답 확률을 더 효율적으로 재현하는가?**

이 질문은 유효하다. 다만 global+local latent, coarse latent diffusion/FM, stochastic structural surrogate, one-to-many buckling 자체에는 직접 선행연구가 있다. Hybrid의 우수성을 미리 전제하거나 세 모델의 기본 설정 비교만으로 배치 위치의 인과적 효과를 주장해서는 안 된다.

현재 원고도 용량 차이와 여러 대조군을 이미 인정한다: [원고의 연구질문](../../paper/HI-MGN-V/HI-MGN-V.tex), RQ1–RQ4 및 A1–A6. 아래 제안은 그 계획을 확인하고, 현재 구현·데이터에서 특히 중요한 추가 해석을 정리한 것이다.

## 2. 현재 구현은 무엇을 비교하는가

| 유형 | 실제 stochastic state | ex3 설정의 좌표 수 | prior의 velocity network |
|---|---|---:|---|
| global | 그래프별, V-cycle level별 전역 slot | 3×32 = 96 | 전역 joint vector를 처리하는 MLP; mesh GNN이 condition 제공 |
| local | coarsest node별 regional state | 256×4 = 1,024 | coarsest graph GNN |
| hybrid | 두 state의 결합 | 96+1,024 = 1,120 | regional stream과 graph token이 양방향 통신하는 하나의 coupled graph FM |

256은 설정상 coarse cluster 목표다. 실제 그래프가 그 수를 유지하는지 확인해야 하며, 일반적으로 local 차원은 실제 n_c×C다. Local은 fine-node IID noise가 아니다. Coarsest ancestor의 동일한 regional code를 여러 V-cycle level과 block에서 다시 읽는다. Global 역시 level별 slot이므로 단일 32차원 code라고 설명하면 부정확하다.

근거: [MeshGraphNets.py](../../methods/MeshGraphNets_Variational/model/MeshGraphNets.py) 268–326, 467–480, 615–725; [latent.py](../../methods/MeshGraphNets_Variational/model/latent.py) 1–19, 56–69; [conditional_prior.py](../../methods/MeshGraphNets_Variational/model/conditional_prior.py) 488–508, 638–656.

Hybrid posterior는 shared trunk에서 두 Gaussian head를 계산하고 각각 독립 noise로 reparameterize한다. 따라서 고정된 관측 (u,c)에 대한 posterior noise는 mean-field다. 그러나 u를 적분한 aggregated posterior에서 global/local이 독립이라는 뜻은 아니다. 반대로 prior는 독립 base noise에서 출발해도 coupled velocity로 의존성을 학습할 수 있다. 근거: [vae.py](../../methods/MeshGraphNets_Variational/model/vae.py) 142–170.

## 3. 반드시 비교해야 할 선행연구

| 1차 문헌 | 원문에서 확인한 선행 내용 | 우리에게 남는 구체적인 질문 |
|---|---|---|
| [HPU-Net, 2019](https://arxiv.org/pdf/1905.13077) | 계층적 spatial latent를 쓰는 conditional VAE. Appendix F가 global-only/local-only/hierarchy를 직접 비교하며 global 차원도 맞춘다. | 세 배치 비교 자체의 최초성은 약하다. 비정형 물리 메시에서 capacity·통신·joint law를 분리하는 실증이 필요하다. |
| [LION, NeurIPS 2022](https://arxiv.org/abs/2210.06978) | Global shape vector와 point-structured latent; p(z)p(h|z)의 계층적 latent diffusion; 여러 층의 modulation. | Global+spatial latent 자체는 선행. 기존 mesh 위 응답장, coarse ancestry, 하나의 coupled graph FM이라는 조합과 효과를 평가한다. |
| [INADE, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Tan_Diverse_Semantic_Image_Synthesis_via_Probability_Distribution_Modeling_CVPR_2021_paper.html) | Instance별 noise를 normalization에 주입하고 여러 층에서 동일 sample을 일관되게 재사용. | 지역별 noise의 반복 주입도 선행이다. Mesh ancestry에 정렬한 재주입의 효과를 단회 주입과 비교한다. |
| [DualDiff, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/file/af2bb2b2280d36f8842e440b4e275152-Paper-Conference.pdf) | Graph의 node/cluster 두 denoising branch가 cross-conditioning하고 번갈아 갱신된다. | Global/local 양방향 coupling 자체의 최초성도 피한다. Graph-global token+regional state의 동시 joint FM과 조건부 물리장이라는 범위를 명시한다. |
| [LDGN, ICLR 2025](https://arxiv.org/html/2504.02843v1) | Coarse-node variational latent와 conditional diffusion 또는 FM을 이용한 mesh flow-field 생성; OOD 조건과 분포를 평가. | Local arm의 가장 가까운 선행이다. Coarse latent+FM 자체보다 구조 불안정성과 배치 비교가 차이다. |
| [SAR, 2026 preprint](https://arxiv.org/html/2604.11403v1) | 비정형 mesh 물리장의 scale-autoregressive 생성, FM, 전역 attention과 공간 통계 평가. | Stochastic support, 통신 범위, scale별 생성 순서를 구분해야 한다. |
| [Bi-FORK, 2026-10-08 preprint](https://arxiv.org/html/2610.12449v1) | 보 좌굴·메타물질·Allen–Cahn의 다중해 궤적을 latent FM으로 생성. Repulsion-guided sampling과 mode JSD 평가. | 좌굴 one-to-many latent FM은 선행이다. 일반적인 imperfection law의 pushforward 확률과 공간 의존성을 정확히 재현하는지가 우리의 검증 대상이다. |
| [Diffusion Last Layer, ICML 2026](https://arxiv.org/html/2602.04139v2) | Input-dependent basis의 계수에 conditional diffusion을 적용해 stochastic PDE field를 생성. | Global vector도 국소 패턴과 공간 상관을 표현할 수 있다. Local의 필요성을 표현 불가능성 대신 효율·학습 편향으로 검증한다. |
| [PNO, TMLR 2025](https://arxiv.org/html/2502.12902v2) | Function-space proper energy score로 stochastic operator를 학습. | Reconstruction+regularizer+latent prior가 direct proper-score stochastic surrogate 대비 어떤 장점이 있는지 비교 가능하다. |
| [GeoLAMP, 2026 preprint](https://arxiv.org/html/2609.00297v2) | Global/local GNO encoder grid를 더한 뒤 하나의 Gaussian latent bottleneck으로 압축하고 latent dynamics를 생성. | Global/local 입력 encoding과 별도 stochastic state 배치는 다르다. 논문 제목의 유사성만으로 같은 모델이라 분류하면 안 된다. |
| [Patrignani & Pinho, CMAME 2026](https://spiral.imperial.ac.uk/entities/publication/41490ed8-1bdd-44ff-b20f-1106abe1554d) | 구조 응답을 위한 deterministic mesh GNN의 hybrid local-global attention. | 전역 통신의 효과와 전역 확률상태의 효과를 분리해야 한다. |

HPU-Net Appendix F의 global/local/hierarchy 비교는 특히 직접적이다. 다만 이미지 segmentation이며 global arm의 reconstruction constraint도 조정했으므로 결과 숫자를 물리 mesh의 보편 법칙으로 가져올 수 없다. LION도 순차적 조건부 latent prior이므로, 현재의 하나의 joint ODE와 동일한 구현은 아니다. 이런 구체적인 차이가 기여 후보이지 개별 구성요소의 최초성은 아니다.

Bi-FORK는 mode 확률을 JSD로 평가한다. 따라서 '선행은 모드만 찾고 우리는 확률을 처음 평가한다'는 주장은 부정확하다. 그 논문은 방향 선호 없는 perturbation에서 symmetry orbit의 균등분포를 목표로 삼고, K개 sample의 repulsion을 사용한다. 우리 목표가 지정된 결함법칙에 따른 비균등 확률·tail이라면 unguided sampling의 calibration과 별도로 비교해야 한다. Guidance를 사용하면 실제 법칙의 보존 여부를 다시 검증해야 한다.

## 4. 현재 데이터에서 직접 확인한 중요한 점

[dataset_metadata_audit.json](dataset_metadata_audit.json)은 현재 설정의 두 HDF5를 읽은 결과이며, [재현 스크립트](audit_dataset_metadata.py)를 함께 둔다.

- 학습용 파일: 150 geometry×64 draw = 9,600. 평가용 infer 파일: 8×64 = 512. 둘 다 T=1이다.
- 전체 158 geometry에서 K=1이다. 단일 dimple의 폭은 geometry에 종속되고 random recipe는 진폭 a, 각도 θ, 축방향 위치의 세 변수로 기술된다.
- 전체 geometry에서 같은 seed의 진폭 a와 각도 θ가 정확히 같다. 축방향 위치를 (z−2σ)/(L−4σ)로 정규화했을 때 최대 차이는 약 2.56e−5다. 따라서 현재 평가 파일을 독립적인 새 imperfection recipe 검증으로 해석할 근거가 없다.
- 원고 711–724행에는 별도의 dense/OOD 8×200 평가와 fresh recipe가 기술돼 있다. 현재 config에 연결된 8×64 HDF5와 동일한 artifact라고 취급하면 안 된다. 별도 데이터·예측 archive·checkpoint provenance를 연결해야 한다.
- 학습 geometry의 λ 범위는 약 0.526–3.488이고 infer는 0.441–2.754다. 원시 R,t,L,Δ의 out-of-box 외삽과 무차원 물리 regime의 외삽은 같은 의미가 아니다. 각 test가 어느 무차원 축에서 벗어나는지도 보여야 한다.
- Terminal KE/IE 메타데이터는 학습 median 0.0338, max 0.1758이고 infer median 0.0281, max 0.1129다. 이는 기술 통계이며 특정 acceptance threshold 위반 판정이 아니다. 다만 T=1 저장만으로 정적 평형 데이터라고 결론낼 수 없으므로, loading-rate·damping·정착 기준과 응답의 물리적 의미를 명시해야 한다.

단일 dimple라는 점은 연구 해석에 중요하다. Fixed condition에서 random input이 세 변수이면 응답장의 고차원성이 곧 높은 intrinsic stochastic dimension을 뜻하지 않는다. Global 96차원이 정보를 담기에 원리적으로 부족하다고 주장하기 어렵다. Local이 더 좋다면 위치 표현의 학습 효율, pooling 병목, decoder 접근성, prior 학습 난도 때문일 수 있다. (a,θ,z)를 드러낸 privileged-input surrogate와 Monte Carlo가 유용한 기준이다. 이 기준의 성능이 수학적인 upper bound라는 보장은 없다.

## 5. 우선 해결해야 할 실험 해석 문제

### 5.1 용량, 배치 위치, 통신, 주입 횟수

기본 여섯 arm은 좋은 native-recipe 비교지만 순수한 placement 비교는 아니다. 차원, head, prior 구조, 파라미터, FLOPs가 함께 달라진다. 차원·파라미터·시간을 동시에 완전히 맞추기 어려우므로 다음을 분리해 보고한다.

1. 실제 기본 설정의 품질–시간–메모리 Pareto 비교.
2. 대략 동일한 latent coordinate budget에서의 비교.
3. 전체 파라미터/학습비용을 맞춘 비교.
4. Local+deterministic global communication 대 local+stochastic global state.
5. 같은 hybrid decoder와 branch 크기에서 coupled prior 대 uncoupled prior.
6. Coarsest 단회 주입 대 여러 V-cycle block에서의 재주입.

Global 크기를 1,024 근처로 키우는 것만으로 모든 confound가 사라지지는 않는다. Prior의 MLP/GNN·pooling·conditioning 폭도 함께 보고해야 한다. 통신 대조군은 prior와 simulator 중 어느 쪽에 global path를 추가했는지도 명시한다.

### 5.2 정규화와 FM의 크기 및 의미

- Global MMD는 slot별 graph sample을 비교한다. 현재 한 GPU의 batch_size=2이면 grad_accum_steps=4여도 MMD 통계의 graph 수는 8이 아니다.
- Local MMD는 [coarse rows,C]를 표본으로 삼는다. 위치와 geometry가 섞인 node marginal을 정규화하며, 전체 latent field의 joint law를 맞추지 않는다. 서로 상관된 coarse rows를 독립 realization 수백 개로 세면 안 된다.
- Hybrid는 MMD_g+MMD_l, KL_g+KL_l, FM_g+FM_l을 합한다. KL은 branch 안의 coordinate mean 뒤 합이다. 동일 coefficient가 동일 total rate나 동일 regularization pressure를 의미하지 않는다.
- Global MMD도 level slot별 평균이므로 slot 사이 의존성을 직접 정규화하지 않는다. Hybrid part-wise MMD 역시 g–l joint dependence를 정규화하는 항은 아니다.
- Weighted loss뿐 아니라 branch별 gradient 크기, active dimensions, unnormalized total KL도 기록한다. Alpha_recon=1000 대 다른 계수=1이라는 숫자만으로 후자의 영향이 0.1%라고 결론낼 수 없다.

근거: MeshGraphNets.py 395–442; latent.py 104–113; vae.py 203–232; conditional_prior.py 689–718.

### 5.3 현재 목적함수와 학습 일정의 정확한 명칭

현재 ex3는 prior_grad_to_encoder=0이다. Posterior sample을 detach하고 conditional prior가 따라간다. KL arm은 learned conditional prior가 아니라 N(0,I)에 대한 KL을 사용한다. 'Regularized variational encoder + conditional latent flow'라는 설명이 정확하며, exact conditional ELBO라고 부르려면 별도 유도가 필요하다.

FM endpoint로 gradient를 열었다고 FM regression MSE가 자동으로 KL(q||p)가 되지 않는다. Decoded-field PV auxiliary도 증명된 mutual-information lower bound가 아니다. Aggregate MMD 역시 conditional calibration이나 latent utilization을 보장하지 않는다. [Flow Matching 원 논문](https://arxiv.org/abs/2210.02747), [InfoVAE 원 논문](https://arxiv.org/abs/1706.02262)과 구분해 기술해야 한다.

설정상 120 epoch 중 처음 80 epoch는 joint, 이후 40 epoch는 simulator/posterior를 고정하고 prior만 학습한다. Freeze 때 standardization, optimizer와 LR 일정도 바뀐다. Prior-only loop는 grad accumulation 없이 microbatch마다 optimizer step을 수행한다. 따라서 freeze 하나만의 효과를 나타내는 실험이라고 해석하면 안 된다. 근거: single_training.py 227–264; training_loop.py 511–555, 639–653.

현재 문서 중 `docs/reference/config/meshgraphnets_variational.md` 139행의 single-stage 설명, 161·243행의 FM-gradient/CVAE-rate 보장, 233행의 PV/MI-floor 표현과 `vae.py` 177–180행의 MMD 보장은 수정 검토 대상이다. 이번 검토에서는 해당 파일을 편집하지 않았다.

## 6. Hybrid의 역할 분담을 검증하는 법

'Global은 mode, local은 detail', 'global은 저주파, local은 고주파'는 유용한 가설이지만 현재 구조가 보장하지 않는다. Node-conditioned global modulation은 localized pattern도 만들 수 있고 regional latent도 decoder 통신을 통해 먼 영역에 영향을 줄 수 있다.

- 최종 decoder 입력 z_g를 고정하고 z_l를 교체하는 실험과 그 반대를 수행한다. Mode coefficient, buckle 위치·방향, 파장, 지역 variance, 전체 QoI의 변화와 spatial Jacobian/sensitivity를 측정한다.
- 단, coupled FM에서 base noise ε_g를 고정해도 최종 z_g는 고정되지 않는다. V_g가 local stream을 읽기 때문이다. Base-noise intervention과 최종-latent decoder intervention을 구분한다.
- 독립적으로 뽑은 z_g,z_l를 임의로 교차하면 joint support 밖으로 나갈 수 있다. 이를 p(z_l|z_g,c)의 조건부 sampling으로 부르면 안 된다. 조건부 sampling이 없으면 decoder stress test라고 표시한다.
- Local code를 바꾸었을 때 먼 영역이 덜 변하는 것이 항상 좋은 것은 아니다. 구조 역학의 올바른 응답은 비국소적일 수 있다. 실제 FEA covariance나 민감도와 비교해야 한다.
- Hybrid 두 branch가 실제 쓰이는지 branch removal, active units, Jacobian과 within-condition 변화로 확인한다. 단순히 두 branch를 만들었다고 해석 가능한 disentanglement가 생기지 않는다.
- FPS seed, cluster 수, hierarchy variant, remeshing에 따른 결과와 sample 분산을 별도로 측정한다. Hierarchy randomization으로 생긴 spread를 물리적 imperfection uncertainty와 혼합하지 않는다.

[RGFM, 2026](https://arxiv.org/html/2608.23696v1)도 multiscale local computation이 장거리 상관을 재현할 수 있음을 보인다. 따라서 'local은 장거리 상관을 표현할 수 없다'보다, 현재 depth·ODE step·coarsening·비용에서 언제 global state가 도움이 되는지를 시험하는 것이 적절하다.

## 7. 분포 평가에서 빠지면 안 되는 것

| 평가 축 | 권장 측정 | 주의점 |
|---|---|---|
| 조건별 scalar calibration | max magnitude·radial displacement·좌굴 확률·quantile의 CRPS/W1/coverage | 조건을 섞은 histogram은 conditional mismatch를 가린다. 확률 event 정의를 고정한다. |
| 장거리 joint dependence | area-weighted field ES + 거리별 variogram score, covariance, modal spectrum | Nodewise CRPS만으로 공간 coherence를 입증할 수 없다. ES도 correlation에 둔감할 수 있다. |
| 샘플 물리적 품질 | BC 오차, admissibility, mode/파장, solver correction·reference 대비 residual | 숨겨진 결함이 있는 기하에서의 해이므로 nominal geometry residual=0을 무조건 요구하면 안 된다. |
| 지리적/물리적 shift | In-box unseen shapes, transition, parameter extrapolation, mesh shift를 구별 | 숨겨진 결함법칙 자체가 바뀌면 conditioning으로 식별할 수 있는지 따로 검토한다. |
| 통계적 신뢰도 | 여러 training seed, geometry별 점수와 CI, sample-count sweep, matched real–real reference | 9,600개 draw를 9,600개의 독립 geometry처럼 세지 않는다. Common recipes의 dependence를 반영한다. |

현재 best_by=crps는 scalar node/component 점수다. 기본 validation prior draw 수는 별도 키 vae_valid_prior_samples의 기본값 8이며, train config의 num_vae_samples=16이나 inference64와 다르다. 고정된 evaluation noise와 충분한 draw로 model selection의 Monte Carlo noise를 줄여야 한다. 노드 평균은 큰 mesh에 더 큰 가중치를 줄 수 있으므로 geometry 평균과 면적 가중 평균도 함께 보고한다. 근거: training_loop.py 94–145, 947, 980–1060.

현재 PV auxiliary는 기본 uz의 max−min이고 ex3 최종 scalar는 max||u||다. 이 둘을 동일한 좌굴 크기라고 부르면 안 된다. Max||u||에는 imposed axial shortening도 포함되므로, 물리적 buckling 판정과 radial deformation·임계하중 관계도 교차 확인한다.

Finite real–real 점수는 동일 sample count·동일 조건으로 구성한 비교 기준이며 보편적인 hard lower bound가 아니다. CRPS/ES는 완벽한 예측분포에서도 기대점수가 0이 아니다. 분할 크기와 seed 불확실성을 보고한다. 64 FEA draw로는 드문 failure의 tail 확률을 정밀 추정하기 어렵다. [Scheuerer & Hamill, 2015](https://repository.library.noaa.gov/view/noaa/22327/)은 ES와 variogram score를 함께 써야 할 직접적인 근거다.

## 8. 권하는 실행 순서와 주장 범위

1. **데이터·평가 계약 고정.** Current 64-draw 파일과 별도 200-draw archive, field metric, validation sample count, conditioning·hidden law를 확정한다. Fresh recipes를 가진 held-out reference를 확보한다.
2. **기본 6-arm을 해석 가능한 결과로 만든다.** Reconstruction/posterior-decoded/prior-decoded를 같은 condition에서 비교한다. 각 branch의 loss·active units·시간·메모리·여러 training seed를 기록한다. 특정 arm을 사전 승자로 정하지 않는다.
3. **관찰된 차이를 설명하는 최소 대조군을 추가한다.** Capacity matching, deterministic global path, coupled/uncoupled prior, 반복 주입 중 실제 가설을 구별하는 대조군을 한 번에 한 가지씩 수행한다.
4. **단일 dimple 밖으로 검증 범위를 확장한다.** 여러 dimple 또는 상관길이가 다른 random imperfection field, 다른 geometry/BC에서 결과가 유지되는지 본다. Hidden law가 다른 경우 law hyperparameter를 입력할지 평가 범위로만 둘지 정한다.

현재 가능한 기여 문장:

> 동일한 계층적 메시 생성모델에서 graph-global, coarse-regional, hybrid 확률 표현을 비교하고, 결함으로 유도된 조건부 응답장의 국소 변동과 장거리 의존성을 재현하는 데 필요한 표현·통신·decoder 접근성을 구분하여 검증한다.

Hybrid 결과가 나온 뒤에만 추가할 문장:

> 동일 예산과 전역 통신 대조군에서도 joint global–regional prior와 mesh-ancestry 기반 반복 주입이 조건부 공간 상관 및 구조 QoI 분포를 개선했다.

피해야 할 현시점 주장: 최초의 global+local latent, 최초의 stochastic mesh surrogate/one-to-many buckling, global의 국소 표현 불가능성, hybrid의 자동 disentanglement, exact conditional ELBO/MI 보장, 검증하지 않은 mesh independence·OOD calibration·우수성.

## 9. 이번 검증의 범위

기존 `methods/MeshGraphNets_Variational/tests/test_latent_type.py`를 CPU에서 실행하여 **13 passed**를 확인했다. 작은 합성 메시에서 여섯 조합의 latent shape, 선택한 regularizer, 학습·prior validation 경로, freeze/standardization 및 prior-only 업데이트를 확인하는 구조 검증이다. 실제 ex3 field quality, 학습 완료, GPU capacity 또는 hybrid의 우수성을 보여주는 결과는 아니다.

검토에서 생성한 파일은 이 폴더의 연구 메모·메타데이터 요약·재현 스크립트다. 기존의 모델 코드, 설정, 원고와 기존 작업 변경사항은 편집하지 않았다.
