# cHI-MGNflow: 계층적 메시 그래프 위의 2단계 잠재 Flow-Matching 모델

*내부 기술 리포트 — AI-CAE4ALL, `methods/HI_MGNFlow`. 2026-09-18.*

## 초록

메시 기반 시뮬레이션 대체 모델(surrogate model) 중 일부 문제는 하나의
형상·경계조건에 유일한 정답이 대응하지 않고, **가능한 결과들의 분포**가
대응한다(접촉, 좌굴 등 경로·초기조건에 민감한 물리). 본 리포트는 이런
문제를 위해 이 코드베이스가 채택한 조건부 생성 모델 cHI-MGNflow의 구조를
기술한다. cHI-MGNflow는 Lino, Pfaff & Thuerey (ICLR 2025)의 논문 *Learning
Distributions of Complex Fluid Simulations with Diffusion Graph Networks*가
제안한 LDGN(그 논문이 자신의 압축-latent 변형에 붙인 이름)의 2단계 설계
— 근손실(near-lossless) 그래프 압축기(stage 1)와, 그 압축된 잠재 공간 위에서만 동작하는
flow-matching prior(stage 2) — 를 그대로 채택하고, 논문 자체의 멀티스케일
백본을 이 리포지토리의 계층적 HI-MGN V-cycle로 대체한 것이다. 이전 버전
(field-space flow matching, 압축 없음)은 학습·추론 시점의 잠재 분포는
일치시켰지만 압축을 전혀 하지 않아 ODE 적분 스텝마다 전체 V-cycle을
다시 계산해야 했고, reconstruction 단계에서부터 결과가 맞지 않았다. 본
아키텍처는 분포 일치와 압축을 동시에 만족시켜, 샘플 하나당 전체 V-cycle
forward를 정확히 2회(encode 1회, decode 1회)로 고정하고, K-step ODE
적분은 가장 coarse한 메시 레벨 하나에만 국한시킨다.

## 1. 서론

MeshGraphNets류 모델은 결정론적 encode-process-decode 구조로 메시 위의
물리량을 예측한다. 그러나 결과가 초기 미세조건이나 경로에 따라 갈리는
문제(접촉, 좌굴 모드 선택 등)에서는 하나의 예측값이 아니라 조건부 분포
`p(y|c)`를 모델링해야 하며, 학습과 추론에서 이 분포로부터 표본을 뽑는
절차 자체가 아키텍처의 일부가 된다.

이 코드베이스는 이 문제에 두 가지 설계로 접근해왔다.

1. **MeshGraphNets-Variational** — VAE로 posterior `q(z|y,c)`를 학습하고,
   추론용으로 별도의 근사 prior `p(z|c)`를 학습한다. 두 분포가 정확히
   같을 이유가 없어 aggregate-posterior drift가 발생한다.
2. **cHI-MGNflow (이전 버전)** — 이 불일치를 없애기 위해 필드 공간에서
   직접 conditional flow matching을 적용했다: 학습·추론 모두 `N(0,I)`에서
   출발해 같은 ODE로 목표 분포에 도달하므로 분포 불일치 자체가 없다. 다만
   압축이 전혀 없어 `[N_nodes, output\_var]` 전체 크기에서 flow가 동작했고,
   ODE 적분 스텝 하나마다 전체 V-cycle을 다시 계산해야 했다. 이 설계는
   reconstruction 단계에서부터 결과가 맞지 않았다.

참고 논문(Lino, Pfaff & Thuerey, LDGN)을 다시 검토한 결과, 이 논문은
필드 공간이 아니라 **압축된 coarse 잠재 위에서** flow matching을 수행하고
있었다. 본 리포트가 기술하는 아키텍처(이하 v2)는 이 논문의 설계를 그대로
채택하고 논문의 자체 멀티스케일 백본만 이 리포지토리의 HI-MGN V-cycle로
교체한 것이며, 이전 버전(v1)을 완전히 대체한다.

## 2. 관련 연구

- **MeshGraphNets** (Pfaff et al., ICLR 2021) — 메시 그래프 위의 결정론적
  encode-process-decode 시뮬레이션 대체 모델.
- **계층적(멀티스케일) 확장** — 코스닝을 통해 만든 여러 레벨 사이를
  오가며 장거리 상호작용을 저비용으로 전달하는 V-cycle 구조(이 리포지토리의
  HI-MGN). 본 아키텍처는 이 V-cycle을 압축기·prior 양쪽의 공통 백본으로
  그대로 사용한다.
- **MeshGraphNets-Variational** — 풀링된 전역 벡터에 대한 VAE + 별도
  학습된 prior. 위에서 설명한 posterior/prior 불일치의 근원.
- **Flow matching** (Lipman et al., ICLR 2023) 및 **rectified flow**
  (Liu, Gong & Liu, ICLR 2023) — 직선 경로를 따라 단순 분포(`N(0,I)`)를
  데이터 분포로 옮기는 continuous normalizing flow를 시뮬레이션-프리
  회귀만으로 학습하는 기법. 학습이 ODE를 한 번도 적분하지 않고, 적분 스텝
  수 K는 추론 시점에만 결정되는 하이퍼파라미터라는 점이 이 설계 전체를
  가능하게 하는 전제다.
- **Diffusion Graph Networks (DGN) / LDGN** (Lino, Pfaff & Thuerey,
  "Learning Distributions of Complex Fluid Simulations with Diffusion
  Graph Networks," ICLR 2025, arXiv:2504.02843) — 메시 기반 물리 문제를
  위해 "먼저 압축, 그다음 생성"을 제안한다: 근손실 그래프
  오토인코더(VGAE)로 coarse 잠재를 만들고, 그 잠재 위에서만 동작하는
  diffusion prior를 명시적으로 학습한다. 논문은 압축 없이 필드 위에서
  바로 diffusion을 돌리는 자신의 기본형(DGN)과, 압축 후 그 latent 위에서
  diffusion을 돌리는 변형(LDGN)을 직접 대조한다 — 이 내부 대조가 정확히
  이 리포트의 v1(필드 공간) → v2(coarse latent) 전환과 같은
  motivation이다. 핵심 ablation: 같은 멀티스케일 구조를 쓰지만 명시적으로
  학습된 prior가 없는 VGAE(오히려 더 무거운 KL, 더 coarse한 latent를
  쓴)는 간단한 과제에서는 통하지만 복잡한 과제에서는 여전히 collapse한다
  — 즉 계층 구조 자체는 이 설계가 이기는 이유가 아니고, 결정적 요인은
  "prior를 명시적으로 학습시키는가"이다. 5절에서 이 논문과 본 구현의
  구체적인 같은 점·다른 점을 정리한다.
- **AdaLN-Zero** (Peebles & Xie, "Scalable Diffusion Models with
  Transformers", ICCV 2023) — 조건부 정보를 시간에 따라 변하는
  scale/shift/gate로 주입하고, 학습 초기에는 블록이 항등함수가 되도록
  0 근처로 초기화하는 기법. Stage 2 prior의 트렁크 블록에 사용된다.

## 3. 구조

### 3.1 표기

| 기호 | 의미 |
| --- | --- |
| $G=(V,E)$ | 입력 메시 그래프. 레벨 $0$(가장 fine)부터 $L$(가장 coarse)까지의 계층을 가짐 |
| $V_l$ | 레벨 $l$의 노드 집합 (코스닝으로 생성, 압축기·prior 공통) |
| $y \in \mathbb{R}^{|V_0|\times F}$ | 예측 대상 물리량 (finest 레벨의 노드 필드) |
| $c$ | 조건 — 형상/경계조건/재료 등, 노드·엣지 특성에 인코딩됨 |
| $z \in \mathbb{R}^{|V_L|\times C}$ | coarse 레벨의 노드별 잠재. `latent_ch`(=$C$) 채널짜리 벡터가 coarsest 레벨 노드마다 하나 — **풀링된 벡터 하나가 아니다** |
| $E_\theta$ | 압축기 인코더 (공유 가중치, 두 번 호출됨 — 3.3절) |
| $D_\theta$ | 압축기 디코더 |
| $v_\psi$ | stage 2의 flow-matching 속도장 (coarsest 레벨에서만 동작) |

```text
레벨:   0 ──▶ 1 ──▶ … ──▶ L ──▶ … ──▶ 1 ──▶ 0
      (finest)   코스닝(pooling)   (coarsest)   unpool+skip 융합   (finest)
                                       │
                                       ▼
                    stage 1의 z가 사는 곳
                    stage 2 prior가 동작하는 유일한 레벨
```
*그림 1: 계층적 V-cycle 백본. 코스닝으로 레벨 0부터 L까지 내려간 뒤,
unpooling과 skip 융합으로 다시 레벨 0까지 올라온다. 압축기(stage 1)는 이
V 전체를 쓰고, stage 2 prior는 맨 밑의 L 하나에서만 동작한다.*

### 3.2 개요: 2단계 분해

```text
                    ┌─────────────── Stage 1: 압축기 (E_θ, D_θ) ───────────────┐
 y, c ──────────────▶  E_θ(y-aware)  ──▶ (μ, logσ²) ──▶ z ~ N(μ,σ²) ──▶ D_θ ──▶ ŷ
      └───────────▶  E_θ(y-blind)   ──▶ skip 특징 {h_l}, prior 조건 g ──┘   (레벨별 skip)
                    └──────────────────────────────────────────────────────────┘
                                              │  z만 아래로 전달, skip은 위에서 그대로 재사용
                    ┌─────────────── Stage 2: prior (v_ψ, coarsest 레벨만) ─────┐
 g ──────────────────────────────────▶  z0 ~ N(0,I) --[K-step ODE]--> z1'      │
                    └──────────────────────────────────────────────────────────┘
                          학습 시: z1' 대신 stage-1이 만든 실제 z를 목표로 회귀
                          추론 시: z1'을 z 대신 D_θ에 넣어 디코딩
```

압축기는 전체 V-cycle(레벨 $0 \to L \to 0$)을 쓰고, prior는 그 V의 가장
바닥(coarsest 레벨) 하나에만 존재한다. 이 비대칭이 3.5절에서 설명하는
비용 구조의 근원이다.

### 3.3 Stage 1 — 계층적 압축기

`HierarchyEncoder`는 V-cycle의 하강(코스닝) 팔로, 각 레벨에서
`mp_per_level`만큼 메시지 패싱을 하고 레벨 사이를 코스닝 연산자(bi-stride
/ Voronoi 등, HI-MGN 백본이 이미 제공하는 것을 그대로 재사용)로 풀링한다.
가장 coarse한 레벨 $L$에 도달하면 선형 head가 노드별 $(\mu,\log\sigma^2)$를
내고, $z \sim \mathcal{N}(\mu,\sigma^2)$로 재파라미터화한다 — 이는
MeshGraphNets-Variational의 "풀링된 전역 벡터 하나"와 대비되는, **노드별
분포를 갖는 공간적 잠재**다.

같은 가중치의 $E_\theta$를 **한 스텝에 두 번** 통과시킨다:

```text
outs_y = E_θ(x = [state | y        | cond | pos | ...])   -- y를 아는 pass
outs_0 = E_θ(x = [state | zeros(y) | cond | pos | ...])   -- y를 모르는(blind) pass
```

`outs_y`의 coarsest 출력만 $(\mu,\log\sigma^2)$로 간다. `outs_0`의 **레벨별
전체 출력**이 디코더의 skip 연결과 stage 2 prior의 조건 $g$로 간다. 이
분리가 핵심이다: skip 경로가 단순한 fine-level 재사용이면 디코더가 $z$를
거치지 않고 skip만으로 $y$를 복원하도록 학습해버릴 수 있다(LDGN 논문
자체의 VGAE collapse 메커니즘과 동일). skip을 만드는 pass 자체가 $y$를
구조적으로 모르기 때문에 이 우회가 불가능하고, 추론 시점에는 $y$가
애초에 없으므로 이 "blind" pass는 학습 때와 추론 때 정확히 동일하게
동작한다.

`MultiscaleDecoder`(V-cycle 상승 팔)는 $z$(학습 때는 posterior 샘플,
추론 때는 stage 2가 적분한 샘플)를 coarsest 레벨 노드 수로 lift한 뒤,
레벨을 올라가며 `outs_0`의 skip과 융합해 $\hat y$를 만든다.

목적함수:

```text
L_AE = E_{q_θ(z|y,c)}[ ||D_θ(z) - y||² ]  +  β · KL( q_θ(z|y,c) ‖ N(0,I) )
```

$\beta$(`ae_kl_weight`)는 아주 작게(논문 기준 $\sim 10^{-6}$) 잡는다 —
이 압축기는 그 자체로 생성적 병목이 아니라 근손실 압축이 목적이며, KL은
$z$를 stage 2의 목표(=$N(0,I)$에 가까운 분포)에 부드럽게 앵커링하는
정도의 역할만 한다.

### 3.4 Stage 2 — Coarse-Latent Flow-Matching Prior

Stage 1이 수렴하면 그 가중치를 얼려 고정된 특성 추출기로 쓴다. Stage 2는
그 frozen 압축기가 만든 posterior 샘플 $z_1 \sim q_\theta(z|y,c)$(gradient
없이 draw)를 목표점으로 하는 flow-matching 속도장 $v_\psi$를 coarsest
레벨 하나에서만 학습한다.

```text
z0 ~ N(0, I)                              노이즈 끝점, z1과 같은 크기(|V_L| × C)
s  = 1 - σ_min,  σ_min = 1e-4
t  ~ U[0,1)  (또는 logit-normal)
z_t = (1 - s·t)·z0 + t·z1                 직선 경로 위의 점
u   = z1 - s·z0                            그 경로가 갖는 속도(회귀 목표)

v_ψ(z_t, t, g) ≈ u                         AdaLN-Zero GnBlock 스택(prior_blocks개),
                                            coarsest 레벨 그래프 + 시간 임베딩 + g로 조건화

L_prior = E[ w(t) · ||v_ψ(z_t,t,g) - u||² ]
   w(t) = 1                (flow_loss_weighting = uniform, velocity 파라미터화)
   w(t) = (1 - s·t)²        (flow_loss_weighting = x0, data 파라미터화 — 속도 head는 그대로 두고
                             손실 재가중만으로 t=0 근방 정확도에 예산을 집중)
```

일부 학습 그래프는 $t=0$으로 고정해서 뽑는다(`flow_det_prob`) — 이 경우
목적함수는 $E[z_1|g]$에 대한 순수 결정론적 회귀로 붕괴하며, 이는 아래
3.5절의 결정론적 readout을 실제로 *학습시키는* 장치다(추론 시점에 그냥
읽어내는 것이 아니라).

AdaLN-Zero 블록은 초기화 시 항등함수가 되도록(gate/weight를 0 근처로)
구성된다 — 전역 kaiming 초기화를 먼저 적용한 뒤 이 근-항등 초기화를
덮어씌우는 순서로 진행되며, 순서를 바꾸면 kaiming 초기화가 이 성질을
지워버린다.

### 3.5 추론 경로

```text
generate(c):
    g            = E_θ, blind(0, c)                 # 1) encode ONCE
    z0          ~ N(0, I)                            #    (coarsest 레벨 크기)
    z1'          = integrate(v_ψ(·,·,g), z0, K steps) # 2) coarsest 레벨에서만 K번 (Heun/Euler)
    ŷ            = D_θ(z1', {h_l})                    # 3) decode ONCE
```

이 경로에서 **K번 반복되는 것은 coarsest 레벨 하나만 다루는 작은
네트워크뿐**이며, 전체 V-cycle(인코더·디코더)은 샘플 하나당 정확히
1회씩만 호출된다. 이것이 field-space flow(레벨 0 크기에서 매 ODE
스텝마다 전체 V-cycle을 다시 계산해야 하는)와 이 설계 사이의 핵심적인
구조적 차이다.

같은 학습된 네트워크에서 결정론적 readout도 닫힌 형태로 나온다: $t=0$
에서는 경로 위의 점이 곧 $z_0$이므로 회귀 최적점은
$v_\psi^*(z_0,0,g)=E[z_1|g]-s z_0$이고, 크기 1의 Euler 스텝을 밟으면

$$z_0 + v_\psi(z_0,0,g) = E[z_1|g] + \sigma_{min} z_0$$

즉 $\sigma_{min}$($=10^{-4}$) 스케일의 잔차 하나만 남는 조건부 평균이다.
같은 가중치 하나가 (i) 분포를 샘플링하는 생성 모델과 (ii) 적분 없이 1회
forward로 끝나는 결정론적 예측기를 동시에 서빙한다.

### 3.6 학습 모드

Stage 1과 stage 2를 지나는 세 가지 경로가 있다: stage 1만(압축기·디코더
학습), stage 2만(디스크에서 완료된 stage-1 체크포인트를 얼려 로드한 뒤
prior만 학습), 또는 하나의 프로세스에서 둘 다(정해진 epoch 동안 stage 1
→ 메모리 내에서 즉시 freeze → stage 2, 체크포인트 왕복 없음). 어느
시점이든 두 파라미터 집합 중 정확히 하나만 gradient를 받으므로, 분산
학습 래퍼는 파라미터 그래프의 한쪽 가지가 매 스텝 unused여도 견뎌야
한다.

## 4. 설계 근거: 왜 먼저 압축하는가

LDGN 논문의 ablation을 정확히 인용하면: **같은 멀티스케일 구조를 쓴
VGAE**(오히려 더 무거운 KL, 더 coarse한 latent를 쓴)도 여전히
collapse한다. 즉 계층 구조 자체는 이 설계가 이기는 이유가 아니다.
결정적인 lever는 "$N(0,I)$를 직접 가정해서 샘플링하는 대신, 압축된
공간 위에서 **prior를 명시적으로 학습시키는 것**"이다.

| | MeshGraphNets-V | cHI-MGNflow v1 (필드 공간) | cHI-MGNflow v2 (본 문서) |
| --- | --- | --- | --- |
| 학습 시 잠재의 출처 | posterior $q(z\|y,c)$ | 필요 없음 (필드에 직접 flow) | posterior $q(z\|y,c)$ (stage 1) |
| 추론 시 잠재의 출처 | 별도 학습된 근사 prior $p(z\|c)$ — **posterior와 불일치 가능** | $N(0,I)$ → 학습된 ODE (필드 크기) | $N(0,I)$ → 학습된 ODE (coarse 잠재 크기) |
| 압축 | 있음 (풀링된 벡터) | **없음** | 있음 (coarse 레벨 노드별 벡터) |
| 샘플당 전체 V-cycle forward | 1 (prior는 flat) | **K** (ODE 스텝마다) | **2** (encode 1 + decode 1, K는 coarse 레벨에서만) |

v1은 이 표의 "분포 일치"는 만족했지만 "압축"을 하지 않아 비용과
reconstruction 신뢰성 양쪽에서 문제가 있었다. v2는 두 축을 모두
만족시키는, 논문의 ablation이 요구하는 조합이다.

```text
v1 (필드 공간):
  [V-cycle] ──▶ [V-cycle] ──▶ … ──▶ [V-cycle]
  ODE 스텝마다 전체 V-cycle을 K번 (encode+decode 겸용)

v2 (본 구현):
  [encode(V-cycle)] ──▶ (·)─▶(·)─▶ … ─▶(·) ──▶ [decode(V-cycle)]
                         coarsest 레벨에서만 K번 (작음)
```
*그림 2: 샘플 하나를 생성하는 데 필요한 전체 V-cycle 호출 횟수. v1은 매
ODE 스텝마다 전체 V-cycle을 다시 계산해 K회가 필요하고, v2는 encode
1회 + decode 1회로 고정되며 반복은 coarsest 레벨의 작은 네트워크에서만
일어난다.*

## 5. 원본 LDGN 논문과의 비교: 같은 점과 다른 점

정확한 서지사항을 다시 밝히면, 참고 논문의 제목은 *Learning Distributions
of Complex Fluid Simulations with Diffusion Graph Networks* (Lino, Pfaff &
Thuerey, ICLR 2025, arXiv:2504.02843)이다. "LDGN"은 그 논문이 자신의
latent-압축 변형에 붙인 이름이고, 압축 없이 필드 위에서 바로 동작하는
기본형은 논문 안에서 "DGN"이라 불린다. 즉 원 논문 자체가 이미
DGN(필드 공간) vs LDGN(압축된 latent 공간)이라는 대조를 갖고 있고, 이
대조는 정확히 본 리포트의 v1 → v2 전환과 같은 motivation을 공유한다. 이
절은 그 LDGN 설계와 본 구현(cHI-MGNflow v2)이 정확히 어디까지 같고
어디부터 다른지를 정리한다.

### 5.1 같은 점

- **2단계 분해 자체.** 근손실 그래프 오토인코더(VGAE)로 먼저 압축하고,
  그 압축된 latent 위에서만 별도의 생성 prior를 명시적으로 학습한다는
  큰 그림은 그대로 채택했다.
- **잠재의 모양.** 잠재는 풀링된 전역 벡터 하나가 아니라, 가장 coarse한
  레벨 $G_L$의 **노드마다** 분포 $(\mu_i,\sigma_i)$를 갖는 공간적
  잠재다.
- **KL을 아주 작게 유지.** posterior collapse를 피하되 $z$를 prior의
  목표 분포에 앵커링하는 정도로만 KL 항의 가중치를 $\beta \sim 10^{-6}$
  수준으로 아주 작게 잡는다.
- **핵심 motivating ablation.** 같은 계층 구조를 쓰지만 명시적 prior
  없이 $\mathcal{N}(0,I)$에서 직접 샘플링하는 VGAE는(간단한 과제에서는
  통하지만) 복잡한 과제에서 여전히 collapse한다는 논문의 관찰을 그대로
  이 구현의 설계 근거(4절)로 인용한다.
- **생성 시 비용 철학.** 비싼 전체-그래프 연산(encode/decode)은 샘플당
  고정된 소수 회로 묶고, 반복이 필요한 샘플링 과정은 이미 압축된 latent
  공간 안에 가둔다는 비용 구조의 기본 방향은 동일하다.

### 5.2 다른 점

| 항목 | LDGN 원본 | cHI-MGNflow v2 (본 구현) |
| --- | --- | --- |
| 멀티스케일 백본 / coarsening | 자체 U-Net형 multi-scale GNN, Guillard 알고리즘으로 코스닝 (1D ~2배, 2D ~4배 압축/레벨) | 이 리포지토리의 기존 HI-MGN V-cycle(bi-stride/Voronoi 등)을 그대로 재사용 — 논문의 백본 자체를 이것으로 대체한 것이 본 리포트의 출발점 |
| 압축기 인코더 구조 | 조건 인코더 / 노드 인코더 / 노드 디코더, 세 개의 **별도** 컴포넌트 — 조건 인코더가 생성 시점에 필요한 $g$ 인코딩을 전담 | 하나의 `HierarchyEncoder`를 **같은 가중치**로 두 번($y$-aware/$y$-blind) 호출 — 별도의 조건 인코더 네트워크가 없다(3.3절) |
| 생성 prior의 주 형태 | **diffusion**(DDPM류)이 주 모델이고 논문 실험 전부가 이 경로를 씀; flow matching은 부록에서 "더 빠른 샘플링"을 위한 미래 방향으로만 짧게 언급되며 구체적 공식은 없음 | **flow matching**이 유일한 생성 경로; 그 공식(직선 경로, velocity 목표, $x_0$/균등 재가중)은 논문이 아니라 flow-matching/rectified-flow 문헌(Lipman et al.; Liu, Gong & Liu)에서 직접 채택 |
| prior network 내부 구조 | coarse latent 위에서 **자체적으로 또 한 번** 4-scale까지의 멀티스케일 U-Net을 돌림(단일 스케일은 복잡한 과제에서 실패했다고 보고) | `LatentFlowPrior`는 coarsest 레벨 하나에서 **flat하게**(추가 pooling 없이) 블록만 반복 — 더 단순하고 저비용, 단 논문의 "단일 스케일 실패" ablation과 아직 직접 대조되지 않음 |
| 시간/조건 주입 방식 | 매 메시지 패싱 레이어에 시간·조건 임베딩을 **덧셈으로** 주입($v_i \leftarrow v_i + \mathrm{Linear}(r_{emb})$) | **AdaLN-Zero**(Peebles & Xie)로 scale/shift/gate 변조 + 항등 초기화 — 논문에는 없는, 이 구현이 별도로 채택한 조건화 |

정리하면, 본 구현이 LDGN에서 가져온 것은 **"먼저 압축하고, 그 압축된
latent 위에서만 prior를 명시적으로 학습시킨다"는 2단계 설계 원칙과, 그
원칙을 정당화하는 VGAE-collapse ablation** 자체다. 그 원칙을 실현하는
**구체적인 부품들**(멀티스케일 백본, 인코더 내부 구조, 생성 모델의
수학적 형태, prior의 내부 구조, 조건 주입 방식)은 대부분 이 리포지토리
자체의 기존 구성요소(HI-MGN V-cycle)이거나 별도 문헌(flow matching,
AdaLN-Zero)에서 가져와 독자적으로 재구성한 것이다.

## 6. 평가 프로토콜과 현재 검증 상태

이 아키텍처가 이미 계산하는 지표는 다음과 같다(`training_profiles/training_loop.py`):

- **recon** — stage 1의 held-out reconstruction MSE.
- **det** — 3.5절의 1-forward 결정론적 readout의 MSE.
- **crps** — 샘플링 분포에 대한 proper score(CRPS), 앙상블 평가와 직접
  대응.
- **spread/gt** — 앙상블 표준편차와 실제(참) 분포 표준편차의 비율 — 이
  값이 0에 가까우면 모델이 노이즈 채널을 무시하고 있다는 뜻이다.

**현재까지 실제로 확인된 것**은 CPU 정합성 스모크 테스트와 이 방법
자체의 단위 테스트 스위트 통과, 그리고 런처 설정-계약 검증(preflight)의
정합성뿐이다. **이 아키텍처로 GPU 규모의 실제 학습은 아직 수행되지
않았다.** 따라서 본 리포트는 정량적 결과 표를 포함하지 않는다 — 이전
아키텍처(v1)에서 측정된 수치는 압축이 없는 다른 구조에서 나온 것이라
본 아키텍처로 이전되지 않으며, 실측은 실제 GPU 학습이 수행된 뒤 별도
리포트에서 다룬다.

## 7. 코드베이스 내 위치

구현은 `methods/HI_MGNFlow/`에 있다. 설정 키, CLI 실행법, 학습 로그
포맷, 그리고 (6절보다 더 세부적인) 현재 검증 현황 등 **운영 관점의
문서**는 `methods/HI_MGNFlow/README.md`가 맡는다. 본 리포트는 그
아키텍처가 왜 이런 모양이고 무엇을 하는지에만 집중한다.

## 참고문헌

- Pfaff, T., Fortunato, M., Sanchez-Gonzalez, A., Battaglia, P. W.
  "Learning Mesh-Based Simulation with Graph Networks." ICLR, 2021.
- Lino, M., Pfaff, T., Thuerey, N. "Learning Distributions of Complex
  Fluid Simulations with Diffusion Graph Networks." ICLR, 2025.
  arXiv:2504.02843.
- Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. "Flow
  Matching for Generative Modeling." ICLR, 2023.
- Liu, X., Gong, C., Liu, Q. "Flow Straight and Fast: Learning to
  Generate and Transfer Data with Rectified Flow." ICLR, 2023.
- Peebles, W., Xie, S. "Scalable Diffusion Models with Transformers."
  ICCV, 2023. (AdaLN-Zero)
- Kipf, T. N., Welling, M. "Variational Graph Auto-Encoders." NeurIPS
  Bayesian Deep Learning Workshop, 2016. (§4에서 언급한 ablation의
  baseline)
