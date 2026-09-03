# Transolver를 physics foundation model로 만들 수 있는가

> 질문: "어떤 토큰은 boundary condition을, 어떤 토큰은 physical slice를 나타내게 해서
> 임의의 BC에서도 결과를 내는, 완전히 extrapolation·generalize된 모델을 만들 수 있을까?"
>
> 이 문서는 그 질문을 (1) 현재 코드베이스 실측, (2) 2024–2026 문헌, (3) 이 repo의
> 실제 `Transolver` 클래스로 돌린 수치 실험으로 답한다.

---

## 0. 결론

**아키텍처로서는 된다. 그리고 Transolver는 이 목적에 이례적으로 잘 맞는 골격이다.**
Physics-Attention은 이미 perceiver형 latent bottleneck이라, "BC 토큰"을 끼워넣을
자리가 코드에 정확히 한 곳(`_slice_attend`) 존재하고 그 비용은 N에 무관하다.

**하지만 "토큰을 넣으면 임의 BC로 외삽된다"는 건 사실이 아니다.** §3에서 이 repo의
실제 모델로 측정했다: BC를 토큰으로 준 모델(B)은 학습 범위 안에서 per-node 채널(A)과
**대등하고**(6.77% vs 5.38%), 범위를 2.7×/5.3× 벗어나면 **똑같이 무너진다**
(56%/97% vs 69%/94%). 반면 **진폭을 해석적으로 분리하면** 같은 외삽에서
69.00% → **10.34%** (6.7배)로 떨어진다.

즉 외삽을 만드는 건 토큰이 아니라 **출력 스케일 구조(무차원화·진폭 분리)** 와
**in-context 증거**다. 토큰의 가치는 정확도가 아니라 **확장성**(BC 개수 가변, semantic
타입, 공간 support, open vocabulary)에 있다 — 그리고 그것만으로도 충분히 값있다.

실제 순서로 정리하면:

| 순위 | 해야 할 일 | 성격 | 이 repo 상태 |
| --- | --- | --- | --- |
| 1 | 데이터에 BC/파라미터를 기록 | 배관 작업, ML 아님 | **전무** (dp0..dpN만 있음) |
| 2 | 타깃 정규화를 instance/analytic로 교체 | 외삽의 실제 원천 | dataset-global z-score |
| 3 | BC 토큰 주입 | **확장성** (정확도 아님, §3) | 없음 (추가 쉬움) |
| 4 | 토큰 붕괴 방지 (Ada-Temp) | 용량 확보 | per-head 고정 τ |
| 5 | in-context 조건화 | 진짜 범위 밖 외삽 | **이미 반쯤 구현됨** |
| 6 | 채널 무관 다물리 | 하나의 backbone | 고정 `input_var` |

**"모델을 키우고 데이터를 다 넣으면?"에 대한 답은 §12에 있다: 그 실험은 이미 됐고,
백본이 Transolver다.** GeoPT(ICML 2026)가 Transolver 백본으로 100만+ 무라벨 샘플에
사전학습해 자동차·항공기·선박·크래시를 커버한다. 얻은 것은 **라벨 20–60% 절감과 수렴
2× 가속**이고, BC 외삽은 아니다. 그리고 같은 논문이 **"Transolver는 데이터가 제한된
산업 시뮬레이션에서 overfitting으로 scaling bottleneck에 걸린다"**고 진단한다 —
ex1 100개 + ex2 50개에서 모델을 키우면 **더 나빠진다.** 가장 값싼 큰 승리는
**GeoPT 가중치 이식 검토**다(백본이 같다).

이종 데이터셋 통합(§13)과 스케일 범위(§14)는 별도 문제이고, 결론은:
part-id로 BC를 인코딩하는 현재 스키마는 **두 파일을 합치는 순간 손상된다**(실측 증거 있음).
그리고 meso→100 m는 물리 단위로는 불가능하고, **상사(similarity) 클래스 안에서만** 가능하다.

---

## 1. Transolver가 구조적으로 유리한 이유

Physics-Attention은 이미 토큰 시퀀스 모델이다. `model/physics_attention.py`:

```
node [N, C] --(W: soft assignment)--> slice tokens [H, M, D]
                                          |
                                     self-attention          <-- 토큰들이 만나는 유일한 지점
                                          |
node [N, C] <--(Wᵀ deslice)--------- attended tokens [H, M, D]
```

- 토큰화: `_chunk_stats` ([physics_attention.py:139-154](../../../methods/Transolver/model/physics_attention.py#L139-L154))
- 토큰 간 attention: `_slice_attend` ([physics_attention.py:156-164](../../../methods/Transolver/model/physics_attention.py#L156-L164))
- 역투사: `_deslice` ([physics_attention.py:166-179](../../../methods/Transolver/model/physics_attention.py#L166-L179))

**핵심**: `_slice_attend`는 `[H, M, D]` → `[H, M, D]`이고 N이 등장하지 않는다.
여기에 조건 토큰 K개를 concat하면 attention이 `[H, (M+K)²]`이 되는데, M=128·K=16이면
1.27×다. 그리고 이 layer의 지배적 메모리 항은 `[H, N, M]` slice weight(그래서
`chunk_size`가 THE memory lever)이므로, **조건 토큰의 추가 비용은 사실상 0이다.**

### 이건 Transolver 계열 자체의 미해결 지점이다

Transolver-3 논문(이 repo가 구현한 그 논문)은 NASA-CRM 실험에서 **"six input
parameters — Mach number, angle of attack, control surface deflections"** 를 쓴다고
적어놓고 **그 파라미터를 모델에 어떻게 넣는지는 서술하지 않는다.** 그리고 future work에서
스스로를 *"a scalable backbone for grand-scale physics foundation models"* 로 지목한다.

즉 "BC/파라미터 조건화 메커니즘"은 이미 해결돼 있어서 안 쓰는 게 아니라, **Transolver
계열이 명시적으로 비워둔 칸**이다. 이 문서가 채우려는 게 정확히 그 칸이다.

한 가지 규칙만 지키면 된다: **조건 토큰은 deslice 하지 않는다.** `W`는 `[H, N, M]`로
physics slice에 대해서만 정의되므로, attend 결과의 앞 M개만 `_deslice`로 넘긴다.
그러면 tiling(`make_tile_ranges`), node sharding(`_shard_all_reduce_sum`),
amortized 2-stream — N에 걸린 모든 기계장치가 **손대지 않은 채로** 유지된다.

---

## 2. 토큰 분류: LLM 비유를 제대로 하기

| 토큰 | 개수 | 출처 | LLM 대응 | 현재 |
| --- | --- | --- | --- | --- |
| **node** | N (10⁴–10⁸) | 메시 절점 | byte/character | `graph.x`, `pos_normalized` |
| **physics slice** | M = `slice_num` (layer·head별) | 데이터 의존 soft 집계 | content token | 있음 |
| **condition (BC)** | K (10~100) | 해석 설정값 | **system prompt** | **없음** |
| **field/channel** | C_out | 채널 semantic | output vocabulary | 고정 `output_var` |
| **context (demo)** | M per demo | 다른 BC의 해 | few-shot example | `compute_tokens` 재활용 가능 |

LLM 비유가 깨지는 지점을 분명히 해야 한다: **LLM 토큰은 이산·유한·닫힌 vocabulary인데,
물리 조건은 연속이고 차원(단위)을 가진다.** "이 토큰을 본 적 있다"에 대응하는 것은
"이 **무차원수**가 학습 범위 안이다"다. 그래서 vocabulary 설계의 핵심은 임베딩 테이블이
아니라 **무차원화 스키마**다 (§5, §6).

### 왜 BC를 그냥 node 채널로 안 주는가

지금 코드로 할 수 있는 유일한 방법은 BC 값을 절점마다 상수 채널로 붙이는 것이다
(`input_var` 증가). 보간 구간에서는 동작하지만:

1. graph-level 상수 정보에 N×K 메모리를 낭비한다.
2. BC 값이 node 통계로 z-score 되어 물리 상태 채널과 같은 임베딩에 섞인다 →
   "상태"와 "설정"을 분리할 수 없다.
3. **공간적 support가 다른** BC(면 A는 100°C, 면 B는 대류)를 표현할 자리가 없다.
4. BC 개수가 바뀌면 `input_var`가 바뀌고 → **다른 모델**이 된다. foundation model의
   정의에 정면으로 어긋난다.

토큰 시퀀스는 이 네 개를 동시에 푼다: K 가변, 조건별 semantic 임베딩, open vocabulary.

> **주의**: 이 네 개는 전부 **확장성** 논거다. §3의 실측에서 BC가 스칼라 1개일 때
> 토큰과 per-node 채널의 정확도는 대등했다. 토큰을 "더 정확한 조건화"로 팔면 안 되고,
> "BC 스키마가 바뀌어도 같은 모델"로 팔아야 한다. foundation model의 정의가 후자다.

---

## 3. 실험: 토큰만으로는 외삽되지 않는다

이 repo의 실제 `Transolver` 클래스에 §1의 조건-토큰 패치를 붙여 검증했다.
스크립트: `misc/bc_token_extrapolation.py` (아래 결과 재현 가능).

> 이 실험은 **메커니즘 분리 실험**이다 — 절대 정확도 주장이 아니다. 3-layer·64-dim의
> 소형 모델, 합성 데이터, 출력 1채널. 목적은 "BC를 어떻게 주는가(A vs B)"와 "출력
> 스케일을 어떻게 두는가(B vs C)"를 **다른 모든 조건을 고정한 채** 비교하는 것이다.

**문제 설정** (ex1.h5와 같은 static T=1 형태, 열변형 모사)

```
u(x) = g(ΔT) · shape(x; geometry)          출력 1채널
  linear   regime: g(ΔT) = ΔT              (미소변형 열팽창)
  nonlinear regime: g(ΔT) = ΔT(1 + 0.15ΔT)
학습 ΔT ~ U[1, 2] / 평가는 처음 보는 geometry에서 ΔT = 1.5, 4, 8
```

**변형**

- **A** — ΔT를 per-node 입력 채널로, 타깃은 dataset z-score → *지금 repo로 가능한 최선*
- **B** — ΔT를 **condition token**으로, 타깃은 dataset z-score → *"LLM식 토큰" 제안*
- **C** — ΔT를 condition token으로 + **해석적 진폭 분리** (u/ΔT를 예측하고 되곱함)

**결과** (처음 보는 geometry, 물리 단위 relative L2)

**regime = linear**, g(ΔT) = ΔT — 학습 ΔT∈[1,2]

| 변형 | ΔT=1.5 (보간) | ΔT=4 (2.7×) | ΔT=8 (5.3×) |
| --- | --- | --- | --- |
| A  per-node 채널 + z-score (지금 repo) | **5.38 %** | 69.00 % | 93.76 % |
| B  condition token + z-score | 6.77 % | 55.69 % | 96.66 % |
| C  condition token + 진폭 분리 | **4.32 %** | **10.34 %** | **30.43 %** |

**regime = nonlinear**, g(ΔT) = ΔT(1+0.15ΔT) — 분리한 법칙과 실제 응답이 다른 경우

| 변형 | ΔT=1.5 (보간) | ΔT=4 | ΔT=8 |
| --- | --- | --- | --- |
| A  per-node 채널 + z-score | 6.94 % | 64.23 % | 93.56 % |
| B  condition token + z-score | 5.98 % | 95.58 % | 99.42 % |
| C  condition token + 진폭 분리 | 5.27 % | 75.20 % | 99.97 % |

(101k–118k 파라미터, 2500 step, 처음 보는 geometry 12개 평균)

**읽는 법 — 다섯 가지 결론, 그중 4번이 반직관적**

1. **보간 구간에서 A ≈ B ≈ C (5~7%)다. 토큰이 정확도를 주지 않는다.**
   BC 1개를 조건화하는 문제에서 per-node 채널과 condition token은 대등하다
   (linear에서는 A 5.38 vs B 6.77로 오히려 A가 낫다). **그러므로 토큰을 쓸 근거는
   정확도가 아니라 확장성이다** — §2의 네 가지(K 가변, semantic 타입, 공간 support,
   open vocabulary). 이건 공학적 논거이고, 정확도 논거로 포장하면 안 된다.

2. **외삽에서 A와 B가 같이 무너진다 (69→94%, 56→97%).** 토큰은 외삽을 만들지 못한다.
   원인은 조건 경로가 아니라 **출력 경로**다 — 타깃이 train-split 통계로 z-score 되어
   ([mesh_dataset.py:499](../../../methods/Transolver/general_modules/mesh_dataset.py#L499),
   통계는 [:252](../../../methods/Transolver/general_modules/mesh_dataset.py#L252)) head가 한 번도 낸 적 없는
   정규화 값을 내야 한다.

3. **진폭 분리만이 실제로 작동한다.** linear regime 2.7× 외삽에서 69.00% → 10.34%
   (**6.7배** 개선), 5.3×에서 93.76% → 30.43%. §4 장벽 2가 최우선인 이유가 이 한 줄이다.

4. **그런데 C도 평평하지 않다 (4.32 → 10.34 → 30.43%). 이게 반직관적인 부분이다.**
   C의 타깃 û = u/ΔT는 **ΔT에 전혀 의존하지 않는데도** 오차가 ΔT와 함께 커진다.
   원인은 **조건 토큰 자신이 OOD로 나간 것**이다: v̂ = ΔT/1.5가 학습 시 [0.67, 1.33]
   이었는데 평가에서 5.33이 들어오고, 모델이 û에 있지도 않은 ΔT 의존성을 spurious하게
   학습해 두었기 때문에 그 값이 예측을 흔든다.
   → **조건 토큰은 값이 범위를 벗어나면 타깃이 그 조건에 의존하지 않아도 능동적으로
   해를 망친다.** 대책: `condition_dropout`(학습 중 조건 랜덤 제거로 의존성 자체를
   정규화), 조건 값 clamp, 그리고 진폭 분리로 이미 설명된 양은 토큰에서 빼는 것.
   Phase 1에 `condition_dropout`을 넣는 이유가 이 수치다.

5. **분리한 법칙이 실제 응답과 다르면 전부 무너진다** (nonlinear regime: C가 75%, 100%).
   여기서는 B(95.58%)가 A(64.23%)보다 **더 나쁘다** — 조건 토큰이 OOD 값으로
   해를 적극적으로 오염시킨다. 이게 §10의 "regime 전이는 안 된다"에 대한 실측 근거다.

한 줄 요약: **"BC 토큰"은 조건화의 *인터페이스·확장성* 문제를 풀고, 외삽은
*무차원화·진폭 분리*가 푼다. 둘은 다른 문제이고, 토큰을 외삽 해법으로 착각하면
(nonlinear regime처럼) 오히려 손해다.**

---

## 4. 진짜 장벽 세 개

### 장벽 1 — 데이터에 BC가 없다 (blocking)

실측:

```
ex1.h5   100 samples,  T=1,  25,203 nodes,  source_filename = dp0 … dp99
ex2.h5    50 samples, T=50, 199,993 nodes,  source_filename = dp0 … dp49
feature_names = [x,y,z coord, x/y/z disp(mm), stress(MPa), Part No.]
```

`dp*`는 ANSYS design point다. 즉 **각 샘플을 만든 파라미터 표는 Workbench 안에 있고
HDF5에는 없다.** 그리고 두 데이터셋 어디에도 **온도 채널이 없다.**

BC 토큰의 입력이 존재하지 않으므로 이건 선행 조건이다. 필요한 스키마 확장:

```
data/{id}/conditions              float [K]      해당 샘플의 BC/파라미터 값
metadata/condition_schema/
    names                         str [K]        'mold_temp', 'melt_temp', 'inlet_velocity', …
    units                         str [K]        'K', 'm/s', 'MPa', …
    support                       int [K]        적용되는 part/surface id (-1 = 전역)
    reference                     float [K]      무차원화 기준값
    group                         str [K]        유도 무차원수 ('alpha_dT', 'Re', 'Bi', 'Fo')
```

`dataset/DATASET_FORMAT.md`와 `cae_suite/dataset_probe.py`의 교차검증을 같이 갱신해야 한다.

### 장벽 2 — 정규화가 외삽을 막는다 (실험이 지목한 그 지점)

세 곳이 모두 dataset 전역 통계에 묶여 있다:

- 타깃: `target_norm = (target_delta - delta_mean) / delta_std`
  ([mesh_dataset.py:499](../../../methods/Transolver/general_modules/mesh_dataset.py#L499))
- 좌표: 전역 RMS 반경 하나로 나눔
  ([mesh_dataset.py:22-32](../../../methods/Transolver/general_modules/mesh_dataset.py#L22-L32))
- node type: **닫힌 vocabulary** — 처음 보는 part number면 하드 에러
  ([mesh_dataset.py:312-328](../../../methods/Transolver/general_modules/mesh_dataset.py#L312-L328))

마지막 것은 foundation model 설계의 정반대다. 새 부품이 하나 들어오면 크래시한다.

처방:

1. **instance normalization** (MPP의 RevIN): 샘플별로 스케일을 빼내 저장하고, 형상만
   예측한 뒤 출력에서 복원. 다물리 사전학습이 되게 만든 결정적 트릭으로 보고돼 있다.
2. **analytic scaling**: 응답 법칙을 알 때는 더 좋다. `u = g(conditions) · û`로 두고 û만
   예측. 선형 물리(선형 탄성, 정상 전도, 미소변형 열팽창)에서는 **외삽이 구성상 정확**해진다.
3. **per-sample position_scale** 옵션 (geometry 스케일 일반화).
4. node type one-hot → **재료 물성 벡터**(E, ν, α, k, ρ, cp). 새 재료가 categorical
   미지 토큰이 아니라 물성 공간의 보간점이 된다.

### 장벽 3 — slice 토큰이 붕괴한다 (Transolver++가 지적한 것)

Transolver++는 N이 커질수록 slice softmax가 균일해져 토큰이 **평균 풀링으로 degenerate**
한다고 보고한다("if w tends to be uniform, attention will degenerate to average pooling").
처방은 두 개:

```
Ada-Temp   τ = τ₀ + Linear(x_i)                                   절점별 온도(=softmax 예리도)
Rep-Slice  Softmax((Linear(x) − log(−log ε)) / τ)                 Gumbel 재매개화
```

이 repo는 **head별 스칼라 τ 하나**를 학습하고 [0.1, 5]로 clamp한다
([physics_attention.py:102](../../../methods/Transolver/model/physics_attention.py#L102),
[:110-113](../../../methods/Transolver/model/physics_attention.py#L110-L113)). foundation model에서는 토큰
구별력이 곧 용량이다 — 토큰이 전부 전역 평균으로 수렴하면 **조건 토큰이 변조할 대상이
사라진다.** 그래서 Ada-Temp는 선택이 아니라 선행 조건이다.

구현상 좋은 소식: τ는 `x_g` 행에서 계산되는 절점별 스칼라라서 `_slice_weights`
([physics_attention.py:130-137](../../../methods/Transolver/model/physics_attention.py#L130-L137)) 안에서 `[H, N, 1]`
브로드캐스트로 들어가고, `_fused_slice_weights`의 융합(C→[H,M] 가중치 융합)과 충돌하지
않으며 tile 단위로도 성립한다. 단 `misc/verify_v3.py`의 **L1(naive == slice_space)**
동등성 검사를 양쪽 커널에 동일하게 반영해서 유지해야 한다.

---

## 5. BC 토큰 설계

BC는 스칼라가 아니다. (종류, 값, 공간적 support, 단위)의 4-튜플이다.

```
token_k = TypeEmb(semantic_name_k)          # 무엇인가:  'fixed_temperature', 'inlet_velocity'
        + ValueEnc(v̂_k)                     # 얼마나:    무차원화된 값
        + SupportPool({x_i : i ∈ ∂Ω_k})      # 어디에:    해당 면의 절점 집계
```

**TypeEmb — open vocabulary로.** `nn.Embedding(num_types)`를 쓰면
`node_type_to_idx`와 똑같은 닫힌 vocabulary 함정에 빠진다. 조건 이름+단위 문자열의
**고정(frozen) 텍스트 임베딩**이나 해시 n-gram을 쓰면, 처음 보는 BC 종류가 의미적으로
가까운 것 근처로 매핑된다. (MOL-LLM류가 BC를 metadata 토큰으로 다루는 방식.)

**ValueEnc — 단조·선형 tail로. random Fourier feature를 쓰면 안 된다.**
좌표 인코딩의 기본값이라 무심코 쓰기 쉬운데, 물리 크기값에 sinusoidal encoding을 쓰면
**주기성 때문에 학습 범위의 4× 밖 값이 학습된 값으로 aliasing** 되어 모델이 자신 있게
틀린 regime을 예측한다. `[v̂, sign(v̂)·log1p|v̂|]`를 GELU MLP에 넣으면 조각선형이라
범위 밖에서 선형으로 이어진다. (실험의 B/C가 쓴 방식.)

> **§12 갱신 반영**: GeoPT는 산업 규모에서 BC를 **per-node 조건장(prompt)** 으로 준다
> (공력=유입 속도장, 크래시=충격점 감쇠장). §3에서 per-node 채널이 전역 토큰과 대등했던
> 것과 합쳐 보면, **1차 경로는 per-node 조건장**이고 전역 토큰은 per-node로 표현할 수
> 없는 것(물성, 무차원수, regime, log L)에만 쓰는 게 맞다. 아래 SupportPool은 그
> per-node 장을 토큰 쪽에서 보완하는 장치로 읽을 것.

**SupportPool — BC는 면에 산다.** 전역 스칼라 토큰만 쓰면 "어디에" 걸리는지가 사라진다.
support 절점들을 slice 방식(또는 AB-UPT식 supernode pooling)으로 집계해 토큰에
기하 정체성을 준다. 그리고 **per-node BC 채널**로 보완한다: 각 BC 면까지의 부호 거리 +
최근접 BC 절점의 값. 전역 토큰이 "어떤 regime인가", node 채널이 "어디서 얼마나 가까운가"를
담당한다. 둘 중 하나만으로는 부족하다.

> 위 실험의 변형 B는 BC가 균일해서 **전역 스칼라 경로만** 검증했다. support pooling
> 경로는 별도 검증이 필요하다.

**무차원화가 실제 일반화 엔진이다.** `condition_schema`의 단위에서 무차원수(Re, Pr, Bi,
Fo, Pe, α·ΔT, t/τ)를 유도해 모델에는 **무차원 값만** 넣는다. 그러면 물리 단위로는
"새로운" BC가 무차원 공간에서는 in-distribution일 수 있다 — 하나의 학습 regime이
**상사(similarity) 클래스 전체**를 덮는다. DimINO·scale-consistent learning 계열이
보고하는 이득이 이것이고, 뒤집으면 **OOD 판정도 물리 단위가 아니라 무차원 공간에서**
해야 한다는 뜻이다.

---

## 6. 온도 (두 가지 의미, 둘 다 실재한다)

### (a) 물리 온도 — 이 데이터에서 가장 중요한 BC

- warpage/사출 문제에서 mold temp, melt temp, cooling time, ambient는 **바로 그
  파라미터**다. 그런데 ex1/ex2 export에 온도 채널이 아예 없다 → 재export 필요.
- 온도는 **입력 BC이면서 출력 field**다. 하나의 backbone으로 열+구조를 하려면 §8 Phase 5의
  채널 무관 설계가 필요하다.
- **절대온도를 z-score로 넣지 말 것.** 역학이 반응하는 건 ΔT다. 기준값 대비 상대값으로 넣는다.
- **α·ΔT가 무차원 변형 구동량**이고, 이게 곧 §3의 변형 C가 분리해낸 진폭이다.
  즉 열변형은 *보장된 외삽이 가능한* 바로 그 경우에 해당한다. 이 데이터셋에서 가장
  값이 큰 한 방이다.
- 다물리 단계화: 열 해석 → 온도장 → 구조. (1) 온도를 채널로 갖는 단일 모델, 또는
  (2) 1단계의 slice 토큰을 2단계의 context로 넘기는 2-stage. **(2)는
  `forward_with_tokens`로 이미 가능하다.**
- 무차원수: Biot, Fourier(t/τ_thermal), Péclet, Nusselt.

### (b) 코드의 softmax temperature — 이름 충돌 경고

`temperature_init` / `temperature_min` / `temperature_max`는 **이미 slice 배정 softmax의
예리도**를 뜻한다 ([Transolver.py:30-32](../../../methods/Transolver/model/Transolver.py#L30-L32)). 여기에 열 BC를
추가하면서 `temperature_*` 계열 config 키를 쓰면 치명적으로 헷갈린다.
**BC 쪽 키는 `cond_*` / `bc_*` 접두어로 분리할 것.** (그리고 §4 장벽 3의 Ada-Temp는
바로 이 (b) 쪽 temperature를 절점별로 만드는 작업이다.)

---

## 7. In-context 조건화 — 이 repo에 이미 반쯤 구현돼 있다

가장 큰 발견은 이것이다. `model/blocks.py`의 2단 분리:

```python
compute_tokens(fx, ptr, ...)                  -> [H, M, D] per graph   # blocks.py:64
forward_with_tokens(fx, ptr, tokens, ...)     -> 다른 node 집합을 그 토큰에 디코드  # blocks.py:77
```

이건 메모리 절약(decoupled inference / amortized training)을 위해 만들어졌지만,
**정확히 in-context 조건화 인터페이스**다:

1. **demonstration** 샘플을 하나 잡는다 — 같은 geometry, 다른(이미 아는) BC, 해까지 있는 것.
2. 그 샘플로 `compute_tokens`를 돌린다 (해 field를 입력 채널로 임베딩).
3. 그 M개 토큰을 query 샘플의 토큰에 `_slice_attend`에서 concat한다.
4. 모델이 응답을 **암기**하는 대신 **측정**할 수 있게 된다.

ICON/Zebra 계열이 "학습 파라미터 범위 밖으로 일반화된다"고 보고하는 이유가 이것이다 —
demo 쌍이 응답의 국소 기울기를 실어 나른다. CAE에서는 대단히 실용적이다: **거친/저해상도
해석 1회**(또는 인접 BC의 수렴해 1개)를 주면 surrogate가 거기서 외삽한다. "임의의 BC"에
가장 가까이 가는 현실적 경로다.

비용은 토큰 빌드 스트림 1개 추가 — amortized training이 이미 예산화한 cache-stream 비용과
같은 크기다.

> **함정**: context 스트림도 autograd를 타야 한다. `no_grad` 캐시를 쓰면
> aggregate pass에만 등장하는 `in_project_fx`가 조용히 얼어붙는다 —
> [amortized.py:37-39](../../../methods/Transolver/model/amortized.py#L37-L39)에 이미 기록된 교훈이 그대로 적용된다.

---

## 8. 단계별 구현 계획

### Phase 0 — 데이터 계약 (blocking, ML 아님)

- `data/{id}/conditions` + `metadata/condition_schema` (§4 장벽 1의 스키마)
- 가능한 곳에서 온도 field export, `feature_names` 갱신
- `dataset/DATASET_FORMAT.md`, `cae_suite/dataset_probe.py` 교차검증 갱신

> 참고: 이 monorepo에는 **샘플별 파라메트릭 입력 패턴이 이미 있다** — DeepONet의
> branch 입력(`train/branch_values` [samples, dim],
> `dataset/benchmarks/deeponet_fractional2d/prepare_fractional2d.py:356`,
> 소비부 `Neural_Operator/model/deeponet_fractional2d.py:68`). 다만 benchmark 전용
> 레이아웃이라 **공유 mesh HDF5 계약 밖에 있다.** Phase 0은 발명이 아니라 그 개념을
> 공유 계약 안으로 끌어오는 작업이다.

### Phase 1 — condition token (하위 호환: K=0이면 수치적으로 동일)

- 신규 `Transolver/model/conditioning.py`: `ConditionEncoder` (TypeEmb + ValueEnc + SupportPool)
- `_slice_attend(tokens, cond)` → `[M ; K]` concat, attend, **앞 M개만** deslice로.
  호출부가 세 곳이고 **세 곳 모두** 같은 `cond`를 받아야 한다 —
  [`_forward_naive`:203](../../../methods/Transolver/model/physics_attention.py#L203),
  [`_forward_slice_space`:301](../../../methods/Transolver/model/physics_attention.py#L301),
  [`decode_with_tokens`:341](../../../methods/Transolver/model/physics_attention.py#L341).
  한 곳이라도 빠지면 L1 커널 동등성이 깨진다(그리고 naive는 구 checkpoint 복원 경로다).
- `blocks.py`의 `forward` / `compute_tokens` / `forward_with_tokens`에 `cond` 배선
- `Transolver.py`: forward당 1회 graph-level로 생성해 내려보냄
- config 키: `condition_tokens`, `condition_source`, `condition_dropout`
  (조건 일부를 랜덤 드롭 → BC 결측/부분 지정에 강건, classifier-free guidance식).
  **`condition_dropout`은 편의 기능이 아니다** — §3 결론 4에서 조건 토큰이 값 범위를
  벗어나면 *타깃이 그 조건에 의존하지 않을 때조차* 해를 망치는 게 측정됐다. 드롭아웃은
  그 spurious 의존성을 정규화하는 직접적인 대책이다. 조건 값 clamp도 같이 둘 것.
- `cae_suite/specs/transolver.py`의 `known_keys`
  ([specs/transolver.py:193](../../../cae_suite/specs/transolver.py#L193)) 갱신 — 안 하면
  `CFG-UNKNOWN-001`
- 검증: K=0 수치 동일 / `verify_v3.py` L1 (naive == slice_space) 유지
- 병렬성 주의: 조건 토큰은 graph-level이라 **all-reduce 불필요**하지만 rank 간 동일해야
  한다. amortized에서는 cache/query 두 스트림이 **같은** 조건 토큰을 공유한다.

§3 실험에서 실제로 돌려본 최소 패치의 핵심(`misc/bc_token_extrapolation.py`):

```python
def _slice_attend(self, tokens, cond=None):     # tokens [H, M, D], cond [K, C]
    if cond is None:
        return <기존 경로>                       # K=0 → 수치적으로 동일
    H, M, D = tokens.shape
    ck  = self.cond_proj(cond).view(-1, H, D).permute(1, 0, 2)    # [H, K, D]
    seq = torch.cat([tokens, ck], dim=1)                          # [H, M+K, D]
    q, k, v = self.to_q(seq), self.to_k(seq), self.to_v(seq)
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    attn = torch.softmax(_stable(dots), dim=-1).to(v.dtype)
    return torch.matmul(self.dropout(attn), v)[:, :M]   # ← 조건 토큰은 deslice 안 함
```

`[:, :M]` 한 줄이 tiling / node-shard / amortized를 전부 무손상으로 보존하는 지점이다.

### Phase 2 — 외삽을 위한 정규화 (실험이 지목한 최우선 ML 작업)

- `target_scaling {dataset_zscore | instance | analytic}`
- `instance`: 샘플별 RevIN, 스케일 저장·복원
- `analytic`: 스케일 = schema에서 유도한 `g(conditions)` (열변형이면 α·ΔT)
- `position_scale` per-sample 옵션
- node type one-hot → 재료 물성 벡터 (open vocabulary)

### Phase 3 — 토큰 용량 (Transolver++)

- Ada-Temp, Rep-Slice. L1 parity 유지하도록 양쪽 커널에 동일 적용.

### Phase 4 — in-context 조건화 (§7)

- `context_samples` config; 학습 중 (demo BC, query BC) 쌍을 랜덤 샘플링
- context 스트림 autograd 유지

### Phase 5 — 채널 무관 다물리

- 채널별 embed/decode + field token; `input_var`/`output_var` 가변; 채널별 loss 마스킹
- ex1 + ex2 + deepjeb + 열 데이터를 **하나의 모델**로

### Phase 6 — OOD 탐지와 검증 (외삽을 보장할 수 없으므로 탐지해야 한다)

- 무차원 공간에서의 조건 거리 + 학습시 slice-token 분포와의 거리 → confidence, 임계 초과 시 flag/거부
- 값싼 residual 연산자가 있으면 test-time residual check

---

## 9. 데이터 규모 현실 점검

100 + 50 샘플은 foundation model이 아니라 surrogate다. 아키텍처 작업은 쉬운 20%다.
현실적 경로:

- 공개 코퍼스로 사전학습 → design point로 fine-tune. (MPP/Poseidon의 근거: 다물리
  사전학습 모델을 fine-tune하면 처음 보는 물리에서 from-scratch보다 낫다.)
- 이미 `dataset/deepjeb.h5`(730MB)를 갖고 있고, The Well / PDEBench /
  DrivAerNet++ / CarBench가 후보다.
- 목표 규모: geometry × BC × physics 조합으로 O(10⁴) 해석.

역설적으로 좋은 소식: 그 규모를 감당하게 해주는 게 이 repo가 이미 만들어 둔
amortized / decoupled / node-shard 기계장치다.

---

## 10. 정직한 한계

- **"boundary-indexed operator families"**: 데이터 기반 solver는 BC에 대해 불변인
  단일 operator를 배우지 않는다. **BC로 색인된 operator 族**을 배우고, 학습 BC 분포
  밖에서의 거동은 사실상 제약이 없다. 그래서 *데이터·토큰만으로 "완전한 extrapolation"은
  불가능하다.* (이건 학습 부족이 아니라 문제 설정의 성질이다.)
- Re 이동에서 FNO가 단순 baseline보다 못한 사례, PDE FM의 극한 하중 OOD 전이 실패
  사례가 보고돼 있다.
- §3의 nonlinear regime이 이걸 자체 데이터로 재현한다: 응답 법칙이 학습 가정과
  다르면 조건 토큰(B: 95.58%)이 조건화를 아예 안 한 것보다 **더 나쁘고**, 진폭 분리
  (C: 75.20%)도 구해주지 못한다. **틀린 조건화는 없는 조건화보다 나쁘다.**

**얻을 수 있는 것**

1. 해석적으로 분리한 방향으로는 **정확한** 외삽 (선형 응답: α·ΔT, 하중 진폭, 정상 전도)
2. 무차원화로 **상사 클래스 전체** 커버 → 물리 단위의 "새 BC"가 in-distribution이 됨
3. in-context 증거로 **범위 밖** 일반화 (거친 해석 1회)
4. 나머지는 **탐지하고 거부**

**얻을 수 없는 것**: regime 전이 (층류→난류, 탄성→소성, 접촉 개시). 학습 집합에 없는
분기(bifurcation)를 토큰으로 만들어낼 수는 없다.

---

## 12. "Transolver-3를 키우고 데이터를 다 넣으면?" — 이미 답이 나와 있다

**그 실험은 이미 됐고, 백본이 Transolver다.**

**GeoPT** (ICML 2026, *Scaling Physics Simulation via Lifted Geometric Pre-Training*,
코드 공개)는 **Transolver를 백본으로** 100만 개 이상 샘플에 사전학습한 물리 시뮬레이션
foundation model이다.

**물리 범위 — CFD 전용이 아니다:**

| 벤치마크 | 물리 |
| --- | --- |
| DrivAerML | 자동차 외부 공력 (유체) |
| NASA-CRM | 항공기 날개 공력 (유체) |
| DTCHull | 선박 저항/유체동역학 (유체) |
| **Car-Crash** | **충돌 변형 (고체역학)** |
| **Radiosity** | **광 전달** — diffuse inter-reflection 전역 조명, 지배 방정식이 근본적으로 다름 |

Radiosity 전이에서도 from-scratch보다 낮은 오차를 낸다 — 기하 prior가 물리에 특정되지
않는다는 증거다.

> **단, 이 repo의 관심 물리는 그 목록에 없다.** Car-Crash는 고체지만 **동적 충격
> 대변형**이고, ex1/ex2의 warpage는 **정적·준정적 열-구조**다. Radiosity는 열이
> 아니라 **광학**이다(복사 열전달과 적분방정식 구조는 닮았지만 전도+열팽창은 아니다).
> 즉 GeoPT는 "구조/열도 커버한다"의 증거가 아니라 **"기하 prior가 물리 종류를 넘어
> 전이된다"의 증거**로 읽어야 한다. 우리 물리로의 전이는 직접 확인해야 한다.

얻은 것과 얻지 못한 것을 정확히 봐야 한다:

- **얻은 것**: 라벨 데이터 **20–60% 절감**, 수렴 **2× 가속**.
- **얻지 못한 것(주장하지 않음)**: 임의 BC 외삽, 스케일 넘나들기.

즉 **스케일업의 배당금은 "필요한 해석 수가 줄어드는 것"이지 "외삽"이 아니다.** §3의
실측과 정확히 같은 결론이고, scaling law 문헌도 같은 말을 한다 — in-distribution은
power law로 개선되지만 **학습 도메인 밖에서는 개선이 미미하거나 오히려 악화된다.**

그리고 GeoPT 논문의 진단 한 줄이 이 repo에 직접 꽂힌다:

> Transolver는 데이터가 충분하면 좋은 확장성을 보이지만, **데이터가 제한된 산업
> 시뮬레이션에서는 overfitting으로 보이는 scaling bottleneck에 걸린다.**

**ex1 100개 + ex2 50개에서 모델을 키우면 더 나빠진다.** "크기를 늘린다"는 선택지는
데이터가 먼저 커진 다음에야 열린다. 순서를 뒤집을 수 없다.

### GeoPT가 라벨 병목을 우회하는 방식 (이 부분이 진짜 배울 점)

**1M 샘플은 1M번의 CAE 해석이 아니다. 해석은 0회다.**

```
사전학습 코퍼스 = ShapeNet의 산업 관련 기하 10,000+개  ×  기하당 궤적 100개  ≈ 1M 샘플
궤적 생성 = fast ray-tracing. 무작위 속도 벡터로 입자를 쏴서 기하 경계에 닿기까지의
            기하 특징 궤적을 self-supervision 타깃으로 삼는다.
            → "유체 solver보다 orders of magnitude 빠름"
```

병목은 해석 결과(라벨)이고 CAD 기하는 넘쳐난다. GeoPT는 그 비대칭을 정확히 공략한다.
(기하만 주는 사전학습은 dynamics가 없어 negative transfer가 나는데, 이 "lifting"이
그걸 고친다 — 논문의 표현으로 *"geometry-only supervision is meaningless for physics"*.)

**이게 이 프로젝트에 결정적인 이유**: 해석 150개가 병목인 상황에서 **유일하게 열려 있는
스케일 축**이다. 보유 CAD 라이브러리(또는 ShapeNet 그대로)로 자체 사전학습이 가능하고,
CAE 라이선스도 solver 시간도 들지 않는다.

### GeoPT의 조건화 방식은 우리 §5 설계를 수정하게 만든다

fine-tuning에서 GeoPT는 무작위 속도장 대신 **task-specific 속도장 V_S**를 넣는데,
그것이 곧 BC 인코딩이다:

| 물리 | prompt 속도장이 인코딩하는 것 |
| --- | --- |
| 공력 | 유입 조건 — **angle of attack, speed** |
| 유체동역학 | 구조물 주변 물/공기 흐름 |
| 크래시 | **충격점에서 공간적으로 감쇠하는 장** = 힘 전파 |

즉 GeoPT는 BC를 **전역 스칼라 토큰이 아니라 per-node 벡터장(prompt)** 으로 준다.
이건 §5가 제안한 (type, value, **spatial support**) 3-튜플 중 **support 쪽에 무게를
싣는 선택**이고, **§3의 실측과 일치한다** — per-node 채널(A)이 전역 토큰(B)과 대등하거나
더 나았다. 산업 규모에서 검증된 쪽이 per-node라는 사실은 우연으로 보기 어렵다.

**따라서 §5의 처방을 이렇게 조정한다:**

1. **1차 경로는 per-node 조건장** (GeoPT식 prompt). warpage/열이라면 각 BC 면에서
   공간적으로 감쇠하는 장 — 절점에서 각 BC 면까지의 거리로 만든 감쇠 필드에 BC 값을
   실어 보낸다. 크래시의 "충격점 감쇠장"과 같은 구조이고, §5의 SupportPool보다
   구현이 쉽고 이미 검증돼 있다.
2. **전역 조건 토큰은 per-node 장으로 표현 불가능한 것에만** — 재료 물성, 무차원수,
   regime 선언, `log(L/L_ref)`. §3에서 전역 토큰이 값 범위를 벗어날 때 해를 망친
   것을 감안하면, 전역 토큰은 적게 쓰는 편이 안전하다.

**실행 항목**: 백본이 이 repo와 **같은 Transolver 계열**이다. 가중치 이식 가능성
(레이아웃·`slice_num`·`latent_dim` 호환성)을 먼저 확인할 것. downstream 데이터셋도
HuggingFace(`GeoPT/Downstream_Physics_Simulation`)에 공개돼 있어 이식 검증 자체를
남의 데이터로 할 수 있다. 성공하면 Phase 0~2 없이도 즉시 얻는 가장 값싼 승리다.

---

## 13. 이종 데이터셋을 하나로 녹이기 — 지적한 part-name 문제가 맞다

### 실측 증거: 지금 스키마로 두 파일을 합치면 손상된다

```
ex1.h5  part ids = {0,1,2,3}   counts 예: [  461,  24528,  107,  107]   2D 판, bbox 4578×2116×0 mm
ex2.h5  part ids = {0,1,2,3}   counts 예: [38773, 155647, 4096, 1477]   3D,   bbox 1000× 960×1000 mm
두 파일 모두 전 샘플이 동일한 part-id 집합 (distinct sets = 1)
```

- part 2가 ex1에서는 107 노드, ex2에서는 4096 노드인데 **one-hot 슬롯이 같다.** 합치면
  모델은 "이 둘은 같은 것"이라고 배운다. 가설이 아니라 지금 코드의 동작이다
  ([mesh_dataset.py:330-349](../../../methods/Transolver/general_modules/mesh_dataset.py#L330-L349)).
- ex1의 part 2와 3은 노드 수가 **정확히 같다**(107/107, 296/296, 207/207, 286/286) —
  대칭 구속면 또는 하중면 쌍의 서명이다. **BC가 실제로 part id로 인코딩돼 있다.**
- one-hot은 "구속 그룹 2에 속함"만 말하고 **무엇이 얼마나** 걸렸는지(고정? 100 N? 80 °C?)는
  말하지 않는다. 그래서 *같은 기하 + 다른 하중 크기*인 두 샘플은 **입력이 완전히
  동일하고 타깃만 다르다.*

> **지금 확인해야 할 것**: ex1은 T=1이라 `x_phys`가 전부 0이고
> ([mesh_dataset.py:445](../../../methods/Transolver/general_modules/mesh_dataset.py#L445)) part-id 집합도 전
> 샘플 동일하다. 즉 **샘플 간 입력 변화는 geometry뿐이다.** 100개 design point가
> 하중/온도도 함께 흔들었다면 그 변화는 입력에 전혀 나타나지 않고 **라벨 노이즈로**
> 들어간다 — 현재 학습 곡선의 floor가 그것 때문일 수 있다. Workbench 파라미터 표를
> 열어 DP가 geometry만 흔든 건지 확인할 것. (§3에서 관측 불가 변수 하나 때문에 모델이
> 아무것도 학습하지 못한 것과 같은 구조의 문제다.)

### 통일 표현은 "고정 텐서 레이아웃"이 아니라 self-describing manifest다

이게 핵심 통찰이다. 모든 데이터셋을 8행으로 맞추는 게 통일이 아니다. **채널이 스스로
자기 의미와 단위를 신고하게** 만드는 것이 통일이다. The Well이 16개 데이터셋을 통일
HDF5로 묶으면서 per-dataset metadata(fields, BC, physical coefficients)를 physics-aware
conditioning용으로 보존하는 방식이고, MORPH의 UPTF-7도 같은 계열이다.

```
metadata/
  channels/
    names       str   [C]      'x_disp', 'temperature', 'pressure', 'vof', …
    role        str   [C]      coord | state_in | state_out | marker | derived
    dim         int   [C, 7]   SI 차원 지수 [M, L, T, Θ, I, N, J]        ← 이 설계의 핵심
    ref         float [C]      무차원화 기준값(또는 유도 규칙)
  regions/                     part id를 '의미 있는 것'으로 승격
    ids         int   [R]
    kind        str   [R]      volume | surface | edge | nodeset
    material    float [R, P]   E, ν, α, k, ρ, cp, …     ← categorical 대신 물성 벡터
  bcs/                         BC를 명시적 레코드로 (part '이름'이 아니라 참조)
    ids         int   [K]
    type        str   [K]      dirichlet | neumann | robin | periodic | contact
    field       str   [K]      어느 채널에 걸리는가
    region      int   [K]      regions/ids 참조
    value       float [K, T]   시간 의존 허용
    dim         int   [K, 7]   값의 SI 차원
  scales/
    length      float          이 샘플의 대표 길이 L
    time        float          대표 시간 τ
data/{id}/conditions           float [K]   샘플별 BC 값 (schema가 해석)
```

**`dim [C, 7]` SI 차원 지수 벡터가 이 설계에서 가장 값나가는 한 줄이다.** 이게 있으면:

1. **무차원화가 자동 유도된다** — 데이터셋마다 손으로 레시피를 쓸 필요가 없다.
   (채널 차원 + `scales`) 조합으로 로더가 무차원 값을 계산한다.
2. **단위가 안 맞는 병합을 정적으로 검출할 수 있다** — `cae_suite/dataset_probe.py`가
   잡을 수 있는 검사가 된다. `DATA-UNIT-*` 진단 코드를 하나 늘리면 된다.
3. 새 데이터셋 추가가 **코드 변경 없이** 된다. foundation model 파이프라인의 정의다.

즉 "하나의 uniform dataset"의 실질적 의미는 **텐서 모양이 같은 것이 아니라 차원적으로
해석 가능한 것**이다.

### Transolver 쪽 대응

채널 수가 가변이므로 `preprocess`의 고정 `embed_input_size`
([Transolver.py:57](../../../methods/Transolver/model/Transolver.py#L57))를 버리고 **채널별 임베딩 + field token**으로
간다(Phase 5). 채널 c는 `Embed_c(v) = W_type[c]·v + b_type[c]`처럼 semantic 임베딩으로
들어가고 없는 채널은 마스킹한다 — MPP의 "shared embedding space + 1×1 conv"와 같은 아이디어.
`use_node_types`의 닫힌 one-hot은 `regions/material` 물성 벡터로 대체된다.

---

## 14. meso-scale부터 100 m 항공기까지 — 가능한가

### 물리 단위로는 불가능하다

- 좌표 정규화가 **데이터셋 전역 RMS 반경 하나**로 나눈다
  ([mesh_dataset.py:22-32](../../../methods/Transolver/general_modules/mesh_dataset.py#L22-L32),
  `finalize_position_scale`). µm 샘플은 정규화 후 전부 ~0으로 붕괴한다. ex1만 봐도
  bbox가 4578×2116 vs 3112×5908 mm로 이미 2× 퍼져 있는데 전역 스케일 하나를 쓴다.
- float32 동적 범위: 100 m 메시에 µm 변위를 담으면 유효 자릿수가 남지 않는다.

### 유일한 이론적 매개는 상사(similarity)다

무차원수가 같은 두 계는 rescaling을 제외하면 **같은 계**다. 따라서:

1. **좌표는 per-sample 단위 박스로** (형상만) → 스케일 불변이 구성상 성립
2. **절대 크기를 버리지 말고 조건 토큰으로 복원** — `log(L / L_ref)`. 크기는 물리적으로
   중요하다 (Re ∝ L, Bi ∝ L, 좌굴 하중 ∝ L, 표면/체적비 ∝ 1/L). log라서 8자수가
   −9…+2의 좁은 범위로 들어오고, §5의 "단조·선형 tail" 인코딩과 정확히 맞는다.
3. **모델이 보는 값은 무차원만** (§13의 `dim` 벡터로 자동 유도)

### 그러나 상사는 물리 regime 안에서만 성립한다

meso와 항공기는 **같은 상사 클래스가 아니다**:

- meso: 표면장력·점성 지배, Knudsen 효과, 입계 소성, 비연속체 효과 → **지배 방정식 자체가 다르다**
- 항공기: 압축성, 난류, 공탄성

"무차원수가 같다"가 의미를 가지려면 **활성 무차원수 집합이 같아야** 한다. 그래서 하나가
더 필요하다: **regime 토큰** — 어떤 항/무차원수가 지배적인지를 명시적으로 선언. 없으면
모델은 서로 다른 방정식의 데이터를 평균한다.

### 게다가 Transolver 고유의 문제가 겹친다

N이 1e4(meso RVE)와 1e8(항공기)로 **4자수** 차이인데 `slice_num`은 같다. Transolver++의
attention degeneration은 N에 대한 현상이므로 **같은 M 토큰이 두 극단에서 완전히 다르게
거동한다** — 1e8에서는 균일 붕괴, 1e4에서는 과잉 분할. §4 장벽 3의 Ada-Temp가 선택이
아닌 이유가 여기서 한 번 더 나온다. amortized 서브샘플 비율도 자수 단위로 달라야 한다.

### 시연된 사례는 없다

문헌 조사에서 **길이 스케일을 자수 단위로 넘나드는 CAE foundation model은 나오지
않았다.** GeoPT도 cars/aircraft/ships/crash로 전부 매크로다. Luminary SHIFT-SUV는
SUV 한 종·공력 한 물리다.

**현실적 목표 재설정**: "meso→항공기 하나의 모델"이 아니라
**공유 geometry backbone + 상사 클래스별 물리 헤드**. GeoPT가 실제로 하는 게 그것이다 —
기하 표현은 스케일 불변으로 공유하고, 물리는 도메인별 fine-tune.

---

## 15. 다른 CAE foundation model 지형 (2026)

| 이름 | 주체 | 백본 | 스코프 | 규모 | 시사점 |
| --- | --- | --- | --- | --- | --- |
| **GeoPT** (ICML'26) | Physics-Scaling | **Transolver** | 기하 사전학습 → car/aircraft/ship/crash | **1M+ 무라벨** | 백본이 같다 → **가중치 이식 검토** |
| **SHIFT-SUV** | Luminary + Honda + NVIDIA | PhysicsNeMo **DoMINO** | SUV 외부 공력 1종 | ~1k → 25k 해석 | "산업 FM"의 현실 규모 |
| AB-UPT | Emmi AI | UPT (supernode + anchored decoder) | 자동차 CFD 33k~150M cell | — | 조건화 + divergence-free 하드 제약 |
| DoMINO | NVIDIA | multiscale neural operator | 대규모 외부 공력 | — | multi-scale을 아키텍처로 해결 |
| MPP | Polymathic AI | ViT 계열 | 다물리 시공간(격자) | — | 공유 임베딩 + RevIN |
| Poseidon | ETH | hierarchical multiscale ViT | 격자 PDE | — | 사전학습 sample efficiency |
| MORPH | — | UPTF-7 | 임의 modality | — | 통일 텐서 포맷 선례 |
| ICON / Zebra | — | AR transformer / VQ-VAE+AR | parametric PDE | — | in-context가 파라미터 범위 밖으로 |
| Shape | — | self-supervised 3D | 산업 CAD 분석 | — | 무라벨 기하 사전학습 계열 |
| **The Well** | Polymathic AI | (데이터셋) | 16개 데이터셋 통일 HDF5 | 15 TB | **격자 전용** |
| **caemldatasets.org** | 커뮤니티 | (데이터셋) | AhmedML 500 / WindsorML 355 / DrivAerML 500 / HiLiftAeroML | ~1.4k 해석 | 일관 포맷, CC-BY-SA |

**관찰 3개**

1. **메시 기반 CAE에는 The Well이 없다.** 격자 PDE는 통일 코퍼스가 있지만 비정형 메시는
   각자 포맷이다. §13의 스키마 작업은 남이 대신 해주지 않는다.
2. 2026년 산업 "foundation model"의 실제 규모는 **O(10³)–O(10⁴) 해석, 단일 도메인**이다.
   SHIFT-SUV가 1000 → 25000을 목표로 한다. "모든 CAE를 하나로"는 아직 아무도 안 했다.
3. **무라벨 기하 사전학습(GeoPT, Shape)이 라벨 병목의 답으로 수렴하고 있다.** CAD는 많고
   해석 결과는 150개인 이 상황에 정확히 대응한다.

---

## 16. 참고문헌

- Transolver (ICML'24) — Physics-Attention: https://ise.thss.tsinghua.edu.cn/~mlong/doc/Transolver-icml24.pdf
- Transolver++ (attention degeneration, Ada-Temp, Rep-Slice) — https://arxiv.org/abs/2502.02414
- Transolver-3 (이 repo가 구현한 aggregate-then-project) — https://arxiv.org/html/2602.04940v2
- Multiple Physics Pretraining (공유 임베딩 + RevIN) — https://arxiv.org/abs/2310.02994
- ICON: In-context operator learning — https://www.pnas.org/doi/10.1073/pnas.2310142120
- Zebra: in-context generative pretraining for parametric PDEs — https://arxiv.org/pdf/2410.03437
- One Operator to Rule Them All? (boundary-indexed operator families) — https://arxiv.org/html/2603.01406
- DimINO: Dimension-Informed Neural Operator — https://arxiv.org/html/2410.05894
- Striding Across Reynolds Numbers — https://arxiv.org/pdf/2605.30112
- OOD transfer of PDE foundation models under extreme loading — https://arxiv.org/pdf/2603.04354
- AB-UPT (supernode pooling, 산업 규모 조건화) — https://arxiv.org/abs/2502.09692
- MORPH: PDE foundation models with arbitrary data modality (UPTF-7) — https://arxiv.org/pdf/2509.21670
- **GeoPT: Scaling Physics Simulation via Lifted Geometric Pre-Training** (ICML'26,
  Transolver 백본, ShapeNet 10k 기하 × 100 궤적 = 1M+ 무라벨 사전학습;
  downstream = DrivAerML / NASA-CRM / DTCHull / Car-Crash / Radiosity) —
  https://arxiv.org/html/2602.20399v1 · https://physics-scaling.github.io/GeoPT/ ·
  코드 https://github.com/Physics-Scaling/GeoPT ·
  데이터 https://huggingface.co/datasets/GeoPT/Downstream_Physics_Simulation ·
  리뷰 https://openreview.net/forum?id=N9qIqvanBj
- Luminary SHIFT 모델군 (SHIFT-SUV, DoMINO 기반, Honda·NVIDIA 협업) —
  https://luminary.ai/resources/introducing-luminary-shift-models-a-suite-of-physics-ai-foundation-models-to-transform-engineering-design/
- Shape: Self-Supervised 3D Geometry Foundation Model for Industrial CAD — https://arxiv.org/abs/2604.22826
- OOD scaling: "training set size 증가가 도메인 밖에서는 개선 미미/악화" —
  https://arxiv.org/pdf/2406.06489 · 소데이터 PDE surrogate OOD — https://arxiv.org/abs/2601.08404
- The Well (16개 데이터셋 통일 HDF5, 15TB, 격자) — https://arxiv.org/pdf/2511.21861 계열
- CAE ML Datasets (AhmedML / WindsorML / DrivAerML / HiLiftAeroML) — https://caemldatasets.org/
- Scale-Consistent Learning for PDEs — https://arxiv.org/pdf/2507.18813
