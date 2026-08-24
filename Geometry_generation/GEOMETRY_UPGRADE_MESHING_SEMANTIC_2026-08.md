# SDFFlow 개선 · 자동 메슁 · Semantic Conditional 생성 — 실측 기반 연구

**날짜:** 2026-08-24
**대상:** `Geometry_generation/` (SDFFlow), 학습 완료된 `output/geometry_generation/ex1`
**선행 문서:** `SOTA_CONDITIONAL_GEOMETRY_SURVEY_2026-07.md`, `FOUNDATION_MODEL_PLAN.md`
**성격:** 이 문서의 1장·3장 숫자는 **전부 이 저장소의 실제 체크포인트/데이터로 직접 측정한 값**이다.
추정이 아니다. 선행 문서와 결론이 갈리는 곳이 두 군데 있고, 그 근거도 아래에 있다.

---

## 0. 요약

**결론 1 — 병목은 VecSet(latent 구조)이 아니라 인코더다.**
7월 문서는 "global token → VecSet 전환이 가장 큰 레버"라고 결론냈다. 측정해 보니 **아니다**.
동일한 frozen decoder, 동일한 1×256 latent로 held-out 형상에 대해 latent만 test-time 최적화하면
표면오차가 **mean 2.4배, p95 3.1배** 좋아진다. 즉 **decoder와 latent는 이미 그 형상을 표현할 능력이
있는데 인코더가 그 latent를 못 찾는다.** VecSet은 인코더 격차를 메운 *다음*에 의미가 있다.

**결론 2 — 조건 벡터가 사실상 비어 있다. 이게 "내 맘대로 조건 생성"의 진짜 블로커다.**
현재 4개 조건 중 `bbox_x`는 변동계수 **0.45%**, `bbox_z`는 2.4%(극단적 heavy-tail), `bbox_y`는
**표준편차 정확히 0**. 실질 자유도는 **volume + area 약 1.5개**이고 둘의 상관은 0.63이다.
게다가 descriptor를 **스케일 정규화 이후에** 계산하므로 "부피 300 cm³짜리" 같은 절대량 조건은
원리적으로 표현 불가능하다. 아키텍처 문제가 아니라 **라벨링 문제**다.

**결론 3 — 자동 메슁은 gmsh로 된다. 실측으로 11/11, 형상당 4.1초.**
Marching Cubes STL을 **그대로** gmsh에 넣는 게 정답이다. "먼저 decimate 해서 면 수를 줄인다"는
직관적 전처리는 **오히려 gmsh를 깨뜨린다**. 2차 요소(DeepJEB FEA가 쓰는 tet10)는 옵션 조합을
정확히 맞추지 않으면 **negative Jacobian 106개**가 나온다. 정확한 레시피는 3장에 있다.

---

## 1. 현재 상태 — 실측 진단

측정 대상: `output/geometry_generation/ex1/{sdfflow_vae.pth, sdfflow_fm.pth}`
(2026-07-22 학습, VAE 500 epoch / FM 300 epoch, latent 1×256, MLP decoder, `config_train.txt`)
데이터: `dataset/deepjeb.h5`, 2138 shapes, split seed 42 → train 1710 / val 214 / test 214.
길이 단위는 전부 **정규화 좌표**(형상 최대변 = 1.8)이다.

### 1.1 재구성 품질 — held-out에서 무너진다

GT 표면점 → 재구성 표면까지의 **정확한 점-삼각형 거리** (trimesh `closest_point`, MC res 128, 각 8샘플):

| split | mean | p95 | max |
|---|---|---|---|
| TRAIN | 0.0013 | 0.0038 | 0.0225 |
| **VAL** | **0.0053** | **0.0225** | **0.0697** |
| 비율 | **4.1배** | **5.9배** | 3.1배 |

부품 최대치수 대비로 환산하면 held-out에서 mean 0.29%, p95 **1.25%**, max **3.9%**.
200 mm 브래킷이면 평균 0.6 mm, 표면의 5%가 2.5 mm 이상, 최악 7.8 mm 어긋난다.
**필렛 반경이 틀리는 수준이라 이대로는 stress 해석용이 아니다.**

SDF query 부호 정확도도 같은 이야기를 한다: **train 97.9% vs val 92.4%**
(held-out에서는 쿼리점 13개 중 1개가 안/밖을 틀린다).

Marching Cubes 자체는 문제없다 — 검사한 24개 형상 전부 valid + watertight.

### 1.2 결정적 실험: 인코더가 병목이라는 증거

Decoder를 **완전히 얼린 채**, held-out 형상의 latent만 그 형상의 SDF 라벨에 대해
Adam 600 step으로 최적화했다 (DeepSDF auto-decoder 방식). latent 크기는 그대로 1×256.

| held-out shape | encoder mean | opt mean | encoder p95 | opt p95 |
|---|---|---|---|---|
| 446 | 0.0040 | 0.0015 | 0.0137 | 0.0048 |
| 1308 | 0.0102 | 0.0028 | 0.0592 | 0.0100 |
| 781 | 0.0044 | 0.0024 | 0.0172 | 0.0084 |
| 1750 | 0.0057 | 0.0027 | 0.0204 | 0.0088 |
| 375 | 0.0064 | 0.0036 | 0.0269 | 0.0105 |
| 1123 | 0.0047 | 0.0022 | 0.0168 | 0.0069 |
| **평균** | **0.0059** | **0.0025** | **0.0257** | **0.0082** |

**mean 2.4배 / p95 3.1배 개선**. 최적화된 latent의 held-out 성능(0.0025 / 0.0082)은
**인코더의 train 성능(0.0013 / 0.0038)에 근접**한다.

해석:

- 1×256 global latent + MLP decoder의 표현력은 **아직 한계가 아니다.** 남은 여유는 약 2배.
- 인코더가 미학습 형상에 대해 좋은 latent를 못 만든다. **격차의 대부분(2.4배)이 인코더 쪽**이다.
- 주의: 이 최적화는 그 형상의 SDF 라벨을 쓴다(인코더는 못 본다). 따라서 이는 **latent 공간의
  표현력 상한**을 보여주는 oracle이지, 인코더를 그만큼 만들 수 있다는 보장은 아니다.
  그래도 "latent가 좁아서 못 한다"는 가설은 이 실험으로 **기각**된다.

**부수 효과 하나가 더 중요하다.** `train_fm.py`는 인코더의 `mu`를 캐시해서 FM 학습 데이터로 쓴다.
인코더가 부정확하면 **FM은 부정확한 latent 분포를 학습한다.** 즉 인코더 품질은
재구성뿐 아니라 **생성 품질의 상한**이기도 하다.

### 1.3 조건 벡터가 거의 비어 있다

`dataset/deepjeb.h5`의 2138개 `cond` 전수 통계:

| descriptor | mean | std | **CV** | min | max | 상태 |
|---|---|---|---|---|---|---|
| `bbox_x` | 1.0658 | 0.00475 | **0.45%** | 1.0378 | 1.0889 | 사실상 상수 |
| `bbox_y` | 1.8000 | **0.00000** | **0%** | 1.8 | 1.8 | 완전 상수(정규화 산물) |
| `bbox_z` | 0.6369 | 0.01530 | 2.40% | 0.6202 | 1.3030 | 극단적 heavy-tail |
| `volume` | 0.2578 | 0.07629 | 29.6% | 0.1198 | 0.4953 | 유효 |
| `area` | 4.4497 | 0.66362 | 14.9% | 2.9434 | 7.1961 | 유효 |

- `volume`↔`area` 상관 **0.63**. z-score된 4개 조건의 특이값은 `[1, 0.706, 0.553, 0.422]`.
  **실질 자유도 약 1.5개.** 현재 모델은 "볼륨 노브가 달린 무조건부 생성기"에 가깝다.
- `bbox_y`가 정확히 1.8인 이유: `normalize_mesh()`가 **최대 변을 1.8로 스케일**하고,
  `mesh_descriptors()`는 **그 정규화된 메시**에서 계산된다. 브래킷은 항상 y가 최장축이다.
- 따라서 **5개 descriptor가 전부 scale-free**다. 절대 치수/절대 부피/질량 조건은 표현 불가.
  이건 `min_condition_std`(1e-5) 게이트를 **통과한다** — 그래서 안 잡혔다.
  (메모리의 `ex9 cond_var degeneracy`와 같은 계열의 함정이다. 절대 std가 아니라 **상대 변동(CV)**을
  봐야 한다.)

### 1.4 조건 정확도와 CFG의 역효과

held-out 6개 형상의 조건값을 target으로 주고 각 6개씩 생성(총 36개), 디코딩된 메시에서
descriptor를 재측정해 요청값과 비교한 **중앙값 상대오차**:

| cfg_scale | valid+watertight | bbox_x | bbox_z | **volume** | **area** |
|---|---|---|---|---|---|
| **1.0** | 35/36 | 0.2% | 0.2% | **8.5%** | **6.1%** |
| **3.0** | 36/36 | 0.4% | 0.7% | **21.2%** | **11.2%** |

두 가지를 말해준다:

1. **bbox 오차 0.2%는 통제의 증거가 아니다.** 데이터셋에서 bbox가 거의 안 변하니 그냥 맞는 것이다.
2. **CFG를 올리면 조건 정확도가 2.5배 나빠진다.** CLAUDE.md의 "cfg_scale 1.0이 보수적 설정"이라는
   서술은 맞고, 이제 숫자가 붙었다. CFG는 조건 준수 도구가 아니라 **mode-seeking 도구**다.
   조건을 정확히 맞추려면 CFG가 아니라 **4.6절의 guidance**가 필요하다.

### 1.5 데이터셋이 빌더 기본값의 1/8 밀도로 만들어졌다

`build_dataset.py` 기본값 vs `deepjeb.h5` 실제:

| 항목 | 빌더 기본값 | deepjeb.h5 실제 | 비율 |
|---|---|---|---|
| `num_surface` | 16384 | 8192 | 1/2 |
| `num_near` | 65536 | 8192 | **1/8** |
| `num_uniform` | 16384 | 2048 | **1/8** |
| SDF 감독점 합계 | 81920 | **10240** | **1/8** |

학습은 여기서 `num_query_points 4096`을 뽑아 쓴다. 참고로 DeepSDF 계열은 형상당 수십만 점을 쓴다.
**형상당 10,240점은 held-out 일반화에 명백히 부족**하고, 1.1/1.2의 인코더 격차와 직결된다.
`--sharp_edge_fraction`(Dora) 적용도 어차피 데이터셋 재빌드가 필요하므로 같이 처리하면 된다.

> 주의: 원본 DeepJEB STL이 로컬에 없다(`dataset/deepjeb.h5`만 존재). 재빌드하려면
> <https://www.narnia.ai/dataset> 에서 다시 받아야 한다 — 그리고 이건 1.6과 4.2의 기회다.

### 1.6 지금 안 쓰고 있는 것: DeepJEB의 FEA 라벨

DeepJEB(arXiv 2406.09047)는 2138개 설계에 대해 **4개 하중 케이스의 구조해석 결과 + 2개 고유진동수**,
형상당 평균 209,000 절점값의 필드 데이터를 **2차 사면체 요소**로 제공한다.
현재 `deepjeb.h5`에는 이게 **하나도 안 들어 있다** — 기하 descriptor 5개뿐이다.

이걸 넣는 순간 조건 벡터에 **의미 있게 변동하는 물리 조건 7개 이상**(케이스별 최대 von Mises,
최대 변위, 1·2차 고유진동수, 질량)이 생긴다. PhysGen이 하는 일이 정확히 이것이다.
**재다운로드 한 번으로 4장의 Level 2가 통째로 열린다.**

---

## 2. 개선 우선순위 (실측 반영)

### Tier 0 — 데이터 (가장 싸고 가장 큼)

| # | 작업 | 근거 | 예상 효과 |
|---|---|---|---|
| 0.1 | **DeepJEB 재다운로드 → 빌더 기본 밀도로 재빌드** (`--num_near 65536 --num_uniform 16384 --num_surface 16384`) | §1.5 (1/8 밀도) | 인코더 일반화 격차의 상당 부분 |
| 0.2 | 같은 재빌드에 `--sharp_edge_fraction 0.3` (Dora) | 7월 문서 #2, 미적용 상태 | 모서리/필렛 복원 |
| 0.3 | **FEA 라벨을 `cond`에 추가** (4 케이스 최대응력, 최대변위, 2 고유진동수, 질량) | §1.6 | 조건 자유도 1.5 → 8 이상 |
| 0.4 | **절대 스케일을 조건으로 저장** (`normalize_mesh`가 반환하는 `scale`) | §1.3 | 절대 치수 조건 가능해짐 |
| 0.5 | 증강: 미러링(브래킷은 대칭 계열), 소량 이방성 스케일 + SDF 동시 변환 | 1710개 학습 형상 | 인코더 과적합 완화 |

> 0.4는 한 줄짜리다. `build_dataset.py`에서 `mesh_descriptors(mesh)` 호출 시 정규화 전의
> `extent`/`scale`을 `cond`에 함께 실으면 된다. `COND_NAMES`와 spec `known_keys`도 같이 고쳐야 한다.

### Tier 1 — 인코더 (§1.2가 지목한 진짜 병목)

| # | 작업 | 상태 |
|---|---|---|
| 1.1 | 인코더 4블록/512 + latent self-attention | **이미 구현됨**, `config_train_v2.txt`에 있으나 **한 번도 학습 안 함** |
| 1.2 | `num_encoder_points` 4096 → 8192~16384 | config 한 줄 |
| 1.3 | **FPS query token** (학습 파라미터 query → 점군에서 FPS 샘플) | `FOUNDATION_MODEL_PLAN.md` M1. 토큰이 형상에 고정되어 일반화가 좋아짐 |
| 1.4 | 인코더 입력 점 수 랜덤화(dropout) | 강건성 |
| 1.5 | **추론 시 latent refinement** (reconstruct 모드) | §1.2가 2.4배 이득을 이미 증명. 약 600 step, 수 초 |
| 1.6 | **FM 학습용 latent를 인코더 mu 대신 최적화된 latent로** | §1.2 주석. DeepSDF식 Gaussian prior 정규화 필요 — 실험적 |

### Tier 2 — 조건·생성

4장 전체가 여기다. 요약: **descriptor 확장(4.1) → 물리 조건(4.2) → sampling guidance(4.6)** 순서.

### Tier 3 — 그 다음에야 아키텍처

| # | 작업 | 왜 나중인가 |
|---|---|---|
| 3.1 | VecSet(16~64 토큰) + DiT | **이미 구현됨**(`config_train_v2.txt`). §1.2 기준 남은 여유는 2배 — Tier 0/1을 먼저 해야 그 2배가 보인다 |
| 3.2 | Hybrid loss (surface+normal+eikonal) | **이미 구현됨**, 미학습 |
| 3.3 | logit-normal timestep | **이미 구현됨**, 미학습 |
| 3.4 | 구조적/희소 latent (TRELLIS SLat, VoxSet) | 2138개로는 과적합. Tier 0의 데이터 확대 이후 |

> **가장 저렴한 다음 한 걸음:** `config_train_v2.txt`는 이미 3.1~3.3을 전부 켜 놓았는데
> **아직 한 번도 돌린 적이 없다**(`output/geometry_generation/ex2` 없음). Tier 0의 재빌드 데이터로
> ex2를 돌려 ex1과 A/B 하는 게 첫 작업이어야 한다. 그때 §1.1/§1.2 지표를 그대로 재측정하면
> "VecSet이 실제로 얼마나 기여하는가"에 답이 나온다.

---

## 3. 자동 메슁 — gmsh로 된다 (실측)

질문: "SDFFlow가 뱉은 STL을 gmsh 같은 걸로 자동 메슁할 수 있나?"
답: **된다.** 다만 직관과 반대되는 함정이 세 개 있다.

### 3.1 실측 결과 — retry ladder 오토메셔

`classifySurfaces` 각도를 40°→60°→30°→50°→70° 순으로 재시도하는 래더:

| STL | angle | tets | minSICN | p1 | bad(<0.05) |
|---|---|---|---|---|---|
| epoch00000_sample0 | 40 | 19592 | 0.153 | 0.528 | 0 |
| epoch00000_sample1 | 40 | 14115 | 0.000 | 0.479 | 15 |
| epoch00100_sample0 | **60** | 25471 | 0.145 | 0.505 | 0 |
| epoch00100_sample1 | 40 | 22198 | 0.036 | 0.499 | 7 |
| epoch00200_sample0 | 40 | 24468 | 0.072 | 0.489 | 0 |
| epoch00200_sample1 | 40 | 22259 | 0.101 | 0.513 | 0 |
| epoch00299_sample0 | 40 | 19334 | 0.342 | 0.533 | 0 |
| epoch00299_sample1 | 40 | 19081 | 0.062 | 0.515 | 0 |
| sample_0_000 | 40 | 19773 | 0.062 | 0.498 | 0 |
| sample_0_000_001_alpha0p5 | 40 | 26347 | 0.189 | 0.493 | 0 |
| sample_0_001 | 40 | 28860 | 0.099 | 0.468 | 0 |

**성공 11/11 (100%), 총 45초 → 형상당 4.1초.** 고정 40°만 쓰면 8/9였다
(1개는 `PLC Error: A segment and a facet intersect`). **각도 래더만으로 100%가 된다.**

입력은 MC res 96 STL (F 약 21k~30k, genus 6~9, 전부 watertight).

### 3.2 함정 1 — decimate 하면 **깨진다**

"면 27k는 너무 많으니 먼저 20k로 줄이자"는 자연스러운 전처리가 gmsh를 죽인다:

| 전처리 | 결과 |
|---|---|
| 없음 (raw MC STL) | 성공. 40 patches, 20181 tets, minSICN 0.017 / p1 0.350 / med 0.789, 2.4초 |
| quadric decimation → 20k면 (volume 오차 0.01%) | 실패: `Wrong topology of boundary mesh for parametrization` |
| decimation + Taubin smoothing | 실패: 동일 |

`classifySurfaces` + `createGeometry`는 각 패치에 **재매개변수화(reparametrization)**를 붙인다.
Decimation이 패치 경계 근처 토폴로지를 깨서 매개변수화가 불가능해진다.
**MC 출력을 그대로 넣어라.** gmsh가 어차피 표면을 다시 메싱한다.

### 3.3 함정 2 — `MeshSizeFromCurvature`는 **행이 걸린다**

`Mesh.MeshSizeFromCurvature = 12`로 놓으면 작은 형상 하나에서 **4분 넘게 안 끝나 강제 종료**했다.
MC 표면은 계단(staircase) 아티팩트 때문에 국소 곡률이 가짜로 매우 크다 → 사이즈 필드가 붕괴한다.
**MC 출력에는 curvature-adaptive sizing을 쓰지 마라.** `Mesh.MeshSizeFromCurvature = 0` 고정.
적응 메싱이 필요하면 곡률이 아니라 §4.3의 **SDF 기반 국소 두께 필드**로 sizing 하는 게 맞다.

### 3.4 함정 3 — 2차 요소(tet10)는 옵션 조합이 정확해야 한다

DeepJEB의 FEA가 **2차 사면체**를 쓰므로 이건 선택이 아니다. 같은 형상, 같은 19334 요소:

| 요청 방식 | minSICN | p1 | **negative Jacobian** |
|---|---|---|---|
| `generate(3)` 후 `setOrder(2)`, HighOrderOptimize=0 | −0.864 | 0.177 | **106** |
| `generate(3)` 후 `setOrder(2)`, HighOrderOptimize=2 | −0.864 | 0.177 | **106** |
| `Mesh.ElementOrder=2` 사전 설정, HighOrderOptimize=0 | −0.864 | 0.177 | **106** |
| **`Mesh.ElementOrder=2` + `HighOrderOptimize=2`** | **0.025** | **0.397** | **0** |
| `Mesh.ElementOrder=2` + `HighOrderOptimize=4` | −0.864 | 0.177 | **106** |

**유일하게 동작하는 조합은 `Mesh.ElementOrder=2`를 `generate(3)` 전에 설정 + `HighOrderOptimize=2`.**
사후 `setOrder(2)`는 곡면 경계에서 반전 요소를 만들고 어떤 solver도 이 메시를 받지 않는다.

### 3.5 그 밖에 측정된 것

| 설정 | 결과 |
|---|---|
| `MeshSizeMax` 0.10 / 0.05 / 0.025 | 6.6k / 19.3k / 95.4k tets, 1.9s / 3.4s / 12.3s — 예측 가능하게 선형 |
| `Mesh.Optimize`+`OptimizeNetgen` **끔** | minSICN 0.342 → **0.006**, bad 0 → 46. **필수다** |
| `Algorithm3D=10` (HXT) | 약 2배 빠름(1.8s), 품질 낮음(p1 0.398 vs 0.533), 실패 케이스에서 더 취약 |
| `classifySurfaces` angle 20° | 실패(topology). 각도를 낮추면 패치가 잘게 쪼개져 오히려 나빠진다 |

### 3.6 권장 파이프라인

```text
FM latent ──► SDF grid (res 128~192)
                │
                ├─[A] Marching Cubes ─► STL ─► gmsh classify+reparam+tet   ◄── 검증됨, 4.1s/shape
                │                                (angle ladder, Optimize on,
                │                                 curvature off, ElementOrder=2
                │                                 + HighOrderOptimize=2)
                │
                ├─[B] MC STL ─► fTetWild ─► tet                            ◄── A가 전부 실패할 때 fallback
                │                                (triangle soup도 받음, FEM용
                │                                 valid float mesh 보장)
                │
                └─[C] SDF grid ─► mmg3d -ls ─► body-fitted tet             ◄── MC를 아예 건너뜀
                                                 (level-set 직접 이산화 + 이방성 적응)
```

- **[A]가 기본.** 이미 `dataset/geometry_ingest/readers.py::read_gmsh`가 gmsh를 쓰고 있고,
  `clean.py`에 watertight 복구 경로가 있으며, `wheels/`에 오프라인 gmsh 휠까지 커밋되어 있다.
  **새 의존성이 필요 없다.** 다만 `read_gmsh`는 파일을 `gmsh.open()`으로 열 뿐
  `classifySurfaces`/`createGeometry`를 하지 않으므로 **STL→volume 경로에는 재매개변수화 단계가
  빠져 있다.** 이 부분을 추가하는 게 실제 작업이다.
- **[B] fTetWild**는 비-watertight/자기교차 입력에도 항상 유효한 부동소수 tet 메시를 보장한다.
  §3.1의 실패 케이스(PLC error)류에 대한 안전망.
- **[C] mmg3d `-ls`**는 배경 tet 메시 + 절점 level-set 값에서 **body-fitted** 메시를 직접 만든다.
  MC 계단 아티팩트를 원천적으로 피하고 이방성 적응까지 된다. **중장기적으로 가장 CAE-native한 경로**다.
  (CGAL/`pygalmesh`도 implicit domain 메싱을 지원하지만 `eval()`이 점 단위 호출이라
  신경망 SDF에는 배칭이 안 돼 비현실적이다. 대신 `generate_from_array` 경로는 가능하다.)

### 3.7 CAE 품질 게이트 (생성물 필터의 일부로)

DeepJEB 자체도 생성 브래킷의 약 43%를 FEM 전에 버렸다. 게이트를 **명시적 stage**로 만들어야 한다:

1. watertight / manifold / 단일 컴포넌트 (`mesh_report`가 이미 대부분 계산)
2. genus가 학습 분포 범위 내인가 (측정된 생성물은 genus 6~9)
3. 최소 벽두께 ≥ 임계값 (SDF에서 직접: 내부 점의 `|SDF|` 분포)
4. **tet 메싱 성공 + `minSICN` p1 ≥ 0.2 + negative Jacobian 0**
5. 실패는 **버리지 말고 실패 클래스와 함께 저장** — feasibility 모델의 학습 데이터가 된다

---

## 4. Semantic Conditional 생성 — "내 맘대로" 만드는 방법

### 4.0 지금 왜 안 되는가

§1.3이 답이다. 조건 자유도가 1.5개이고 전부 scale-free이며, 그중 두 개는 상수에 가깝다.
**모델을 안 바꾸고 라벨만 바꿔도 통제력은 크게 올라간다.** 아래를 통제의 "레벨"로 정리한다.

### 4.1 Level 1 — 풍부한 기하 descriptor (오늘, 데이터 재수집 없이)

`deepjeb.h5`에 **이미 저장된 표면점/법선/SDF만으로** 계산 가능한 후보 14개를 실제로 뽑아 봤다
(500개 형상 샘플):

| descriptor | CV | 현재 4조건으로 설명되는 정도(R²) | **새 정보** |
|---|---|---|---|
| `planar_frac` (법선이 주축에 정렬된 표면 비율) | **40.9%** | 0.39 | 61% |
| `sym_2` (2주평면 대칭성) | 26.9% | 0.89 | 11% |
| `max_inradius` (**최소 벽두께 / 내접반경**) | 26.5% | 0.81 | 19% |
| `sec_std` (주축 단면적 프로파일 변동) | 19.5% | 0.05 | **95%** |
| `sym_3` | 17.8% | 0.31 | 69% |
| `pca_l3_l1` (편평도) | 17.0% | 0.43 | 57% |
| `sec_min` (최소 단면 — 하중경로 병목) | 13.7% | 0.32 | 68% |
| `pca_l2_l1` (세장비) | 12.0% | 0.30 | 70% |
| `n_axis3` | 7.6% | 0.25 | 75% |
| `sec_max` | 9.0% | 0.02 | **98%** |
| `sym_1` | 8.5% | 0.02 | **98%** |
| `convexity` | 7.6% | 0.69 | 31% |

이 14개의 PCA: **95% 분산에 9차원**이 필요하다. 즉 **현재 1.5 → 9 자유도**로 확장 가능하고,
`sec_std`/`sec_max`/`sym_1`은 현재 조건이 **거의 설명하지 못한다**(새 정보 95~98%).

엔지니어링 의미도 있다:
`max_inradius` = 최소 벽두께(제조성), `sec_min` = 하중경로 최소 단면(강성),
`sym_*` = 대칭 요구, `planar_frac` = 평면 가공면 비율, `sec_std` = 재료 분포 프로파일.

**작업량:** `sdf_sampling.py::mesh_descriptors` + `COND_NAMES` 확장, 데이터셋 재빌드,
`specs/sdfflow.py`의 `known_keys`/validator 갱신. 모델 코드는 **한 줄도 안 바뀐다**
(`cond_dim`이 config에서 유도된다).

**추가로 반드시 넣을 것:** §1.3의 절대 스케일(정규화 전 최대 변, 또는 3축 실치수).
이게 없으면 "이 크기로 만들어줘"가 영원히 불가능하다.

### 4.2 Level 2 — 물리·성능 조건 (재다운로드 한 번)

§1.6. DeepJEB의 4 하중케이스 + 2 고유진동수를 `cond`에 실으면
**"최대응력 250 MPa 이하 & 1차 고유진동수 800 Hz 이상인 브래킷을 생성"**이 표현 가능해진다.
이것이 Siemens Simcenter PhysicsAI Generate가 파는 것이고(설계 파라미터 + KPI 조건부 diffusion),
PhysGen(CVPR 2026)이 하는 것이다(공유 latent에서 SDF + 압력 + 항력 디코더).

이 저장소는 여기서 유리하다: **GINO / Transolver / MeshGraphNets surrogate가 이미 있다.**
생성 → 메싱(3장) → surrogate 채점 → 랭킹 루프가 벤더들이 파는 "generate + predict"다.
게다가 §3의 메셔가 그 루프의 빠진 연결고리였다.

### 4.3 Level 3 — 공간적/구조적 조건 (엔지니어링에서 가장 유용)

"내 맘대로"의 진짜 의미가 "여기에 볼트홀, 저 영역은 비워, 이쪽에 하중"이라면 스칼라 조건으로는 안 된다.
필요한 것은 **공간 필드 조건**:

- **keep-in / keep-out 볼륨**과 **인터페이스 패치**(장착면, 하중점)를 voxel 또는 점군으로 인코딩
- 그 토큰들을 velocity net에 **cross-attention**으로 주입 (현재 DiT는 AdaLN만 있고
  cross-attention이 없다 — 추가해야 하는 유일한 아키텍처 변경)
- 추론 시에는 **SDF inpainting**처럼: 구속 영역의 SDF를 고정하고 나머지를 모델이 채우게 한다
  (flow matching의 mask-guided sampling)

Sizing에도 재사용된다: §3.3에서 곡률 대신 쓸 국소 두께 sizing 필드가 같은 SDF에서 나온다.

이 레벨은 **위상최적화(topology optimization)의 생성형 대응물**이고, 3D TO diffusion 문헌이
거의 그대로 적용된다(설계영역 + 하중/구속 + 부피분율 조건).

### 4.4 Level 4 — 파트 수준 제어, 파트 라벨 없이

SPAGHETTI(SIGGRAPH 2022) / SALAD(ICCV 2023) 계열이 답이다:
형상을 **N개의 Gaussian(외재: 위치·스케일·회전) + 각 Gaussian의 내재 특징 벡터**로 분해하고
그 합집합에서 SDF를 디코딩한다. **파트 감독 없이 자기지도로 분해가 학습**되고,
편집은 "Gaussian을 옮기고/키우고/섞는" 것이 된다 — 재학습 없이.

SDFFlow와의 궁합이 좋다: VecSet 토큰 각각에 Gaussian(위치·공분산)을 붙이면
**공간에 고정된 구조적 latent**가 되고, 이건 TRELLIS SLat / VoxSet가 가는 방향과 같다.
즉 **3.1(VecSet)과 4.4는 같은 작업으로 합칠 수 있다.**

### 4.5 Level 5 — 텍스트/멀티모달

Text2CAD와 NURBGen의 라벨링 파이프라인이 검증된 레시피다:
멀티뷰 렌더 → VLM으로 형상 서술 생성 → LLM으로 다단계 텍스트 지시문 생성.
NURBGen은 캡션에 **관통홀 개수, 전체 치수, 표면적, 부피** 같은 기하 메타데이터를 명시적으로 넣는다.

여기서는 **4.1의 descriptor에서 템플릿 문장을 만들고 LLM으로 패러프레이즈**하면
2138개 형상에 캡션을 붙이는 데 외부 라벨이 전혀 필요 없다. 그 다음 frozen 문장 인코더 →
cross-attention(4.3에서 만든 것과 같은 경로). **단, 2138개로 자유 텍스트 일반화는 기대하지 마라.**
템플릿 기반 통제 언어("두꺼운 웹, 대칭, 홀 4개")까지가 현실적이다.

CAD 프로그램 생성(Text2CAD, CAD-Coder, TOOLCAD)은 **편집 가능한 STEP**이 목표일 때의
별도 분기다. 7월 문서 결론대로 SDF 주경로에서 분리해 유지.

### 4.6 조건을 **정확히** 맞추기 — rejection sampling에서 guidance로

현재 `sample.py`는 `candidate_multiplier`배로 생성 → 전부 128³ 디코딩 + MC → descriptor 재측정 →
정렬해서 상위 K개를 고른다. **K개 중 최선보다 나아질 수 없고**, 후보마다 전체 그리드를 디코딩한다.

문헌이 가리키는 대안은 두 가지다:

1. **Sampler-guidance (soft).** Euler ODE 각 step에서 목표 descriptor에 대한 gradient를
   latent에 더한다. 저해상도 SDF에서 미분 가능한 근사 descriptor를 쓰면 된다:
   부피 = `Σ sigmoid(−sdf/τ)`, 면적 = `Σ ‖∇ sigmoid‖`. Dflow-SUR가 flow ODE 전체를 통과하는
   미분을 하고, HeatGen이 multiphysics 목표로 유도한다. 구현 위치는
   `velocity_net.sample_latents`의 루프 하나뿐이다.
2. **Lagrangian Dual Flows (hard).** arXiv 2607.04513(2026-07)은 생성 ODE에 **dual 변수 λ와 slack**을
   함께 적분해 비선형 등식/부등식 제약을 만족시킨다. step마다 projection 최적화가 없고
   autodiff VJP 한 번이면 되며, `t→1`에서 제약 위반이 대수적 속도로 0에 수렴한다는 보장이 있다.
   projection 기반 대비 **5~10배 빠르고** 위반은 1e-3 수준.
   `max_condition_z` 게이트와 candidate ranking을 **원리적으로 대체**할 수 있는 것이 이것이다.

§1.4에서 CFG를 올려도 조건이 나빠진다는 걸 측정했으니, **조건 정확도의 해법은 CFG가 아니라
이 guidance 경로**라는 게 데이터로 확인된 셈이다.

### 4.7 외삽 — 학습 범위 밖 조건

`config_sample_extrapolation.txt`는 `max_condition_z` + `error` 정책으로 **거부**한다. 안전하지만 무능하다.
대안 두 가지:

- **LAMP** (arXiv 2510.22491, 2026-06): 사례별로 SDF decoder를 공유 초기화에서 overfit시켜
  **정렬된 weight space**를 만들고, 목표 파라미터를 만족하는 affine 혼합계수를 최소자승으로 푼다.
  단일 파라미터 **±100% 외삽에서 R²=0.90**(baseline 0.14), 4파라미터 동시 50% 외삽에서 R²=0.87
  (baseline은 붕괴). **샘플 50개로도 통제 가능**하고, weight-space 국소선형성을 벗어났는지 감지하는
  안전 지표(ROC AUC 0.989)까지 있다. 풀이는 10 ms 미만.
  **한계: 모든 사례가 동일 위상(topology)이라고 가정한다.** DeepJEB 브래킷은 genus가 6~9로 변하므로
  전면 대체는 안 되고, **위상이 고정된 하위 계열**이나 SDFFlow가 만든 후보 주변의 국소 탐색기로 유용하다.
- **Diagonal Flow Matching + abstention** (2603.15925): 설계→성능 예측과 성능→설계 생성을
  하나의 가역 모델로 하고, 목표가 실현 불가능하면 **거부(abstain)**한다.
  `max_condition_z`의 원리적 대체재.

### 4.8 방법별 비교

| 레벨 | 통제 대상 | 새 데이터 필요? | 모델 변경 | 예상 효과 |
|---|---|---|---|---|
| **1. 확장 descriptor** | 두께·대칭·단면·편평도·**절대치수** | 없음(재빌드만) | 없음 | 자유도 1.5 → 9 |
| **2. 물리 조건** | 응력·변위·고유진동수·질량 | DeepJEB 재다운로드 | 없음 | "성능 스펙으로 설계" |
| **3. 공간 조건** | keep-in/out, 인터페이스, 하중점 | voxel 라벨 생성 | **cross-attention 추가** | 진짜 "여기에 이렇게" |
| **4. 파트 수준** | 파트 이동/스케일/혼합 | 없음(자기지도) | Gaussian-anchored latent | 대화형 편집 |
| **5. 텍스트** | 자연어 | VLM/LLM 캡션 자동생성 | text encoder + cross-attn | 데모용, 2138개론 제한적 |
| **6. Guidance** | 위 전부를 **정확히** | 없음 | sampler 루프만 | 8.5% → 1% 미만 목표 |
| **7. 외삽(LAMP)** | 범위 밖 파라미터 | 없음 | 별도 경로 | ±100% 외삽 |

---

## 5. 통합 로드맵

**Phase 1 (데이터 + 검증) — 이번 주**

1. `config_train_v2.txt`를 **현재 데이터로 그대로 학습**해 ex2 생성 → ex1과 §1.1/§1.2/§1.4 지표 A/B.
   (7월에 만들어 놓고 안 돌린 상태다. VecSet/DiT/hybrid loss의 실제 기여도를 여기서 확정한다.)
2. DeepJEB 재다운로드 → **기본 밀도 + `--sharp_edge_fraction 0.3` + 절대 스케일 + 확장 descriptor(4.1)
   + FEA 라벨(4.2)**로 한 번에 재빌드.
3. `--audit-configs`, spec `known_keys` 갱신, `pytest -q tests/test_sdfflow_pipeline.py` 통과 확인.

**Phase 2 (인코더 + 메셔)**

4. Tier 1(인코더 점 수, FPS query token, 증강)을 적용해 §1.2의 2.4배 격차를 얼마나 회수했는지 재측정.
5. **`geometry_ingest`에 `sdf_to_cae_mesh` 경로 추가** — §3.6 [A]를 retry ladder,
   `Optimize`/`OptimizeNetgen` on, `MeshSizeFromCurvature` off, `ElementOrder=2` + `HighOrderOptimize=2`로.
   §3.7 품질 게이트를 `mesh_report`에 붙인다.

**Phase 3 (조건 정확도 + 물리 루프)**

6. Sampler guidance(4.6-1) → 조건 오차 목표 1% 미만. LDF(4.6-2)는 그 다음.
7. 생성 → 메싱 → **기존 GINO/Transolver surrogate 채점** 루프 연결. 이게 벤더들이 파는 제품 형태다.

**Phase 4 (연구급)**

8. 공간 조건(4.3) cross-attention, 파트 수준 Gaussian latent(4.4), LAMP 국소 탐색기(4.7).

---

## 6. 선행 문서와 달라지는 점 (명시)

| 항목 | 7월 문서 | 이 문서 (실측 근거) |
|---|---|---|
| 최우선 아키텍처 레버 | "global token → VecSet이 가장 큰 레버" | **인코더 일반화가 먼저.** latent 최적화 실험에서 mean 2.4배 / p95 3.1배 회수 → latent 표현력은 아직 한계가 아님 (§1.2) |
| 조건부 생성의 상태 | "descriptor 조건은 이미 있고, 물리 조건 추가가 다음" | **현재 조건은 실질 자유도 1.5개이고 전부 scale-free.** 물리 조건 이전에 기하 descriptor부터 비어 있다 (§1.3, §4.1) |
| CFG | "강도/다양성 trade-off" | 맞음. 추가로 **조건 정확도를 2.5배 악화**시킨다는 수치 (§1.4) |
| 자동 메슁 | 다루지 않음 | **11/11, 4.1s/shape로 검증.** 함정 3개 문서화 (§3) |

---

## 7. 출처

**메싱**

- Gmsh 4.15.2 — <https://gmsh.info/> , 튜토리얼 t13 (CAD 없이 재분류+재매개변수화 remeshing)
- fTetWild — <https://arxiv.org/abs/1908.03581> , <https://github.com/wildmeshing/fTetWild>
- mmg3d 암시적 도메인/level-set 메싱 — <https://www.mankier.com/1/mmg3d> ,
  Dapogny et al., *J. Comput. Phys.* (2014) <https://www.sciencedirect.com/science/article/abs/pii/S0021999114000266>
- CGAL 3D Mesh Generation — <https://doc.cgal.org/latest/Mesh_3/index.html> , pygalmesh — <https://github.com/meshpro/pygalmesh>
- 재매개변수화 기반 자동 표면 메싱 — <https://arxiv.org/pdf/2001.02542>

**조건부/의미론적 생성**

- Constrained Flow Matching via Lagrangian Dual Flows — <https://arxiv.org/html/2607.04513>
- LAMP (파라미터 통제 + 외삽) — <https://arxiv.org/html/2510.22491v3>
- PhysGen (CVPR 2026) — <https://arxiv.org/pdf/2512.00422>
- SALAD — <https://arxiv.org/abs/2303.12236> · SPAGHETTI — <https://dl.acm.org/doi/abs/10.1145/3528223.3530084>
- UniPart — <https://arxiv.org/html/2512.09435> · OmniPart — <https://omnipart.github.io/>
- Text2CAD (NeurIPS 2024) — <https://arxiv.org/pdf/2409.17106> · Text2CAD-Bench — <https://arxiv.org/abs/2605.18430>
- NURBGen (VLM 캡션 파이프라인) — <https://arxiv.org/html/2511.06194v2>
- 생성형 위상최적화 (human-guided diffusion) — <https://academic.oup.com/jcde/article/13/7/55/8704128>
- Latent Space Diffusion for Topology Optimization — <https://arxiv.org/html/2508.05624v1>

**데이터**

- DeepJEB — <https://arxiv.org/abs/2406.09047> , 데이터 <https://www.narnia.ai/dataset>
  (2138 설계, 4 하중케이스 + 2 고유진동수, 2차 tet FEA, 형상당 평균 209k 절점값)
