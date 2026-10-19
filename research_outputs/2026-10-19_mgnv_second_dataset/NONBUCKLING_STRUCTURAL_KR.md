# MGN-V 추가 데이터셋: 좌굴을 제외한 구조 문제

검토일: 2026-10-19 (Asia/Seoul). 최신 요청은 좌굴 계열을 제외한 추가 구조 dataset이다. iFEM은 inverse FEM으로 확인됐다.

## 권고

**공간적으로 불균질한 항복강도를 가진 유공 시편의 인장–제하 탄소성 응답분포**를 가장 추천한다. 공개 데이터 활용이 우선이면 **Mechanical MNIST–Cahn–Hilliard의 이질적 hyperelastic 재료 인장**을 대안으로 둔다. **Sparse-strain inverse shape sensing**은 사용자의 관심에 맞는 세 번째 선택이다.

| 후보 | 물리적 변동 원인 | 생성할 field | 확보 상태 | 판단 |
|---|---|---|---|---|
| 유공 시편의 탄소성 인장–제하 | 숨겨진 spatial yield-strength field | u, stress, equivalent plastic strain | 공개 solver·예제 있음; conditional MC corpus 생성 필요 | 가장 추천 |
| 이질적 hyperelastic 재료 인장 | 숨겨진 미세 재료분포 | u; reaction·energy | 공개 field corpus·생성 코드 있음; 확률 조건 재정의 필요 | 빠른 공개 자산 대안 |
| Inverse FEM shape sensing | 센서가 관측하지 못하는 load/state와 측정 noise | sparse strain에 조건화한 u posterior | 공개 선행·FE 모델 있음; reference posterior 구축 필요 | 별도 inverse 기여에 적합 |
| 거친 표면의 접촉 | 숨겨진 roughness/material/friction | contact pressure·displacement | 공개 solver 있음; FE/mesh 및 same-condition 확장 필요 | 후순위 |

현재 저장소에는 이미 deterministic `ex8_elasticity`, `ex9_plasticity`, `ex5_deforming_plate`가 있다. 이들을 새로운 dataset으로 다시 세지 않는다. 제안한 탄소성 인장 문제는 기존 rigid-die forging ex9와 형상·하중·숨은 확률원인이 다르다. 이번 추천의 기준은 새 파일 수가 아니라 다른 구조 메커니즘에서 조건부 응답분포를 검증하는 것이다.

## 1. 탄소성 유공 시편의 인장–제하

직접 확인한 공개 출발점:

- [COMET-FEniCSx의 2D von Mises plasticity](https://bleyerj.github.io/comet-fenicsx/tours/nonlinear_problems/plasticity/plasticity.html): 선형 isotropic hardening, analytical return mapping, Newton equilibrium iteration과 quadrature stress/internal variables. 원래 geometry는 plane-strain 내압 중공 원통이다. 인장 시편·random field·제하 경로는 추가 작업이다.
- [dolfinx_materials의 perforated plate 예제](https://bleyerj.github.io/dolfinx_materials/demos/jax/elastoplasticity/plane_elastoplasticity.html): 유공판 mesh, 인장, von Mises+Voce hardening과 stress/plastic field·reaction을 제공하는 더 가까운 geometry 예제. 현재 코드는 균질 물성·정해진 하중이며 확률 ensemble은 아니다.
- [JAX-FEM의 plasticity 인장–제하 예제](https://deepmodeling.github.io/jax-fem/learn/plasticity/example.html): 3D perfect J2 plasticity와 history update, displacement loading–unloading, 변위 VTK 출력. 우리가 원하는 hardening과 heterogeneous yield-strength extension은 별도 검증 대상이다.

권장하는 하나의 통제된 계약:

- **조건:** nominal geometry, E/nu, positive hardening law, supports, 전체 load protocol 또는 그 parameter, yield-field law의 mean/CV/correlation length.
- **숨긴 원인:** 양수인 spatial yield-strength field 한 종류. 예를 들어 unit-variance Matérn field g를 lognormal로 변환한다. 초기에 E, hardening, 마찰, damage 등을 동시에 랜덤화하지 않는다.
- **출력:** full displacement, stress, equivalent plastic strain p, 반력과 하중경로. 제하 후 residual stress와 plastic deformation도 보존한다.
- **같은 조건 반복:** geometry·BC·load history·material-law hyperparameters를 고정하고 independent material realization만 바꾼다.

이는 한 material realization을 알면 deterministic한 FE 문제지만 nominal 정보만 주면 one-to-many다. 같은 nominal 시편의 재료 불균질성 때문에 소성영역과 하중 재분배가 달라지는 자연스러운 구조 응용이다. 여러 buckling branch나 multimodality가 있어야만 유효한 conditional generative benchmark가 되는 것은 아니다.

검증할 연구 가설은 지역 소성영역과 전체 반력/변형의 공동분포를 어떤 latent placement가 같은 예산에서 더 잘 재현하는가이다. Global=전체 반력, local=소성영역이라는 역할이 자동으로 성립하지 않는다. 같은 variance에서 correlation length를 바꾸어 필요한 표현이 달라지는지 확인하는 편이 좋다. Random field law에 global/local 두 항을 억지로 배치하여 hybrid를 유리하게 만들지 않는다.

중요한 구현·평가 조건:

1. Plane strain solver를 thin plane stress plate라고 relabel하지 않는다. 첫 구현은 plane-strain coupon 또는 검증된 plane-stress constitutive update 중 하나를 선택한다.
2. Random material realization이 대칭을 깨면 원래 quarter symmetry BC를 그대로 적용하지 않는다. 전체 geometry 또는 대칭성을 유지한 prior가 필요하다.
3. Displacement control에서는 u가 거의 affine일 수 있다. stress·p·plastic-zone distribution을 함께 보아야 한다. 원래 quadrature/cell fields를 보존하고 nodal projection이 hotspot을 평탄화하는지 확인한다.
4. Plasticity는 history-dependent다. Current load 값만 같아도 loading/unloading/peak history가 다르면 같은 condition이 아니다. 정해진 protocol부터 시작하고 history 정보를 명시한다.
5. 한 trajectory는 하나의 material realization을 유지한다. Frame별로 독립 latent를 새로 뽑는 방식으로 물리 경로를 만들지 않는다. Snapshot 모델부터 평가하면 시간적 coupling과 latent placement를 분리할 수 있다.
6. 처음에는 positive hardening, small strain, damage/fracture 없는 범위로 설정한다. Mesh/step convergence와 실제 uncertainty 크기를 확인한 뒤 범위를 넓힌다.

제안 pilot은 3–5개 nominal geometry/load condition에서 각각 32–64 draw다. 이는 ensemble 다양성·FE 수렴·비용을 확인할 계획이며, 완료 결과나 최종 학습에 충분한 표본 수라는 뜻은 아니다. 초기 평가에는 field ES/variogram, plastic-zone 면적·위치, 반력, 최대 p와 stress quantile, unload residual statistics를 포함한다.

## 2. 공개 field corpus 대안: 이질적 hyperelastic 재료 인장

[Mechanical MNIST 원 데이터](https://open.bu.edu/items/ad5fb596-9255-4c25-8789-4ac20fcc3419)는 heterogeneous neo-Hookean 재료의 인장과 full displacement·reaction·energy를 제공한다. 숫자 모양 패턴보다 물리적인 미세조직을 원하면 [Cahn–Hilliard extension 공식 코드](https://github.com/elejeune11/Mechanical-MNIST-Cahn-Hilliard)와 [공식 데이터](https://open.bu.edu/items/058a478a-49c3-48fa-ad50-b79fb886dfb1)가 더 적절한 후보다.

Cahn–Hilliard extension은 104,813개의 두 상 재료 패턴에 equibiaxial extension을 적용한다. Full displacement는 최종 d=0.5에서 제공하고 reaction·energy는 7개 하중단계에 제공한다. 미세조직 생성 코드와 FEniCS 해석 코드가 공개돼 있다. 다만 여러 pattern이 같은 CH run의 snapshot에서 나왔으므로 독립성·split grouping을 확인해야 한다.

Fine material map 전체를 입력하면 원래 문제는 deterministic이다. 확률모델 실험을 위해서는 nominal material statistics만 주거나 coarse observation에 조건화하는 법칙을 정해야 한다. 서로 다른 volume fraction·correlation length를 무분별하게 섞어 하나의 conditional ensemble이라고 부르면 안 된다. 기존 crack-path ex2와 물리현상은 다르지만 같은 Mechanical MNIST 계열이라는 한계도 있다.

별도 공개 finite-strain plasticity corpus인 [Lesueur et al. 2025 데이터](https://doi.org/10.57745/HQJMYA)도 확인했다. 공개 목록에 약 3.7 GB data archive, 약 1.7 MB code, displacement/plastic-flow/stress field와 dataloader가 있다. 같은 조건의 random material ensemble인지는 확인되지 않았으므로 ready-made stochastic dataset으로 추천하지 않는다.

## 3. Inverse FEM은 여전히 의미 있는 선택

Sparse strain에서 가능한 전체 displacement를 복원하는 목표는 p(u | strain observations, sensor layout, geometry, BC)다. 구조 응용이라는 점은 유지되지만, forward material-response distribution에서 observation-conditioned posterior로 연구 질문이 바뀐다.

[Poloni et al., MSSP 2023](https://doi.org/10.1016/j.ymssp.2022.110056)의 GP strain extrapolation과 stochastic iFEM이 직접 선행이다. [ISTAR Demo Wing 공개 FE 모델](https://research.tudelft.nl/en/datasets/istar-demo-wing-abaqus-model)도 있다. 그러나 모델 공개와 conditional posterior reference corpus 공개는 다르다.

가장 통제된 시작은 known linear FE operator, 명시한 Gaussian load prior와 sensor-noise law를 사용해 sparse strain에 대한 analytic displacement posterior를 만드는 것이다. 이 경우 mean뿐 아니라 covariance와 calibration을 정확히 평가할 수 있다. 센서가 부족하다는 사실만으로 다중해가 보장되는 것은 아니므로 observation rank와 posterior covariance를 확인해야 한다.

## 4. 접촉 문제의 우선순위

[ContactMechanics](https://contactengineering.github.io/ContactMechanics/) 같은 공개 solver는 rough-surface contact pressure와 displacement를 생성하는 출발점이다. 다만 FFT half-space 표면 grid와 현재 unstructured structural FE mesh의 차이를 고려해야 한다. 기존 MGN DeformingPlate는 이미 저장소에 있으므로 그 공개본을 새 dataset으로 세지 않는다. Friction/random roughness를 추가하는 경우 contact graph, active set, 접촉 알고리즘의 민감도까지 추가 검증해야 하므로 첫 신규 캠페인으로는 탄소성 인장보다 부담이 크다.

## 검증 범위

공개 논문·dataset listing·공식 solver/생성 코드와 현재 로컬 dataset 목록을 확인했다. 대용량 corpus 다운로드, 신규 FEA, random-material extension, 변환 또는 학습은 수행하지 않았다. 위 확률 ensemble 설계는 제안이며 구현·성능 결과와 구분한다.
