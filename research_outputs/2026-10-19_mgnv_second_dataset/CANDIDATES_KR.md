# MGN-V 추가 데이터셋 후보: 기존 좌굴과 가까운 문제

> 이전 후보 검토 기록. 후속 요청에서 좌굴 계열을 제외하고 구조 문제를 찾도록 범위가 바뀌었다. 현재 추천은 [NONBUCKLING_STRUCTURAL_KR.md](NONBUCKLING_STRUCTURAL_KR.md)를 따른다.

검토일: 2026-10-19 (Asia/Seoul).

사용자 요구: 기존 데이터 외에 하나를 추가한다. 처음 관심을 표현한 iFEM은 inverse FEM이며, 이후 현재 buckling과 비슷한 종류의 데이터도 우선 조사하도록 범위를 조정했다.

## 추천

**공개 데이터 활용이 우선이면 다공성 메타물질의 압축 좌굴, 새 확률 ensemble 구축이 가능하면 박막–연성기판의 wrinkling을 권한다.** 후자는 현재 단일 dimple 쉘과 다른 공간적 확률입력을 검증할 수 있다는 점에서 global/local/hybrid 연구질문에 더 강한 추가 사례가 될 가능성이 있다. 이는 아직 검증되지 않은 연구 가설이다.

| 후보 | 현재 쉘 좌굴과 공통점 | 추가로 시험할 내용 | 현재 확보 상태 |
|---|---|---|---|
| Porous mechanical metamaterial | 압축, 불안정성, 여러 post-buckling pattern | Ligament별 변형과 전체 pattern coherence | 공개 mesh/field trajectory, 생성 코드 |
| Bilayer wrinkling | 압축에서 불안정성으로 새로운 변형 패턴 발생 | 분포한 imperfection의 지역 위상·진폭과 장거리 상관 | 공개 FEA 예제/solver; 확률 corpus 생성 필요 |
| Shallow arch/cap snap-through | 임계점, 다중 평형과 급격한 전이 | Branch 선택·전이 하중·shape | Solver 출발점 있음; 적절한 repeated-condition corpus 미확인 |
| Sparse-strain inverse FEM | 구조 변위장과 uncertainty | 관측에 조건화한 복원 posterior | 별도 역문제 생성기가 필요; 후속 연구에 적합 |

## 1. Wallpaper porous metamaterial

1차 자료:

- [Scientific Data 논문 DOI](https://doi.org/10.1038/s41597-025-06150-x), [저자 원문](https://arxiv.org/html/2507.11195v1)
- [공개 데이터 v5](https://zenodo.org/records/18172947)
- [원본 schema README](https://zenodo.org/records/18172947/preview/README.md?include_deleted=0), [usage notebook](https://zenodo.org/records/18172947/preview/Usage_notes.ipynb?include_deleted=0)
- [MATLAB FE solver](https://github.com/FHendriks11/mechmetamat_homogenization), [Python geometry generator](https://github.com/FHendriks11/wallpaper_microstructures)

데이터는 17개 wallpaper group의 1,020개 geometry와 geometry별 12개 하중경로를 포함한다. 노드별 변형 좌표와 메시가 있으므로 scalar curve 전용 자료는 아니다. v5의 다운로드 목록은 46.0 GB ZIP이며, README에 CC BY 4.0이 명시된다. 구버전보다 접촉·수렴·time ordering·periodicity 수정이 반영된 v5를 고정하는 편이 적절하다.

원본의 12개 경로는 서로 다른 F_final에 해당한다. 반복 sample 축이 있는 MC corpus가 아니다. 형상과 하중을 고정한 뒤 branch 분포를 정의하는 작업이 남는다. 원본 2×2 RVE, quadratic triangular mesh, periodic source/image-node mapping, contact 이후 제외 규칙도 유지해야 한다. 원문 solver의 고유모드 방향 perturbation/relaxation은 물리적인 제조 불완전성의 확률법칙과 구분해야 한다.

이 후보가 제공할 실험은 두 가지로 나뉜다.

### A. 공개 branch benchmark

[Bi-FORK 원문](https://arxiv.org/html/2610.12449v1)은 같은 자료에서 symmetry-related 1/2/4 mode를 학습한다. Mode 확률을 균등하게 정의한 branch-coverage 실험은 유효한 별도 benchmark다. 그러나 그것을 제조 오차에서 유도된 실제 branch 빈도라고 해석할 수는 없다.

이번에 [공개 코드](https://github.com/ml-jku/Bi-FORK)도 확인했다. 검토한 commit은 `0c90edf95f9f2ac6d1de933ea8d1b846ba2ad8b2`다.

- [microstructures.py](https://github.com/ml-jku/Bi-FORK/blob/0c90edf95f9f2ac6d1de933ea8d1b846ba2ad8b2/src/bifurcation/datasets/microstructures.py): 네 rollout은 큰 cell 안의 subcell에서 얻으며 모든 mode를 반드시 덮지 않는다고 명시한다.
- 해당 reader는 `microfluctuation`, `canonical_subtraj_indices`, `num_unique_modes`, `after_contact` 등을 갖는 가공 HDF5를 기대한다. Raw Zenodo schema에는 별도의 independent branch 축이 없다.
- [modes.py](https://github.com/ml-jku/Bi-FORK/blob/0c90edf95f9f2ac6d1de933ea8d1b846ba2ad8b2/src/bifurcation/datasets/utils/modes.py): 발견한 mode 사이에서 균등 sampling한다.
- 확인한 tree에서 전체 raw-to-microstructures 변환기는 식별하지 못했다. 공개 reader가 있다는 사실을 가공 데이터 재현 완료로 표현하면 안 된다.

따라서 사용 전에 subcell correspondence, 중복 mode 제거 tolerance, 실제 branch coverage, invalid/pre-contact mask를 확인해야 한다. 데이터 전체나 변환을 실행하여 검증한 상태는 아니다.

### B. 현재 쉘과 같은 imperfection-induced response law

입력은 nominal mesh·재료·하중 프로토콜·BC로 고정하고, 작은 hole-boundary/ligament imperfection 등 물리적 random field 하나를 숨긴다. 같은 조건에서 반복 FEA를 수행하여 응답장 분포를 얻는다. 이는 새로운 생성 캠페인이며 공개본의 기존 확률성으로 주장할 수 없다.

먼저 작은 pilot에서 실제로 다른 응답이 발생하고 수치 tolerance·branch-switch rule보다 물리적 perturbation이 결과를 지배하는지 확인한다. 긴 파장 또는 specimen 전체의 좌굴을 연구하려면 2×2 periodic RVE의 한계도 확인해야 한다. 4×4/6×6 specimen 등은 가능한 확장 제안이지 현재 공개 데이터의 내용이 아니다.

MGN-V에 연결할 때는 known affine deformation을 제거한 fluctuation w를 예측하는 구성이 적절한 출발점이다. Column-vector convention에서 x=F X+w로 복원한다. 전체 displacement를 쓸 수도 있으나 두 convention을 혼합하면 안 된다. 초기에는 T=1의 fixed load snapshot으로 시작하고 geometry별로 모든 loading·branch를 같은 split에 둔다. Periodic graph, 고차 삼각요소의 중간 node, hole을 가로지르지 않는 연결성을 보존한다. 현재 `periodic_box`의 축정렬 minimum-image가 모든 oblique lattice를 처리한다고 가정하면 안 된다.

단지 균등한 2/4개 branch만 있는 문제는 global latent로도 충분할 수 있다. 이 자료를 채택했다고 hybrid의 필요성이 자동으로 입증되는 것은 아니다.

## 2. Bilayer wrinkling: 새로 만든다면 가장 강한 후보

1차 자료:

- [FEniCS-arclength, JOSS](https://joss.theoj.org/papers/10.21105/joss.05727)
- [공개 MIT solver](https://github.com/pprachas/fenics_arclength)
- [Bilayer 예제 문서](https://fenics-arclength.readthedocs.io/en/stable/examples/displacement_control/Bilayer_wrinkling.html)
- [실제 validation 코드](https://github.com/pprachas/fenics_arclength/blob/master/validation/validate_bilayer.py)

현재 예제는 plane-strain neo-Hookean film/substrate, lateral compression, 고정된 중앙 point perturbation을 사용한다. 따라서 공개 예제 자체는 stochastic benchmark가 아니다. 문서/코드에는 full displacement와 임계 strain·파장을 확인할 수 있는 출발점이 있다. 현재 2D 단면 예제로 2D 표면의 wrinkle 방향성까지 검증했다고 주장할 수는 없다.

권하는 신규 데이터 계약:

| 구분 | 제안 |
|---|---|
| 관측 condition | Nominal geometry, film/substrate 두께·강성비, 압축률, BC |
| 숨기는 원인 | 작은 초기 surface imperfection field 한 종류 |
| 확률법칙 | 명시한 covariance/PSD와 amplitude; 먼저 하나의 law를 고정 |
| 출력 | 같은 condition에서의 full displacement field 또는 변형 경로 |
| 핵심 변화 | Domain 길이, imperfection correlation length, 임계점 전후 압축률 |
| 평가 | Wrinkle phase/amplitude, wavelength spectrum, localization, 거리별 covariance, energy/반력, field ES/variogram |

기존 shell의 단일 dimple 세 변수와 달리 공간적으로 분포한 확률입력을 시험할 수 있다. 다만 random field를 주어도 최종 응답이 항상 같은 mode로 수렴할 수 있으므로 pilot에서 conditional variability가 실제 존재하는지 확인해야 한다. Global+local latent에 유리하도록 prior를 맞춰 만들지 말고, 물리적으로 정한 한 random field의 correlation length를 통제 변화시키는 것이 좋다.

제안 pilot은 3–5개 nominal condition에서 각 32–64개의 독립 imperfection draw를 비교하는 것이다. 수렴, mesh/refinement, 실제 ensemble 다양성, 비용을 확인하기 위한 제안이며 학습용으로 충분한 표본 수나 측정 runtime이라는 뜻은 아니다. Surface perturbation은 FE mesh/BC의 유효성을 유지하고, true/nominal geometry 및 displacement convention을 진단 metadata로 보존한다.

## 3. Snap-through와 다른 공개 좌굴 자료를 후순위로 둔 이유

단일 shallow arch의 좌우 또는 전이 전후 두 상태는 물리적으로 가깝지만 적은 global 변수만으로 설명될 수 있다. 여러 arch가 결합된 array나 비축대칭 shell이라면 지역 state의 상호작용을 추가할 수 있으나 새 corpus가 필요하다. Arc-length로 얻은 모든 평형점이 실제 하중 프로토콜에서 실현되는 stable response인 것은 아니다. Dynamics, loading history, stability와 branch selection을 정의해야 한다.

- [ABC dataset 공식 코드](https://github.com/pprachas/ABC_dataset): heterogeneous column의 압축 좌굴이지만 공개 ML 목표는 buckling-direction classification이다. 현재 목적의 field-distribution benchmark로 바로 사용 가능하다고 확인하지 못했다.
- [24-bar snap-through 자료](https://github.com/ldriemeier/FEA-using-DNN): 공개 출력의 중심은 apex displacement·reaction curve다. Full field와 repeated-condition ensemble을 별도 확보해야 한다.
- [Concrete thin-shell buckling 자료](https://researchdata.bath.ac.uk/1533/): 공개된다는 사실과 우리에게 필요한 conditional field ensemble이라는 사실은 별개다. 이번 조사에서 해당 ensemble을 확인하지 못했으므로 우선 추천하지 않는다.
- [Bi-FORK Beam3D](https://arxiv.org/html/2610.12449v1): 분기 sampling sanity check로는 유용하나 3–11 node의 작은 beam이어서 coarse regional latent를 연구하는 주력 추가 데이터로는 제한적이다.

## 4. Inverse FEM을 이번 주력 후보와 분리한 이유

Inverse FEM의 목표는 p(u | sparse strain, sensor layout, geometry, BC)이며, 현재 좌굴의 p(u | nominal geometry, loading)와 conditioning/uncertainty의 원인이 달라진다. 센서 관측에 따른 posterior 복원을 연구하려면 좋은 후속 방향이다. 이미 [Poloni et al., MSSP 2023](https://doi.org/10.1016/j.ymssp.2022.110056)이 GP strain extrapolation과 stochastic iFEM을 제시한다.

관련 공개 자산은 [ISTAR Demo Wing Abaqus Model](https://research.tudelft.nl/en/datasets/istar-demo-wing-abaqus-model)이다. 공개 FE 모델이며 동일 관측의 reference posterior가 있는 corpus로 확인된 것은 아니다. Analytic posterior를 만들려면 알려진 선형 FOM, load prior와 noise law, sparse strain observation을 새로 정의해야 한다.

현재 로컬 DDFEM/POC_ifem_v0의 IFEM은 **immersed FEM**이다. 작은 read-only smoke에서는 그 planar linear FOM으로 strain observation H를 만드는 것이 가능했으나, shell shape-sensing이나 snap-through가 이미 구현됐다는 뜻은 아니다. 이번 요청에서는 이 코드의 변경이나 dataset 생성은 수행하지 않았다.

## 검증 범위

논문, 공개 file listing, 작은 schema 문서, solver 및 dataset reader를 확인했다. 대용량 corpus 다운로드·전체 로딩, 새 FEA, 모델 학습, 데이터 변환은 수행하지 않았다. 확보된 corpus, 공개 generator, 우리가 제안한 신규 stochastic extension을 구분했다.
