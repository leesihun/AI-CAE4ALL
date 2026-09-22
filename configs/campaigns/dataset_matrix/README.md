# Dataset별 baseline config matrix

2026-09-22 작업본. `manifest.json`이 86개 train/infer 쌍(172개 config)의 전체 경로를 담는다.
기존 config는 교체하지 않고 각 사례의 `baseline/`에 추가했다. Shell buckling은 제외했다.
**현재 검토 사항: Flag 일반 모델의 1-frame 입력을 2-frame 입력으로 확장할지 사용자 확인 중이다.**
따라서 이 문서는 모든 설정의 물리적 타당성이나 학습 완료를 인증하는 문서가 아니다.

## 공통 필드 계약

저장 순서는 항상 **input → output → conditioner → partNo**이다.
Mesh HDF5는 `nodal_data[field, time, node]`, 아래 행 번호는 0-based이며 범위는 끝을 제외한다.
input은 좌표 `x,y,z` 세 행이다. `output`은 예측할 물리장 순서이다.
`conditioner`는 예측하지 않는 알려진 조건이다. `partNo`는 존재할 때 마지막 행의 범주형 node type이다.

주의: native config의 `input_var`는 좌표 수가 아니라 **모델에 들어가는 현재 물리 상태의 채널 수**이다.
좌표는 별도 geometry/position 입력이다. 같은 물리장을 input/output 두 블록으로 중복 저장하지 않는다.
정적 T=1 문제는 물리 상태 입력을 0으로 하고 저장된 output을 직접 예측한다.
시간 문제는 현재 상태 `u[t]`에서 `u[t+1]-u[t]`를 예측한 뒤 누적한다.
조건은 rollout 중 고정하며 미래 정답을 입력하지 않는다. `partNo`는 one-hot 입력으로 사용한다.

| 분류/사례 | input | output (정확한 순서) | conditioner (정확한 순서) | partNo | input_var/output_var/cond_var |
|---|---|---|---|---|---|
| deterministic/ex1 thermoelastic | 0:3 xyz | 3:7 ux, uy, uz, stress | 없음 | 7 | 4/4/0 |
| deterministic/ex2 contact | 0:3 xyz | 3:7 ux, uy, uz, stress | 없음 | 7 | 4/4/0 |
| deterministic/ex3 full, mid CRM | 0:3 xyz | 3:7 cp, cf_x, cf_y, cf_z | 7:17 Mach, AoA, inboard aileron, outboard aileron, elevator, HTP, normal_x, normal_y, normal_z, surface_area | 없음 | 4/4/10 |
| deterministic/ex4 cylinder | 0:3 xyz | 3:6 velocity_x, velocity_y, pressure | 없음 | 6 | 3/3/0 |
| deterministic/ex5 plate | 0:3 xyz | 3:7 ux, uy, uz, stress | 없음 | 7 | 4/4/0 |
| deterministic/ex6 flag | 0:3 xyz | 3:6 ux, uy, uz | 없음 | 6 | 3/3/0 |
| deterministic/ex7 AirfRANS | 0:3 xyz | 3:7 velocity_x, velocity_y, pressure, nu_t | 7:12 u_inf_x, u_inf_y, sdf, normal_x, normal_y | 12 | 4/4/5 |
| deterministic/ex8 elasticity | 0:3 xyz | 3 sigma | 없음 | 없음 | 1/1/0 |
| deterministic/ex9 plasticity | 0:3 xyz | 3:5 ux, uy | 5:7 uz(항상 0), die_profil | 없음 | 2/2/2 |
| deterministic/ex10 DeepJEB | 0:3 xyz | 3:5 stress, displacement_magnitude | 5:9 lc_ver, lc_hor, lc_dia, lc_tor | 없음 | 2/2/4 |
| probabilistic/ex1 turbulent | 0:3 xyz | 3:7 density, pressure, velocity_x, velocity_y | 7 tcool | 없음 | 4/4/1 |
| probabilistic/ex2 crack | 0:3 xyz | 3:6 damage, ux, uy | 없음 | 없음 | 3/3/0 |
| geometry_generation/ex1 DeepJEB | surface_points, surface_normals, query_xyz (이름 있는 배열) | signed_distance | volume, area | 없음 | mesh 행 설정을 사용하지 않음 |

CRM 각도 조건의 단위는 degree이다. Plasticity의 오래된 root 속성보다 실제 row 배치를 우선한다.
SDF generation 시 입력은 latent noise와 query 좌표이며, surface point cloud는 VAE 학습 입력이다.
SDF 추론 기본값은 unconditional branch이다. 조건 생성은 `cond_values volume, area`의 **수치값**을
같은 순서로 추가한다. 학습 세트의 조건 통계와 OOD 검사를 사용한다.

## 적용 기법과 개수

| 대상 | 적용 기법 | 쌍 수 |
|---|---|---:|
| deterministic 11사례 (CRM full/mid 각각) | DeepONet, Point-DeepONet, FNO, GINO, Transolver **3**, MeshGraphNets, HI-MGN | 77 |
| CRM full/mid, Flag, Plasticity | LSH-VAE + latent conditioner (코드 경로 `SimulGenVAE`) | 4 |
| probabilistic 2사례 | HI-MGN-V, cHI-MGNflow | 4 |
| geometry_generation DeepJEB | SDFFlow | 1 |

LSH-VAE는 train/infer 전체에 대해 node 수, T, connectivity의 동일성을 `prepare.py`에서 검사한다.
Elasticity는 node 수가 같아도 좌표/연결 및 node 대응이 달라 이 목록에 포함하지 않았다.
Plasticity는 reference xy가 case마다 달라져 이를 conditioner에 포함한다.

LSH-VAE 입력/예측은 일반 stepwise 모델과 다르다:

- VAE 학습: 전체 output trajectory를 인코딩/복원한다.
- LC 학습: 알려진 조건 → VAE latent (주 latent와 계층 latent)를 학습한다.
- 추론: LC 조건 → latent → 전체 output trajectory. 정답 field를 encoder에 넣는 reconstruction 평가가 아니다.
- CRM 조건: canonical 행 7:13의 여섯 global parameter만 사용한다. 고정 mesh의 normal/area는 LC에서 중복 입력하지 않는다.
- Flag 조건: `[frame 0, frame 1] → [ux, uy, uz] → node` 순서로 평탄화한다. 예측 평가는 조건으로 관측한 두 프레임을 제외한 `t>=2`를 별도로 보고해야 한다.
- Plasticity 조건: `[reference x, reference y, initial ux, initial uy, die_profil] → node` 순서이다.
- CSV 행은 정수 sample_id 오름차순이며 동봉된 `*_conditions.json`에 source와 ID 순서가 있다.
- VAE/LC 모두 같은 train/val/test ID를 사용하고 모든 scaler는 train에서만 fit한다.
  이전 split provenance가 없는 checkpoint의 재사용은 명시적으로 거부한다.

## 필요한 데이터 준비와 실행

Repository root에서 실행한다:

```powershell
python configs/campaigns/dataset_matrix/prepare.py
python configs/campaigns/dataset_matrix/generate.py --check
python configs/campaigns/dataset_matrix/audit.py
python configs/campaigns/dataset_matrix/audit.py --data
python configs/campaigns/dataset_matrix/audit.py --native Transolver
```

`--native`는 각 독립 runtime 이름을 받는다: MeshGraphNets, MeshGraphNets_Variational,
HI_MGNFlow, Neural_Operator, Transolver, SimulGenVAE, SDFFlow.
`generate.py --json`은 파일 내용을 출력만 한다. 기존 파일을 자동으로 덮어쓰지 않는다.

`prepare.py`는 source HDF5를 변경하지 않고 `dataset/derived/config_matrix/`에 다음을 만든다:

- CRM mid canonical VDS: 원본 row 순서 `[x,y,z,nx,ny,nz,cp,area,cfx,cfy,cfz,Mach,AoA,ailIn,ailOut,elevator,HTP]`를 위 표로 변환.
- Flag/Plasticity train/infer LC conditioner CSV와 provenance JSON.

CRM VDS는 상대 경로로 원본 데이터를 참조하는 zero-copy 파일이다. **VDS만 복사하면 안 된다.**
원본을 포함한 `dataset/`의 상대 디렉터리 구조를 유지해야 한다. 원본이 없으면 검증에서 실패한다.
HDF5 preprocessing 저장은 새 mesh config에서 껐다. Source 데이터에 통계나 split을 쓰지 않는다.

예시(학습 실행은 사용자가 별도로 수행):

```powershell
python AI_CAE4ALL_main.py --config configs/Transolver/deterministic/ex4/baseline/config_train_transolver3.txt
python AI_CAE4ALL_main.py --config configs/Transolver/deterministic/ex4/baseline/config_infer_transolver3.txt
```

경로는 native method working directory 기준이다. Suite launcher가 working directory를 맞춘다.
출력은 `output/dataset_matrix/<category>/<example>/<method>/`로 분리된다.
새 추론 config에는 학습 전 checkpoint가 없으므로 **학습 이후** 실행해야 한다.

## 논문 근거와 구현상 조정

여기서 baseline은 논문에 근거한 **native implementation adaptation**이며 모든 dataset에 대한 원문 재현이나
검증된 최적 hyperparameter라는 뜻이 아니다. Loss, optimizer, sampling, mesh adapter가 달라질 때 이를 구분한다.

| 기법 | 근거 | 이번 설정과 차이 |
|---|---|---|
| MGN | [Learning Mesh-Based Simulation](https://arxiv.org/abs/2010.03409) | latent 128, message passing 15. Eulerian field를 displacement로 해석하지 않는다. Flag의 원문 history/second-order 처리와 현재 1-frame delta runtime은 다르며 수정 여부 확인 중이다. |
| HI-MGN | [HI-MGN](https://arxiv.org/abs/2608.13827) | FPS/Voronoi seed hierarchy, learned interpolation, 2-level V-cycle. `[2,3,5,3,2]`로 총 15 processing block을 맞췄다. cluster 수는 각 mesh 규모에 맞춘 baseline 선택이다. |
| Transolver 3 | [Transolver 3](https://arxiv.org/abs/2602.04940) | slice-space/tiled attention, decoupled inference. 큰 contact/AirfRANS에는 amortized training. CRM은 원문 full-mesh 설정에 따라 24 layers, width 256, 8 heads, 64 slices, 500 epochs, LR .001, WD .05, warmup 25. Native normalized MSE와 scheduler floor는 원문의 relative-L2/min-LR와 다르다. |
| DeepONet | [DeepONet](https://doi.org/10.1038/s42256-021-00302-5) | fixed sensor branch + coordinate trunk, vector output split_both. Irregular mesh의 sensor interpolation은 이 저장소 adapter이다. |
| Point-DeepONet | [Point-DeepONet](https://arxiv.org/abs/2412.18362) | PointNet branch + SIREN trunk, mesh_state variant. 원문의 특수 geometry/SDF 실험을 동일하게 재현한다고 주장하지 않는다. |
| FNO | [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) | 4 layers, width 64, mesh→grid→mesh adapter. 원본의 structured-grid 직접 입력과 구분한다. FFT 안정성을 위해 AMP를 끈다. |
| GINO | [GINO](https://arxiv.org/abs/2309.00583) | input/output radius GNO + latent FNO. mesh_state, unweighted reductions이며 모든 데이터에서 SDF/quadrature가 갖춰진 논문 버전은 아니다. Sparse surface를 감싸는 volume grid의 빈 input cell을 허용하되 query coverage는 별도 검사 대상이다. |
| LSH-VAE | [LSH-VAE](https://doi.org/10.1007/s00366-023-01916-6) | 7 hierarchy levels, main latent 32, hierarchical latent 8. 작은 width와 native AdamW를 사용하며 원문의 Adamax/107-layer 상세 recipe와 다르다. LC는 저장소 확장이다. T=1, batch=1에서도 GroupNorm이 유효하도록 각 block에 적어도 16 channels를 둔다. |
| HI-MGN-V | [InfoVAE](https://arxiv.org/abs/1706.02262), [Flow Matching](https://arxiv.org/abs/2210.02747) | 저장소의 conditional-prior variational hierarchy. Global latent 32, MMD, auxiliary reconstruction, conditional FM; posterior reconstruction과 prior sampling 성능을 구분해야 한다. 단일 논문의 동일 명칭 recipe로 오인하지 않는다. |
| cHI-MGNflow | [Latent Diffusion Graph Networks](https://arxiv.org/abs/2504.02843) | coarse-node AE latent 4 channels, KL 1e-6, AE 500 epochs 후 latent flow 500 epochs. 저장소 HI hierarchy/flow adaptation이며 원문의 모든 architecture와 동일하지 않다. |
| SDFFlow | [3DShape2VecSet](https://arxiv.org/abs/2301.11445), [Flow Matching](https://arxiv.org/abs/2210.02747) | 32×32 latent token set, FPS encoder, attention SDF decoder, DiT flow. Surface/normal/eikonal loss와 condition dropout은 저장소의 조합이며 이 조합 전체를 원문 recipe로 주장하지 않는다. |

메모리/계산량을 고려한 grid, sensor 수, batch 크기이며 성능 튜닝 결과는 아니다.
80GB급 GPU를 목표로 했지만 해당 장치에서 전체 train/infer 메모리는 아직 실측하지 않았다.

## 데이터별 안전 장치

- MGN 계열에서 fixed geometry와 displacement geometry를 분리한다.
  Crack은 `[damage,ux,uy]` 중 `[ux,uy,0]`만 좌표에 더한다. Plasticity는 `[ux,uy,0]`이다.
  이 규칙은 train stats, graph, AR rollout, inference, checkpoint에 동일하게 전달한다.
- Turbulent는 저장 좌표의 x 주기가 `128/127`이다. x 128개가 `[-.5,.5]`에 놓인 데이터의 seam edge를
  minimum-image 길이 `1/127`로 처리한다. y는 비주기이다. Fine/coarse edge vector를 감싸지만
  FPS/Voronoi partition 자체를 toroidal partition으로 바꾼 것은 아니다. World edges는 끈다.
- DeepJEB surrogate의 네 load case는 같은 `metadata.bracket`을 공유한다. 이를 기준으로 grouped split하여
  같은 형상이 train/val/test에 섞이지 않는다. 외부 infer bracket 120개와 train source bracket 301개는 중복 0이다.
- SDFFlow는 parent 단위 split을 사용하고 거의 상수인 bbox 조건은 제외하여 volume/area만 conditioning한다.
- Cylinder/Flag의 7-row 형식에서도 마지막 node-type 행을 정확히 읽는다.
- Mesh 계열 train loss는 normalized channel MSE이며 equal feature weights를 사용한다. 물리 단위 metric은
  inverse normalization 후 채널별로 보고해야 한다. 여러 field의 MSE 합만으로 dataset 간 우열을 비교하지 않는다.

## 검증 범위와 남은 항목

완료한 검사:

- 172개 config의 generated-source 일치, launcher schema: 오류/경고 없음.
- 13개 mesh 사례의 train/infer 전체 sample shape, field 수, T 검사; 유한값은 결정론적 node subset만 검사.
- Native parser 161개 통과. Transolver 추론 11개는 아직 없는 실제 학습 checkpoint가 필요하여 해당 검증을 유보.
- LSH 후보 4사례의 전체 train/infer 고정 topology 검사, derived VDS/CSV read-back.
- 수정한 geometry, node type, grouped split, train-only scaler에 대한 회귀 테스트.
- Transolver 3 tiled/decoupled 출력 일치 및 amortized forward/backward의 target 대응 테스트.

남은 항목:

- Flag 일반 모델 2-frame history 여부 결정 및 해당 경로 검증.
- 전체 NO coordinate-domain/GINO coverage 확인, 대표 config의 실제 model forward 검증 추가.
- 80GB GPU 실측, 장기 학습 안정성, 실제 checkpoint inference, held-out 성능/확률 calibration.
  마지막 항목들은 아직 실행하지 않았으며 config 생성/단위 테스트로 대신 인증할 수 없다.
