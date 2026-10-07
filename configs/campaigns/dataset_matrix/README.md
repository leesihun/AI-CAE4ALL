# Dataset별 baseline config matrix

2026-09-22 작업본. `manifest.json`이 generate.py가 만드는 75개 train/infer 쌍(150개 config)과
손으로 관리하는 SDFFlow geometry_generation ex2/ex3 2쌍(`"generated": false`)의 경로를 담는다.
2026-10-07에 SDFFlow ex4(Thingi10K)를 삭제했다: 데이터와 출력은 `_set_aside_20261007/`로 옮겼고 lane 2에는 arm이 없다.
2026-09-23에 GINO가 suite에서 삭제되어 GINO 11쌍(22개 config)을 matrix에서 뺐다. GPU lane 2는 재배정하지 않았다.
기존 config는 교체하지 않고 각 사례의 `baseline/`에 추가했다. Shell buckling은 제외했다.
Flag(ex6) 일반 모델은 2026-09-24 결정대로 다른 사례와 같은 1-frame delta 예측을 유지한다(아래 MGN 행).
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
| probabilistic/ex1 turbulent | 0:3 xyz | 3:7 density, pressure, velocity_x, velocity_y | 7 tcool | 없음 | 4/4/1 |
| probabilistic/ex2 crack | 0:3 xyz | 3:6 damage, ux, uy | 없음 | 없음 | 3/3/0 |
| geometry_generation/ex1 DeepJEB | surface_points, surface_normals, query_xyz (이름 있는 배열) | signed_distance | volume, area | 없음 | mesh 행 설정을 사용하지 않음 |

CRM 각도 조건의 단위는 degree이다. Plasticity(ex9)의 root `builder_*` 속성은 2026-08-19 재빌드
이전의 4/4/0을 그대로 들고 있었는데, 2026-09-23에 파일 안의 `metadata/feature_names`
(`[x,y,z | ux,uy | uz, die_profil]`)와 실측(987개 샘플 전체에서 `uz`는 정확히 0, `die_profil`은
시간에 대해 상수)에 맞춰 2/2/2로 제자리 수정했다. 배열은 건드리지 않았다.
SDF generation 시 입력은 latent noise와 query 좌표이며, surface point cloud는 VAE 학습 입력이다.
SDF 추론 기본값은 unconditional branch이다. 조건 생성은 `cond_values volume, area`의 **수치값**을
같은 순서로 추가한다. 학습 세트의 조건 통계와 OOD 검사를 사용한다.

## 적용 기법과 개수

| 대상 | 적용 기법 | 쌍 수 |
|---|---|---:|
| deterministic 11사례 (CRM full/mid 각각) | DeepONet, Point-DeepONet, FNO, Transolver **3**, MeshGraphNets, HI-MGN | 66 |
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
- Flag 조건: `[frame 0, frame 1] → [ux, uy, uz] → node` 순서로 평탄화한다 (`prepare.py::make_conditions`, `arr[3:6, :2, :]`).
- Plasticity 조건: `[reference x, reference y, initial ux, initial uy, die_profil] → node` 순서이다
  (`prepare.py::make_conditions`, `arr[:, 0, :][[0,1,3,4,6]]` — 즉 **frame 0 한 장**을 관측한다).
- CSV 행은 정수 sample_id 오름차순이며 동봉된 `*_conditions.json`에 source와 ID 순서가 있다.
- VAE/LC 모두 같은 train/val/test ID를 사용하고 모든 scaler는 train에서만 fit한다.
  이전 split provenance가 없는 checkpoint의 재사용은 명시적으로 거부한다.

## 채점 프레임 규칙 (scoring frame)

**규칙은 하나다: 모델이 입력으로 *본* 프레임은 채점하지 않는다.** 본 프레임을 포함하면
그 프레임의 오차는 0에 가깝고, 짧은 trajectory일수록 평균이 크게 낙관적으로 나온다
(ex9는 20프레임 중 1장 = 5%, ex6는 401프레임 중 2장 = 0.5%).
본 프레임 수는 dataset이 아니라 **route(방법)** 가 정하므로, 같은 exN에서도 AR 계열과
LSH-VAE의 채점 구간이 달라질 수 있다. ex6이 바로 그 경우다.

- **AR mesh/operator 계열** (MGN, HI-MGN, Transolver 3, DeepONet, Point-DeepONet, FNO,
  HI-MGN-V, cHI-MGNflow): frame 0을 초기조건으로 받아 `infer_timesteps = T-1` 스텝을 굴린다
  (`generate.py`의 `case(..., steps=...)`). 채점 구간은 `t = 1 .. T-1`이고 **frame 0은 절대
  채점하지 않는다** — 입력을 그대로 되돌려 쓴 값이라 어떤 모델이든 오차 0이다.
- **LSH-VAE (`dense=True` 4사례)**: 조건 → latent → 전체 trajectory이므로 "스텝"이 없다.
  본 프레임은 LC 조건에 들어간 프레임뿐이고, 그 수는 `prepare.py::make_conditions`가 정한다.
- **T=1 정적 사례**: rollout이 없다. `infer_timesteps 1`이고 그 한 장이 전부 예측이므로
  전 구간을 채점한다. 여기서 "frame 0을 빼라"를 기계적으로 적용하면 채점할 것이 남지 않는다.

| 사례 | 파일 T | 구조 | AR 계열 관측/채점 | LSH-VAE 관측/채점 |
|---|---:|---|---|---|
| deterministic/ex1 thermoelastic | 1 | static | — / 그 1장 | 해당 없음 |
| deterministic/ex2 contact | 50 | trajectory | t=0 / `t=1..49` | 해당 없음 |
| deterministic/ex3 full, mid CRM | 1 | static | — / 그 1장 | 프레임 관측 없음 (global param 6개) / 그 1장 |
| deterministic/ex4 cylinder | 600 | trajectory | t=0 / `t=1..599` | 해당 없음 |
| deterministic/ex5 plate | 400 | trajectory | t=0 / `t=1..399` | 해당 없음 |
| deterministic/ex6 flag | 401 | trajectory | t=0 / `t=1..400` | **t=0,1** / **`t=2..400`** |
| deterministic/ex7 AirfRANS | 1 | static | — / 그 1장 | 해당 없음 |
| deterministic/ex8 elasticity | 1 | static | — / 그 1장 | 해당 없음 |
| deterministic/ex9 plasticity | 20 | trajectory | t=0 / `t=1..19` | t=0 / `t=1..19` (AR과 동일) |
| probabilistic/ex1 turbulent | 101 | trajectory | t=0 / `t=1..100` | 해당 없음 |
| probabilistic/ex2 crack | 20 | trajectory | t=0 / `t=1..19` | 해당 없음 |

**ex6만 route 간 채점 구간이 다르다.** LSH-VAE는 frame 0,1을 조건으로 받으므로 AR 계열과
같은 `t=1..400`으로 채점하면 t=1 한 장을 공짜로 얻는다. ex6에서 두 계열을 비교할 때는
**양쪽 모두 `t=2..400`** 으로 맞춰 보고한다. ex9는 양쪽 다 frame 0 한 장만 관측하므로
`t=1..19`로 그대로 비교 가능하다.

위 T 값은 `dataset/`의 실제 HDF5에서 읽은 값이며 `audit.py --data`의
`max(T-1, 1) == c['steps']` 단언이 매 실행마다 이를 다시 확인한다. 정적 사례의 `max(...,1)`이
바로 "T=1은 frame 0을 빼지 않는다"를 코드로 적어 둔 부분이다.

> `methods/HI_MGNFlow/misc/score_rollouts.py`는 AR 규칙(step 0 제외, `t=1..`)을 이미 구현한다.
> 다만 **T=1 파일에는 쓸 수 없다** — `m[1]`에서 IndexError가 난다. 정적 사례는 rollout 채점
> 대상이 아니므로 그 스크립트를 겨누지 않는다. 이 matrix의 채점은 두 경우와 ex6 예외를 모두
> 구현한 `score_rank.py`로 한다(아래 "채점과 순위").

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

## 전체 campaign runner (8-GPU Linux 두 대)

전체 train→infer는 `run_matrix.py`가 돌리며, 머신마다 wrapper 하나씩이다(Linux 전용):

```bash
bash configs/run_all_135.sh     # deterministic ex1, ex2, ex3_full, ex5, ex8 + probabilistic ex1 + geometry ex2  (34 arm / 68 stage)
bash configs/run_all_136.sh     # deterministic ex3_mid, ex4, ex6, ex7, ex9 + probabilistic ex2 + geometry ex1, ex3  (37 arm / 74 stage)

DRY_RUN=1 bash configs/run_all_135.sh        # 계획만 출력, 아무것도 쓰지 않음 (nvidia-smi는 조회함)
CHECK=1 bash configs/run_all_135.sh          # 남은 모든 stage에 launcher --check
SKIP_GPU_GATE=1 bash configs/run_all_135.sh  # idle 대기 없이 바로 시작
PYTHON=/path/to/python bash configs/run_all_135.sh   # 기본은 PATH의 python3 (launcher는 3.10+)
```

Wrapper는 git mode 100644라 `bash ...`로 실행한다(`./run_all_135.sh`로 쓰려면 `chmod +x`).
Method별 interpreter는 기존대로 `ai_cae4all.local.toml`에서 고른다.

- **Lane.** config의 `gpu_ids` 숫자가 home lane이다: 0 deeponet+fno, 1 point_deeponet, 2 (arm 없음),
  3 transolver3, 4 meshgraphnets, 5 himgn, 6 himgn_v+lsh_vae, 7 chi_mgnflow+sdfflow.
  손으로 관리하는 SDFFlow geometry ex2/ex3는 자기 config의 `gpu_ids` 0/1을 그대로 lane으로 쓴다
  (135: ex2 lane 0 / 136: ex3 lane 1). 출력은 `output/geometry_generation/<ex>_<dataset>/sdfflow/`이다.
  GPU마다 worker 하나가 자기 lane을 manifest 순서로 돌고, 끝나면 가장 뒤처진 lane의 다음 arm을 가져간다.
- **Start gate.** 즉시 한 번, 그 뒤 `GATE_INTERVAL`(3600 s)마다 확인한다. GPU 8개가 보이고 모두 util 0%여야
  시작한다. 조건은 util뿐이다: 메모리만 잡고 계산하지 않는 process는 gate를 막지 않는다.
  한 번의 확인은 `GATE_SAMPLES`(3)회 × `GATE_SAMPLE_GAP`(20 s) 간격 측정이다.
- **실행.** Stage마다 카드 하나. config를 `output/dataset_matrix/_campaign/<machine>/launch/`에 복사하되
  `gpu_ids`만 0으로 바꾸고, 학습이면 config에 없는 `resume_training True`/`resume_interval_minutes 15`를 끝에 붙인다.
  원본 경로와 sha256은 머리 주석에 적는다. 그 복사본을 `CUDA_VISIBLE_DEVICES=<카드 UUID>`로
  launcher에 넘긴다. 체크인된 config는 절대 수정하지 않는다(그래서 marker의 sha256도 그대로다). 학습이 실패하면 그 arm의 추론은 건너뛴다.
- **GPU 사망.** 실행 중 `WATCH_INTERVAL`(300 s)마다 nvidia-smi를 보고, 성공하지 못한 stage 뒤에는 CUDA driver로
  카드를 직접 열어 본다. 죽은 것이 확인되면 그 worker만 멈추고, arm은 자기 lane 맨 앞으로 돌아가 다른 카드가 가져간다
  (arm당 1회). 나머지 worker는 계속 돈다.
- **Strike.** 한 카드에서 stage가 `STRIKE_LIMIT`(3)번 연속 실패(failed, not launched, interrupted)하면 그 worker는
  퇴역하고 나머지 카드가 그 lane을 가져간다. 성공한 stage가 하나라도 있으면 횟수는 0으로 돌아간다.
  퇴역 전에 실패한 arm은 다시 시도하지 않고 report에 실패로 남는다. 다음 실행이 그것들을 돌린다.
- **재개.** 끝난 stage는 `_campaign/<machine>/done/<arm>__<stage>`에 marker(JSON: config sha256, 시작·종료 시각)를
  남긴다. stage를 건너뛰는 조건은 둘이다: marker의 sha256이 **지금의 config와
  byte 단위로 같고**(주석 한 줄을 고쳐도 다시 돈다), 추론이면 그 추론이 **현재 학습이 끝난 뒤에 시작**했어야 한다.
  그 밖의 marker(다른 config의 것, 읽을 수 없는 것, 예전 bash runner의 빈 파일)는 다시 돈다. 재학습하면 추론도 다시 돈다.
  같은 명령을 다시 실행하면 남은 것만 돈다.
- **학습 도중 재개.** marker는 마지막 epoch까지 돌고 최종 checkpoint를 쓴 stage만 남긴다. 중간에 끊긴 학습(정지, 카드 사망,
  머신 다운)은 trainer가 15분마다 epoch 경계에서 checkpoint 옆에 남긴 `<checkpoint>.resume`(model·EMA·optimizer·
  scheduler·AMP scaler·RNG·best/patience tracker·epoch)을 다음 실행이 이어받아 **마지막 epoch까지** 돈다. 잃는 것은
  마지막 state 이후의 epoch(최대 15분 + 1 epoch)뿐이다. 최종 checkpoint를 쓰면 `.resume`은 지워진다. 학습은 시작 전에
  config sha256을 `_campaign/<machine>/started/<arm>__train`에 적고, 그 기록과 config가 맞을 때만 이어받는다.
  trainer도 `.resume`에 적힌 config fingerprint(`gpu_ids`·resume key 제외)가 다르면 거부한다. 두 stage route
  (LSH-VAE VAE→LC, SDFFlow VAE→FM, cHI-MGNflow AE→Prior)는 끝난 앞 stage를 다시 돌리지 않고 끊긴 stage부터 잇는다.
  앞 stage checkpoint가 그 사이 다시 쓰였으면 뒤 stage state는 `.resume.stale`로 밀려나고 그 stage는 epoch 0부터 돈다.
  재개는 단일 GPU 학습만 지원한다(matrix의 모든 stage가 그렇다). 다중 GPU 경로는 `resume_training`을 거부한다.
- **Set aside.** stage를 돌리기 전에 이전 실행이 그 stage에 남긴 것을
  `_campaign/<machine>/set_aside/<run id>/<arm>__before_<stage>/`로 옮긴다: 맞지 않는 marker, 그 arm의 예측 디렉터리,
  학습 전이면 추론 marker도. `_campaign/<machine>/started/<arm>__train`이 **바로 이 config가 시작한 것**이라고
  적고 있지 않으면 resume state(config가 가리키는 경로 옆의 `.resume`/`.resume.tmp`/`.resume.stale`)와
  SimulGenVAE의 `vae.pth`/`lc.pth`도 함께 옮긴다. 그래서 고친 config의 학습이 이전 config의 끊긴 학습을 잇지 않는다.
  `skip_completed_stages`는 자기 호환 key만 비교하므로 두면 다른 config의 checkpoint가 이 실행에 이어진다.
  아무것도 지우지 않는다(`set_aside/`는 확인 후 손으로
  지운다). config가 가리키는 checkpoint·dataset·log가 든 경로나 `output/` 밖의 경로는 옮기지 않으며, 옮기지 못한 stage는
  strike 없이 실패로 남는다. SDFFlow geometry는 자기 stage 기록을 따로 가지므로 marker와 resume state만 옮긴다.
  `DRY_RUN=1`은 옮길 것과 함께 이어받을 `.resume`과 그 시각을 보여 준다(`train would continue from ...`).
- **정지.** Ctrl-C/SIGTERM(nohup이 아니면 SIGHUP도)은 모든 stage에 SIGTERM, `KILL_GRACE`(30 s) 뒤 SIGKILL
  (두 번째 신호면 즉시). 그 stage들은 기록하지 않는다.
- **출력.** `_campaign/<machine>/logs/runner.log`, stage별 로그는 `logs/stages/`, 결과표는 `report.txt`/`report.json`.
- **채점과 순위.** 신호로 정지하지 않았으면 report 뒤에 `score_spread.py`(probabilistic, `spread_scores.csv`)와
  `score_rank.py`(아래 "채점과 순위", `ranking.txt`/`ranking.csv`/`convergence.csv`)를 돌린다. 둘 다 참고용이라 exit code를 바꾸지 않는다.
  geometry 사례도 넘기며, 그 arm은 순위 없이 수렴 절(아래 "수렴 확인")에만 나온다.
  `score_rank.py`는 두 marker가 지금의 config와 맞고 예측 파일이 추론 시작 뒤에 쓰인 arm만 채점한다.
  `score_rank.py`는 numpy/h5py가 필요해서 launcher가 고르는 MeshGraphNets interpreter(`ai_cae4all.local.toml`)로 돈다.
  모든 stage가 끝난 뒤 같은 명령을 다시 실행하면 학습 없이 다시 채점만 한다.
- **Exit code.** 0 전부 완료, 1 일부 미완료, 2 아무것도 시작하지 못함, 130 신호로 정지.
- 조정 변수: `GATE_INTERVAL GATE_SAMPLES GATE_SAMPLE_GAP EXPECTED_GPUS WATCH_INTERVAL KILL_GRACE DEAD_CONFIRM_GAP`.

알고 둘 동작 (의도된 것):

- 추론 rollout HDF5의 `meta.attrs['config_file']`과 train log의 config dump는 launch copy를 가리킨다.
  원본은 그 파일 머리의 `source`/`source_sha256` 줄에 있다.
- 시작 시 카드 하나가 이미 죽어 있으면 gate가 열리지 않는다. `EXPECTED_GPUS=7`로 시작한다.
  util이 `[N/A]`처럼 숫자가 아닌 카드도 gate를 닫힌 채로 둔다.
- Strike는 카드별 연속 실패만 센다. config 자체가 잘못된 arm이 여러 카드에 흩어져 있으면 어느 카드도 퇴역하지 않는다.
- marker가 생긴 뒤 train config를 고치면(주석만 고쳐도) 그 arm은 학습부터, infer config만 고치면 추론만 다시 돈다.
  이전 결과는 지워지지 않고 `set_aside/`로 옮겨진다. 재학습이 실패하면 추론 marker는 이미 옮겨진 뒤라 그 arm은 채점되지 않는다(`stale`이 아니라 `unverified`).
- 학습이 exit 0으로 끝난 직후 카드가 죽어도 그 stage는 ok이다.
- report의 "this run" 집계는 requeue된 stage의 마지막 결과만 센다.
- runner 쪽 libcuda가 driver와 맞지 않으면 stage가 실패할 때마다 그 카드가 lost로 판정된다(결국 전부). 먼저 `CHECK=1`/`DRY_RUN=1`로 확인한다.
- 남은 stage가 없어도 nvidia-smi는 있어야 한다. 이전 runner가 남긴 고아 stage process가 보이면 시작을 거부한다.
- gate의 nvidia-smi 호출 중 받은 신호는 최대 60 s 안에 반영된다. 할 일이 없는 worker는 10 s마다 다시 본다.
- `parse_lane`은 native parser보다 엄격하다: `gpu_ids <0-7>`이 소문자 key로 줄 맨 앞에 있어야 하고 뒤 주석을 허용하지 않는다.
- runner 자체는 표준 라이브러리만 쓴다. numpy/matplotlib은 `score_spread.py`의 그림에만 필요하고 없으면 표만 낸다.
  `score_rank.py`는 numpy/h5py가 없으면 채점하지 못하고 그 사실만 runner.log에 남긴다.
  `PYTHON=/없는/경로`면 exit 127.
- 끊긴 학습은 다음 실행에서 마지막 `.resume`부터 이어 돈다(위 "학습 도중 재개"). 최종 checkpoint를 쓰고 `.resume`을
  지우기 전에 끊기면 마지막 `.resume`부터 남은 epoch를 다시 돌아 같은 checkpoint를 다시 쓴다. `.resume`을 지운 뒤
  marker를 쓰기 전에 끊긴 stage는 처음부터 다시 돈다(드물고, 보수적인 쪽이다).
  가장 긴 stage는 Transolver 3 ex4(20.1M update)다(아래 "학습 예산 기록").
- 이어 돈 학습의 marker `started`는 마지막 시도의 시작 시각이다. 로그의 `Elapsed`는 시도들을 이어서 센다.
- 재개는 끊지 않은 실행과 같은 궤적을 잇는다. 끊지 않은 실행과 끊고 이은 실행의 최종 checkpoint가 bit 단위로 같다
  (`num_workers 2` 포함; method별 검증은 아래 "검증 범위"). 예외는 셋이다. 앞의 둘은 DataLoader worker 안에서 뽑는
  난수다. persistent worker는 그 난수열을 epoch 사이에 이어 쓰는데, 이은 실행은 새 worker로 시작하므로 그 난수열을 되살릴 수
  없다. 그래서 같은 분포에서 뽑는 통계적으로 같은 학습이지만 같은 궤적은 아니다(끊지 않은 두 실행끼리는 0).
  - SDFFlow VAE stage는 surface subsample을 worker 안에서 뽑는다. `vae_num_workers 2`(ex1–ex3)에서 VAE를 끊고 이으면
    작은 합성 시험 두 번에서 VAE checkpoint가 3–5e-3, 그 VAE를 쓰는 FM도 같은 크기로 달랐다.
    FM stage만 끊긴 경우는 bit 단위로 같다.
  - HI-MGN-V와 cHI-MGNflow는 sample마다 쓸 coarsening hierarchy variant를 worker 안에서 고른다
    (`mesh_dataset._pick_hierarchy_variant`). 해당 arm은 probabilistic ex1·ex2의 `himgn_v`/`chi_mgnflow` 4개로,
    `num_workers 2`·`hierarchy_variants 2`다. 작은 합성 시험에서 checkpoint가 HI-MGN-V 9.5e-3, cHI-MGNflow 4.4e-2
    달랐고, `hierarchy_variants 1`이면 bit 단위로 같았다. bit 단위로 맞추려면 variant를 (seed, epoch, sample) 함수로
    골라야 하는데, 그러면 재개를 끈 학습이 뽑는 variant도 바뀌므로 하지 않았다.
  - FNO(`grid_sampler` backward)와 SDFFlow(memory-efficient SDPA backward)는 GPU kernel 자체가 비결정적이라, 끊지 않은
    두 실행끼리도 1–2 ULP 다르다. 이은 실행도 그 폭 안에 든다.

## 채점과 순위 (`score_rank.py`)

```bash
python configs/campaigns/dataset_matrix/score_rank.py                                  # 전 사례
python configs/campaigns/dataset_matrix/score_rank.py --examples deterministic/ex6     # 일부
python configs/campaigns/dataset_matrix/score_rank.py --out output/dataset_matrix/_campaign/all
python configs/campaigns/dataset_matrix/score_rank.py --no-provenance                  # runner 없이 만든 출력
```

Runner는 자기 머신의 사례만 `--examples`로 넘긴다. 다른 머신의 arm은 그 머신의 `output/`에 있으므로,
**두 머신을 합친 순위**는 두 `output/dataset_matrix/` 트리를 **marker(`_campaign/<machine>/done/`)까지 포함해**
한곳에 모은 뒤 위 첫 명령을 손으로 한 번 돌려 얻는다. marker가 없으면 전부 `unverified`가 된다.

- **출처 검증(provenance).** arm은 runner가 두 stage를 **지금의 config로** 끝냈을 때만 채점한다. 조건은 셋이다:
  어느 한 머신의 `done/`에 train·infer marker가 모두 있고 둘의 sha256이 지금 config와 byte 단위로 같다;
  추론이 학습 종료 뒤에 시작했다; 채점하는 예측 파일이 모두 그 추론 시작 뒤에(2 s 여유) 쓰였다.
  runner가 stage를 건너뛰는 기준과 같은 규칙이라, 한 표 안에 현재 arm과 이전 실행의 잔여물이 섞이지 않는다.
  어긋나면 `stale`(다른 config, 이전 checkpoint, 이전 예측), marker가 아예 없으면 `unverified`다.
  손으로 만든 출력은 `--no-provenance`로 채점한다(출처 검사를 모두 끈다).
- **읽는 것.** 각 infer config가 가리키는 정답(`infer_dataset`, LSH-VAE는 `dataset_dir`)과 예측
  (`inference_output_dir`의 `rollout_sample<id>_steps<n>.h5`, LSH-VAE는 `output_dir/reconstructions.h5`).
  `_vaesample` 파일은 읽지 않는다. 같은 sample의 rollout이 여럿이면 가장 최근 것을 쓰고 detail에 적는다.
- **채점 행.** 물리 출력 행 `3:3+output_var`만. 좌표와 node-type 행은 writer가 그대로 복사하므로 채점하지 않는다.
  rollout의 `output_var` 속성이 config와 다르면 채점하지 않는다(mismatch).
- **채점 프레임.** 위 "채점 프레임 규칙" 그대로다. 정적 사례는 그 1장(MGN/operator rollout의 frame 0은 0 입력이므로 마지막 frame),
  AR 사례는 `t=1..T-1`, ex6만 **모든 method**를 `t=2..400`으로 채점한다.
- **지표.** sample·채널마다 `rel_l2 = ‖pred−gt‖/‖gt‖`, `nrmse = rmse/std(gt)`, `r2`. sample 값은 채널 macro 평균
  (채널 크기 차이가 skill로 계산되지 않도록), arm 값은 sample 평균이고 median을 옆에 적는다.
  CSV에는 채널별 값과 pooled 값(`*_pooled`)도 있다.
- **free node 열(ex4, ex5, ex6).** 이 세 사례는 경계 node를 writer가 주어진 상태에서 그대로 복사하므로 전체 node 평균이
  오차를 희석한다. 그래서 MeshGraphNets가 학습하는 node type(ex4 NORMAL·OUTFLOW = 0, 5; ex5·ex6 NORMAL = 0)만으로 계산한
  `rel_l2_free`, `nrmse_free`를 옆에 적는다. node type은 sample마다 정답 파일 마지막 행의 frame 0에서 읽는다.
  이 열은 참고용이고 순위에는 쓰지 않는다.
- **순위.** 사례 안에서 `rel_l2`가 낮은 순이며 값이 정확히 같으면 평균 순위를 준다. 모든 held-out sample이 있고 모든 값이
  유한하며 출처가 확인된 arm만 채점된다. 나머지(missing / incomplete / mismatch / non-finite / stale / unverified / error)는
  사유와 함께 표에 남고 **공동 꼴찌** 순위 (k+1+N)/2를 받는다(채점된 arm k개, 전체 N개). 빠진 route가 조용히 사라지지 않고
  그 method의 점수를 깎는다. 채점된 arm이 없는 사례는 순위를 매기지 않는다. 같은 방식의 순위를 `nrmse`로도 낸다(`rank_nrmse`).
- **정규화 순위와 종합표.** LSH-VAE가 있는 4사례는 7개, 나머지는 6개 method가 겨루므로 순위를 (rank−1)/(N−1)로
  정규화한다(0 최고, 1 꼴찌). 종합표는 method별 **평균 정규화 순위**(`nrank rel_l2`, `nrank nrmse`), 순위에 든 사례 묶음 수 /
  그 method가 도는 묶음 수, 1위 횟수다. ex3_full과 ex3_mid는 같은 CRM 문제의 두 해상도이므로 `ex3 (CRM)` 한 묶음으로 세고
  그 값은 둘의 평균이다(그러지 않으면 CRM이 두 번 계산된다). 1위는 묶음 값이 가장 낮은 method이며 동률이면 나눠 갖는다.
- **`updates` 열.** generate.py가 manifest에 적은 arm별 optimizer update 수(아래 "학습 예산 기록")다. 예산이 같다는 것은
  비교가 공정하다는 뜻이지 그 arm이 수렴했다는 뜻이 아니다(수렴은 아래 "수렴 확인"으로 본다).
- **`val 60-80%` 열과 수렴 절.** 각 arm의 학습 로그에서 계획 epoch 60%→80% 사이 val loss 변화율과 flag를 순위표 옆에 적고,
  `ranking.txt` 끝의 수렴 절과 `convergence.csv`에 stage별로 자세히 적는다(아래 "수렴 확인" 1). 순위에는 쓰지 않는다.
- **작은 held-out 세트.** ex1은 1개, ex2는 5개다. 이 사례 머리에는 그 사실이 찍히며, 근소한 차이를 우열로 읽지 않는다.
- **Probabilistic.** HI-MGN-V와 cHI-MGNflow의 추론이 쓰는 `spread_metrics.json`의 `crps_norm`(낮은 순)으로 deterministic과
  같은 규칙(출처 검증, 공동 꼴찌, 정규화 순위)으로 순위를 매기고 `sd_ratio`, `spread_skill`을 옆에 적는다. json이 추론 시작보다
  오래됐으면 `stale`이다. 전체 calibration 표는 `score_spread.py`다. geometry 사례는 method가 하나라 순위가 없고 수렴 절에만 나온다.
- Exit code는 순위 대상 arm 중 `ok`가 아닌 것이 하나라도 있으면 1이다(runner는 이 값을 참고로만 적는다). 수렴 flag는 exit code를 바꾸지 않는다.

## 학습 예산 기록 (update 수)

2026-09-25 개정. 예산은 epoch 수가 아니라 **optimizer update 수**로 판단한다:
stage마다 update = ⌈학습 표본 × window / batch⌉ × epochs (카드 1장 기준. 학습 표본 = seed-42 split의 0.8 S;
window = T−1, 정적 사례는 1). generate.py가 이 값을 arm마다 `manifest.json`의
`updates`(두 stage arm은 `updates_note`에 stage별 값)로 적고 `score_rank.py`가 순위표 옆에 찍는다.

**규칙.** 각 slot의 모든 method는 그 dataset을 쓴 원문의 예산을 **최소한** 받는다. 모자란 곳만 올렸고 넘치는 곳은
깎지 않고 아래에 기록한다. slot 기준은 다음과 같다(generate.py `CASES` 위 주석, 각 config의 머리 주석에도 적힌다):

| slot | 학습 표본 × window | epochs × batch | update | 기준 |
|---|---|---|---:|---|
| ex1 | 80 × 1 | 5000 × 1 | 400k | HI-MGN 2D 정적 열탄성: "5000 epochs, a batch size of 1", 모든 비교 모델 공통 |
| ex2 | 40 × 49 | 500 × 1 (MGN/HI-MGN 100 × 1) | 980k (196k) | HI-MGN 3D 동적 접촉 100 epochs가 기준, 나머지는 과제 공통 500 유지 |
| ex3_full, ex3_mid | 84 × 1 | 1250 × 1 | 105k | HI-MGN NASA-CRM 1000 epochs × 학습 105개 = 105k를 우리 84개로 맞춤 |
| ex4 | 80 × 599 | 420 × 2 | 10.06M | MGN CylinderFlow 10M step × batch 2 |
| ex5 | 80 × 399 | 627 × 2 | 10.01M | MGN DeformingPlate 10M step × batch 2 |
| ex6 | 80 × 400 | 313 × 1 | 10.02M | MGN FlagSimple 10M step × batch 1 |
| ex7 | 643 × 1 | 500 × 1 | 321.5k | 원문 예산 없음, 과제 공통 |
| ex8 | 800 × 1 | 625 × 1 | 500k | Transolver elasticity 500 epochs × batch 1 × 학습 1000개 = 500k를 우리 800개로 맞춤 |
| ex9 | 720 × 19 | 500 × 8 | 855k | 원문 예산 없음, 과제 공통 |

ex3과 ex8은 원문보다 학습 표본이 적다(ex3 105 → 84, ex8 1000 → 800; 나머지는 val/test로 뗀다). 그래서 둘 다 epoch를
그만큼 늘려 원문 update 수에 맞췄다(1000 × 105 = 1250 × 84 = 105k, 500 × 1000 = 625 × 800 = 500k).

arm별 update 수(`manifest.json`의 `updates`):

| slot | NO 3종 (DeepONet, Point-DeepONet, FNO) | Transolver 3 | MGN, HI-MGN | LSH-VAE (VAE + LC) |
|---|---:|---:|---:|---:|
| ex1 | 400k | 400k | 400k | — |
| ex2 | 980k | 980k | 196k | — |
| ex3_full, ex3_mid | 105k | 105k | 105k | 235k (210k + 25k) |
| ex4 | 10.06M | **20.13M** | 10.06M | — |
| ex5 | 10.01M | **20.01M** | 10.01M | — |
| ex6 | 10.02M | **16.0M** | 10.02M | 75k (50k + 25k) |
| ex7 | 321.5k | 321.5k | 321.5k | — |
| ex8 | 500k | 500k | 500k | — |
| ex9 | 855k | **6.84M** | 855k | 675k (450k + 225k) |

| probabilistic (batch 16) | HI-MGN-V (1000 epochs, 한 stage) | cHI-MGNflow (AE 500 + flow 500) |
|---|---:|---:|
| ex1 (57 × 100) | 357k | 357k (178.5k + 178.5k) |
| ex2 (1400 × 19) | 1.663M | 1.663M (831.5k + 831.5k) |

geometry(SDFFlow)는 `updates`를 적지 않는다: VAE 322.5k / 58.5k / 131k, FM 13.5k / 2.5k / 8.5k(ex1 / ex2 / ex3; ⌈학습 형상 / batch⌉ × epochs, DataLoader에 `drop_last`가 없다).

- **2026-09-25에 올린 곳.** NO 3종 ex1(80k → 400k, 2000 × 2 → 5000 × 1), T3 ex1(160k → 400k), ex8 전 method
  (NO 3종·MGN·HI-MGN 100k → 500k, 2000 × 16 → 625 × 1; T3 400k → 500k, 2026-09-26), ex5 전 method(8.0M → 10.01M, 627 epochs;
  T3는 batch 1이라 20.01M), ex6 NO 3종·MGN·HI-MGN(8.0M → 10.02M, 313 × 1), ex3 전 method(84k → 105k, 1250 epochs),
  LSH-VAE 3사례(VAE·LC 모두 5000 epochs, LSH-VAE 원문 Table 1).
- **깎지 않은 초과 예산.** T3는 원문 recipe대로 항상 batch 1이라 같은 표본 제시 수에서 update가 batch 2 method의 2배다:
  ex4·ex5 2×, ex9 8×(다른 method batch 8). ex6 T3는 500 epochs를 유지해 1.6×. ex2는 NO 3종·T3가 과제 공통 500 epochs라
  MGN/HI-MGN 원문 예산(100)의 5×. 이 차이가 순위에 영향을 줄 수 있으므로 표의 `updates` 열과 함께 읽는다.
- **적은 쪽.** SDFFlow FM(ex2 2.5k, ex3 8.5k, ex1 13.5k)이다. val loss가 일찍 바닥을 찍고 다시 오르는 것이 측정되어
  `fm_best.pth`를 쓰므로 에폭을 늘리지 않았다. LSH-VAE LC는 `drop_last=True`라 epoch당 ⌊학습 표본/16⌋ update이다.
- **Warmup.** mesh/operator route는 `warmup_epochs 5`(epoch 수와 무관), Transolver 3는 epoch의 5%(ex1 250, ex3 62,
  ex4 21, ex5 31, ex8 31, 나머지 25)다.
- **긴 쪽.** ex4–ex6의 AR 사례는 10M 이상이고 T3는 16–20M이다. 중단되면 마지막 `.resume`(15분 간격)부터 이어 돈다.

### 수렴 확인 (예산이 같다고 수렴한 것은 아니다)

update 수를 맞추는 것은 method끼리 **공정하게** 비교하기 위해서이고, 그 예산이 **충분한지**는 loss 곡선으로 따로 본다.
모든 route가 cosine schedule로 LR을 1e-8까지 내리므로, 마지막 구간의 train/val loss는 예산이 모자라도 평평해진다.
평평한 끝부분만 보고 수렴했다고 판단하지 않는다. 대신 다음 두 가지를 본다.

1. **cosine 끝 20% 이전의 val 기울기 (`score_rank.py`가 자동으로 적는다).** 학습 80% 지점까지 val loss가 아직 뚜렷하게
   내려가고 있었다면 예산이 모자랐을 수 있다.
   - **읽는 로그.** train config의 `log_file_dir`(두 stage route는 `vae_`/`lc_`/`fm_log_file_dir`)이다. LSH-VAE는
     `vae.log`/`lc.log`, SDFFlow는 `vae.log`/`fm.log`, cHI-MGNflow는 한 `train.log`의 `[AE]`(`ae_epochs`)와 `[Prior]`
     (`training_epochs`) 줄을 따로 본다. 로그는 **마지막 실행만** 읽는다: append 로그의 `==== Run` 머리, 또는 epoch가
     줄어드는 곳부터 새 실행이다. 단 `==== Resume [TAG] at epoch K` 뒤는 같은 실행의 연속이다: 그 뒤 처음 적힌
     epoch부터의 이전 줄(state 이후에 적혔다가 다시 도는 epoch)만 버리고 나머지는 이어 붙인다. Transolver 3 multi-GPU 로그는 검증하지 않은 epoch에도 직전 val 값을 다시 적으므로
     `val_interval` epoch(와 마지막 epoch)만 쓴다. 다른 route는 검증하지 않은 epoch에 `Valid skipped`라고 적는다.
   - **지표.** 계획 epoch를 5%씩 20개 구간으로 나누고 구간마다 loss의 median을 잡는다. `val 60-80%`는 55–60% 구간에서
     75–80% 구간으로의 val 변화율, `train 60-80%`는 같은 것의 train loss다. `tail`은 마지막 구간(95–100%)이 가장 낮은
     구간보다 얼마나 높은지, `best`는 그 가장 낮은 구간, `final val`은 마지막으로 기록된 val이다.
   - **Flag.** `F` val 60-80%가 −10%보다 더 떨어짐(예산 의심). `R` deterministic arm(마지막 epoch를 쓴다)에서 tail이 +5% 초과.
     `I` 로그가 계획한 마지막 epoch 전에 끝남. `N` NaN/Inf loss. `X` 계획보다 epoch가 많음. `O` 로그가 기록된 이번 학습의
     시작보다 오래됨(`skip_completed_stages`로 재사용된 stage 또는 잔여물). `?` 로그가 없거나 epoch 줄이 없음.
   - **표시.** 순위표의 `val 60-80%` 열은 stage가 둘 이상이면 가장 많이 떨어진 stage의 값과 이름, 그리고 모든 stage의 flag다.
     수렴 절과 `convergence.csv`는 stage마다 한 줄이며 geometry arm도 들어간다. `ranking.csv`에는
     `conv_val_60_80`/`conv_stage`/`conv_flags`가 붙는다.
   - **해석 주의.** cosine schedule에서는 후반 하강의 일부가 LR annealing 자체의 효과다. 그래서 `F`는 의심 표시일 뿐
     부족 판정이 아니고, 판정은 2의 두 배 시험으로 한다. 임계값 −10%/+5%는 잠정값이며 첫 실행의 실제 곡선을 보고 다시 정한다.
     구간당 검증 epoch가 적은 arm은 `val 60-80%`가 사실상 두 epoch의 비교라 잡음이 크다: ex2 MGN/HI-MGN(100 epochs,
     `val_interval 5`)은 구간당 1개, cHI-MGNflow(500, 10)는 2–3개다. 그곳은 모든 epoch로 계산하는 `train 60-80%`를 함께 본다.
2. **예산 두 배 시험.** 의심되는 slot 하나에서 epochs만 두 배로 올려 다시 돌린다. held-out 점수가 크게 좋아지면
   그 slot의 순위는 예산 부족을 반영하는 것이다.

### 마지막 epoch vs best-val checkpoint

deterministic mesh/operator route(MGN, HI-MGN, Transolver 3, NO 3종)는 학습이 끝난 **마지막 epoch**의 checkpoint를
저장하고(`use_ema True`이므로 추론은 그 EMA 가중치), LSH-VAE도 마지막 epoch를 저장한다. best-val이 항상 낫지는 않으므로
이 선택을 유지한다.

- LR은 warmup 뒤 **한 번의** cosine 주기(`T_0 = epochs − warmup`, `T_mult 1`)로 1e-8까지 내려간다. 그래서 마지막 epoch가
  annealing을 끝낸 해이다. 그보다 앞의 best-val 시점은 LR이 아직 큰 상태라 그 뒤의 annealing 이득을 버린다.
  (restart가 있는 schedule이었다면 마지막 epoch가 restart 직후일 수 있어 판단이 달라지지만, 이 설정에는 restart가 없다.)
- val 세트가 작다(ex1 학습 파일의 val 10개, ex2 5개). 거기서 고른 best-val은 잡음에 맞춘 선택이 되기 쉽다.
- AR 사례의 val loss는 한 step 예측 오차이고 채점은 긴 rollout이다. 한 step val이 가장 낮은 checkpoint가 rollout에서도
  가장 좋다는 보장이 없다.
- 예외: 과적합이 측정된 SDFFlow FM은 `fm_best.pth`, probabilistic 두 route는 `best_by crps`로 고른다.
  마지막 epoch를 쓰는 arm이 학습 후반에 val이 다시 오르면 수렴 절이 `R`(마지막 구간이 가장 낮은 구간보다 5% 넘게 높음)로
  표시한다(위 "수렴 확인" 1). 그것은 과적합 신호로 따로 기록한다.

## 논문 근거와 구현상 조정

여기서 baseline은 논문에 근거한 **native implementation adaptation**이며 모든 dataset에 대한 원문 재현이나
검증된 최적 hyperparameter라는 뜻이 아니다. Loss, optimizer, sampling, mesh adapter가 달라질 때 이를 구분한다.

| 기법 | 근거 | 이번 설정과 차이 |
|---|---|---|
| MGN | [Learning Mesh-Based Simulation](https://arxiv.org/abs/2010.03409) | latent 128, message passing 15, `weight_decay 0` (원문은 plain Adam). ex1–ex3은 MGN 원문에 없는 데이터라서 HI-MGN 원문의 MGN 비교군 예산(ex1 5000 / ex2 100 epochs, batch 1)을 따르고, ex3은 원문의 1000 epochs × 학습 105개를 이 split의 학습 84개로 옮긴 1250 epochs다(1250 × 84 = 1000 × 105 = 105k update). Eulerian field를 displacement로 해석하지 않는다. Flag(ex6)는 원문이 가속도를 예측해 두 번 적분하고 속도 history(h=1)를 넣지만, 이 설정은 의도적으로 다른 사례와 같은 1-frame delta(1차) 예측을 쓴다. 원문은 pressure(ex4)와 von-Mises stress(ex5)를 입력에 넣지 않고 매 step 값 자체를 직접 예측하는 보조 출력으로 다루지만, 이 runtime은 T>1에서 모든 출력이 입력이어야 하므로 (`input_var == output_var`) 이들도 state로 입력받아 delta를 예측하고 rollout에서 누적한다. 원문 세 데이터셋은 원문의 10M step과 batch를 그대로 맞춘다: Cylinder(ex4) 80 trajectory × 599 window / batch 2 × 420 epochs = 10.06M, DeformingPlate(ex5) 80 × 399 / 2 × 627 = 10.01M, FlagSimple(ex6) 80 × 400 / 1 × 313 = 10.02M update(batch는 원문과 같은 2 / 2 / 1). 모든 학습이 GPU 1장이라 world size로 나누지 않는다. epoch 수를 500으로 "맞추지" 않는다. LR은 native 고정 schedule(linear warmup `warmup_epochs 5` → cosine, 하한 1e-8)이며 원문은 5M step에 걸친 1e-4 → 1e-6 exponential decay다. Training noise는 slot별 `std_noise`다. 정규화 단위라 채널별 σ<sub>i</sub> = `std_noise` × node std<sub>i</sub>이고, `output_var`의 모든 state 채널에 들어간다(pressure·stress도 여기서는 state 입력이라 noise를 받는다). 값은 원문 Table 2의 σ를 noise 대상 field의 학습셋 최대 node std로 나눈 것이라 그 field의 어느 채널도 원문 σ보다 크게 흔들리지 않는다: ex4 0.0416(0.02 / velocity_x 0.4812), ex5 0.126(0.003 / u_z 0.02382), ex6 0.00134(0.001 / u_y 0.7437; 공식 코드의 3e-3은 noise_gamma 0.1과 짝이고 이 runtime은 γ = 1이다). ex2·ex9는 0.01(HI-MGN 원문 값; ex9는 원문 과제가 아니다), 정적 사례는 0이다. MGN, HI-MGN, Transolver 3, NO 3종이 slot마다 같은 값을 쓴다. ex5의 world edge 반경은 최소 edge 0.017679 × `world_radius_multiplier 1.7` = 0.0300으로 원문 r<sub>W</sub> 0.03과 같다(ex2는 2.0). **코드가 필요해 바꾸지 않은 차이**: 원문은 kinematic node의 다음 step 속도를 입력으로 주고 rollout에서 경계·kinematic node를 참값으로 고정하지만, 이 runtime에는 두 기능이 없다. ex4–ex6의 모든 mesh/operator method에 같은 차이가 있다. |
| HI-MGN | [HI-MGN](https://arxiv.org/abs/2608.13827) | FPS/Voronoi seed hierarchy, learned interpolation, 2-level V-cycle. 블록 구성은 원문과 같은 `[4,6,8,6,4]`(총 28)이다. 원문은 MGN 비교군도 28로 맞췄지만(시간전진 과제에는 MGN 원문 권장값 15를 추가로 평가), 여기서는 MGN이 자기 원문의 15를 유지하므로 두 모델의 블록 수는 같지 않다. cluster 수는 원문 세 과제(ex1 2D 정적 열탄성, ex2 3D 동적 접촉, ex3 NASA-CRM)에서 원문과 같은 `[N, 5000, 100]`이다. 원문에 없는 ex4–ex9는 원문 값이 없고, 대부분 노드가 5,000개 이하라(ex8 972, ex4–ex6 약 1–2k, ex9 3,131) 각 mesh 규모에 맞춰 줄인 baseline 선택이다(ex7은 약 18만 노드지만 원문 과제가 아니므로 `4096, 256`을 유지한다). MGN과 같이 `weight_decay 0`(원문은 AdamW라고만 하고 weight decay 값이 없다. 0이면 AdamW가 Adam과 같다). noise σ는 원문 과제 ex2에서 원문과 같은 정규화 단위 0.01이다(이 runtime의 `std_noise`도 정규화 단위다; ex9도 0.01). ex4–ex6은 MGN과 같은 slot 값(0.0416 / 0.126 / 0.00134)과 같은 예산(420 × batch 2 / 627 × batch 2 / 313 × batch 1)을 쓴다. LR warmup + cosine은 HI-MGN 원문 schedule과 같다. 원문 세 과제의 학습 예산도 원문과 같다: ex1 5000 epochs, ex2 100, ex3_full·ex3_mid 1250, 모두 batch 1이다. ex3의 1250은 원문 1000 epochs × 학습 105개의 105k update를 학습 84개(21개는 val)에서 맞춘 값이다. 원문처럼 MGN 비교군도 같은 예산을 쓴다. **loss는 원문과 다르다**: 원문은 채널 가중 Huber<sub>η</sub>(식 9, η 값은 원문에 없음)이고, 이 runtime은 같은 채널 가중(합 1) MSE만 지원하므로 MSE로 학습한다. ex1(400k)과 ex3(105k)은 모든 mesh/operator method의 update 수가 같다. ex2만 다르다: 다른 method는 과제 공통 500 epochs(980k)라 HI-MGN·MGN(196k)의 5×다. 이 차이는 깎지 않고 `updates`에 기록한다. **augmentation도 원문과 다르다**: 원문은 지배 물리와 경계조건이 보존될 때 무작위 회전·반전을 썼다(과제별 변환은 원문에 없음). 이 runtime의 `augment_geometry`는 Z축 0–360° 회전과 x·y 반전 하나로 고정돼 있다. 이 변환은 ex1의 고정 edge와 heat-flux edge를 뒤바꾸고, ex2의 하중 축(Y)을 돌리며, ex3에서는 자유류 방향과 조건 row의 법선 벡터를 보존하지 않는다. 따라서 세 과제 모두 `augment_geometry False`로 학습한다. **ex3_mid mesh도 원문과 다르다**: 원문 §3.3은 454,404 노드를 122,778 노드로 subsample한 mesh(105 train / 44 test)를 썼다. 그 mesh는 edge의 27.7%가 face 3개 이상에 걸린 비-manifold 그래프라서 SDF와 시각화가 깨졌고, `rebuild_crm_mid.py`가 이를 123,217 노드 quad mesh로 교체했다(105/44와 값은 full과 동일). 모든 method가 한 slot에서 같은 데이터를 쓰도록 이 mesh를 유지한다. ex3_full(454,404)은 원문 Appendix D와 같은 mesh다. |
| Transolver 3 | [Transolver 3](https://arxiv.org/abs/2602.04940) | slice-space/tiled attention. inference는 `direct`: 현 구현의 `decoupled`는 같은 mesh로 cache와 query를 만들어 결과가 direct와 같고(embed만 두 번), T>1 rollout이 없다. 원문은 NASA-CRM(ex3, 454k 노드)을 full mesh로 학습하고 수백만 노드 benchmark에만 amortized training(subset 100k)을 쓰므로, 그보다 작은 ex2(200k)/ex7(185k)를 포함해 전 사례 full mesh. 원문은 모든 benchmark에 한 recipe를 쓰므로 전 사례 LR .001, WD .05, batch 1, warmup = `epochs // 20`(ex1 5000→250, ex3 1250→62, ex4 420→21, ex5 627→31, ex8 625→31, 나머지 500→25). **epoch 수는 ex1·ex3·ex4·ex5·ex8에서 원문과 다르다**: 원문은 모든 benchmark를 500 epoch로 학습한다(Table 9, NASA-CRM 포함). 여기서는 method 간 update 수를 같게 두려고 slot 공통 예산을 쓴다: ex1 5000(400k), ex3 1250(105k), ex4 420, ex5 627(ex4·ex5는 MGN 원문의 10M step 예산; 원문에는 시간전진 benchmark가 없다), ex8 625(원문 elasticity 500 epochs × 학습 1000개 = 500k를 학습 800개에 맞춤). T3는 원문 recipe대로 항상 batch 1이므로 다른 method가 batch 2·8인 slot에서는 update가 더 많다: ex4 20.13M, ex5 20.01M(2×), ex9 6.84M(8×). ex6은 원문의 500 epochs를 유지해 16.0M으로 다른 method(10.02M)의 1.6×다. 이번 조정은 모자란 곳만 올렸으므로 이 초과분은 깎지 않고 기록한다. ex8은 모든 method가 625 epochs × batch 1 = 500k다. CRM은 원문 full-mesh 설정에 따라 24 layers, 그 외 8 layers (depth는 의도적으로 사례별 유지); width 256, 8 heads, 64 slices, `mlp_ratio 2`(원문에 기재가 없어 공식 코드 `MODEL_KWARGS` 값을 따름). Native normalized MSE와 scheduler floor는 원문의 relative-L2/min-LR와 다르다: cosine 최소 LR은 native 코드에 고정된 1e-8이고 원문은 1e-6이다(공식 코드 `train_surface.py`는 warmup 없이 `eta_min = lr×0.01` = 1e-5, WD는 torch 기본 0.01이므로 이 config는 공식 코드가 아니라 논문 recipe를 따른다). LR 1e-3 cosine에서 1e-6과 1e-8의 차이는 학습 마지막 약 2% 구간에만 나타난다. Slice temperature는 head별 학습 스칼라(init 0.5)를 forward에서 `clamp(0.1, 5.0)`한다. 공식 T3 코드는 clamp가 없고(`softmax(logits / temperature)`), 이 범위는 v1 공식 코드의 정규 격자용 attention 관례다. 학습값이 범위 안이면 공식 코드와 같고, 벗어나면 그 head는 경계값에 고정되어 gradient가 0이 된다. 경계 도달 여부는 측정하지 않았다: 체크포인트 `model_state_dict`(및 `ema_state_dict`)의 `blocks.*.attn.temperature`가 0.1 또는 5.0이면 걸린 것이다. 좌표 정규화도 원문 출처끼리 일치하지 않는다: 논문 §A.2는 min-max(선택적으로 상수배), 공식 T3 코드는 정규화 없는 원 좌표(`--pos_norm` 미사용, 라벨만 z-score), v1/Transolver++ 코드는 채널별 z-score다. native는 `centered_isotropic`만 지원한다: 샘플마다 자기 중심을 빼고 학습셋 RMS 반경 스칼라 하나로 나누므로 종횡비는 보존하지만 샘플 간 평행이동 정보는 입력에서 사라진다. 학습의 `use_amp True`는 원문의 16-bit mixed precision(fp16 또는 bf16)을 따른다. native는 bf16 고정이며, 입력 행([좌표 | x])이 bf16으로 반올림되어 ex3 노드의 3.0%, ex7의 24~25%가 같은 입력이 된다. 그래서 ex3_full·ex3_mid·ex7은 `use_amp False`(fp32)로 학습하고 나머지 slot은 원문대로 bf16이다. 추론 코드(`inference_profiles/`)에는 autocast가 없으므로 infer config의 `use_amp`와 무관하게 모든 slot이 fp32로 추론한다. |
| DeepONet | [DeepONet](https://doi.org/10.1038/s42256-021-00302-5), [Lu et al. 2022](https://arxiv.org/abs/2111.05512) | fixed sensor branch + coordinate trunk, vector output split_both (Lu et al. 2022 §3.1.6의 두 번째 방식: branch와 trunk 출력을 모두 출력 수만큼 나눈다). Irregular mesh의 sensor interpolation은 이 저장소 adapter이다. `deeponet_activation relu`. Sensor 격자는 train bbox를 덮으며 정사각 도메인인 ex1/ex8은 32×32, 나머지 2D slot은 32×16, 3D는 16×16×8이다. 32×16(2:1)은 ex7(가로:세로 2.0)과는 맞지만 ex4(3.9), ex9(3.3), ex6(1.5)의 도메인 비율과는 다르다. `use_amp False`(fp32): bf16이면 trunk 입력 행(좌표 + condition + positional + one-hot)이 반올림되어 ex3 노드의 3.0%, ex7의 24~25%가 서로 같은 입력이 된다(좌표만 보면 37% / 40%). LR은 원문대로 1e-3, T>1 AR 사례만 1e-4. Optimizer/scheduler는 native AdamW + warmup 5 + cosine이며 config로 바꿀 수 없다 (원문: Adam, 고정 LR). |
| Point-DeepONet | [Point-DeepONet](https://arxiv.org/abs/2412.18362) | PointNet branch + SIREN trunk, mesh_state variant. 원문의 특수 geometry/SDF 실험을 동일하게 재현한다고 주장하지 않는다. 원문 recipe의 sensor 5000개는 복원 추출이라 노드가 5000개보다 적은 mesh에서는 일부 노드만 본다: ex4/5/6/8/9(727–3131 노드)는 노드의 80–99%만 본다. 그래서 이 다섯 slot은 `point_sensor_count 0`(모든 노드)이고 ex1/2/3/7은 5000을 유지한다. `point_siren_omega0 10`과 `weight_decay 1e-5`는 공식 코드 값이며 원문에는 둘 다 없다. `use_amp False`(fp32): bf16이면 trunk 입력 행(좌표 + condition + positional + one-hot)이 반올림되어 ex3 노드의 3.0%, ex7의 24~25%가 서로 같은 입력이 된다(좌표만 보면 37% / 40%). LR은 원문대로 1e-3, T>1 AR 사례만 1e-4. Optimizer/scheduler는 native AdamW + warmup 5 + cosine이며 config로 바꿀 수 없다 (원문: AdamW, batch 16, scheduler 미기재). |
| FNO | [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) | 4 layers, 2D width 32 / 3D width 20, mesh→grid→mesh adapter. `fno_use_channel_mlp False`: 원문 layer는 σ(W v + K v)이고 layer별 channel MLP가 없다. Width와 modes는 원문 값이다: 2D는 modes 12×12 · width 32(Darcy/2D NS, 2.4M params), 3D는 축별 modes 8 · width 20(3D NS, 6.6M). 3D의 얇은 마지막 축은 이전 격자의 0.375배 값인 3 / 7로 제한했다(ex3 8×8×3, ex5 8×8×7; ex2 8×8×8). 모든 modes는 Nyquist(res // 2, 마지막 축은 res // 2 + 1) 안이다. Projection 폭은 width와 같고 GELU이며, native `mesh` variant에 고정이다(원문 128, ReLU; 원문 projection은 `fno_variant paper_darcy`에만 있다). Grid는 train bbox를 축별로 덮으므로 shape을 도메인 비율에 맞춘다(2D 약 4k, 3D 약 16k 점): ex4/ex9 128×32, ex7 96×48, ex6 78×52, ex1/ex8 64×64; ex2 26³, ex3 48×44×8(z 최소 8), ex5 24×32×20. 원본의 structured-grid 직접 입력과 구분한다. FFT 안정성을 위해 AMP를 끈다. LR은 원문대로 1e-3, T>1 AR 사례만 1e-4. Optimizer/scheduler는 native AdamW + warmup 5 + cosine이며 config로 바꿀 수 없다 (원문: Adam, 100 epoch마다 LR 절반). |
| LSH-VAE | [LSH-VAE](https://doi.org/10.1007/s00366-023-01916-6) | 7 hierarchy levels, main latent 32, hierarchical latent 8. VAE와 LC 모두 5000 epochs(원문 Table 1). 원문 Eq. 14의 KL ramp는 `kl_warmup_start_frac 0.3`(기본값)과 `kl_warmup_epochs 3500`(= 0.7 × 5000)이라 학습 끝에서 β가 목표값에 닿는다(기본 0이면 0.8n에서 끝난다). 작은 width와 native AdamW를 사용하며 원문의 Adamax/107-layer 상세 recipe와 다르다. `alpha 1e6`: 원문 Eq. 13의 α/β_target ≈ 1e6이고 `beta_target` 기본값이 1.0이다. LC는 저장소 확장이다. 모든 GroupNorm은 1 group(샘플별 channel×time LayerNorm)이다: 8 group이면 T=1(ex3)에서 16-channel 층의 group당 값이 2개라 ±1로 양자화된다. decoder 출력 head는 `Conv1d → Tanh`로 정규화가 없다(Tanh 직전 GroupNorm은 샘플별 진폭을 지워, 크기만 다른 두 응답이 같게 복원됐다). 계층 KL은 원문 Eq. 15의 KL(q(z_i|z<i,x) ‖ p(z_i|z>i))를 residual 형태 ½Σ(Δμ²/σ² + Δσ² − log Δσ² − 1)로 계산한다(q = N(μ+Δμ, σ²Δσ²), p = N(μ, σ²)). LC 입력은 MinMax만 하고 LayerNorm하지 않는다(서로 다른 조건 사이의 LayerNorm은 ex3 6개 비행조건의 샘플별 평균·표준편차를 지웠다). |
| HI-MGN-V | [InfoVAE](https://arxiv.org/abs/1706.02262), [Flow Matching](https://arxiv.org/abs/2210.02747) | 저장소의 conditional-prior variational hierarchy. Global latent 32(hyperparameter_sweep/SAOI 학습은 16이었으므로 그 결과를 그대로 옮길 수 없다), MMD, auxiliary reconstruction, conditional FM; posterior reconstruction과 prior sampling 성능을 구분해야 한다. 한 stage로 `training_epochs 1000`을 학습해 cHI-MGNflow의 AE 500 + flow 500과 update 수가 같다. `prior_freeze_epoch 700`부터 prior만 학습하고, checkpoint는 `best_by crps`로 고른다. 단일 논문의 동일 명칭 recipe로 오인하지 않는다. `alpha_recon 1000`/`lambda_mmd 1`은 hyperparameter_sweep base 그대로다. InfoVAE의 기준은 숫자가 아니라 "loss on X와 loss on Z가 비슷한 크기가 되도록 λ를 고른다"이며, 원문 λ=1000은 784픽셀 합산 Bernoulli NLL에 맞춘 값이라 정규화 필드의 평균 MSE인 여기로 옮길 수 없다. 이 비율에서 두 항이 균형을 이루는지는 측정되지 않았고, sweep2의 `mmd10`/`mmd100` arm이 그 판단 근거다(Adam + group별 clip이라 비율만 의미가 있다). |
| cHI-MGNflow | [Latent Diffusion Graph Networks](https://arxiv.org/abs/2504.02843) | coarse-node AE latent 4 channels, KL 1e-6, AE 500 epochs 후 latent flow 500 epochs. `val_flow_steps 30` = `flow_steps 30`이라 validation이 추론과 같은 sampler로 채점된다. 저장소 HI hierarchy/flow adaptation이며 원문의 모든 architecture와 동일하지 않다. **latent 채널은 원문과 다르다**: LDGN은 세 과제 모두 F<sub>L</sub>=1이다(Appendix Table 1; 3필드 u,v,p인 ELLIPSEFLOW도 1). 여기서는 4를 유지한다. 원문의 F<sub>L</sub>=32·KL 1e-3/1e-8은 VGAE **베이스라인** 값이지 LDGN 값이 아니다. 전체 압축비는 F<sub>L</sub>뿐 아니라 latent mesh 노드 수(원문 1D ≈4×, 2D ≈16×)에도 달려 있고, 이 저장소의 voronoi coarsest 노드 비율은 측정하지 않았으므로 원문과 압축비가 같다고 가정하지 않는다. |
| 확률 추론 공통 | — | HI-MGN-V와 cHI-MGNflow의 infer config는 `num_vae_samples` = held-out scene 수(probabilistic/ex1 18, ex2 250)이다. 이는 **scene당** draw 수이며, 각 scene의 ensemble 크기를 그것이 채점되는 held-out 세트 크기에 맞춘 것이다. |
| SDFFlow | [3DShape2VecSet](https://arxiv.org/abs/2301.11445), [Flow Matching](https://arxiv.org/abs/2210.02747) | ex1은 512×32 latent token set(원문 M=512, C0=32; width는 256으로 원문 C=512보다 작다), FPS encoder, attention SDF decoder, DiT flow. KL은 latent 원소의 **합**에 곱해진다. ex1 `kl_weight 0.000001` × 16,384원소 = 0.0164로 원문(공식 코드: 원소 **평균** KL × 1e-3)의 약 16배다. ex2–ex3는 32×32 = 1,024원소에 `kl_weight 0.0000000001`이라 평균 기준 약 1.0e-7, 원문의 약 1/10,000로 사실상 KL이 없다. 근거는 ex3(MCB, 1,024원소) KL sweep이다: 1e-4는 epoch 50–75에서 posterior collapse, 1e-6·1e-8·1e-10은 1,024차원 모두 활성, valid L1 6.8–7.0e-3, sbr는 1e-4 .375, 1e-6 .81, 1e-10 .91. ex1의 1e-6은 이 sweep 결과를 옮긴 값이며 ex1에서 sweep하지 않았다. infer는 seed-42 parent split의 test 개수(209)만큼 무조건부 생성해 참조 집합과 같은 크기로 비교할 수 있게 한다. Surface/normal/eikonal loss와 condition dropout은 저장소의 조합이며 이 조합 전체를 원문 recipe로 주장하지 않는다. **학습 스케줄도 원문과 다르다**: 원문은 AE batch 512 / 1600 epochs / lr<sub>max</sub> 5e-5 (warmup 80 epochs), diffusion batch 256 / 8000 epochs / lr<sub>max</sub> 1e-4 (warmup 800 epochs), 둘 다 cosine으로 1e-6까지다. 여기서는 VAE batch 8 / 1500 epochs / lr 1e-4 (warmup 20), FM batch 64 / 500 epochs / lr 1e-4 (warmup 10), cosine으로 1e-8까지다. ex1의 학습 형상 1,713개 기준으로 원문 update는 AE ⌈1713/512⌉ × 1600 = 6,400, diffusion ⌈1713/256⌉ × 8000 = 56,000이고, 여기서는 VAE 215 × 1500 = 322.5k(약 50배), FM 27 × 500 = 13.5k(약 1/4)다. 이 스케줄은 ex1 값이며 ex3 VAE는 1000 epochs다. 원문은 ShapeNet-v2로 학습했고, 이 저장소에서는 FM val loss가 일찍 바닥을 찍고 다시 오르는 것이 측정되었으므로(ex4: epoch 200 0.660 → epoch 999 2.127) 에폭을 원문만큼 늘리지 않고 infer는 `fm_best.pth`를 읽는다. |

메모리/계산량을 고려한 grid, sensor 수, batch 크기이며 성능 튜닝 결과는 아니다.

소표본 정적 사례는 epoch 수가 아니라 update 수(= ⌈학습 표본 × window / batch⌉ × epochs)로 판단한다. ex1은 5000 × batch 1 = 400k, ex3은 1250 × batch 1 = 105k(HI-MGN 원문 CRM 예산), ex8은 모든 method가 625 × batch 1 = 500k(Transolver elasticity 500 epochs × 학습 1000개를 학습 800개에 맞춤)이다. ex2·ex4–ex6은 궤적 수가 적지만 window가 많아 update가 충분하다(ex2 980k, ex4–ex6 10M 이상).
학습용 8-GPU 박스의 카드는 B300이며, 그 장치에서 전체 train/infer 메모리는 아직 실측하지 않았다.

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

완료한 검사 (2026-09-23, GINO 삭제 후 150개 config 기준):

- 150개 config의 generated-source 일치(`generate.py --check`), launcher schema(`audit.py`): 오류/경고 없음.
- 13개 mesh 사례의 train/infer 전체 sample shape, field 수, T 검사(`audit.py --data`); 유한값은 결정론적 node subset만 검사.
- Native parser 139개 통과(`audit.py --native` 7개 runtime). Transolver 추론 11개는 아직 없는 실제 학습 checkpoint가 필요하여 해당 검증을 유보.
- Runner launch copy 150개가 launcher parser와 각 native parser 양쪽에서 원본과 `gpu_ids`만 다르게 읽힘.
- 손으로 관리하는 SDFFlow ex2/ex3 train config의 launch copy가 launcher preflight(`--check`) 통과.
  infer config는 학습 산출 checkpoint가 아직 없어 PATH-INPUT-001만 나며, 그 경로는 train 출력 경로와 같다.
- LSH 후보 4사례의 전체 train/infer 고정 topology 검사, derived VDS/CSV read-back.
- 수정한 geometry, node type, grouped split, train-only scaler에 대한 회귀 테스트.
- Transolver 3 tiled/decoupled 출력 일치 및 amortized forward/backward의 target 대응 테스트.

2026-09-25 재점검 (runner 수정 전후):

- `generate.py --check`, `audit.py`(150개), `audit.py --data`, suite 전체 `--audit-configs`(392개, 오류 0;
  경고 54개는 모두 이 matrix 밖의 SAOI_run/hyperparameter_sweep).
- 두 머신의 launch copy 78개 train config 전부 launcher `--check` 통과(GPU 1장 box에서 `gpu_ids 0`으로).
  원본 config에 대한 `CHECK=1`은 원본 `gpu_ids`(0–7)를 그대로 검사하므로 GPU가 8장 미만인 box에서는 ENV-CUDA-002가 난다.
- `DRY_RUN=1`: 135는 41 arm / 82 stage, 136은 37 arm / 74 stage.
- `score_rank.py`: 합성 데이터(정적·AR·ex6 예외·LSH·결측·NaN·output_var 불일치·probabilistic)에서 독립 재계산과 일치.
  실제 repo에서 74개 채점 대상 arm 전부의 config와 정답 파일을 열고, 아직 예측이 없으므로 전부 missing으로 표시.
- Neural_Operator 단위 테스트 30개 파일 통과.

2026-09-25 개정 후 재점검 (update 예산, `std_noise`, SDFFlow KL, runner marker/set-aside 개정 뒤):

- `generate.py --check`(생성 75쌍 + 손으로 관리하는 SDFFlow 3쌍), `audit.py`(150개), `audit.py --data`,
  suite 전체 `--audit-configs`(393개, 오류 0; 경고 54개는 모두 이 matrix 밖의 SAOI_run/hyperparameter_sweep).
- `audit.py --native` 139개 통과(MGN 44, MGN-V 4, HI_MGNFlow 4, NO 66, Transolver 11, SimulGenVAE 8, SDFFlow 2).
  Transolver 추론 11개의 checkpoint 의존 검사는 계속 유보.
- 두 머신의 launch copy 78개 train config 전부 launcher `--check` 통과(GPU 1장 box에서 `gpu_ids 0`으로).
- `DRY_RUN=1`: 135는 41 arm / 82 stage, 136은 37 arm / 74 stage, 둘 다 종료 코드 0이며 파일을 쓰지 않음.
- `score_rank.py` 합성 테스트 59항목: 정적·AR·ex6 frame 예외·LSH, free-node 열(독립 계산과 일치), 결측·중단·NaN·
  frame 수/`output_var` 불일치, 중복 rollout(최신 파일 채점), probabilistic CRPS, CRM 묶음, `updates` 열,
  provenance(sha 불일치, 학습 전 추론, 빈/반쪽 marker, 시각 여유, 예측 mtime)와 tied-last 순위.
- `run_matrix.py` 상태 테스트 58항목: 실제 plan에서 74개 예측 디렉터리가 config에 적힌 296개 경로 중 어느 것도
  포함하지 않고 서로 겹치지 않음; 가짜 root에서 marker 판정, set-aside 대상·보호 경로·거부 조건, 삭제 없는 이동,
  이름 충돌 시 `.2` 접미사, LSH started record와 재개, 하나라도 거부되면 아무것도 옮기지 않음.

2026-09-26 (ex8 625 epochs, 수렴 리포트):

- ex8 개정 뒤 `generate.py --check`, `audit.py`(150개), suite 전체 `--audit-configs`(393개, 오류 0, 경고 54개는 이전과 같음),
  native parser(MGN 44, NO 66, Transolver 11), ex8 launch copy 6개 launcher `--check` 통과.
- 수렴 리포트 합성 테스트 56항목: 모든 trainer의 로그 줄 형식(MGN single/multi-GPU, Transolver 3 multi-GPU의 반복 val,
  HI-MGN-V, cHI-MGNflow `[AE]`/`[Prior]`, LSH-VAE `vae.log`/`lc.log`와 이전 실행, SDFFlow hybrid/FM, model_split의
  train-only)과 모든 flag, 60-80%·tail·best 구간을 numpy 독립 재계산과 대조. 순위와 exit code는 그대로.
- 실제 manifest 78개 arm(88 stage) 전부 로그 경로가 `output/` 아래이고 계획 epoch가 읽히며, 20개 구간 모두에 검증 epoch가
  1개 이상 들어감(가장 적은 곳은 ex2 MGN/HI-MGN 구간당 1개, cHI-MGNflow 2개). runner hook이 두 머신 모두 geometry를 포함해 `convergence.csv`를 씀(135: 45 stage, 136: 43 stage).
- `DRY_RUN=1`: 135는 41 arm / 82 stage, 136은 37 arm / 74 stage, 둘 다 종료 코드 0이며 파일을 쓰지 않음.

2026-10-07 (학습 도중 재개):

- 7개 학습 repo(MGN, HI-MGN-V, cHI-MGNflow, NO, Transolver, SDFFlow, SimulGenVAE)를 process를 죽여서 시험했다.
  끊지 않은 실행 A, 그 반복 A2(noise floor), 매 epoch state를 쓰다가 epoch K 뒤에 죽이고 다시 실행한 B를 두고,
  최종 checkpoint(stage별 bank 포함)의 모든 tensor를 비교했다. 모든 시나리오에서 B 뒤에 `.resume`/`.resume.tmp`가
  남지 않았고, 재개를 끈 A/A2는 resume 파일을 만들지 않았으며, B 로그는 `==== Resume [TAG] at epoch K` 한 줄로 이어졌다.
- floor 0이고 A와 B가 bit 단위로 같았다: MGN(`num_workers` 0/2), Transolver, point_deeponet/deeponet(`num_workers` 0/2),
  SimulGenVAE(VAE·LC 각각 끊기, EMA+fp16, VAE `num_workers 2`), HI-MGN-V(joint 구간, prior-only tail, moment 구간,
  tail checkpoint 위의 오래된 state, multiscale 끔, `num_workers 2`+`hierarchy_variants 1`), cHI-MGNflow(AE, Prior,
  Prior epoch 0, AE→Prior 경계 직후, 단독 `train_ae`/`train_prior`, `num_workers 2`+`hierarchy_variants 1`),
  SDFFlow(SDPA math 고정, `num_workers 0`). 나머지는 위 "알고 둘 동작"의 예외 셋이다.
- `train_prior`의 `ae_checkpoint`가 끊긴 뒤 다시 쓰이면 state가 `.resume.stale`로 밀려나고 그 stage는 epoch 0부터 돈다.
- 최종 helper로 단위 테스트: MGN 93, HI-MGN-V 67, cHI-MGNflow 19, Transolver 23, NO 195(skip 6), SDFFlow 130(skip 2),
  SimulGenVAE 29, Studio backend 109 통과.
- suite 전체 `--audit-configs`(355개, 오류 0, 경고 54개는 이전과 같음). `DRY_RUN=1`: 135는 34 arm / 68 stage,
  136은 37 arm / 74 stage, 둘 다 종료 코드 0.
- 두 머신의 launch copy 71개 train config(`resume_training`/`resume_interval_minutes` 포함) 전부 launcher `--check`
  통과(GPU 1장 box에서 `gpu_ids 0`으로).

남은 항목:

- 전체 NO coordinate-domain coverage 확인, 대표 config의 실제 model forward 검증 추가.
- 첫 실행의 수렴 절에서 `F`/`R` arm을 보고 임계값을 다시 정한 뒤, 의심 slot에 예산 두 배 시험(위 "수렴 확인" 2).
  지금의 update 수는 논문 근거이지 수렴 증명이 아니다.
- B300 실측, 장기 학습 안정성, 실제 checkpoint inference, held-out 성능/확률 calibration.
  마지막 항목들은 아직 실행하지 않았으며 config 생성/단위 테스트로 대신 인증할 수 없다.
