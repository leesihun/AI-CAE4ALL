# iFEM_dense48 — AI-CAE4ALL native sweep

dense48 데이터셋에 대한 **33 arm 스윕**이다. 데이터셋·config·모델·로그·train/test 시각화·
held-out 예측이 **전부 AI-CAE4ALL 트리 안**에 들어간다. 이전 캠페인 결과는 일절 참조하지
않고 처음부터 다시 잡았다.

## 설치와 실행

```bash
# 번들 폴더에서 (한 번만)
bash install.sh /path/to/AI-CAE4ALL

# 이후 전부 AI-CAE4ALL 루트에서
cd /path/to/AI-CAE4ALL
export PYBIN=/home/sonic/anaconda3/envs/AARL/bin/python   # torch/h5py/torch_geometric 환경

bash configs/campaigns/iFEM_dense48/prepare.sh     # 체크섬 + 출력 트리 + MGN 전용 사본 10개
bash configs/campaigns/iFEM_dense48/check_all.sh   # 66개 config preflight (train은 전부 PASSED여야 함)
bash configs/campaigns/iFEM_dense48/train_all.sh   # GPU 4개 큐 동시 기동, detached
# ... 약 7.6일 ...
bash configs/campaigns/iFEM_dense48/infer_all.sh   # 미지 형상 8,482 case 추론
$PYBIN configs/campaigns/iFEM_dense48/baselines.py   # 기준선 (2분, 한 번만)
$PYBIN configs/campaigns/iFEM_dense48/score_infer.py # 채점 → output/iFEM_dense48/scores.json
bash configs/campaigns/iFEM_dense48/pack_results.sh  # 회수용 아카이브
```

## 모든 것이 AI-CAE4ALL 안에 있다

config 경로는 전부 상대경로다. 런처가 각 method를 `methods/<Name>/`을 cwd로 실행하므로
(`cae_suite/cli.py:291`) `../../`가 정확히 suite 루트를 가리킨다. 절대경로가 하나도 없어서
박스를 옮겨도 그대로 동작하고, 경로 재작성 스크립트가 필요 없다.

| 무엇 | 어디 |
|---|---|
| 데이터셋 | `dataset/iFEM_dense48/{train,infer}.h5` |
| MGN 전용 사본 | `dataset/iFEM_dense48/native_train/<ARM>/train.h5` |
| config | `configs/{Transolver,MeshGraphNets,Neural_Operator}/iFEM_dense48/config_{train,infer}_<arm>.txt` |
| 로스터·스크립트 | `configs/campaigns/iFEM_dense48/` |
| 모델·로그 | `output/iFEM_dense48/<ARM>/{model.pth, train.log, launcher.log}` |
| **train/test 시각화** | MGN `…/<ARM>/dumps/{test,train}/<gpu>/<epoch>/`, DeepONet `…/<ARM>/{test,train}/<gpu>/<epoch>/` |
| held-out 예측 | `output/iFEM_dense48/<ARM>/infer/` |
| 점수 | `output/iFEM_dense48/{scores.json, baselines.json}` |

시각화는 `test_interval 25`, `test_max_batches 32`, `test_batch_idx 0..7`,
`display_trainset True`(+ MGN·DeepONet은 `display_testset True`)로 켜져 있어 250 epoch 런에서
11회 덤프된다. 회당 test 8 + train 8 샘플이다.

**method마다 나오는 것이 다르다.**

- **MGN·DeepONet** — 샘플마다 `.h5`(pos/edge/face + 예측·정답의 정규화/역정규화)와 그 옆에
  같은 이름의 `.png`가 같이 나온다. PNG는 2×2(예측 vs 정답 × 정규화 vs 역정규화)이고 색은
  `plot_feature_idx 0` 채널 하나다. 학습 로그의 `All visualizations complete!`가 그 확인이다.
- **Transolver** — 렌더러가 아예 없다(`training_profiles/training_loop.py`의 `test_model`은
  숫자 배열만 쓴다). `write_test_predictions True`로 `.h5`만 남고 PNG는 나오지 않는다.

dense48은 9×9 정렬 격자라 대각선 에지가 없어서 삼각형 3-cycle이 0개다. 렌더러가 삼각형을
요구하므로 원래는 PNG가 한 장도 안 나오면서 로그만 성공이라고 찍혔다. `mesh_utils_fast.py`의
`quads_to_triangles`(4-cycle → 삼각형 2개) fallback이 그 경우를 메운다 —
`methods/MeshGraphNets/tests/test_surface_reconstruction.py` 참고.

## 스윕 33 arm

`candidates.tsv`가 정본(`make_configs.py`가 생성하므로 손으로 고치지 말 것). 각 arm은 자기
model의 BASE와 **정확히 한 key만** 다르다.

| model | BASE | 축 (아래로/위로) |
|---|---|---|
| transolver | latent 256, layers 4, heads 8, slice 64, batch 32, 250 ep | `num_layers` 2/8/16 · `latent_dim` 128/512 · `num_heads` 4/16 · `slice_num` 16/32/128 · `batch_size` 16/8 · `training_epochs` 600/1000 |
| meshgraphnets | latent 128, mp 8, batch 32, 250 ep | `message_passing_num` 4/2/16 · `latent_dim` 64/256/512 · `batch_size` 16/8 · `training_epochs` 600 |
| deeponet | hidden 256, basis 128, batch 32, 250 ep | `hidden_channels` 128/512 · `basis_dim` 64/256 · `batch_size` 16/8 · `training_epochs` 600 |

- Transolver 15 arm (T01–T15), MeshGraphNets 10 arm (M01–M10), DeepONet 8 arm (D01–D08)
- 합계 **727 GPU-hour**, GPU 4개에 LPT 패킹으로 균등 배분 → makespan **183 h ≈ 7.6일**
- 네 GPU 부하가 181/183/183/181 h로 놀고 있는 장치가 없다

GPU 배정은 `make_configs.py`가 계산해 `queues.sh`에 쓰고 `train_all.sh`가 그대로 읽는다.
arm을 추가·삭제하면 `python make_configs.py`만 다시 돌리면 config·candidates.tsv·큐가 함께
갱신된다.

### Transolver 비용 모델 (왜 batch/width 스윕이 싸고 depth 스윕만 비싼가)

`PhysicsAttentionIrregular.forward`가 배치 안의 그래프마다 **Python 루프**를 돈다
(`model/physics_attention.py:361-362`, layer마다 1회). 그래서 epoch 비용 ≈ sample 수 × layer 수이고
`batch_size`·`latent_dim`·`num_heads`·`slice_num`에는 거의 무관하다(16 layer에서 1632 s/epoch 실측).

- `num_layers` 2/4/8/16 → epoch 비용이 그대로 1/2/4/8배
- `batch_size`를 16·8로 줄여도 epoch 비용은 거의 그대로이면서 optimizer step은 2·4배가 된다
- 폭·head·slice 스윕은 사실상 공짜 (batch 32에서 VRAM 0.1 GB)

## 데이터

| 항목 | 값 |
|---|---|
| train | 39,447 case / 709 형상 |
| infer | 8,482 case / 177 형상 — train과 φ 겹침 0 |
| `nodal_data` | `[20, 1, 81]`, 전 sample 동일 |
| `mesh_edge` | `[2, 144]`, 전 sample 비트 단위 동일 |
| 좌표 | 9×9 단위 격자, z ≡ 0, 전 sample 비트 단위 동일 |
| row 계약 | 0–2 `x,y,z` / 3–6 **target** `u_eps1_x,y`·`u_eps2_x,y` / 7–19 **condition** 13채널 |
| var | `input_var 4`, `output_var 4`, `cond_var 13` → `3+4+13 = 20`, 여유 0 |

`num_timesteps 1`이므로 loader는 `graph.x`의 앞 4열(state)을 **0으로 채운다**. 모델이 보는 것은
좌표와 13개 condition 채널뿐이고, 타깃은 row 3–6의 절대 변위장이다.

**`metadata/splits`는 어떤 loader도 읽지 않는다.** 세 method 모두 저장된 split을 무시하고
`split_seed`로 매번 재분할하므로, 형상 분리는 파일을 나눠서 달성했다. train.h5는 내부적으로
다시 31,557 / 3,944 / 3,946으로 무작위 분할된다 → **로그의 `Valid`는 학습에 쓴 형상의 미지
하중에 대한 값이지 미지 형상 성능이 아니다.**

## 지표

로그의 `TrainOpt`/`Valid`는 채널별 전역 z-score 공간의 **절대** 가중 MSE를 node로 평균낸
값이다. 두 가지 이유로 일반화 지표가 아니다.

1. 형상이 새지 않은 split이 아니다(위 참조).
2. 진폭 가중이다. 이 데이터의 case별 타깃 norm은 4자릿수에 걸쳐 있어서 소수의 큰 case가
   값을 지배한다. 또 `delta_std`가 train split에서 적합되므로 **0을 예측했을 때의 기준값이
   val에서 1.0이 아니다.**

보고용 숫자는 `score_infer.py`의 per-case relative L2를 쓴다.

```
rel_L2(case) = ||u_pred - u_true||_F / ||u_true||_F      # [4, 81] 블록 전체
```

0 예측은 case마다 정확히 1.000이다. 채널별 수치는 해당 채널 truth가 0이 아닌 case에서만
계산하고 0인 case 수를 따로 보고한다(`u_eps2`는 case의 약 80%에서 0). 분모를 바꾸지 말 것.

**arm 선택과 보고를 분리한다.** arm이 33개라 같은 case로 고르고 보고하면 test set에 대한
selection이 된다. `score_infer.py`가 held-out을 형상 단위로 다시 반으로 갈라
`gval`(88형상 / 4,266 case)과 `gtest`(89형상 / 4,216 case)를 따로 낸다.
**`gval`로 고르고 고른 arm의 `gtest`를 보고한다.** 분할은 φ 다이제스트 정렬 후 번갈아
배정하는 결정적 규칙이라 저장 파일이 필요 없다.

기준선은 `baselines.py`가 이 데이터에서 직접 계산한다(0 예측 / train 평균장 / k-NN k=1,3).
**다른 데이터셋이나 다른 split에서 나온 기준선 수치를 가져다 쓰지 말 것** — 둘 다에 의존한다.

## 운영상 주의

- **Transolver와 MeshGraphNets는 마지막 epoch이 끝나야 checkpoint를 쓴다.** 중간에 죽이면
  `No checkpoint saved`가 찍히고 그 arm은 통째로 사라진다. DeepONet만 `checkpoint_interval 25`로
  주기 저장을 한다. `train_all.sh`는 `setsid nohup`으로 띄워 로그아웃에도 살아남는다.
  가장 긴 arm은 T04/T15로 각각 113 h다.
- **MGN은 `dataset_dir`를 `r+`로 열고 normalization 메타데이터를 쓴다.** opt-out 키가 없어서
  `prepare.sh`가 arm마다 전용 사본을 만든다(10 × 409 MB ≈ 4.1 GB). 공유 `train.h5`는 Transolver·
  DeepONet용으로 읽기 전용을 유지한다.
- **`gpu_ids`는 물리 device 번호**이고 개수가 아니다. `CUDA_VISIBLE_DEVICES` 재매핑도 없다.
  GPU가 4개 미만이면 preflight가 `ENV-CUDA-002`로 막는다 — `make_configs.py`의 `NUM_GPUS`를
  고치고 재생성할 것.
- 각 arm은 GPU 하나를 쓴다(`world_size 1`). 한 arm에 여러 GPU를 묶는 DDP는 쓰지 않았다:
  `distributed_training.py:8-11`에 "world_size > 1에서 실행해본 적 없음"이라는 자체 경고가
  있고, DDP에서 `batch_size`는 rank당 값인데 `learningr`는 자동 스케일되지 않아 레시피가
  조용히 바뀐다. 4개 큐를 동시에 돌리는 지금 구성이 총 처리량은 같으면서 정보량은 4배다.
- config에 명시해 둔 기본값 함정 세 가지: `attention_kernel`의 저장소 기본값은
  `slice_space`(약 40% 느림), `use_amp`의 기본값은 `True`, `use_compile`은 physics attention의
  per-graph `.item()` 때문에 그래프가 깨진다. 셋 다 명시적으로 지정해 두었다.
- `std_noise`는 `graph.x`의 앞 `output_var` 열만 흔드는데 이 정적 데이터에서 그 열은 **0**이다.
  즉 정규화 효과가 없어서 스윕 축으로 넣지 않았다.
- `augment_geometry`는 반드시 False. `x[:, :3]`과 `y[:, :3]`을 하나의 xyz 벡터로 회전시키는데
  여기서 그 셋은 서로 무관한 2-D 장 두 개이고 방향성 condition 채널은 아예 회전되지 않는다.
- `use_node_types`도 반드시 False. 20행뿐이라 node-type 행이 없고, True로 두면 loader가
  `gamma_traction_y`를 one-hot 해버린다.
- 추론은 case마다 HDF5 하나를 쓴다 → arm당 8,482개. `pack_results.sh`가 arm별로 묶는다.
  예측값은 항상 **마지막 timestep**에 있다(Transolver `(8,1,81)`, MGN·DeepONet `(8,2,81)`의 t=1).
