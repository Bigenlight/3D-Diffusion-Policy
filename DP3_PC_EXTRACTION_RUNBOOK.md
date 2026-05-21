# DP3 × RoboCasa — PC 전처리 재실행 Runbook

> **이 문서**는 RoboCasa human_raw demo.hdf5 → DP3 학습용 `processed_demo_128_128.hdf5` 추출을 **0부터 재현하는 절차**다. CloseDrawer 1 task 기준이며, 다른 task로 확장 시 §[다른 task로 확장](#다른-task로-확장) 참고.
>
> 검증된 결과 (2026-05-13): 55/55 demos, 10,411 frames, 9.97 GB, verify 12/12 PASSED.

---

## 0. 목적 & 산출물

- **목적**: DP3 (3D Diffusion Policy) reader가 기대하는 hdf5 구조로 RoboCasa point cloud 데이터를 생성.
- **산출물**:
  - `<DATASET_BASE_PATH>/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/processed_demo_128_128.hdf5` — 학습용 (DP3 reader 자동 인식 경로)
  - 검증 스크립트 `/home/woonsang/theo/scripts/verify_processed_hdf5.py`
  - 시각화 스크립트 `/home/woonsang/theo/scripts/visualize_pc_to_mp4.py` (선택)

---

## 1. 사전 셋업 (이미 끝났음, 재구축 시 참고)

### 1.1. 시스템 전제
- Ubuntu 24.04, headless
- NVIDIA driver 동작 (8× RTX A4000 16GB)
- 시스템에 `nvcc` 없어도 무방 — conda env 안에 설치
- 디스크 여유 ≥ 50 GB

### 1.2. 리포 클론
```bash
cd /home/woonsang/theo
git clone --recurse-submodules https://github.com/ALRhub/FPVNet
```
이 한 줄로 `FPVNet/` + `FPVNet/custom_robocasa/` + `FPVNet/custom_robocasa/custom_robosuite/` 전부 받음.

### 1.3. conda env
```bash
conda create -n fpvnet python=3.10 -y
conda activate fpvnet
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124
```
> 원 FPVNet 가이드는 `mamba`를 권하지만 시스템에 `mamba`가 없으면 `conda`로 대체해도 동등. (검증됨)

### 1.4. FPVNet requirements
`FPVNet/requirements.txt`의 `# 3DA` 섹션(dgl/flash-attn/openai CLIP)은 **주석 처리** (DP3 불필요, flash-attn 빌드 무거움):
```diff
- # 3DA
- --find-links https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
- dgl
- packaging
- ninja
- flash-attn==2.7.2.post1
- git+https://github.com/openai/CLIP.git
+ # 3DA (DP3에 불필요 — 주석)
+ # --find-links https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
+ # dgl
+ packaging
+ ninja
+ # flash-attn==2.7.2.post1
+ # git+https://github.com/openai/CLIP.git
```
```bash
cd /home/woonsang/theo/FPVNet
pip install --no-build-isolation -r requirements.txt
```

### 1.5. CUDA toolkit (env-local) — pytorch3d GPU 빌드 전제 ★
시스템에 `nvcc`가 없으므로 **conda env 안에** 설치:
```bash
conda install -n fpvnet -c "nvidia/label/cuda-12.4.0" \
  cuda-nvcc cuda-cudart-dev cuda-libraries-dev \
  libcusparse-dev libcublas-dev libcusolver-dev \
  libcurand-dev libcufft-dev cuda-nvrtc-dev -y
```
`cuda-nvcc`만 깔면 빌드 중 `cusparse.h not found`로 실패. **dev 헤더 패키지까지 함께 깔아야** 한다.

### 1.6. custom_robocasa install.sh
```bash
cd /home/woonsang/theo/FPVNet/custom_robocasa
sh install.sh
```
install.sh가 하는 일: `custom_robosuite editable` + `custom_robocasa editable` + open3d/pytorch3d 빌드 + kitchen assets 다운 + setup_macros×2.

**함정 두 개:**
- (a) install.sh의 3번째 줄 `pip install -r requirements.txt`가 `--no-build-isolation`을 안 줘서 pytorch3d 빌드가 CPU only로 나옴 → §1.7에서 별도 재빌드.
- (b) `download_kitchen_assets`가 `Proceed? (y/n)` 인터랙티브 prompt에서 EOFError → 별도로 `yes |` 파이프로 재실행.

후속 명령:
```bash
# open3d (이미 install.sh가 깔았을 수도, 보장 차원)
pip install open3d

# kitchen assets — install.sh가 EOFError로 실패했으면 재실행
cd /home/woonsang/theo/FPVNet/custom_robocasa
yes | python -m robocasa.scripts.download_kitchen_assets
```
받는 자산: `textures.zip`, `fixtures.zip`, `objaverse.zip`, `generative_textures.zip` → 총 ~8.2 GB.
※ `objects/aigen_objs`는 받지 않아도 CloseDrawer엔 영향 없음 (확인됨).

### 1.7. pytorch3d **CUDA 빌드** ★ (가장 자주 깨지는 곳)
```bash
cd /home/woonsang/theo/FPVNet
conda activate fpvnet
pip uninstall -y pytorch3d
export CUDA_HOME="$CONDA_PREFIX"
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6"            # A4000 = Ampere SM_86
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
pip install --no-build-isolation --no-cache-dir \
  git+https://github.com/facebookresearch/pytorch3d.git
```
검증:
```bash
python -c "
import torch
from pytorch3d.ops import sample_farthest_points
pts = torch.randn(2, 5000, 3, device='cuda')
out, _ = sample_farthest_points(pts, K=1024)
print('GPU FPS OK', out.shape, out.device)
"
```
"GPU FPS OK" 안 뜨면 추출이 무조건 실패한다 (FPS sampler가 `device='cuda'` 하드코드).

### 1.8. dataset_states_to_obs.py **L475 패치** ★
race condition 버그. **마지막 demo(num_procs개)** 가 silent하게 누락된다.

`FPVNet/custom_robocasa/custom_robocasa/robocasa/scripts/dataset_states_to_obs.py:475`:
```diff
ind = retrieve_new_index(process_num, current_work_array, work_queue, lock)
- while (not work_queue.empty()) and (ind != -1):
+ while ind != -1:
```
패치 안 하면 `--num_procs 5` 본 처리 시 마지막 1~5개 demo가 빠진다 (관측: 5명 worker, demo_55 누락).

### 1.9. macros_private.py — DATASET_BASE_PATH
`FPVNet/custom_robocasa/custom_robocasa/robocasa/macros_private.py`:
```python
DATASET_BASE_PATH = "/home/woonsang/theo/robocasa_data"
```
이 경로가 **DP3 reader가 자동으로 찾는 root**이며, 출력 hdf5는 그 안 정해진 경로에 떨어진다. 본 문서의 모든 경로는 이 값을 기준으로 함.

### 1.10. RoboCasa human_raw 데이터 다운로드
```bash
conda activate fpvnet
yes | python -m robocasa.scripts.download_datasets --ds_types human_raw
```
받는 위치: `${DATASET_BASE_PATH}/v0.1/single_stage/<group>/<task>/<date>/demo.hdf5` 24 single + 5 multi task. ~1.8 GB.

---

## 2. 환경 검증 (실행 전 한 줄 가드)

PC 추출 시작하기 직전에 한 번 돌려서 모든 사전 셋업이 살아있는지 확인:
```bash
conda activate fpvnet
cd /home/woonsang/theo/FPVNet
python -c "
import os, h5py, robocasa, torch
from robocasa.utils.camera_utils import get_robot_cam_configs
from robosuite.renderers.context.egl_context import create_initialized_egl_device_display
from pytorch3d.ops import sample_farthest_points

# 1) 카메라 이름
c = get_robot_cam_configs('PandaOmron')
need = {'robot0_agentview_left','robot0_agentview_right','robot0_eye_in_hand'}
assert need.issubset(c.keys()), f'camera missing'

# 2) 자산
root = os.path.join(robocasa.__path__[0], 'models/assets')
for p in ['textures','fixtures','objects/objaverse','generative_textures']:
    full = os.path.join(root, p)
    assert os.path.isdir(full) and len(os.listdir(full))>0, f'asset missing: {p}'

# 3) input demo
f = h5py.File('/home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5','r')
demos = list(f['data'].keys())
g = f['data'][demos[0]]
assert 'model_file' in g.attrs
f.close()

# 4) EGL + pytorch3d CUDA  (-1 = device 자동선택, 0으로 줘도 무방)
assert create_initialized_egl_device_display(-1), 'EGL fail'
pts = torch.randn(1, 5000, 3, device='cuda')
sample_farthest_points(pts, K=1024)

print('ALL GUARDS PASSED')
"
```
"ALL GUARDS PASSED" 뜨면 §3 진행.

---

## 3. 본 추출 — CloseDrawer

### 3.1. (선택) Sanity check — 3 demos만 빠르게
```bash
conda activate fpvnet
cd /home/woonsang/theo/FPVNet
PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m robocasa.scripts.dataset_states_to_obs \
  --dataset /home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5 \
  --output_name processed_demo_128_128_test.hdf5 \
  --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand \
  --camera_width 128 --camera_height 128 \
  --num_procs 2 --n 3 \
  --keep_full_pc --dont_store_image --dont_store_depth
```
- 출력: 입력과 같은 디렉토리에 `processed_demo_128_128_test.hdf5` (~30-60s).
- 검증:
```bash
python /home/woonsang/theo/scripts/verify_processed_hdf5.py \
  /home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/processed_demo_128_128_test.hdf5
```
"OVERALL [OK] 12/12 checks passed" 떠야 §3.2 진행. 안 뜨면 §[문제 해결](#문제-해결).

> ⚠ §1.8 L475 패치를 안 한 상태에서는 sanity check도 race condition으로 **마지막 1개 demo가 누락**됨 (관측 사례: `--n 3 --num_procs 2` → demo_1/demo_2만 출력, demo_3 없음). 본 처리 전 반드시 §1.8 적용.

### 3.2. 본 처리 — 55 demos 전체
```bash
conda activate fpvnet
cd /home/woonsang/theo/FPVNet
PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m robocasa.scripts.dataset_states_to_obs \
  --dataset /home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5 \
  --output_name processed_demo_128_128.hdf5 \
  --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand \
  --camera_width 128 --camera_height 128 \
  --num_procs 5 \
  --keep_full_pc --dont_store_image --dont_store_depth
```
- 출력: `processed_demo_128_128.hdf5` (DP3 reader 자동 인식 경로)
- **검증된 결과**: 55 demos, 10,411 frames, 9.97 GB, 9분 17초.
- 검증:
```bash
python /home/woonsang/theo/scripts/verify_processed_hdf5.py \
  /home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/processed_demo_128_128.hdf5
```
"OVERALL [OK] 12/12 checks passed" 확인 + demo 개수 55.

### 3.3. 누락 진단 (안전망)

> §1.8 패치된 환경에선 항상 `missing=[]`이 나옴. 이 셀은 패치 누락 시 catch용.
```bash
python -c "
import h5py
IN  = '/home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5'
OUT = '/home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/processed_demo_128_128.hdf5'
with h5py.File(IN) as fi, h5py.File(OUT) as fo:
    src = sorted(fi['data'].keys(), key=lambda x: int(x[5:]))
    dst = sorted(fo['data'].keys(), key=lambda x: int(x[5:]))
    miss = [d for d in src if d not in set(dst)]
    print(f'src={len(src)} dst={len(dst)} missing={miss}')
"
```

---

## 4. 명령 옵션 의미 표

| 옵션 | 값 | 이유 |
|---|---|---|
| `--dataset` | demo.hdf5 절대경로 | 원본 human_raw demo |
| `--output_name` | `processed_demo_128_128.hdf5` | DP3 reader가 `processed_demo_{w}_{h}.hdf5` 패턴으로 찾음 (img_width/height 128 기본값) |
| `--camera_names` | left/right/eye_in_hand | FPVNet DP3 표준 3-cam fused PC |
| `--camera_width/height` | 128 | DP3 config 기본값과 일치 |
| `--num_procs` | 5 | A4000 16GB 1장에 안전한 worker 수. 8 GPU 있어도 코드가 GPU 0번만 사용 |
| `--keep_full_pc` | flag | `obs/point_cloud` (T, 49152, 6) 보존. DP3 reader는 sampled만 쓰지만 다른 FPV/3DA reader 호환 |
| `--dont_store_image` | flag | `*_image` key 제거 (디스크 절약, PC 생성에는 이미 사용됨) |
| `--dont_store_depth` | flag | `*_depth` key 제거 (동상) |
| `MUJOCO_GL=egl` | env | 헤드리스 EGL 강제 (robosuite가 자동 설정하지만 명시 안전) |
| `OMP/MKL/OPENBLAS_NUM_THREADS=1` | env | spawn worker × BLAS thread oversubscription 방지 |
| `PYTHONUNBUFFERED=1` | env | tqdm 즉시 flush |

---

## 5. 출력 hdf5 구조 (DP3 reader가 기대하는 것)

```
processed_demo_128_128.hdf5
├── data/
│   ├── attrs: {total: 10411}
│   ├── demo_1/
│   │   ├── attrs: {num_samples: 189, ep_meta: '{"lang":"close the left drawer", ...}', model_file: ...}
│   │   ├── states                (189, 167)
│   │   ├── actions               (189, 12)     # reader는 [:, :7] 슬라이스
│   │   ├── action_dict/...
│   │   ├── datagen_info/...
│   │   └── obs/
│   │       ├── point_cloud                  (189, 49152, 6)  XYZ + RGB(0~255)
│   │       ├── sampled_point_cloud          (189, 1024, 6)   FPS sampled
│   │       ├── uniform_sampled_point_cloud  (189, 1024, 6)   Uniform sampled
│   │       ├── robot0_gripper_qpos          (189, 2)
│   │       ├── robot0_base_to_eef_pos       (189, 3)
│   │       ├── robot0_base_to_eef_quat      (189, 4)
│   │       └── (... 기타 proprio keys)
│   ├── demo_2/  ...
│   └── demo_55/
└── mask/
    └── (robomimic 호환 filter keys — DP3 학습에선 무시)
```

**중요한 약속**:
- RGB 채널은 **0~255 스케일** (reader가 `/255.0` 정규화). 절대 미리 0~1로 만들지 말 것.
- demo 이름은 `demo_<int>` 패턴 (reader가 `int(name.split('_')[1])`로 정렬).
- `actions`는 reader가 `[:, :7]`만 사용. RoboCasa 원본은 (T, 12).

---

## 6. 다른 task로 확장

CloseDrawer 외 task로 가려면 §3.1/§3.2 명령의 `--dataset` 경로만 바꿈. 경로 패턴:
```
${DATASET_BASE_PATH}/v0.1/single_stage/<kitchen_*>/<TaskName>/<date>/demo.hdf5
```

### 6.1. FPVNet DP3 학습용 env_group (8개)

`FPVNet/environments/robocasa_env_groups.py`:

| group | task 수 | tasks (모두 추출 필수) |
|---|---|---|
| **drawer** | 2 | OpenDrawer, CloseDrawer |
| **stove** | 2 | TurnOnStove, TurnOffStove |
| **coffee** | 2 | CoffeeSetupMug, CoffeeServeMug |
| **sink** | 3 | TurnOnSinkFaucet, TurnOffSinkFaucet, TurnSinkSpout |
| **buttons** | 3 | CoffeePressButton, TurnOnMicrowave, TurnOffMicrowave |
| **pnp1** | 4 | PnPCounterToCab, PnPCabToCounter, PnPCounterToSink, PnPSinkToCounter |
| **pnp2** | 4 | PnPCounterToMicrowave, PnPMicrowaveToCounter, PnPCounterToStove, PnPStoveToCounter |
| **doors** | 4 | OpenSingleDoor, CloseSingleDoor, OpenDoubleDoor, CloseDoubleDoor |

> 한 group을 학습하려면 그 group의 **모든** task processed hdf5가 있어야 함. reader가 한 group 전체 순회 (`robocasa_dp3_dataset.py:55-67`).

### 6.2. task → 정확한 date (모든 24 task)

`dataset_registry.py`의 `SINGLE_STAGE_TASK_DATASETS`에서 그대로 추출:

| group | task | date (디렉토리명) |
|---|---|---|
| pnp1 | PnPCounterToCab | 2024-04-24 |
| pnp1 | PnPCabToCounter | 2024-04-24 |
| pnp1 | PnPCounterToSink | 2024-04-25 |
| pnp1 | PnPSinkToCounter | **`2024-04-26_2`** ⚠ suffix `_2` 주의 |
| pnp2 | PnPCounterToMicrowave | 2024-04-27 |
| pnp2 | PnPMicrowaveToCounter | 2024-04-26 |
| pnp2 | PnPCounterToStove | 2024-04-26 |
| pnp2 | PnPStoveToCounter | 2024-05-01 |
| doors | OpenSingleDoor | 2024-04-24 |
| doors | CloseSingleDoor | 2024-04-24 |
| doors | OpenDoubleDoor | 2024-04-26 |
| doors | CloseDoubleDoor | 2024-04-29 |
| drawer | OpenDrawer | 2024-05-03 |
| drawer | **CloseDrawer** | **2024-04-30** (이 runbook의 기준 task) |
| sink | TurnOnSinkFaucet | 2024-04-25 |
| sink | TurnOffSinkFaucet | 2024-04-25 |
| sink | TurnSinkSpout | 2024-04-29 |
| stove | TurnOnStove | 2024-05-02 |
| stove | TurnOffStove | 2024-05-02 |
| coffee | CoffeeSetupMug | 2024-04-25 |
| coffee | CoffeeServeMug | 2024-05-01 |
| buttons | CoffeePressButton | 2024-04-25 |
| buttons | TurnOnMicrowave | 2024-04-25 |
| buttons | TurnOffMicrowave | 2024-04-25 |

추가:
- **NavigateKitchen** (`2024-05-09`)은 registry에는 있으나 어떤 ENV_GROUP에도 미포함 → 현행 DP3 학습 무관.

### 6.3. multi_stage 5개는 현행 DP3 학습 미지원

`MULTI_STAGE_TASK_DATASETS` (registry L262-308)에 정의되어 `human_raw` 다운로드 시 함께 받지만:
- `robocasa_dp3_dataset.py:15`는 **`SINGLE_STAGE_TASK_DATASETS`만 import**
- `ENV_GROUPS`도 single만 나열

→ multi_stage task (ArrangeVegetables, MicrowaveThawing, RestockPantry, PreSoakPan, PrepareCoffee)를 DP3로 학습하려면 reader + env_groups 코드 패치 필요.

### 6.4. 다른 task 추출 시 추가 가드

§2 가드의 자산 점검(textures/fixtures/objects/objaverse/generative_textures)은 CloseDrawer 기준으로 통과 검증된 것. 다른 task가 추가 fixture 카테고리를 요구할 가능성 → 새 task 첫 sanity 실행 시 stderr에 missing asset 메시지 나오면 `python -m robocasa.scripts.download_kitchen_assets` 재실행으로 보충.

---

## 7. 문제 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `RuntimeError: Not compiled with GPU support.` | pytorch3d CPU only | §1.7 CUDA 빌드 재실행 |
| `fatal error: cusparse.h: No such file or directory` | CUDA dev 헤더 없음 | §1.5의 `cuda-libraries-dev libcusparse-dev …` 설치 |
| `Could not create GL context` | EGL 미설정 | `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl` |
| 마지막 N개 demo 누락 | `dataset_states_to_obs.py:475` race | §1.8 패치 |
| `EOFError: input ... Proceed? (y/n)` | install.sh / download script 인터랙티브 | `yes \|` 파이프 |
| `wandb login` 거부 (key 86 chars) | wandb 0.19.1이 신형 키 미지원 | `pip install -U wandb` (0.26+) |
| `wandb` 업그레이드 후 `protobuf~=3.19.0` 충돌 (tianshou) | 의존성 충돌 경고 | tianshou 미사용 코드 경로면 무시 가능. 학습 시 깨지면 `pip install protobuf==3.19.0` 롤백 |

---

## 8. 시각화 (선택) — 추출 결과 영상화

검증 후 점검용 mp4 (정상 추출 시 5분 이내):
```bash
# 1) image 데이터 별도 추출 (RGB 1024×1024, 1 demo 만)
cd /home/woonsang/theo/FPVNet
PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
python -m robocasa.scripts.dataset_states_to_obs \
  --dataset /home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo.hdf5 \
  --output_name demo_viz_image.hdf5 \
  --camera_names robot0_agentview_left \
  --camera_width 1024 --camera_height 1024 \
  --num_procs 1 --n 1 --dont_store_depth

# 2) side-by-side 4K 영상
python /home/woonsang/theo/scripts/visualize_pc_to_mp4.py \
  /home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/processed_demo_128_128.hdf5 \
  demo_1 \
  /home/woonsang/theo/scripts/demo_1_4k.mp4 \
  --key point_cloud --max_points 8000 \
  --image_hdf5 /home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_viz_image.hdf5 \
  --image_key robot0_agentview_left_image \
  --size 9.6 --dpi 200 --marker_size 7 --fontsize 14 \
  --stride 3 --fps 12 --rotate --total_azim 180 --quality 9
```

---

## 9. Docker 이전 시 챙길 것

### 9.1. 이미지 빌드 시 (Dockerfile)
1. **base image**: `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` (devel — nvcc/dev 헤더 포함하지만 그래도 conda 채널 사용이 더 안전)
2. **시스템 패키지** — EGL/OpenGL 런타임 lib:
   ```dockerfile
   RUN apt-get update && apt-get install -y --no-install-recommends \
       libegl1 libgles2 libglvnd0 libglib2.0-0 git build-essential ca-certificates \
       && rm -rf /var/lib/apt/lists/*
   ```
   `libegl1`이 빠지면 헤드리스에서도 EGL context 생성 실패.
3. **conda env** (§1.3) + `cuda-nvcc cuda-libraries-dev libcusparse-dev …` (§1.5 동일)
4. **소스 위치**: FPVNet/custom_robocasa/custom_robosuite는 `/opt/fpvnet`처럼 고정 경로에 클론 후 `pip install -e` editable. 호스트 볼륨으로 덮어쓰지 말 것 (덮어쓸 거면 editable 자체가 무의미 + macros_private도 함께 덮임).
5. **requirements.txt 패치를 이미지에 굽기**:
   - `# 3DA` 섹션 주석 (§1.4)
   - `wandb==0.19.1` → `wandb>=0.26.0` (신형 키)
6. **`dataset_states_to_obs.py:475` 패치도 이미지에 굽기** (§1.8). RUN sed/patch로 적용.
7. **pytorch3d 빌드는 이미지 빌드 단계에서** (§1.7 명령 그대로 `RUN`). 런타임 빌드 금지: 매 컨테이너 5~10분 낭비 + `TORCH_CUDA_ARCH_LIST` 불일치 위험.
8. **macros_private.py**는 editable install 안 파일이므로 컨테이너 빌드 시 작성. `DATASET_BASE_PATH`는 빈 문자열 또는 컨테이너 내 표준 경로(예: `/data/robocasa`)로 두고, 실제 데이터는 볼륨 마운트.

### 9.2. 컨테이너 실행 시
```bash
docker run --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility \
  -e MUJOCO_GL=egl \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -v /host/path/to/robocasa_data:/data/robocasa:ro \
  fpvnet-dp3 ...
```
- `NVIDIA_DRIVER_CAPABILITIES`에 **`graphics` 필수** (utility만이면 EGL fail). 헤드리스 컨테이너도 graphics caps 필요.
- 호스트의 `~/.netrc`는 컨테이너 안으로 자동 전파되지 않음 → `-e WANDB_API_KEY=$WANDB_API_KEY` 사용. (호스트 netrc 매핑은 `-v $HOME/.netrc:/root/.netrc:ro`로 가능하나 비추 — env var이 더 깔끔.)
- RoboCasa kitchen assets (~8.2 GB)는 이미지에 굽기보다 볼륨 마운트 권장. 또는 별도 dataset image로 분리.

---

## 10. 검증된 버전 핀 (재현용)

검증된 시점: **2026-05-13**, Ubuntu 24.04, 8× RTX A4000 16GB.

| 컴포넌트 | 버전 | 비고 |
|---|---|---|
| OS | Ubuntu 24.04.1 LTS | |
| GCC | 13.3.0 | pytorch3d 요구 9+ 충족 |
| NVIDIA driver | (driver-only, toolkit 시스템 미설치) | |
| conda | 25.3.1 | |
| Python | 3.10 | conda env |
| torch | 2.4.0+cu124 | `--index-url https://download.pytorch.org/whl/cu124` |
| torchvision | 0.19.0+cu124 | |
| CUDA (env-local) | 12.4 | nvidia/label/cuda-12.4.0 채널 |
| pytorch3d | 0.7.9 | source build, FORCE_CUDA=1, sm_86 |
| open3d | 0.19.0 | |
| mujoco | 3.2.6 | |
| robosuite | 1.5.0 (editable, custom_robosuite fork) | |
| robocasa | 0.2.0 (editable, custom_robocasa fork) | |
| numpy | 1.23.3 | robocasa 핀 |
| numba | 0.56.4 | |
| h5py | 3.16.0 | |
| hydra-core | 1.2.0 | |
| diffusers | 0.11.1 | |
| wandb | 0.26.1 | FPVNet 핀(0.19.1)에서 업그레이드 — 신형 키 필요 |
| imageio | 2.37.3 + imageio-ffmpeg 0.6.0 | 시각화용 |

**디스크 실측**: kitchen assets 8.2 GB + human_raw 1.8 GB + processed (CloseDrawer 1 task) 9.97 GB ≈ **20 GB**. (50 GB는 안전 마진.)

### 10.1. 1년 후 재현 안전망 (commit 핀)

`git clone` 시 main HEAD가 깨질 가능성 대비:

| repo | 검증된 commit | 핀 명령 예시 |
|---|---|---|
| `ALRhub/FPVNet` | (확인된 HEAD — 클론 직후 `git -C FPVNet rev-parse HEAD` 결과를 별도 기록) | `git -C FPVNet checkout <sha>` |
| `ALRhub/custom_robocasa` (submodule) | `3069f96bfb82b3f0711b0117ca0c67a2983f6ae2` | `git -C FPVNet/custom_robocasa checkout <sha>` |
| `facebookresearch/pytorch3d` | 최신 main; torch 2.4 호환 깨지면 `git+...@<sha>` 형태로 핀 | `pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git@<sha>` |

> 새 환경 처음 셋업 후 `pip freeze > /home/woonsang/theo/FPVNet/fpvnet_freeze.txt` + 각 submodule의 `git rev-parse HEAD`를 별도 메모로 남길 것. Docker 이미지 reproducibility의 핵심.

---

## 11. 변경 이력

- 2026-05-13: CloseDrawer 1 task 검증 완료 (55/55, 12/12 PASSED). L475 패치, pytorch3d CUDA 재빌드 함정 정리.
- 2026-05-13 (검수 반영): §6 PnPSinkToCounter `_2` suffix 추가 + 완전판 24 task 표, §9 Docker 보강(EGL 시스템 lib, NVIDIA caps, editable install, pytorch3d 빌드 위치), §10 버전 핀 표 + commit SHA 추가.
