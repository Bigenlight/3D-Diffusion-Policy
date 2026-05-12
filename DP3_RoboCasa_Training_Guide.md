# DP3 × RoboCasa 학습 가이드 (FPVNet 코드베이스)

## 목표
FPVNet 리포의 DP3 (3D Diffusion Policy) 파이프라인으로 RoboCasa 24개 kitchen task를 학습 + 평가.

---

## 필요 리포지토리

| # | 리포 | 역할 | 비고 |
|---|------|------|------|
| 1 | `ALRhub/FPVNet` | 학습 코드 본체 (메인) | - |
| 2 | `ALRhub/custom_robocasa` | RoboCasa + robosuite fork + PC wrappers | submodule, init 완료 |
| 3 | OpenAI CLIP (pip git+) | — | **DP3 불필요** (BESO 내장 CLIP 사용) |
| 4 | pytorch3d (pip git+) | FPS 샘플링 | PC 전처리 + **평가 rollout 모두에 필수** (sim wrapper가 FPS 사용) |
| 5 | SUGAR Pointnet2_PyTorch | — | **DP3 불필요** (FPV-SUGAR 전용, FPVNet `requirements.txt`에는 없음 / README §SUGAR에만 별도 설치) |

---

## 외부 의존성 (DP3 기준)

### 필수
| 패키지 | 버전/조건 |
|--------|----------|
| python | 3.10 (mamba env 권장) |
| torch | 2.4 + CUDA 12.4 |
| hydra-core | 1.2.0 |
| wandb | 0.19.1 (기본 mode:disabled) |
| diffusers | 0.11.1 |
| zarr | 2.12.0 |
| einops, dill, huggingface_hub | — |
| timm, ftfy, regex | BESO 내장 CLIP 토크나이저 |
| PyOpenGL-accelerate, h5py, tqdm, open3d | — |
| pytorch3d | GCC 9+ 필요. `custom_robocasa/requirements.txt` 라인 2에서 자동 설치됨 |

### 생략 가능 (DP3에 불필요)
FPVNet `requirements.txt`의 `# 3DA` 섹션 (라인 33~39):
```
--find-links https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
dgl
packaging
ninja
flash-attn==2.7.2.post1
git+https://github.com/openai/CLIP.git
```
→ DP3만 돌릴 거면 위 라인들을 주석 처리해도 됨 (단 `packaging`/`ninja`는 다른 빌드에서 유용하므로 남겨도 무방).
SUGAR (Pointnet2_PyTorch + Dropbox pretrain)는 `requirements.txt`에 없음 — README §SUGAR 섹션의 별도 스크립트만 skip하면 됨.

---

## 핵심 코드 파일

### 실행 / 설정
| 경로 | 역할 |
|------|------|
| `run.py` | Hydra entry. agent/trainer/sim instantiate → `trainer.main(agent)` → `env_sim.test_agent(agent)` |
| `scripts/dp3.sh` | DP3 학습 명령 (8 env_group × 3 seed multirun) |
| `configs/robocasa_dp3_config.yaml` | top-level (obs/act seq, batch, epoch, PC opt, target) |
| `configs/agents/dp3_agent.yaml` | DP3Agent (AdamW 1e-4, cosine LR, CLIP lang encoder) |
| `configs/agents/model/dp3/dp3_policy.yaml` | DP3Encoder + ConditionalUnet1D `[512,1024,2048]` + DDIM 100 steps |
| `configs/trainers/dp3_trainer.yaml` | DP3Trainer (EMA, `save_every_n_epochs=100`) |

### 모델
| 경로 | 역할 |
|------|------|
| `agents/base_agent.py` | 공통 BaseAgent (nn.Module) |
| `agents/dp3_agent.py` | obs_dict → point_cloud + agent_pos(gripper+eef_pos+eef_quat) + lang_emb |
| `agents/models/dp3/dp3_policy.py` | DP3 메인 (Encoder + Unet1D + DDIM + LinearNormalizer + LowdimMaskGenerator) |
| `agents/models/dp3/model/diffusion/conditional_unet1d.py` | 1D U-Net noise predictor |
| `agents/models/dp3/model/vision/pointnet_extractor.py` | DP3Encoder / PointNetEncoderXYZ(RGB) |
| `agents/models/dp3/model/common/normalizer.py` | LinearNormalizer |
| `agents/models/dp3/model/diffusion/ema_model.py` | EMAModel |

### 학습 / 데이터 / 평가
| 경로 | 역할 |
|------|------|
| `trainers/dp3_trainer.py` | DataLoader + Normalizer.fit + train loop + EMA + ckpt + wandb |
| `environments/dataset/robocasa_dp3_dataset.py` | RoboCasa hdf5 reader (sampled/full/segmented PC, obs_dim=8, action 7-DoF) |
| `environments/robocasa_env_groups.py` | ENV_GROUPS 8개 (pnp1/pnp2/doors/drawer/sink/stove/coffee/buttons) |
| `simulation/robocasa_pc_sim.py` | eval rollout (Robosuite/PC/Seg wrapper 체인, `num_episode=50`) |

### PC 전처리 (custom_robocasa)
| 경로 | 역할 |
|------|------|
| `custom_robocasa/custom_robocasa/robocasa/scripts/dataset_states_to_obs.py` | ★ 패치된 PC 추출 스크립트 |
| `custom_robocasa/env_wrappers/point_cloud_sampling_wrapper.py` | gym.Wrapper로 PC obs 추가 |
| `custom_robocasa/utils/point_cloud/pc_generator.py` | Open3D 기반 multi-view depth → PC |
| `custom_robocasa/utils/point_cloud/sampling/fps_pc_sampler.py` | pytorch3d FPS 샘플러 |

---

## 워크플로우

### Step 1 — Python env
```bash
mamba create -n fpvnet python=3.10 -y
mamba activate fpvnet
# PyTorch 2.4 + CUDA 12.4
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124
```

### Step 2 — FPVNet 의존성 (3DA 섹션 skip 가능)
```bash
cd /home/theo_lab/Downloads/3d_diffusion_in_robocasa/FPVNet
pip install --no-build-isolation -r requirements.txt
# DP3만 돌릴 경우 requirements.txt 라인 33~39 (`# 3DA` 섹션: dgl/flash-attn/openai CLIP) 주석 처리 가능
# SUGAR 관련 라인은 requirements.txt에 없음 → README의 별도 SUGAR 스크립트만 skip
```

### Step 3 — custom_robocasa 설치 (~5-7GB 에셋 + editable install)
```bash
cd /home/theo_lab/Downloads/3d_diffusion_in_robocasa/FPVNet/custom_robocasa
sh install.sh
cd ..
```
`install.sh`가 하는 일 (검증됨):
1. `pip install -e custom_robosuite/`
2. `pip install -e custom_robocasa/`  (← `custom_robocasa/requirements.txt`가 여기서 pytorch3d를 git+로 빌드)
3. `pip install -r requirements.txt` (custom_robocasa의 requirements — open3d + pytorch3d)
4. `python -m robocasa.scripts.download_kitchen_assets`
5. `python -m robosuite.scripts.setup_macros` (1회)
6. `python -m robocasa.scripts.setup_macros` (1회)

### Step 4 — RoboCasa human_raw 데이터셋 (~15-25GB, 30분~1시간)
```bash
python -m robocasa.scripts.download_datasets --ds_types human_raw
```

### Step 5 — macros_private.py 설정
`setup_macros`가 만든 파일을 직접 편집:
```
custom_robocasa/custom_robocasa/robocasa/macros_private.py
```
```python
# DATASET_BASE_PATH 를 데이터셋 root로 지정
DATASET_BASE_PATH = "/abs/path/to/robocasa/datasets"
```
(원본 `macros.py`에는 `DATASET_BASE_PATH = None`이며, `dataset_registry.py` 라인 340에서 None이면 에러를 띄움)

### Step 6 — ★ PC 전처리 (가장 무거움)
24개 task 각각에 대해 반복 (FPVNet README와 동일):
```bash
OMP_NUM_THREADS=1 MPI_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -m robocasa.scripts.dataset_states_to_obs \
    --dataset <ds-path>/demo.hdf5 \
    --output_name processed_demo_128_128.hdf5 \
    --camera_names robot0_agentview_left robot0_agentview_right robot0_eye_in_hand \
    --camera_width 128 --camera_height 128 \
    --keep_full_pc --dont_store_image --dont_store_depth
```
주요 argparse 옵션 (스크립트 라인 691~860 직접 검증):
- `--dataset` (필수), `--output_name` (※ README는 `--output`로 적혀있으나 argparse 정식 이름은 `--output_name`)
- `--camera_names` (nargs+, default = 3개 카메라), `--camera_height` / `--camera_width` (default 128)
- `--num_procs` (default 5, 멀티프로세싱 process 수)
- `--pc_size` (default 1024), `--pc_obj_max_size` (default 512), `--pc_in_global_frame`
- `--keep_full_pc`, `--segmentation` (segmented PC 별도 저장)
- `--dont_store_image`, `--dont_store_depth` (디스크 절약)
- `--n` (debug용 demo 개수 제한), `--filter_key`, `--done_mode`, `--no_compress`, `--global_actions`, `--add_datagen_info`, `--generative_textures`, `--randomize_cameras`, `--shaped`, `--copy_rewards`, `--copy_dones`, `--include-next-obs`

- task당 30~60분 × 24 task = **총 12~24시간**
- 결과: ~**50GB** hdf5
- pytorch3d 회피 여부: **불가능**. 평가 단계 `simulation/robocasa_pc_sim.py`가 `FPSPointCloudSampler` (pytorch3d.ops 사용) 를 직접 import하므로 학습/평가 환경에 pytorch3d 필수.

### Step 7 — wandb
기본 `mode:disabled`. 그대로 두거나, 로깅 원하면 `wandb.mode=online`으로 override.

### Step 8 — 학습
원본 (24 runs, 매우 무거움):
```bash
cd /home/theo_lab/Downloads/3d_diffusion_in_robocasa/FPVNet
bash scripts/dp3.sh
```

축소판 (권장, 단일 GPU) — `scripts/dp3.sh`의 모든 override를 보존하고 `env_group`/`seed`만 줄이는 형태:
```bash
python run.py --config-name=robocasa_dp3_config \
    --multirun agents=dp3_agent \
    trainers=dp3_trainer \
    agent_name=dp3 \
    agents/model=dp3/dp3_policy \
    scale_data=False \
    use_full_point_cloud=False \
    use_sampled_point_cloud=True \
    use_segmented_point_cloud=False \
    use_pc_color=False \
    env_group=doors,drawer \
    seed=0
```
(top-level config의 `defaults`는 `agents: beso_agent` / `trainers: base_trainer`이므로 DP3에는 위 override 그룹을 반드시 같이 줘야 함.)

### Step 9 — 평가
`run.py` 마지막에 자동으로 `env_sim.test_agent(agent)` 실행. task당 50 rollouts.

---

## Loss / Eval / 체크포인트

### Loss
- 정의 위치: `agents/models/dp3/dp3_policy.py:238` `compute_loss(batch)`
- 형태: **`F.mse_loss(pred, target, reduction="none")`** (line 343) → `loss_mask` 곱 → 평균
- `prediction_type`은 `dp3_policy.yaml`의 `noise_scheduler` 설정 따름 (`sample` 모드면 action 자체, `epsilon`이면 노이즈)
- 반환: `loss_dict = {"bc_loss": loss}` (이름은 BC지만 실제는 diffusion MSE)
- wandb 키:
  - `bc_loss` — **per-batch** (매 step 로깅, `trainers/dp3_trainer.py:115`)
  - `epoch_train_loss` — **per-epoch** 평균 (`trainers/dp3_trainer.py:96`)

### 학습 루프 (`trainers/dp3_trainer.py:60~105`)
- `for num_epoch in range(100)`: 모든 배치 forward → `loss.backward()` → `optimizer.step()` → `scheduler.step()` (cosine)
- EMA: `if_use_ema=False` (현재 기본값) → EMA 가중치 미생성
- **중간 평가 없음** — 학습 도중 sim rollout 안 함. epoch별 wandb 로깅은 train loss뿐.

### 체크포인트
- 저장 시점: **마지막 epoch (= 100) 끝난 후 1번만** (`save_every_n_epochs=100`)
- 파일: `last_model.pth` (EMA 켜진 경우 `last_model_ema.pth` 추가)
- 저장 방식: `torch.save(self.model, path, pickle_module=dill)` — **state_dict가 아니라 model 객체 전체를 dill pickle** (`agents/dp3_agent.py:130-131`)
- 저장 경로 (Hydra working_dir):
  ```
  logs/${env_group}/sweeps/${agent_name}/${date}/${time}/${hydra.job.override_dirname}/last_model.pth
  ```
  예: `logs/doors/sweeps/dp3/2026-05-12/14-30-00/seed=0/last_model.pth`
- **Resume 기능 없음** — 중간 epoch에서 끊기면 처음부터.
- 로딩: `dill`로 pickle된 모델 객체이므로 같은 코드베이스(같은 모듈 경로) 환경에서만 `torch.load(path, pickle_module=dill)` 가능.

### 평가 (`simulation/robocasa_pc_sim.py:92` `test_agent`)
- 흐름: `run.py` 끝에서 자동 호출 → `env_group`의 모든 task에 대해 환경 생성 → task당 `num_episode=50` rollouts → `env._check_success()` 체크
- 메트릭: **success rate = success_count / 50**
- 결과:
  - stdout: `print(f"Success rate: {success_rate}")`
  - wandb: `wandb.log({f"{env_name}_average_success": success_rate})` (task별로 별도 키)
- 한 sweep run은 한 env_group의 모든 task를 다 돌리고 끝남 → 결과 비교는 wandb 또는 stdout 로그 수집해야 함.
- wandb를 켜두면 task별 평균 성공률이 자동으로 한 프로젝트에 모임. `mode:disabled`로 두면 stdout만으로 결과 추적해야 하니 **eval 결과 보려면 wandb online 권장**.

---

## 주의사항 / 함정

- **PC 전처리가 병목**: 12~24h. 백그라운드로 돌리거나 task subset만 처리.
- **pytorch3d 빌드 실패** 잦음 → GCC 9+ 필수. 회피 불가 (`simulation/robocasa_pc_sim.py` → `FPSPointCloudSampler` → `pytorch3d.ops` 의존 체인이 평가 단계에서 강제됨).
- `--no-build-isolation` 누락 시 일부 패키지 (특히 pytorch3d) 빌드 실패.
- `macros_private.py`의 `DATASET_BASE_PATH`가 잘못되면 dataset_states_to_obs 단계에서 침묵 실패. 파일 정확한 경로: `custom_robocasa/custom_robocasa/robocasa/macros_private.py`.
- 8개 env_group × 3 seed 그대로 돌리면 단일 GPU에선 비현실적 → 반드시 축소.
- DP3는 **BESO 내장 CLIP** 사용 → OpenAI CLIP git+ 설치 불필요 (3DA와 혼동 주의).
- SUGAR Pointnet2 / Dropbox pretrain weight → FPV-SUGAR 전용, DP3에서는 만지지 말 것.

---

## 리소스 추정

| 항목 | 논문 기준 | 단일 RTX 4090 (추정) |
|------|----------|---------------------|
| GPU | 4× RTX 6000 Ada | 1× RTX 4090 |
| Batch | 512 | 256 |
| 시간 / group | ~6h | ~3-5h |
| 전체 (8g × 3s) | — | ~480h (비현실적, 축소 필수) |
| 디스크 (raw) | — | 15~25GB |
| 디스크 (PC hdf5) | — | ~50GB |
| 디스크 (kitchen assets) | — | 5~7GB |

---

## 참고 링크

- FPVNet: https://github.com/ALRhub/FPVNet
- custom_robocasa: https://github.com/ALRhub/custom_robocasa
- RoboCasa 원본: https://github.com/robocasa/robocasa
- DP3 (3D Diffusion Policy) 원논문: https://arxiv.org/abs/2403.03954
- BESO (CLIP lang encoder 출처): https://github.com/intuitive-robots/beso
