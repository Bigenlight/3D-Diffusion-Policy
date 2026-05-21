# DP3 × RoboCasa drawer — Docker 학습 + Eval Runbook

> PC 추출 절차는 별도: [`DP3_PC_EXTRACTION_RUNBOOK.md`](./DP3_PC_EXTRACTION_RUNBOOK.md).
> 이 문서는 추출이 끝난 상태에서 **Docker 빌드 → 학습 → eval(+영상)** 까지.

검증된 결과 (2026-05-13, 8× RTX A4000 16GB):
- 학습 100 epoch ≈ **59분** · loss 0.110 → 0.001 plateau
- Eval n=10: CloseDrawer **80%** / OpenDrawer **30%** / drawer 평균 **55%** (FPVNet 페이퍼 53% 일치)

---

## 0. 사전 조건

> 패러다임: PC 추출은 **호스트 conda env**에서 수행 ([`DP3_PC_EXTRACTION_RUNBOOK.md`](./DP3_PC_EXTRACTION_RUNBOOK.md)), 본 문서는 그 산물 hdf5만 **Docker**에서 사용.

- PC 추출 완료 → `processed_demo_128_128.hdf5` 두 task (`CloseDrawer/2024-04-30` + `OpenDrawer/2024-05-03`) 모두 `/home/woonsang/theo/robocasa_data/v0.1/single_stage/kitchen_drawer/` 아래 존재
- `kitchen_assets/` (~8.2 GB)는 PC 추출 단계 산물로 `/home/woonsang/theo/FPVNet/custom_robocasa/.../models/assets/` 아래 존재 (Docker `:rw` 마운트 대상)
- 호스트 NVIDIA driver **≥ 550** (cu124 forward-compat) + `nvidia-container-toolkit` 설치됨
- Docker daemon 접근 가능 (`docker` 그룹 미가입 시: `sudo usermod -aG docker $USER && newgrp docker`)
- wandb 신형 키 (`wandb_v1_...`) 호스트 `~/.wandb_env`에 `export WANDB_API_KEY=...` 저장됨

---

## 1. Docker 빌드

| 항목 | 값 |
|---|---|
| Base | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` (호스트 cu124와 ABI 일치, nvcc/dev 헤더 포함) |
| Python / torch | 3.10 / 2.4.0+cu124 |
| pytorch3d | 0.7.9 source build (FORCE_CUDA=1, `TORCH_CUDA_ARCH_LIST=8.6` Ampere) |
| 빌드 시간 / 이미지 | ~40-60분 / **17.9 GB** |

```bash
docker build -t fpvnet-dp3:latest /home/woonsang/theo/docker
```

> 비-Ampere GPU(예: Hopper sm_90, Ada sm_89, Turing sm_75)는 Dockerfile의 `TORCH_CUDA_ARCH_LIST=8.6`을 해당 arch로 변경 후 `docker build --no-cache ...` 재빌드 필요 (pytorch3d source build 때문).

### 빌드 함정 2개

1. **Anaconda ToS** — `Miniconda3-latest`는 신정책으로 main/r 채널 사용 전 ToS 동의 필요 → 빌드 fail. **Miniforge3** 로 교체(conda-forge default, ToS 회피).
2. **protobuf 충돌** — robocasa 의존성 `tianshou==0.4.10`이 `protobuf~=3.19.0`을 핀해서 wandb 0.26.1의 `.pb2.py`와 ImportError. Dockerfile 마지막에 `pip install --upgrade 'protobuf>=5,<7'` 강제 업그레이드.

---

## 2. Docker 파일 일람

| 경로 | 역할 |
|---|---|
| `/home/woonsang/theo/docker/Dockerfile` | base + apt(EGL stack) + Miniforge + Python 3.10 + torch cu124 + FPVNet clone(submodule) + sed L475 패치 + protobuf 업글 + imageio-ffmpeg |
| `/home/woonsang/theo/docker/run.sh` | wrapper. `smoke` / `train` / `eval` / `video` / `shell` 모드 dispatch + GPU·볼륨·env 마운트 |
| `/home/woonsang/theo/docker/entrypoint.sh` | conda activate base + `WANDB_API_KEY`로 `wandb login --relogin` |
| `/home/woonsang/theo/docker/smoke_test.py` | 8개 sanity 항목 (torch+CUDA / pytorch3d FPS / EGL / MuJoCo render / cameras / macros / kitchen_assets mount / hdf5 / wandb) |
| `/home/woonsang/theo/docker/train_drawer.sh` | 학습 명령 + `WANDB_INIT_TIMEOUT=600` |
| `/home/woonsang/theo/docker/eval_only.py` | eval 전용 (DP3Agent 새로 instantiate + `agent.model = torch.load(.., weights_only=False)` + `env_sim.test_agent`) |
| `/home/woonsang/theo/docker/eval_with_video.py` | eval + RGB 3카메라 + sampled_point_cloud 1024 시각화 mp4 |
| `/home/woonsang/theo/docker/patches/requirements.txt` | FPVNet `requirements.txt` 통째 덮어쓰기. `# 3DA` 섹션(dgl/flash-attn/openai CLIP) 주석, `wandb 0.19.1 → 0.26.1` |
| `/home/woonsang/theo/docker/patches/install_no_assets.sh` | `custom_robosuite`+`custom_robocasa` editable install, open3d만 추가, pytorch3d/assets는 skip (각각 Dockerfile 단계 8/볼륨 마운트로) |
| `/home/woonsang/theo/docker/patches/macros_private.py` | `DATASET_BASE_PATH = "/data/robocasa"` (호스트 bind-mount 경로) |
| `/home/woonsang/theo/docker/.dockerignore` | logs/wandb/robocasa_data/`.git` 제외로 context 슬림화 |

### 호스트 볼륨 마운트 (run.sh에서 자동)
- `/home/woonsang/theo/robocasa_data` → `/data/robocasa` `:ro`
- `/home/woonsang/theo/FPVNet/custom_robocasa/.../models/assets` → 컨테이너 동일 경로 **`:rw`** (MJCFObject 임시 xml write)
- `/home/woonsang/theo/FPVNet/logs` → `/workspace/FPVNet/logs` `:rw`

---

## 3. 학습 실행

```bash
bash /home/woonsang/theo/docker/run.sh train
```

| Hydra override | 값 |
|---|---|
| `env_group` | `drawer` (= OpenDrawer + CloseDrawer) |
| `seed` | `0` |
| `train_batch_size` | `128` (default 256은 A4000 16GB OOM 위험) |
| `num_workers` | `8` |
| `hydra.mode` | `RUN` (default MULTIRUN 회피) |
| `wandb.mode` / `entity` | `online` / `null` → wandb SDK가 사용자 default entity로 자동 fallback (`RwHlabs`) |
| `--config-name` | `robocasa_dp3_config` (default가 BESO/img라 필수) |
| 그 외 | `agents=dp3_agent trainers=dp3_trainer agents/model=dp3/dp3_policy agent_name=dp3 scale_data=False use_full_point_cloud=False use_sampled_point_cloud=True use_segmented_point_cloud=False use_pc_color=False` |

### 결과
- 100 epoch / **59분** / loss **0.110 → 0.001** (epoch 90 이후 plateau)
- ckpt: `/home/woonsang/theo/FPVNet/logs/drawer/runs/dp3/2026-05-13/15-20-48/last_model.pth` (**1.0 GB**, dill pickle된 `self.model` 객체 통째)
- wandb run: https://wandb.ai/RwHlabs/fpvnet_dp3_drawer/runs/po7z9a8r

### 학습 시 발견 함정

| # | 이슈 | 해결 |
|---|---|---|
| 1 | wandb init default 90초 timeout으로 hang | `WANDB_INIT_TIMEOUT=600` env (train_drawer.sh에 반영) |
| 2 | `save_every_n_epochs` 변수만 선언, epoch loop 미사용 — 중간 ckpt 없음 | 마지막 epoch에 `last_model.pth` 1개만 저장. 안전망 필요하면 trainer 루프 직접 패치 |
| 3 | `if_use_ema` 하드코드 False (`configs/trainers/dp3_trainer.yaml:23`) | yaml에서 true로 켜도 무시. EMA 필요 시 코드 수정 |
| 4 | wandb 0.26.1과 `tianshou`의 protobuf 3.19 핀 충돌 | Dockerfile에서 `protobuf>=5,<7`로 업글 |
| 5 | hydra default config = BESO/img | `--config-name=robocasa_dp3_config` + 15개 hydra override 필수 (train_drawer.sh; 표 7개 + agents/trainers/model/scale_data/use_*_point_cloud 등 8개) |
| 6 | 다른 wandb entity로 보내고 싶을 때 | `WANDB_ENTITY=<entity> ./run.sh train` 또는 hydra `wandb.entity=<entity>` override |

> **학습 OOM fallback**: A4000 16GB가 아닌 더 작은 VRAM(예: 12 GB)에서 OOM 발생 시 `train_batch_size=64 num_workers=4`로 낮춰 재시도 (train_drawer.sh 직접 편집 또는 hydra override).

---

## 4. Eval — success rate

```bash
EVAL_NUM_EPISODE=10 EVAL_SEED=0 bash /home/woonsang/theo/docker/run.sh eval
```

| 항목 | 값 |
|---|---|
| 시간 | ~13분 (drawer 2 task × 10 ep) |
| wandb metric | `{Task}_average_success` |
| 결과 (n=10, seed=0) | CloseDrawer **80%** · OpenDrawer **30%** · drawer 평균 **55%** |
| 비교 | FPVNet PointMapPolicy 페이퍼 53% (+2%p 일치) |
| wandb URL | https://wandb.ai/RwHlabs/fpvnet_dp3_drawer/runs/ffyf08hb |

> **Multi-seed 평균/std**: 단일 seed=0 결과는 분산이 크므로 `for s in 0 1 2; do EVAL_NUM_EPISODE=10 EVAL_SEED=$s bash /home/woonsang/theo/docker/run.sh eval; done` 후 wandb summary의 `{Task}_average_success` 3개를 손으로 모아 mean ± std 산출 (자동 집계 스크립트 없음).

---

## 5. Eval — 영상 녹화

```bash
EVAL_TASK=CloseDrawer EVAL_N_EPISODE=3 bash /home/woonsang/theo/docker/run.sh video
```

| 항목 | 값 |
|---|---|
| 영상 구조 | 상단: RGB 3대 카메라(256×256, `robot0_agentview_{left,right}` + `eye_in_hand`) / 하단: `sampled_point_cloud` 1024 점 3D scatter (천천히 회전) |
| 색 | RGB 채널 (시각용. DP3는 `use_pc_color=False`로 XYZ 좌표만 입력) |
| 해상도 | 1440×1080 (1080p) |
| 출력 | `/home/woonsang/theo/FPVNet/logs/eval_videos/{task}_seed{S}_ep{N}.mp4` (각 5-8 MB) |

예시 결과 (CloseDrawer seed=0, 최초 시행분 — 같은 명령으로 재생성 가능): ep0 SUCCESS 219step · ep1 FAIL 500step · ep2 SUCCESS 353step.

---

## 6. Eval 시 발견 함정

| # | 함정 | 해결 |
|---|---|---|
| 1 | `kitchen_assets`를 `:ro`로 마운트 → MJCFObject 임시 xml write fail | `run.sh`에서 `:rw`로 변경 (호스트 영향 무: 임시 xml은 unique 이름·즉시 삭제). 검증: `docker inspect <container> --format '{{range .Mounts}}{{.Source}} -> {{.Destination}} ({{.Mode}}){{"\n"}}{{end}}'`로 `kitchen_assets` 줄이 `rw`인지 확인 |
| 2 | wandb project default `atalay_robocasa_final` (타인) | `eval_only.py`에서 `WANDB_PROJECT` env로 강제 → `fpvnet_dp3_drawer` |
| 3 | `~/.wandb_env`의 `export WANDB_API_KEY=...` 그대로 `--env-file`로 못 줌 | `run.sh`에서 `grep ... \| cut -d= -f2-`로 값만 추출 후 `-e WANDB_API_KEY=$KEY` |
| 4 | `eval_only.py`가 `/workspace/`에 마운트라 FPVNet 패키지 import 실패 | 스크립트 상단 `sys.path.insert(0, "/workspace/FPVNet")` |
| 5 | `agent.predict(obs_dict)` 키 이름 — sampled가 아니라 `"point_cloud"` | `simulation/robocasa_pc_sim.py:181` 패턴 그대로 |
| 6 | torch 2.4+ `torch.load` default `weights_only=True` → dill ckpt 거부 | `torch.load(..., weights_only=False, pickle_module=dill)` 명시 |
| 7 | 컨테이너에 `imageio-ffmpeg` 누락 → mp4 write 실패 | Dockerfile 한 줄 추가 + incremental rebuild |
| 8 | `train_drawer.sh`가 컨테이너에 COPY 안 됨 → No such file | `run.sh`의 train mode에서 호스트 파일 `-v ...:/workspace/train_drawer.sh:ro` mount |

---

## 7. 핵심 한 줄 요약 (다른 task로 확장 시)

다른 group 학습은 §3 명령에서 `env_group=<group>` 만 변경 (`pnp1/pnp2/doors/drawer/sink/stove/coffee/buttons` 8개 중 하나). 단 group의 모든 task가 `processed_demo_128_128.hdf5`로 추출되어 있어야 함 (group → task 매핑은 [`DP3_PC_EXTRACTION_RUNBOOK.md §6`](./DP3_PC_EXTRACTION_RUNBOOK.md)).

## 8. 변경 이력
- 2026-05-13: drawer 1 group 학습+eval 완주 검증 (loss 0.001 plateau, drawer 평균 55%, 페이퍼 53% 일치). Docker 빌드 함정 2개 (Anaconda ToS, protobuf) + 학습/eval 시행착오 13개 기록.
- 2026-05-13 (검수 반영): §0 호스트 conda↔Docker 패러다임/kitchen_assets 출처/NVIDIA driver·toolkit 요건/docker 그룹 가입 명령 추가. §1 이미지 크기 17.8 → 17.9 GB 정정 + 비-Ampere GPU arch 변경 안내. §3 함정 #5 "8개" → "15개 hydra override" 정정 + #6 wandb entity override 추가 + OOM fallback. §4 multi-seed 평균/std 절차. §5 ep 결과를 "예시(최초 시행분)"로 명시. §6 #1에 `docker inspect` 마운트 모드 검증 명령 추가.
