# DP3 → RoboCasa Porting Notes

## 🎯 목적

DP3 (3D Diffusion Policy, Ze et al. RSS 2024)를 **RoboCasa atomic 태스크에 파인튜닝**하는 것이 목표. DP3-on-RoboCasa의 공개 레퍼런스 구현은 없기 때문에, 같은 벤치마크에서 검증된 **DP fork** (`robocasa-benchmark/diffusion_policy`, 2D Diffusion Transformer)를 코드 패턴 레퍼런스로 차용한다 — 데이터 파이프라인(HDF5→중간 포맷), 액션 인터페이스(12D delta + sub-key), eval 프로토콜(HTTP/chunk replay)은 그대로 미러링하고, 정책 모델/관측 표현만 RGB→3D point cloud로 교체.

**시작 태스크**: **`CloseDrawer`** (atomic, 정지 베이스, easy 카테고리). PnP* 계열은 DP3가 1-4%로 실패하므로 v2 이후로 미룸.
**v1 목표 SR**: **40-60%** — PointMapPolicy 페이퍼에서 DP3 setup이 `CloseDrawer` 60.0%, `TurnOffMicrowave` 62.7%, `OpenDrawer` 46.0% 기록. ⚠️ 페이퍼의 "DP3 27.04% 평균"은 **표준 DP3가 아니라** DP3 encoder + xLSTM backbone 그래프트 — 우리(표준 UNet1D)와 직접 비교 불가, 태스크별 수치를 앵커로 사용.

---

## 📁 참고 파일

### A. DP3 본체 (`/home/theo/workspace/3D-Diffusion-Policy/3D-Diffusion-Policy/`)

| 파일 | 설명 |
|---|---|
| `train.py:37` `TrainDP3Workspace` | Hydra 진입점. dataset/policy/env_runner를 yaml로부터 instantiate, EMA + TopKCheckpoint |
| `diffusion_policy_3d/policy/dp3.py` | DP3 정책 — `predict_action` / `compute_loss`, DDIM 10-step inference, `prediction_type=sample` |
| `diffusion_policy_3d/model/vision/pointnet_extractor.py:109` | `PointNetEncoderXYZ` (MLP 3→64→128→256 + LayerNorm + maxpool + Linear→`out_channels`). `out_channels`는 yaml 설정값(DP3 low-dim 디폴트 64). line 204 `DP3Encoder`가 PC+state concat |
| `diffusion_policy_3d/model/diffusion/conditional_unet1d.py` | FiLM-conditioned UNet1D, `down_dims=[512,1024,2048]`, kernel=5 |
| `diffusion_policy_3d/config/dp3.yaml` | 기본 하이퍼파라미터 — Tp=16/To=2/Ta=8, DDIM 100→10, AdamW lr 1e-4, batch 128 |
| `diffusion_policy_3d/config/task/metaworld_assembly.yaml` | task yaml 템플릿 — `shape_meta`, `env_runner._target_`, `dataset._target_`, `zarr_path` |
| `diffusion_policy_3d/dataset/adroit_dataset.py:86` | Dataset 템플릿 — `__getitem__`이 `{obs:{point_cloud, agent_pos}, action}` 반환, `get_normalizer()`가 in-memory buffer에 fit |
| `diffusion_policy_3d/env_runner/dexart_runner.py` | env_runner 템플릿 — MultiStepWrapper To 스택 + Ta chunk replay, `SimpleVideoRecordingWrapper`, `test_mean_score` 반환 |
| `diffusion_policy_3d/gym_util/mujoco_point_cloud.py` | **참고만** — depth→PC 변환 파이프라인 (open3d). 정적 `cam_mat0` 사용 → 모바일 베이스 비호환 |
| `INSTALL.md` | conda env `dp3` 설치 (Python 3.8, mujoco-py 2.1, CUDA 11.7/12.1) |

### B. RoboCasa 본체 (`/home/theo/workspace/robocasa_docker/`)

| 파일 | 설명 |
|---|---|
| `robocasa/wrappers/gym_wrapper.py` | `RoboCasaGymEnv` — `gym.make("robocasa/<TaskName>")` 진입점. 396 태스크 등록됨 |
| `robocasa/utils/env_utils.py:57` `create_env(...)` | env 팩토리. `:118` `camera_depths=False` 하드코딩 — **kwarg pass-through 수정 필요** |
| `robocasa/utils/env_utils.py:134-146` `convert_action(flat12)` | 12D → 5 sub-key dict (eef_pos `[0:3]` / eef_rot 3D axis-angle `[3:6]` / gripper `[6:7]` / **base_motion `[7:11]` 4D** / control_mode `[11:12]`). HTTP `/act` 응답 키 이름과 일치: `action.end_effector_position / action.end_effector_rotation / action.gripper_close / action.base_motion / action.control_mode` |
| `robocasa/scripts/download_datasets.py` | mimicgen_human 데모 공식 다운로더 (S3 → 로컬) |
| `robocasa/scripts/dataset_scripts/dataset_states_to_obs.py` | states + model_file을 sim에서 replay하면서 obs (RGB / depth optional) HDF5 재생성 |
| `robocasa/utils/camera_utils.py` (lines 64–113) | RoboCasa 카메라 extrinsics 하드코딩 (fovy=60°, 128×128 default) |
| `robosuite/robosuite/utils/camera_utils.py:20/39/106` | `get_camera_intrinsic_matrix` / `get_camera_extrinsic_matrix` (dynamic `cam_xpos/cam_xmat`) / `get_real_depth_map` — 우리 PC 생성기가 호출할 함수 |
| `groot_docker_n1.5/serve_groot.py` | eval HTTP 서버 템플릿 (`/health /reset /act`) — `serve_dp3.py` 미러 대상 |
| `examples/run_groot_eval.py` | eval 클라이언트 — obs payload 빌드 → POST /act → action chunk replay → success OR-누적 |
| `VLA_COMMUNICATION_PROTOCOL.md` | HTTP 프로토콜 스펙 (obs/action 키, action chunk shape `[T=16, dim]`) |

### C. DP fork — 2D Diffusion Policy on RoboCasa 레퍼런스

| 파일 / URL | 설명 |
|---|---|
| `https://github.com/robocasa-benchmark/diffusion_policy` | RoboCasa-adapted 2D Diffusion Policy fork. **데이터/액션/eval 패턴 차용 원천** |
| `diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py` | Transformer + ResNet18FiLM 정책 — DP3와 직접 비교용 baseline |
| `diffusion_policy/dataset/lerobot_dataset.py` | LeRobot parquet 로더 — obs 키 명명 규칙(`robot0_*`) 참고 |
| `diffusion_policy/env_runner/robomimic_image_runner.py` | RoboCasa env 호출 패턴 — n_test=50, max_steps=`horizon×1.5`, action chunk 16 |
| `diffusion_policy/config/task/robocasa/pretrain_human300.yaml` | `shape_meta` + obs 키 명명 규칙 — 우리 task yaml의 청사진 |
| `eval_robocasa.py` | task별 평가 루프 — `gym.make("robocasa/<task>")` 직접 호출, episode-level lang_emb 주입 |

### D. ALRhub (KIT) — PointMapPolicy / FPV-Net 그룹 인프라

| URL | 설명 |
|---|---|
| `https://github.com/ALRhub/custom_robocasa` | **RoboCasa PC wrapper 코드** — 우리 `robocasa_point_cloud.py` 짤 때 직접 참고. 같은 그룹이 DP3 baseline 돌릴 때 사용한 PC 전처리 |
| `https://github.com/ALRhub/X_IL` | DDPM/BC/flow matching/BESO IL 프레임워크. DP3 모듈은 없지만 RoboCasa 통합 보일러플레이트 |
| `https://github.com/ALRhub/PointMapPolicy` | 페이퍼 공식 repo — **빈 껍데기** (LICENSE+README만, 코드 0줄). 참고 가치 없음 |
| `https://github.com/ALRhub/FPVNet` | 같은 그룹의 비DP3 3D 정책 RoboCasa 사례 (SUGAR 인코더). DP3 baseline 22.75% 별도 보고. v3 방향성 |

⚠️ **공개 DP3-on-RoboCasa 코드/체크포인트는 어디에도 없음** — 페이퍼 2편 (PointMapPolicy 27.04%, FPV-Net 22.75%) 모두 내부 구현, 비공개. HF / 200+ DP3 fork / Papers-with-Code 전수 확인 완료.

> 이하 §🔗/§🔧 섹션은 위 §C 레퍼런스의 어떤 패턴을 그대로 가져오고 어떤 부분을 교체할지 정리한다.

---

## 🔗 DP fork (2D)와 닮은 점 → 그대로 차용 가능

| 항목 | DP fork | DP3 (우리 v1) | 차용 방식 |
|---|---|---|---|
| Diffusion head | Conditional Transformer + DDPM 100 step | UNet1D + FiLM + DDIM 100→10 step (`prediction_type=sample`) | 학습 루프/EMA/스케줄러 동일, head 구조만 교체 |
| 액션 dim & 의미 | 12D delta `[eef_pos3 + rot_6d6 + grip1 + base_motion1 + ctrl_mode1]` | **10D 평탄** `[eef_pos3 + rot_6d6 + grip1]`, base_motion(4)/ctrl_mode(1)는 wrapper에서 0 채움 | `convert_action()` 12D 계약 그대로 유지 |
| 카메라 키 | `agentview_left/right/eye_in_hand` 3× **256×256 RGB** | `agentview_left + eye_in_hand` 2× **128×128 RGB+Depth** → PC 1024×3 (xyz only) | 카메라 이름 명명규칙 재사용 |
| State 키 형식 | base_pos(3) + base_quat(4) + eef_pos(3) + eef_quat(4) + grip_qpos(2) | **13D 평탄** `eef_pos3 + eef_axis_angle3 + grip_qpos2 + base_pos3 + base_yaw_sin_cos2` | 같은 sub-key 출처, flatten만 다름 |
| 언어 처리 | `lang_emb 768` (DistilBERT, episode 1회 인코딩 후 매 step tile) | **v1 없음** (단일 태스크 — lang_emb 상수), v2에서 PointNetEncoder에 FiLM gate ~80 LoC 추가 | 데이터 변환 시 `ep_meta.lang` 문자열만 보존 |
| Eval 프로토콜 | n_test=50, `max_steps = horizon × 1.5`, action chunk 16 | n_test=20, `max_steps = horizon × 1.5`, action chunk **Ta=8** (yaml 기본) | `examples/run_groot_eval.py` 루프 그대로 미러 |
| Data 포맷 | LeRobot parquet + video + `modality.json` | **DP3 zarr** (`point_cloud / state / action / meta/episode_ends`) | 변환 스크립트로 일회성 변환 |
| 정규화 | action identity / state per-key | `LinearNormalizer(mode='limits')` action·state·point_cloud 전체에 per-dim [-1,1] | DP3 표준 그대로 |

**차용하지 않는 것**: (1) `lang_emb 768` — v1은 단일 태스크라 상수 신호이며 인코더 통합에 ~80 LoC 변경이 필요해 v2로 미룸. (2) RGB 이미지 obs — DP3는 입력이 PC+state로 고정이라 ResNet18FiLM 분기를 통째로 버리고 depth→PC 경로만 살림 (RGB는 video log 용도로만 유지).

### 🧭 구조도 (Side-by-Side)

```mermaid
flowchart LR
    subgraph DPfork ["🟦 RoboCasa DP fork (2D, 기존)"]
        direction TB
        Text["📝 Text<br/>'close the drawer'"]:::lang
        BERT["DistilBERT<br/>→ lang_emb [768]<br/>(episode 1회 + tile)"]:::lang
        RGB["📷 3× RGB [256×256×3]<br/>agentview L / R / wrist"]:::vision2d
        Res["ResNet18ConvFiLM × 3<br/>(lang_emb으로 FiLM 변조)<br/>→ vision tokens [N, 512]"]:::vision2d
        State1["state [12D]<br/>eef_pos + quat + grip + base"]:::state
        Concat1["concat:<br/>lang(768) + vision(N×512) + state(12)<br/>= obs_features"]:::concat
        TF["⚙️ Diffusion Transformer<br/>12 layers × 512 dim, cross-attn<br/>DDPM 100 step (train &amp; infer)<br/>predict ε (epsilon)"]:::head_t
        A1["🎬 action [16 × 12D]<br/>pos3 + rot_6d6 + grip1 + base1 + mode1<br/>→ convert_action()"]:::action

        Text --> BERT
        BERT -.FiLM.-> Res
        RGB --> Res
        Res --> Concat1
        BERT --> Concat1
        State1 --> Concat1
        Concat1 --> TF
        TF --> A1
    end

    subgraph DP3v1 ["🟩 DP3 (3D, v1 우리 계획)"]
        direction TB
        NoLang["❌ 언어 입력 없음<br/>(v1 = 단일 태스크 가정)"]:::nolang
        Depth["📷 1× Depth [128×128×1]<br/>agentview_left 만"]:::vision3d
        PCgen["Depth → Point Cloud unproject<br/>K + T (robosuite dynamic)<br/>bbox crop + FPS 1024<br/>→ PC [1024 × 3] (xyz only)"]:::vision3d
        PNet["PointNetEncoderXYZ<br/>MLP 3 → 64 → 128 → 256<br/>+ LN + MaxPool + Linear → 64<br/>→ pc_emb [64]"]:::vision3d
        State2["agent_pos [13D]<br/>eef_pos + axis_angle + grip + base + yaw_sincos<br/>+ state MLP → 64"]:::state
        Concat2["concat:<br/>pc_emb(64) + state_emb(64)<br/>= global_cond [128]"]:::concat
        UNet["⚙️ Diffusion UNet1D + FiLM<br/>down_dims [512, 1024, 2048]<br/>DDIM 100 → 10 step (infer)<br/>predict sample (x0)"]:::head_u
        A2["🎬 action [8 × 10D]<br/>pos3 + rot_6d6 + grip1<br/>wrapper 0-fill → 12D<br/>→ convert_action()"]:::action

        NoLang -.x.-> PNet
        Depth --> PCgen
        PCgen --> PNet
        PNet --> Concat2
        State2 --> Concat2
        Concat2 -.FiLM.-> UNet
        UNet --> A2
    end

    classDef lang fill:#fff3cd,stroke:#f0ad4e,color:#000
    classDef nolang fill:#f8d7da,stroke:#d9534f,color:#000
    classDef vision2d fill:#cfe2ff,stroke:#0d6efd,color:#000
    classDef vision3d fill:#d4edda,stroke:#198754,color:#000
    classDef state fill:#e2e3e5,stroke:#6c757d,color:#000
    classDef concat fill:#fde2e4,stroke:#dc3545,color:#000
    classDef head_t fill:#f8d7da,stroke:#dc3545,color:#000
    classDef head_u fill:#d1ecf1,stroke:#0dcaf0,color:#000
    classDef action fill:#fff,stroke:#000,color:#000
```

**한눈 요약** (위 도식의 색깔 매핑):
- 🟨 lang (DP fork만) / 🟥 lang 부재 (DP3) ← **가장 큰 차이 #1**
- 🟦 2D vision (RGB+ResNet) / 🟩 3D vision (Depth→PC+PointNet) ← **차이 #2**
- 🟪 cond 주입: DP fork는 lang→vision FiLM, DP3는 PC+state→UNet FiLM
- 🟥 Transformer+DDPM / 🟦 UNet1D+DDIM ← head 구조 + sampler 다름
- ⚪ 출력: 둘 다 chunk이지만 길이/차원 다름 (16×12D vs 8×10D, DP3는 wrapper에서 12D 확장)

---

## 🔧 수정 필요 항목

### 1) DP3 측 신규 파일 (`/home/theo/workspace/3D-Diffusion-Policy/3D-Diffusion-Policy/diffusion_policy_3d/`)

| 파일 | 한 줄 목적 |
|---|---|
| `env/robocasa/__init__.py` | `RobocasaDP3Env` export, Hydra `_target_` 해상도용 |
| `env/robocasa/robocasa_wrapper.py` | `gym.Wrapper`: obs dict → `{point_cloud, agent_pos}` 변환, 10D 정책 출력 → 12D `convert_action` 입력 확장 (base/ctrl_mode 0 채움, 6D→axis_angle 회전 변환) |
| `env/robocasa/robocasa_point_cloud.py` | **신규 PC 생성기** `RoboCasaPointCloudGenerator(sim, cam_names, img_size=(128,128))` — `mujoco_point_cloud.py`를 재사용하지 않는 사유는 §📁 표 A 참조. 내부적으로 `robosuite.utils.camera_utils.get_camera_extrinsic_matrix` (dynamic) + `get_camera_intrinsic_matrix` + `get_real_depth_map`을 매 step 호출, Open3D `create_from_depth_image` 백프로젝션 → 두 카메라 PC concat → bbox crop → FPS 1024 |
| `env_runner/robocasa_runner.py` | `RobocasaRunner(BaseRunner)` — `dexart_runner` 패턴 복제 |
| `dataset/robocasa_dataset.py` | `RobocasaDataset(BaseDataset)`: zarr 로드, `LinearNormalizer` fit, `{obs:{point_cloud, agent_pos}, action}` 반환 |
| `config/task/robocasa_close_drawer.yaml` | `shape_meta: point_cloud:[1024,3] / agent_pos:[13] / action:[10]`, zarr_path, env_runner·dataset `_target_` |

### 2) DP3 측 수정

- `env/__init__.py` — `from .robocasa import RobocasaDP3Env` 한 줄 추가.
- `gym_util/mjpc_wrapper.py` — `ENV_POINT_CLOUD_CONFIG`에 RoboCasa 항목 추가:
  ```yaml
  robocasa_close_drawer:
      num_points: 1024
      use_point_crop: true
      point_cloud_bbox: <dynamic, wrapper에서 주입>
      camera_names: ['robot0_agentview_left', 'robot0_eye_in_hand']
      img_size: [128, 128]
  ```

### 3) RoboCasa 측 (최소 침습) — `/home/theo/workspace/robocasa_docker/`

- `robocasa/utils/env_utils.py:118` — `create_env(...)`에 `camera_depths` kwarg 통과 (default `False` 유지, 변환기와 eval wrapper에서 `True`로 호출).
- `scripts/convert_robocasa_to_dp3_zarr.py` **(NEW)** — `--src demo_obs.hdf5 --task CloseDrawer --out data/robocasa_<task>.zarr`. demo별로 `states+model_file`을 sim에 replay하면서 `camera_depths=True`로 depth 재생성 → `RoboCasaPointCloudGenerator` → 13D agent_pos + 10D action (**데이터셋 변환 방향: `axis_angle → rot_6d`**) → `ReplayBuffer.append` → zarr write.
  - ⚠️ 액션 변환은 `robomimic.utils.torch_utils.TorchUtils.axis_angle_to_rot_6d` (이미 존재)를 재사용하되, `robomimic_dataset_utils.py:60-78`의 action-length gate(7/8D-only)는 우회하고 직접 12D action의 `[3:6]` 슬라이스에 적용.
- `scripts/serve_dp3.py` **(선택, P4 eval)** — `serve_groot.py` 미러. `/health /reset /act` FastAPI. `predict_action` 호출 후 `action: [T=8, 10]` 청크를 반환, client는 기존 `run_groot_eval.py` 패턴 그대로 사용 (`--server` URL만 교체).

### 4) Action sub-key 매핑 (wrapper에서, **추론 시 역변환 방향: `rot_6d → axis_angle`**)

```python
# policy 10D output → 12D env action via convert_action()
a = policy_action  # shape (10,)

eef_pos        = a[0:3]                              # delta xyz
rot_6d         = a[3:9]                              # 6D rotation
rot_matrix     = rotation_6d_to_matrix(rot_6d)       # pytorch3d / 직접 구현
eef_axis_angle = matrix_to_axis_angle(rot_matrix)    # robosuite.utils.transform_utils.mat2axisangle
gripper        = a[9:10]

env_action_12d = np.concatenate([
    eef_pos,                # [0:3]   action.end_effector_position
    eef_axis_angle,         # [3:6]   action.end_effector_rotation
    gripper,                # [6:7]   action.gripper_close
    np.zeros(4),            # [7:11]  action.base_motion  (atomic 정지 베이스)
    np.zeros(1),            # [11:12] action.control_mode
])
# convert_action(env_action_12d) → sub-key dict (env_utils.py:134-146)
```

> 상태(`agent_pos`)도 동일 manifold 유지 위해 axis-angle 사용 (단, 데이터셋에서 quat→axis-angle 시 antipodal 부호 통일 필요 — §⚠️ 검증 위험 #참조).

### 5) bbox crop (단일 태스크 정지 베이스)

```python
# robocasa_wrapper.reset()에서 episode 1회 계산
base_xy = self.env.sim.data.body_xpos[
    self.env.sim.model.body_name2id('mobilebase0_support')
][:2]
self.point_cloud_bbox = np.array([
    [base_xy[0] - 0.8, base_xy[1] - 0.8, 0.7],   # ~1.6m × 1.6m × 0.6m
    [base_xy[0] + 0.8, base_xy[1] + 0.8, 1.3],
])
# RoboCasaPointCloudGenerator.generate(sim, bbox=self.point_cloud_bbox)
```

> 위 수정안의 가정(정지 베이스, bbox 정의 가능 등)이 깨질 수 있는 지점과 외부 검증 자료를 다음 섹션에서 다룬다.

---

## ⚠️ 한계와 고민

### 본질적 미지수

1. ⚠️ **Depth 정밀도 — 거리 양자화 노이즈**: RoboCasa는 `<map znear="0.001" />` + zfar=50 디폴트. `mjStatistic.extent` ≈ 3-5 m인 kitchen 씬에서 실효 near ≈ 5 mm / far ≈ 250 m가 되어 uint24 depth 정밀도가 near plane 근처에 몰린다. 정작 우리가 쓸 cabinet/sink/counter 표면은 1-2 m 구간이라 양자화 노이즈가 ↑. DP3 원 페이퍼는 ~1 m tabletop 가정 → 분포가 다르다. **P1에서 시각화로 표면 두께/노이즈 직접 확인 필수**.
2. ⚠️ **bbox crop이 모든 걸 결정**: DP3 페이퍼 Table VII — Adroit에서 crop 제거 시 SR **78.3 → 45.3** (단일 어블레이션 중 최대 낙폭). Kitchen 스케일에선 crop 정의 자체가 더 어렵다. 너무 좁으면 target object를 놓치고, 너무 넓으면 배경 점이 인코더 입력을 압도한다. 단일 태스크 + 정지 베이스 한정으론 `mobilebase0_support` body pos를 기준 잡고 ±0.8 m 동적 crop이 가능하지만, 베이스가 움직이는 순간 무너진다.
3. ⚠️ **모바일 베이스 미테스트**: DP3 / iDP3 / ManiCM 등 후속작 모두 **정지 베이스에서만** 검증됨. CloseDrawer 같은 atomic은 베이스가 정지 → 회피 가능하지만, 내비 + 매니퓰레이션이 섞이는 composite는 보장 없음.
4. ⚠️ **클러터 위험**: ClutterDexGrasp (arXiv 2506.14317)는 DP3 단독으로 클러터 씬에서 **0% SR**을 보고 — curriculum + synthetic robot PC augmentation + 20k traj가 있어야 동작. RoboCasa atomic은 그 정도 클러터는 아니지만, drawer 내부 / fridge 내부 / 가득 찬 sink는 동일한 위험이 잠재.
5. ⚠️ **언어 부재**: DP3는 lang conditioning이 없어서 멀티태스크 일반화에 약하다 (저자도 페이퍼 limitation에서 "this work doesn't address long horizons" 정도로만 언급). v1 단일 태스크는 OK, 멀티태스크로 확장하려면 별도 lang hook (FiLM into encoder / pooled token) 설계 필요.

### 검증된 위험 (Wave-1 E2)

1. ⚠️ **EGL depth 미검증**: 컨테이너에서 RGB 렌더링은 "OpenGL error 0x501" 우회로 통과했지만 `sim.render(..., depth=True)`는 단 한 번도 검증된 적이 없다. depth FBO가 EGL 컨텍스트에서 silent zero일 가능성 — **P0 첫날 스모크 테스트가 게이트**.
2. ⚠️ **py3.8 (DP3) vs py3.11 (RoboCasa) 공존 불가**: DP3 INSTALL.md는 python3.8 + mujoco-py 2.1 + numpy<1.24, RoboCasa Dockerfile은 python3.11 + mujoco 3.3.1 + numpy 2.2.5. 단일 conda env 불가 → **zarr를 교환 매개로 두 컨테이너 분리** (states_to_obs는 RoboCasa 측, 학습은 DP3 측).
3. ⚠️ **`extract_action_dict`는 7/8D만 처리** → 자체 변환기 필수 (라인 번호·구현 §🔧 §3 참조).
4. ⚠️ **`LinearNormalizer`는 클립 안 함**: `normalizer.py:264-278` mode='limits'가 train buffer의 min/max로 [-1,1] 스케일만 하고 clip은 없다. Test 레이아웃 PC가 train 범위 밖으로 빠지면 OOD 점이 그대로 PointNet에 들어가서 LayerNorm으로도 흡수 불가. **bbox crop을 넉넉히 + 다양한 레이아웃 샘플링**으로 완화.
5. ⚠️ **OSC_POSE 8-step open-loop drift**: 20 Hz × 8 step = 0.4 s 무피드백 delta 적분. 접촉 천이가 잦은 PnP에서 drift 누적이 큼. **Ta=4부터 시작해서 수렴 후 8로 늘리는 게 안전**.

---

## 📚 더 참고할 자료 (PointMapPolicy 외)

> 시간축 정렬: **v1 재현 (PointMapPolicy)** → **v2 인코더 교체 (iDP3)** → **v3 표현/일반화 (EquiBot / 3D Diffuser Actor / PointMapPolicy structured PC)**.

| 자료 | 왜 봐야 하나 |
|---|---|
| **PointMapPolicy** (arXiv 2510.20406, NeurIPS 2025) | ALRhub. ⚠️ 페이퍼의 "DP3 27.04%"는 **DP3 인코더 + xLSTM backbone 그래프트** (표준 DP3 아님). 진짜 가치는 **태스크별 SR 표** — CloseDrawer 60 / TurnOffMicrowave 62.7 / TurnSinkSpout 58.7 / OpenDrawer 46 / TurnOn·OffSinkFaucet 42 / TurnOnMicrowave 39.3 vs **PnP* 1-4%** (전부 실패). 본인들 PMP method는 49.12% — v3 방향성 |
| **FPV-Net** (arXiv 2502.12320, 2025.02) | 같은 ALRhub 그룹 비DP3 3D 정책 RoboCasa 사례. DP3 baseline 22.75% 별도 보고 (PointMapPolicy 수치와 다른 setup). SUGAR 인코더 사용 |
| **iDP3** (arXiv 2410.10803) | 같은 저자 (Yanjie Ze) 후속작. **egocentric PC frame + pyramid CNN encoder + 4096 pts + voxel+uniform sampling**. 단 stationary humanoid만 검증, 모바일 베이스/kitchen scale 미검증. v2 인코더 교체 후보 |
| **ManiCM** (arXiv 2406.01586) | DP3 + consistency model로 inference 25 FPS 가속. 실배포 / HTTP eval 지연이 병목일 때 |
| **HumanoidGen** (arXiv 2507.00833) | DP3가 bimanual humanoid manipulation에서 동작하는 사례 — 양손/멀티핸드 확장 시 참고 |
| **EquiBot** (arXiv 2407.01479) | SIM(3)-equivariant DP3. PC 자체에 회전/병진 불변성 부여 → 레이아웃 일반화 개선 가능성. v3 후보 |
| **Motion Before Action** (arXiv 2411.09658) | DP3 + object motion priors. object-centric 보조 신호 추가 |
| **EL3DD** (arXiv 2511.13312) | language-conditioned 3D diffusion (단, 3D Diffuser Actor 기반이지 DP3 아님). 멀티태스크 lang hook 설계 시 인터페이스 참고용 |
| **3D Diffuser Actor** (arXiv 2402.10885) | DP3 경쟁 모델 — 멀티뷰 PC + 3D 트랜스포머. 같은 RoboCasa에 붙이면 비교 baseline이 될 수 있음 |
| **DP3 GitHub Issues** | #40 EGL/OpenGL, #2 MuJoCo build, #125 GPU 활용도 낮음 — **포팅 함정을 미리 점검** |
| **RoboCasa docs** | `docs/benchmarking/policy_learning_algorithms.md` — DP는 외부 fork로 관리한다는 공식 입장 + API 표면 |
| **DP3 paper Table VII** | 어블레이션 표 전체 정독 권장. 포팅 중 하나라도 빠뜨리면 baseline 못 맞춤 (sample vs epsilon, DDIM vs DPM-solver, LayerNorm 등) |
