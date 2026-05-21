# OpenDrawer Depth-Outlier 버그 디버깅 기록

> **요약** — FPVNet/custom_robocasa 의 PC 추출 코드에서 OpenDrawer 일부 demo가 ±수백 미터 outlier 좌표를 만들어내, drawer group 학습 전체의 normalizer를 망가뜨리고 있었다. 이 문서는 발견 → 진단 → 5번의 패치 시도 → 최종 해결까지의 기록이다. 데이터 fix 후 eval 결과: CloseDrawer **20-30% → 90%**, OpenDrawer **0% → 30%** (30 ep, seed=0).

## 발견

PC 시각화 영상을 비교하다가 발견:

| 영상 | 결과 |
|---|---|
| `scripts/opendrawer_demo_1_4k.mp4` | 정상 — 부엌 + 로봇 + 서랍이 또렷 |
| `scripts/opendrawer_demo_2_4k.mp4` | **거의 까만 화면, 데이터가 안 보임** |

`opendrawer_demo_2_4k.mp4`의 좌표 통계를 보니 `|xyz|max ≈ 488 m`. 정상 demo는 `|xyz|max ≈ 1.3 m`. demo_2의 outlier 한 점 때문에 matplotlib 축 범위가 ±488m로 늘어나 정상 점들이 한 점으로 압축돼 시각화 전체가 깨진 것.

OpenDrawer 55 demo 중 비슷한 outlier가 있는 demo가 **9개**. CloseDrawer 55개는 모두 정상. eval 결과도 일관:

| Task | pre-fix eval (n=30) |
|---|---|
| CloseDrawer | ~20-30% |
| OpenDrawer | **0%** |

## 진단 (5 Opus 병렬 조사)

5개 Opus subagent를 풀어 가능한 원인을 분류. 두 개의 독립적 버그가 동시에 작동 중이었다.

### (C) `eval_with_video.py`의 lang="dummy" 버그 — 평가-레벨

```python
# 깨진 코드 (try가 예외 안 던지면 except 도달 안 함)
try:
    lang = env.get_ep_meta_str()  # 존재하지 않는 메서드 — hasattr False → 그냥 빠져나옴
except Exception:
    lang = "dummy"  # 절대 실행 안 됨
# 결과: lang은 그대로 "dummy" — DP3가 task 구분 못 함
```

DP3는 `lang` (CLIP RN50 임베딩) 을 FiLM 컨디션으로 받아 OpenDrawer/CloseDrawer를 구분한다. 모든 eval이 `"dummy"`로 들어가니 OpenDrawer는 자연히 0%. 수정 (`env.get_ep_meta()["lang"]`) 후 OpenDrawer 0% → 30%로 회복.

### (B) `pc_generator.py`의 depth-outlier 버그 — 데이터-레벨

`FPVNet/custom_robocasa/utils/point_cloud/pc_generator.py` 의 `get_point_cloud()` 가 raw MuJoCo depth를 그대로 Open3D `create_from_depth_image` 에 넘긴다:

```python
o3d_depth = o3d.geometry.Image(depths[cam])  # raw [0, 1] depth
o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(
    o3d_depth, cam_intrinsics  # default depth_scale=1000, depth_trunc=1000
)
```

대부분의 demo에선 float32 depth가 [0, 1] 안에 잘 들어와 Open3D가 미터로 해석 → 부엌 씬 ±1.3 m 범위로 unprojection. 하지만 OpenDrawer 9개 demo에서 raw depth에 [0, 1] 범위를 벗어난 픽셀이 1~수십 개씩 섞여 들어와 (NaN/Inf, 또는 부동소수점 노이즈로 1.0 초과), 이게 ±수백 m 좌표로 튀어나옴 (`demo_2` 의 488 m). 정상 점은 그 demo에 잘 있지만 `LinearNormalizer.fit(mode="limits")`가 outlier에 끌려가 모든 점이 한 점으로 압축돼 학습이 깨짐.

→ **CloseDrawer 까지 같이 망가뜨림** — drawer group은 두 task를 합쳐서 normalizer를 학습하기 때문. 사후 결과(Close 20% → 90%)가 이 가설을 검증.

## 패치 진화 (v1 → v5)

`pc_generator.py` 의 depth 처리를 5번 다시 썼다. 각 단계마다 6 demo로 빠르게 재추출 + 영상 비교.

### v1 — `get_real_depth_map()` 호출 추가
```python
depth_m = get_real_depth_map(self.sim, depths[cam])
```
**결과**: AssertionError. `get_real_depth_map`이 `assert np.all(depth_map <= 1.0)`을 빡빡하게 검사해서, 부동소수점 노이즈로 1.0을 0.0001 초과한 픽셀 하나만 있어도 추출이 전체 중단.

### v2 — clip + NaN 처리 추가
```python
d = np.clip(np.nan_to_num(depth_raw, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
real = near / (1.0 - d * (1.0 - near / far))
o3d_cloud = create_from_depth_image(
    o3d_depth, intrinsics, depth_scale=1.0, depth_trunc=5.0
)
```
**결과**: 추출은 성공하지만 모든 demo가 ±1000 m outlier. `depth_trunc`가 `--keep_full_pc` 경로에서 안 듣고 Open3D가 49152 픽셀을 전부 보존, far-plane 픽셀이 그대로 unprojection.

### v3 — Open3D 우회, NumPy manual unprojection
```python
x = (u - cx) * z / fx
y = (v - cy) * z / fy
xyz = stack([x, y, z]).reshape(-1, 3)
```
**결과**: outlier는 사라졌지만 `|xyz|max = 6.39 m`로 부풀음 (camera-depth 5m clamp이 world coord로는 더 큼). 시각적으로 OLD 대비 배경이 부풀어 모든 demo가 어수선.

### v4 — 최소 clip [0, 1]
```python
def _sanitize_depth(depth):
    return np.clip(np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
```
**결과**: 수치상 `|xyz|max = 1.99 m`로 OLD 범위 일치. demo_2 outlier도 사라짐. **하지만 영상 비교에서 사용자가 잡아냄**: clean demo (demo_3, demo_6) 도 OLD 대비 모양이 미묘하게 일그러짐. `byte-diff` 측정: `max|diff| = 0.79 m`, `median|diff| = 0.05 m`. clip [0, 1] 이 OLD에서 자연스럽게 1.0 살짝 넘던 raw depth (Open3D가 float를 미터로 해석 → 정상 배경 1.0~1.5 m 점들)까지 깎아낸 것. **너무 공격적인 패치**.

### v5 (최종) — gated sanitize
```python
def _sanitize_depth(depth):
    if np.isfinite(depth).all() and depth.max() <= 5.0 and depth.min() >= 0.0:
        return depth                                # clean: OLD 경로 byte-level 보존
    safe = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0)
    safe = np.where(safe > 5.0, 1.0, safe)         # glitch만 1.0으로
    safe = np.maximum(safe, 0.0)
    return safe.astype(depth.dtype)
```

검증 (6 demo):

| demo | OLD `|xyz|max` | v5 NEW `|xyz|max` | byte-diff |
|---|---|---|---|
| 1 | 1.32 m | 1.32 m | identical |
| **2 (망가졌던 거)** | **488.90 m** | **1.46 m** | **fixed** |
| 3 | 1.77 m | 1.77 m | **0.000000** |
| 4 | 1.27 m | 1.27 m | identical |
| 5 | 1.99 m | 1.99 m | identical |
| 6 | 1.79 m | 1.79 m | **0.000000** |

clean demo는 OLD와 byte-level 동일, demo_2만 깨끗하게 수정.

### 핵심 통찰

OLD 코드가 99% 케이스에서 잘 동작하던 이유는, Open3D의 `create_from_depth_image`가 float32 입력을 (uint16과 달리) `depth_scale`로 나누지 않고 그대로 미터로 해석하기 때문. raw depth [0, 1.x] → 카메라로부터 [0, 1.x] m. depth_trunc 기본값(1000m)도 못 트리거. 따라서 정상 depth 값을 임의로 clip 하면 OLD 결과가 살짝 깨지고, 패치 목표는 **out-of-band glitch (>5 또는 NaN) 만** 잡는 것이지 정상 depth를 건드리는 게 아님.

## 부수적 발견 — `dataset_states_to_obs.py`의 lang 무작위화

영상 비교 중 발견: 같은 demo_1이 OLD/NEW에서 서로 다른 lang annotation을 가짐.

| demo | RAW `demo.hdf5` lang | OLD processed lang | NEW processed lang |
|---|---|---|---|
| 1 | open the right drawer | open the left drawer | open the right drawer |
| 2 | open the left drawer | open the right drawer | open the right drawer |
| 3 | open the left drawer | open the left drawer | open the left drawer |
| 4 | open the left drawer | open the right drawer | open the left drawer |
| 5 | open the left drawer | open the left drawer | open the right drawer |
| 6 | open the left drawer | open the left drawer | open the left drawer |

원인: `dataset_states_to_obs.py:77` 이 `env.reset_to(initial_state)` 후 `env.env.get_ep_meta()` 를 다시 호출하는데, robocasa env가 reset 시 `lang` 을 env 내부 random state로 새로 결정한다. RAW와 일치하지 않음. 매 추출마다 결과가 다르다.

→ **학습 영향**: 매 추출마다 model이 보는 task↔lang 매핑이 달라짐. 별도 패치 거리. 이번 단계에선 손대지 않음 (eval은 RAW가 아닌 처리된 hdf5의 lang을 사용하므로 학습/평가 간 일관성은 있음).

## 최종 결과

전체 dataset 재추출 + 재학습 + 30 ep × 2 task × seed=0 eval:

| Task | post-fix | pre-fix | 개선 |
|---|---|---|---|
| **CloseDrawer** | **27/30 = 90%** | ~20-30% | **+60-70%p** |
| **OpenDrawer** | **9/30 = 30%** | 0% | **+30%p** |

OpenDrawer ep_success pattern: `▁▁▁█▁█▁██▁▁▁█▁▁▁▁▁▁█▁█▁▁█▁▁▁█▁`

- wandb run: [`charmed-feather-8`](https://wandb.ai/RwHlabs/fpvnet_dp3_drawer/runs/rfldmnqd) (project `fpvnet_dp3_drawer`)
- ckpt: `FPVNet/logs/drawer/runs/dp3/2026-05-21/01-44-04/last_model.pth` (960 MB, 251.5 M parameter)
- 학습 시간: ~1 h 50 min (100 epoch × 165 step/epoch, 단일 A4000 16 GB, batch 128)
- eval 영상 60개: `FPVNet/logs/eval_videos/post_patch_20260521_033924_{Close,Open}Drawer/`

## 패치된 파일

**상류 FPVNet (외부 repo)** — 우리 fork 아님, 패치는 호스트 파일에만 적용:
- `FPVNet/custom_robocasa/utils/point_cloud/pc_generator.py` (v5 gated sanitize)
- `FPVNet/custom_robocasa/custom_robocasa/robocasa/scripts/dataset_states_to_obs.py:475` (race condition fix, 이전 단계에서 적용됨)

**도커 환경 (`/home/woonsang/theo/docker/`)**:
- `eval_with_video.py` — lang="dummy" 버그 수정 (`env.get_ep_meta()["lang"]`)
- `run.sh` — video 모드에서 `EVAL_OUT_NAME` 미지정 시 timestamp 폴더 자동 생성, `EVAL_CKPT` env 전달

**원본 백업** (`.old.hdf5`, 11 GB × 2): 추후 비교용으로 보관 중.

## 교훈

1. **데이터 outlier 1 픽셀이 그룹 학습 전체를 깎을 수 있다.** drawer group의 두 task가 normalizer를 공유. OpenDrawer 9개 demo의 outlier가 CloseDrawer 성능까지 끌어내림.
2. **"OLD 대비 distortion"은 패치가 너무 공격적이라는 신호.** v4의 clip [0, 1] 이 수치적으론 OK였지만 시각 비교에서 잘림이 드러남. 정상 케이스는 OLD 경로 byte-level 보존이 정답.
3. **시각 검증을 빼먹지 말 것.** 수치 통계 (mean, max, range) 만으론 잘림 같은 미묘한 distortion을 못 잡는다. matplotlib 회전 영상으로 비교한 게 결정적.
4. **lang 같은 부수 메타데이터도 추출 결정성을 검증해야 한다.** RAW와 processed의 lang 불일치는 추적하기 전엔 모르고 지나칠 뻔.
