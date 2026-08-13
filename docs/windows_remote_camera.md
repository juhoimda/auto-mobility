# 🪟 Windows 원격 카메라 구동 가이드 (`docs/windows_remote_camera.md`)

> **목적**: usbipd 문제로 WSL에서 카메라를 직접 붙이지 않고, **Windows 네이티브에서 D435i를 구동해**
> ROS2 DDS(미러링 네트워크)로 토픽을 WSL에 전달하는 방식. WSL은 `run_pipeline_all.sh --remote-camera` 로 촬영.
> 이 문서는 **Windows 쪽에 수행할 내용**만 정리한다 (WSL 쪽 수정은 `scripts/common.sh` / `run_pipeline_all.sh` / `run_live.sh` 에 이미 반영됨).

---

## ✅ 0. 사전 조건

| 항목 | 값 | 확인 방법 |
| :--- | :--- | :--- |
| Windows ROS 2 | Humble 이상 | `ros2 doctor` |
| D435i 드라이버 | Windows용 librealsense + `realsense2_camera` 동작 확인 | Windows에서 `ros2 run realsense2_camera realsense2_camera_node` |
| WSL 미러링 네트워크 | `%USERPROFILE%\.wslconfig` 에 `networkingMode=mirrored` | `.wslconfig` 확인, WSL 재시작 후 `ipconfig` 에서 동일 IP 확인 |
| 압축 republish (권장) | `image_transport` + `depth_image_transport` 설치 | `ros2 pkg prefix image_transport` / `depth_image_transport` |

---

## 📌 1. 환경 변수 — 양쪽 ROS 2 도메인/RMW 일치

Windows에서 PowerShell로:

```powershell
$env:ROS_DOMAIN_ID = "42"          # WSL의 ROS_DOMAIN_ID와 동일해야 함 (예: 42)
$env:RMW_IMPLEMENTATION = "rmw_fastrtps_cpp"   # FastDDS로 통일 (WSL과 동일)
```

- WSL 쪽도 동일한 `ROS_DOMAIN_ID` 를 설정한다. (미설정 시 양쪽 모두 0 이므로 동작은 하나, 명시 권장)
- RMW 미지정 시 Windows Humble 기본값이 FastDDS(`rmw_fastrtps_cpp`)라 생략 가능하나, 다른 값이 잡혀 있지 않은지 확인.

---

## 📷 2. 카메라 노드 — 네임스페이스/노드명을 WSL과 동일하게

**핵심**: WSL이 찾는 토픽은 아래 4개다. 토픽명이 `config/topics.yaml` 의 `/camera/camera/...` 와 **정확히** 일치해야 한다.

| 토픽 | 메시지 | WSL 사용처 |
| :--- | :--- | :--- |
| `/camera/camera/color/image_raw` | `sensor_msgs/Image` (RGB8) | RTAB-Map RGB |
| `/camera/camera/depth/image_rect_raw` | `sensor_msgs/Image` (16UC1) | RTAB-Map Depth |
| `/camera/camera/color/camera_info` | `sensor_msgs/CameraInfo` | RTAB-Map 내부 파라미터 |
| `/camera/camera/imu` | `sensor_msgs/Imu` | imu_filter_madgwick → `/camera/camera/imu/filtered` |

파일 없이 즉시 실행하는 명령 (PowerShell, `ros-args` 로 네임스페이스/파라미터 지정):

```powershell
ros2 run realsense2_camera realsense2_camera_node --ros-args `
  -r __ns:=/camera -r __node:=camera `
  -p depth_module.depth_profile:=640x480x30 `
  -p rgb_camera.color_profile:=640x480x30 `
  -p rgb_camera.color_format:=RGB8 `
  -p align_depth.enable:=false `
  -p enable_infra1:=false -p enable_infra2:=false `
  -p depth_module.emitter_enabled:=1 `
  -p enable_accel:=true -p enable_gyro:=true -p enable_sync:=true `
  -p unite_imu_method:=1 -p global_time_enabled:=false -p initial_reset:=false `
  -p spatial_filter.enable:=true -p temporal_filter.enable:=true `
  -p hole_filling_filter.enable:=true `
  -p spatial_filter.filter_smooth_alpha:=0.5 -p spatial_filter.filter_smooth_delta:=20 `
  -p temporal_filter.filter_smooth_alpha:=0.4 -p temporal_filter.filter_smooth_delta:=20 `
  -p hole_filling_filter.holes_fill:=1
```

> ⚠️ QoS 키(`color_qos` 등)는 v4.58.3에서 FPS 급락 유발 → **절대 넣지 말 것** (WSL `config.py` 의 금지 목록과 동일).
> 기본 RELIABLE QoS를 사용한다. RTAB-Map은 Best Effort로 구독하므로 문제없다.

**권장: 런치 파일로 관리** — 네임스페이스/파라미터를 코드로 유지하고 싶다면
`camera.launch.py`(WSL) 와 동일한 노드 정의(`name='camera', namespace='camera'`, 위 파라미터)의
Python launch 파일을 Windows 워크스페이스에 만들고 `colcon build` 후 `ros2 launch` 로 실행한다.

---

## 🗜️ 3. 압축 토픽 발행 (WSL `use_compressed` 수신용)

WSL이 `--remote-camera` 로 촬영할 때 **기본적으로 압축 토픽을 구독**한다.
Windows에서 raw → 압축으로 중계하는 republish 노드 2개를 추가로 띄워야 한다.

```powershell
# RGB: JPEG 압축 (/camera/camera/color/image_raw/compressed)
ros2 run image_transport republish raw in:=/camera/camera/color/image_raw out:=/camera/camera/color/image_raw compressed

# Depth: PNG 압축 (/camera/camera/depth/image_rect_raw/compressedDepth) — depth_image_transport 필요
ros2 run image_transport republish raw in:=/camera/camera/depth/image_rect_raw out:=/camera/camera/depth/image_rect_raw depth
```

- 결과 토픽: `/camera/camera/color/image_raw/compressed`, `/camera/camera/depth/image_rect_raw/compressedDepth`
- WSL의 `republish.py` 가 이를 수신해 로컬에서 raw로 복원 → RTAB-Map에 공급.
- `depth_image_transport` 가 없어 두 번째 명령이 실패하면 → WSL에서 `use_compressed:=false` 로 대체
  (raw 그대로, 대역폭 약 370 Mbps — 기가비트 미러링 환경에서 가능).

**재사용 tip**: 카메라 노드 + republish 2개를 하나의 Windows launch 파일로 묶으면
한 번의 `ros2 launch` 로 3개가 모두 뜬다.

---

## 🛡️ 4. Windows 방화벽

- ROS/카메라 프로세스 최초 실행 시 Windows 방화벽 **허용 팝업이 뜨면 "액세스 허용"**.
- 또는 미리 규칙 추가 (관리자 PowerShell):

```powershell
netsh advfirewall firewall add rule name="ROS2 FastDDS UDP" dir=in action=allow protocol=UDP localport=7400-7600
```

---

## 🔍 5. 검증 (양쪽에서)

Windows:
```powershell
ros2 topic list | findstr camera
ros2 topic hz /camera/camera/color/image_raw
```

WSL:
```bash
ros2 topic list | grep camera
ros2 topic hz /camera/camera/color/image_raw
```

- WSL에서 `/camera/camera/color/image_raw` 가 보이고 Hz가 나오면 DDS 연결 완료.
- 안 보이면: ① `ROS_DOMAIN_ID` 일치 확인 ② RMW 확인 ③ 방화벽 ④ `.wslconfig` 미러링 적용 후 WSL 재시작 여부.

---

## ▶️ 6. WSL 쪽 실행

```bash
cd ~/auto-mobility
./scripts/pipeline/run_pipeline_all.sh --remote-camera
```

| 옵션 | 의미 |
| :--- | :--- |
| `--remote-camera` | Windows 카메라 토픽 수신 모드. USB 검사 생략, `use_compressed:=true` 자동 적용 |
| `CAMERA_MODE=remote` | 환경변수로도 동일하게 동작 |
| `USE_COMPRESSED=false` | 원격이지만 raw 토픽으로 받을 때 (Windows에서 압축 republish 생략 시) |
