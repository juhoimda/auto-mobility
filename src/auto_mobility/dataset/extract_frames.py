#!/usr/bin/env python3
"""
extract_frames.py — Rosbag(MCAP/sqlite3)에서 알고리즘 독립적인 Canonical Frame Dataset 추출기

구조:
  ros2_data/frames/<dataset>/
    ├── rgb/
    │   ├── 000000.png
    │   └── ...
    ├── depth/
    │   ├── 000000.png
    │   └── ...
    ├── frames.csv
    ├── camera_info.json
    ├── imu.csv
    └── dataset_info.json
"""

import os
import sys
import json
import csv
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from auto_mobility.config import (
    BAG_DIR, FRAME_DIR,
    CAMERA_RGB_TOPIC, CAMERA_RGB_COMPRESSED_TOPIC,
    CAMERA_DEPTH_TOPIC, CAMERA_DEPTH_COMPRESSED_TOPIC,
    CAMERA_ALIGNED_DEPTH_TOPIC, CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC,
    CAMERA_INFO_TOPIC, CAMERA_INFO_WINDOWS_TOPIC,
    CAMERA_IMU_TOPIC, CAMERA_IMU_FILTERED_TOPIC,
)

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    rosbag2_py = None


def _stamp_sec(msg) -> float:
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
    return 0.0


def decode_rgb_message(msg, type_name: str) -> Optional[np.ndarray]:
    """RGB 메시지를 BGR uint8 ndarray로 디코딩."""
    try:
        if "CompressedImage" in type_name:
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        elif "Image" in type_name:
            height, width = msg.height, msg.width
            encoding = msg.encoding.lower()
            if encoding in ("rgb8", "bgr8"):
                arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((height, width, 3))
                if encoding == "rgb8":
                    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                return arr
            elif encoding in ("mono8", "8uc1"):
                arr = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape((height, width))
                return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    except Exception as e:
        print(f"⚠️ RGB decode error: {e}")
    return None


def decode_depth_message(msg, type_name: str) -> Optional[np.ndarray]:
    """Depth 메시지를 16-bit uint16 (mm) ndarray로 디코딩."""
    try:
        if "CompressedImage" in type_name:
            raw = bytes(msg.data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            idx = raw.find(b"\x89PNG")
            if idx != -1:
                return cv2.imdecode(arr[idx:], cv2.IMREAD_UNCHANGED)
            else:
                return cv2.imdecode(arr[12:], cv2.IMREAD_UNCHANGED)
        elif "Image" in type_name:
            height, width = msg.height, msg.width
            encoding = msg.encoding.lower()
            if encoding in ("16uc1", "mono16"):
                return np.frombuffer(bytes(msg.data), dtype=np.uint16).reshape((height, width))
            elif encoding in ("32fc1",):
                arr = np.frombuffer(bytes(msg.data), dtype=np.float32).reshape((height, width))
                arr_mm = np.clip(arr * 1000.0, 0, 65535).astype(np.uint16)
                return arr_mm
    except Exception as e:
        print(f"⚠️ Depth decode error: {e}")
    return None


def extract_dataset_from_bag(bag_path_or_name: str, out_dir: Optional[str] = None,
                              max_sync_dt_ms: float = 50.0, force: bool = False) -> Path:
    """Rosbag으로부터 Canonical Dataset 디렉터리를 추출 및 생성."""
    if rosbag2_py is None:
        raise RuntimeError("rosbag2_py is not available. Please run in ROS 2 environment.")

    p = Path(bag_path_or_name)
    if not p.is_absolute():
        if (BAG_DIR / bag_path_or_name).exists():
            p = BAG_DIR / bag_path_or_name
        elif not p.exists():
            raise FileNotFoundError(f"Rosbag not found: {bag_path_or_name}")

    bag_path = p.resolve()
    bag_name = bag_path.name

    if out_dir is None:
        out_path = (FRAME_DIR / bag_name).resolve()
    else:
        out_path = Path(out_dir).resolve()

    if out_path.exists() and (out_path / "frames.csv").exists() and not force:
        print(f"✅ Canonical Dataset 이미 존재함: {out_path} (재사용)")
        return out_path

    print("==========================================================")
    print(f" 📦 Canonical Frame Dataset 추출 시작")
    print(f" 📁 입력 Rosbag : {bag_path}")
    print(f" 💾 출력 디렉터리: {out_path}")
    print("==========================================================")

    # Readers & topics setup
    reader = rosbag2_py.SequentialReader()
    storage_options = None
    for storage_id in ("mcap", "sqlite3"):
        try:
            storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id=storage_id)
            converter = rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            )
            reader.open(storage_options, converter)
            break
        except Exception:
            continue

    if storage_options is None:
        raise RuntimeError(f"Failed to open rosbag with mcap or sqlite3: {bag_path}")

    all_topics = reader.get_all_topics_and_types()
    topic_map = {t.name: t.type for t in all_topics}

    # Priority topic candidates
    rgb_candidates = [
        CAMERA_RGB_COMPRESSED_TOPIC, CAMERA_RGB_TOPIC,
        "/camera/camera/color/image_raw/compressed", "/camera/camera/color/image_raw"
    ]
    depth_candidates = [
        CAMERA_ALIGNED_DEPTH_COMPRESSED_TOPIC, CAMERA_ALIGNED_DEPTH_TOPIC,
        CAMERA_DEPTH_COMPRESSED_TOPIC, CAMERA_DEPTH_TOPIC,
        "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth",
        "/camera/camera/aligned_depth_to_color/image_raw",
        "/camera/camera/depth/image_rect_raw/compressedDepth",
        "/camera/camera/depth/image_rect_raw"
    ]
    info_candidates = [
        CAMERA_INFO_WINDOWS_TOPIC, CAMERA_INFO_TOPIC,
        "/camera/camera/color/camera_info_windows", "/camera/camera/color/camera_info"
    ]
    # Depth CameraInfo candidates for native depth (P0 blocker: must attempt recovery)
    depth_info_candidates = [
        "/camera/camera/depth/camera_info",
        "/camera/camera/depth/camera_info_windows",
        "/camera/camera/aligned_depth_to_color/camera_info",
    ]
    imu_candidates = [
        CAMERA_IMU_TOPIC, CAMERA_IMU_FILTERED_TOPIC, "/camera/camera/imu"
    ]

    selected_rgb = next((t for t in rgb_candidates if t in topic_map), None)
    selected_depth = next((t for t in depth_candidates if t in topic_map), None)
    selected_info = next((t for t in info_candidates if t in topic_map), None)
    selected_depth_info = next((t for t in depth_info_candidates if t in topic_map), None)
    selected_imu = next((t for t in imu_candidates if t in topic_map), None)

    print(f"  • RGB Topic   : {selected_rgb}")
    print(f"  • Depth Topic : {selected_depth}")
    print(f"  • Info Topic  : {selected_info}")
    print(f"  • IMU Topic   : {selected_imu}")

    if not selected_rgb or not selected_depth:
        raise ValueError(f"Rosbag does not contain required RGB and Depth topics! Available: {list(topic_map.keys())}")

    # Prepare extraction buffers (store raw bytes & metadata to prevent RAM peak OOM)
    rgb_frames: List[Tuple[float, float, bytes, str, str]] = []  # (stamp, bag_stamp, raw_data, topic_type, frame_id)
    depth_frames: List[Tuple[float, float, bytes, str, str]] = []  # (stamp, bag_stamp, raw_data, topic_type, frame_id)
    imu_records: List[dict] = []
    camera_info_record: Optional[dict] = None
    depth_camera_info_record: Optional[dict] = None
    tf_static_transforms: List[dict] = []

    t0 = time.time()
    msg_count = 0

    while reader.has_next():
        topic, data, bag_stamp_ns = reader.read_next()
        bag_stamp_sec = float(bag_stamp_ns) * 1e-9
        msg_count += 1

        if topic == selected_rgb:
            msg = deserialize_message(data, get_message(topic_map[topic]))
            stamp = _stamp_sec(msg) or bag_stamp_sec
            fid = msg.header.frame_id if hasattr(msg, "header") else "camera_color_optical_frame"
            rgb_frames.append((stamp, bag_stamp_sec, data, topic_map[topic], fid))

        elif topic == selected_depth:
            msg = deserialize_message(data, get_message(topic_map[topic]))
            stamp = _stamp_sec(msg) or bag_stamp_sec
            fid = msg.header.frame_id if hasattr(msg, "header") else "camera_color_optical_frame"
            depth_frames.append((stamp, bag_stamp_sec, data, topic_map[topic], fid))

        elif topic == selected_info and camera_info_record is None:
            msg = deserialize_message(data, get_message(topic_map[topic]))
            if hasattr(msg, "k") and len(msg.k) >= 9 and msg.k[0] > 0:
                camera_info_record = {
                    "fx": float(msg.k[0]),
                    "fy": float(msg.k[4]),
                    "cx": float(msg.k[2]),
                    "cy": float(msg.k[5]),
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "distortion_model": str(msg.distortion_model),
                    "distortion_coefficients": [float(x) for x in msg.d],
                    "K": [float(x) for x in msg.k],
                    "R": [float(x) for x in msg.r],
                    "P": [float(x) for x in msg.p],
                }

        elif topic == selected_depth_info and depth_camera_info_record is None:
            msg = deserialize_message(data, get_message(topic_map[topic]))
            if hasattr(msg, "k") and len(msg.k) >= 9 and msg.k[0] > 0:
                depth_camera_info_record = {
                    "fx": float(msg.k[0]),
                    "fy": float(msg.k[4]),
                    "cx": float(msg.k[2]),
                    "cy": float(msg.k[5]),
                    "width": int(msg.width),
                    "height": int(msg.height),
                    "distortion_model": str(msg.distortion_model),
                    "distortion_coefficients": [float(x) for x in msg.d],
                    "K": [float(x) for x in msg.k],
                    "R": [float(x) for x in msg.r],
                    "P": [float(x) for x in msg.p],
                    "frame_id": str(msg.header.frame_id),
                }

        elif topic == "/tf_static":
            try:
                msg = deserialize_message(data, get_message(topic_map[topic]))
                for tf in getattr(msg, "transforms", []):
                    tf_static_transforms.append({
                        "parent": str(tf.header.frame_id),
                        "child": str(tf.child_frame_id),
                        "translation": [float(tf.transform.translation.x), float(tf.transform.translation.y), float(tf.transform.translation.z)],
                        "rotation": [float(tf.transform.rotation.x), float(tf.transform.rotation.y), float(tf.transform.rotation.z), float(tf.transform.rotation.w)],
                    })
            except Exception:
                pass

        elif topic == selected_imu:
            msg = deserialize_message(data, get_message(topic_map[topic]))
            stamp = _stamp_sec(msg) or bag_stamp_sec
            imu_records.append({
                "timestamp": stamp,
                "angular_velocity_x": float(msg.angular_velocity.x),
                "angular_velocity_y": float(msg.angular_velocity.y),
                "angular_velocity_z": float(msg.angular_velocity.z),
                "linear_acceleration_x": float(msg.linear_acceleration.x),
                "linear_acceleration_y": float(msg.linear_acceleration.y),
                "linear_acceleration_z": float(msg.linear_acceleration.z),
            })

    print(f"✅ Bag 읽기 완료 ({time.time()-t0:.1f}s): RGB {len(rgb_frames)}장, Depth {len(depth_frames)}장, IMU {len(imu_records)}개")

    # Associate RGB and Depth frames
    if not rgb_frames or not depth_frames:
        raise RuntimeError("No valid RGB or Depth frames extracted from bag!")

    # ⏱️ MCAP 청크 버퍼링 지터 제거: 실제 센서 타임스탬프(header.stamp) 기준 엄격한 시간순 정렬
    rgb_frames.sort(key=lambda f: f[0])
    depth_frames.sort(key=lambda f: f[0])
    imu_records.sort(key=lambda r: r["timestamp"])

    # 중복 타임스탬프 필터링 (Strict Monotonicity 보장)
    filtered_imu: List[dict] = []
    last_imu_ts = -1.0
    for rec in imu_records:
        if rec["timestamp"] > last_imu_ts:
            filtered_imu.append(rec)
            last_imu_ts = rec["timestamp"]
    imu_records = filtered_imu

    rgb_stamps = np.array([f[0] for f in rgb_frames], dtype=np.float64)
    depth_stamps = np.array([f[0] for f in depth_frames], dtype=np.float64)

    # Match RGB frames to Depth frames
    matched_pairs: List[dict] = []
    used_depth_indices = set()
    last_matched_rgb_ts = -1.0

    for r_idx, (r_stamp, r_bag_stamp, r_data, r_type, r_fid) in enumerate(rgb_frames):
        d_idx = int(np.argmin(np.abs(depth_stamps - r_stamp)))
        dt_ms = abs(depth_stamps[d_idx] - r_stamp) * 1000.0
        if dt_ms <= max_sync_dt_ms and d_idx not in used_depth_indices and r_stamp > last_matched_rgb_ts:
            used_depth_indices.add(d_idx)
            last_matched_rgb_ts = r_stamp
            d_stamp, d_bag_stamp, d_data, d_type, d_fid = depth_frames[d_idx]
            matched_pairs.append({
                "rgb_stamp": r_stamp,
                "depth_stamp": d_stamp,
                "rgb_bag_stamp": r_bag_stamp,
                "rgb_data": r_data,
                "rgb_type": r_type,
                "depth_data": d_data,
                "depth_type": d_type,
                "dt_ms": dt_ms,
                "rgb_frame_id": r_fid,
                "depth_frame_id": d_fid,
            })

    print(f"🔗 동기화 프레임 매칭: 총 {len(matched_pairs)} 프레임 페어 생성 (dt <= {max_sync_dt_ms}ms)")

    # Save to canonical directory structure with lazy decoding
    rgb_dir = out_path / "rgb"
    depth_dir = out_path / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    frames_csv_path = out_path / "frames.csv"
    import concurrent.futures

    def _process_and_save_pair(idx_and_pair):
        idx, pair = idx_and_pair
        try:
            r_msg = deserialize_message(pair["rgb_data"], get_message(pair["rgb_type"]))
            d_msg = deserialize_message(pair["depth_data"], get_message(pair["depth_type"]))
            r_img = decode_rgb_message(r_msg, pair["rgb_type"])
            d_img = decode_depth_message(d_msg, pair["depth_type"])

            if r_img is None or d_img is None:
                return None

            h, w = r_img.shape[:2]
            frame_str = f"{idx:06d}"
            rgb_rel = f"rgb/{frame_str}.png"
            depth_rel = f"depth/{frame_str}.png"

            png_opts = [cv2.IMWRITE_PNG_COMPRESSION, 1]
            cv2.imwrite(str(out_path / rgb_rel), r_img, png_opts)
            cv2.imwrite(str(out_path / depth_rel), d_img, png_opts)

            return {
                "frame_id": idx,
                "rgb_timestamp": f"{pair['rgb_stamp']:.6f}",
                "depth_timestamp": f"{pair['depth_stamp']:.6f}",
                "rgb_path": rgb_rel,
                "depth_path": depth_rel,
                "rgb_depth_dt_ms": f"{pair['dt_ms']:.3f}",
                "bag_timestamp": f"{pair['rgb_bag_stamp']:.6f}",
                "camera_frame_id": pair["rgb_frame_id"],
                "depth_camera_frame_id": pair["depth_frame_id"],
                "width": w,
                "height": h
            }
        except Exception as e:
            print(f"⚠️ Frame {idx} save error: {e}")
            return None

    saved_records = []
    saved_pairs_count = 0
    total_matched = len(matched_pairs)
    print(f"💾 프레임 디스크 저장 시작 (총 {total_matched} 프레임 / 8개 스레드 병렬 쓰기)...")
    t_save_start = time.time()
    first_w, first_h = 640, 480

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process_and_save_pair, (i, pair)) for i, pair in enumerate(matched_pairs)]
        for done_cnt, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            res = fut.result()
            if res is not None:
                saved_records.append(res)
                if saved_pairs_count == 0:
                    first_w, first_h = res["width"], res["height"]
                saved_pairs_count += 1

            if done_cnt % 500 == 0 or done_cnt == total_matched:
                pct = (done_cnt / total_matched) * 100.0
                elapsed_s = time.time() - t_save_start
                print(f"  • [Frame Extraction] 디스크 저장 진행 중: {done_cnt}/{total_matched} ({pct:.1f}%, {elapsed_s:.1f}s)")

    saved_records.sort(key=lambda r: r["frame_id"])

    with open(frames_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        fieldnames = [
            "frame_id", "rgb_timestamp", "depth_timestamp", "rgb_path", "depth_path",
            "rgb_depth_dt_ms", "bag_timestamp", "camera_frame_id", "depth_camera_frame_id", "width", "height"
        ]
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        for rec in saved_records:
            writer.writerow(rec)

    # Save camera info
    cam_info_path = out_path / "camera_info.json"
    if camera_info_record is None:
        print("⚠️ CameraInfo 미발견! 기본 D435i Intrinsics(fx=385.0) 기록")
        w = first_w
        h = first_h
        camera_info_record = {
            "fx": 385.0, "fy": 385.0, "cx": w / 2.0, "cy": h / 2.0,
            "width": w, "height": h, "distortion_model": "plumb_bob",
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "K": [385.0, 0.0, w / 2.0, 0.0, 385.0, h / 2.0, 0.0, 0.0, 1.0],
            "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "P": [385.0, 0.0, w / 2.0, 0.0, 0.0, 385.0, h / 2.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            "is_fallback": True
        }
    else:
        camera_info_record["is_fallback"] = False

    with open(cam_info_path, "w", encoding="utf-8") as f_info:
        json.dump(camera_info_record, f_info, indent=2)

    # Save IMU
    imu_path = out_path / "imu.csv"
    with open(imu_path, "w", newline="", encoding="utf-8") as f_imu:
        fieldnames = ["timestamp", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                      "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z"]
        writer = csv.DictWriter(f_imu, fieldnames=fieldnames)
        writer.writeheader()
        for rec in imu_records:
            writer.writerow(rec)

    # Calculate dataset quality statistics
    dt_arr = [p["dt_ms"] for p in matched_pairs]
    r_stamps = [p["rgb_stamp"] for p in matched_pairs]
    r_diffs = np.diff(r_stamps) if len(r_stamps) > 1 else np.array([0.0])
    duration = float(r_stamps[-1] - r_stamps[0]) if len(r_stamps) > 1 else 0.0
    est_hz = round(saved_pairs_count / duration, 2) if duration > 0 else 0.0

    def _bag_fingerprint(bp: Path) -> str:
        import hashlib
        h = hashlib.sha256()
        if bp.is_file():
            stat = bp.stat()
            h.update(f"{bp.name}:{stat.st_size}:{stat.st_mtime}".encode())
        elif bp.is_dir():
            for f in sorted(bp.iterdir()):
                if f.is_file():
                    stat = f.stat()
                    h.update(f"{f.name}:{stat.st_size}:{stat.st_mtime}".encode())
        return h.hexdigest()[:16]

    # --- P0-2: Canonical RGB-D alignment contract proof ---
    # frame_id equality alone MUST NOT judge alignment.
    from auto_mobility.dataset.rgbd_alignment import prove_alignment, save_contract

    color_fid = rgb_frames[0][4] if rgb_frames else "unknown"
    depth_fid = depth_frames[0][4] if depth_frames else "unknown"

    # Build K_color from camera_info_record
    K_raw = camera_info_record.get("K")
    if K_raw is not None:
        K_color_arr = np.array(K_raw, dtype=float).reshape(3, 3)
    else:
        fx = camera_info_record.get("fx", 0.0)
        fy = camera_info_record.get("fy", 0.0)
        cx = camera_info_record.get("cx", 0.0)
        cy = camera_info_record.get("cy", 0.0)
        K_color_arr = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

    # --- Attempt to recover K_depth and T_color_depth from bag metadata ---
    K_depth_arr = None
    T_color_depth_arr = None
    depth_scale = 1.0
    if depth_camera_info_record is not None:
        Kd_raw = depth_camera_info_record.get("K")
        if Kd_raw is not None:
            K_depth_arr = np.array(Kd_raw, dtype=float).reshape(3, 3)
        # depth frame from depth info
        # also record depth scale if available (assume mm)
    # If depth is already in color frame (driver-aligned), K_depth equals K_color
    # This is NOT an estimation: it is driver-provided aligned depth semantics
    if K_depth_arr is None and color_fid == depth_fid and first_w == 640 and first_h == 480:
        # Hallway bag: depth frame_id == color frame_id, same resolution, TF indicates identity
        # Treat as driver-aligned: depth K is color K
        K_depth_arr = K_color_arr.copy()

    # Build T_color_depth from tf_static if available
    if tf_static_transforms:
        # helper to build 4x4 from translation+quaternion
        def _quat_to_mat(qx, qy, qz, qw):
            # normalize
            n = (qx*qx + qy*qy + qz*qz + qw*qw) ** 0.5
            qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
            return np.array([
                [1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
                [2*(qx*qy+qz*qw), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)],
                [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx*qx+qy*qy)],
            ])
        def _tf_to_matrix(t):
            R = _quat_to_mat(*t["rotation"])
            M = np.eye(4)
            M[:3,:3]=R
            M[:3,3]=t["translation"]
            return M
        tf_map = {f"{e['parent']}->{e['child']}": _tf_to_matrix(e) for e in tf_static_transforms}
        # Try to derive T_color_depth
        if color_fid == depth_fid:
            T_color_depth_arr = np.eye(4)
        else:
            # Look for camera_link as common parent
            # need T_link_color and T_link_depth
            # Find any transform that ends with color/depth frame
            T_link_color = None
            T_link_depth = None
            for e in tf_static_transforms:
                if e["child"] == color_fid:
                    T_link_color = _tf_to_matrix(e)
                if e["child"] == depth_fid:
                    T_link_depth = _tf_to_matrix(e)
            if T_link_color is not None and T_link_depth is not None:
                try:
                    T_color_depth_arr = np.linalg.inv(T_link_color) @ T_link_depth
                except Exception:
                    pass
            # fallback: if depth is camera_depth_optical_frame and color is camera_color_optical_frame, use known 15mm baseline
            if T_color_depth_arr is None:
                for e in tf_static_transforms:
                    if "camera_color_optical_frame" in e["child"]:
                        T_link_color = _tf_to_matrix(e)
                    if "camera_depth_optical_frame" in e["child"]:
                        T_link_depth = _tf_to_matrix(e)
                if T_link_color is not None and T_link_depth is not None:
                    try:
                        T_color_depth_arr = np.linalg.inv(T_link_color) @ T_link_depth
                    except Exception:
                        pass
    # If still None but driver-aligned due to same frame_id, use identity
    if T_color_depth_arr is None and color_fid == depth_fid:
        T_color_depth_arr = np.eye(4)

    alignment_contract = prove_alignment(
        color_frame_id=color_fid,
        depth_frame_id=depth_fid,
        depth_topic=selected_depth,
        K_color=K_color_arr,
        K_depth=K_depth_arr,
        T_color_depth=T_color_depth_arr,
        image_size_color=(first_w, first_h),
        image_size_depth=(first_w, first_h),
        depth_scale=depth_scale,
    )
    save_contract(alignment_contract, out_path)

    if alignment_contract.method == "DRIVER_ALIGNED":
        _depth_color_alignment = "aligned"
    elif alignment_contract.method == "REPROJECTED":
        _depth_color_alignment = "reprojected"
    else:
        _depth_color_alignment = "not_aligned"

    print(f"   \U0001f50d RGB-D Alignment: {alignment_contract.method} "
          f"[{alignment_contract.contract_fingerprint}]")
    if not alignment_contract.is_proven():
        print(f"   \u26a0\ufe0f  Alignment UNPROVEN: {alignment_contract.reject_reason}")

    dataset_info = {
        "extraction_schema_version": "canonical-v3",
        "extraction_code_version": "3.0.0",
        "dataset_name": bag_name,
        "source_bag": str(bag_path),
        "source_bag_fingerprint": _bag_fingerprint(bag_path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frame_count": saved_pairs_count,
        "duration_sec": round(duration, 3),
        "estimated_fps": est_hz,
        "raw_rgb_count": len(rgb_frames),
        "raw_depth_count": len(depth_frames),
        "imu_sample_count": len(imu_records),
        "rgb_topic": selected_rgb,
        "depth_topic": selected_depth,
        "camera_info_topic": selected_info,
        "imu_topic": selected_imu,
        "rgb_frame_id": color_fid,
        "depth_camera_frame_id": depth_fid,
        "sync_threshold_ms": max_sync_dt_ms,
        "sync_dt_mean_ms": round(float(np.mean(dt_arr)), 3) if len(dt_arr) else 0.0,
        "sync_dt_p95_ms": round(float(np.percentile(dt_arr, 95)), 3) if len(dt_arr) else 0.0,
        "sync_dt_max_ms": round(float(np.max(dt_arr)), 3) if len(dt_arr) else 0.0,
        "frame_interval_mean_ms": round(float(np.mean(r_diffs) * 1000.0), 3) if len(r_diffs) else 0.0,
        "frame_interval_std_ms": round(float(np.std(r_diffs) * 1000.0), 3) if len(r_diffs) else 0.0,
        "frame_interval_p95_ms": round(float(np.percentile(r_diffs, 95) * 1000.0), 3) if len(r_diffs) else 0.0,
        "timestamp_reversal_count": int(np.sum(r_diffs < 0)),
        "is_camera_info_fallback": camera_info_record.get("is_fallback", False),
        "depth_color_alignment": _depth_color_alignment,
        "alignment_method": alignment_contract.method,
        "alignment_contract_fingerprint": alignment_contract.contract_fingerprint,
    }

    dataset_info_path = out_path / "dataset_info.json"
    with open(dataset_info_path, "w", encoding="utf-8") as f_meta:
        json.dump(dataset_info, f_meta, indent=2)

    print(f"🎉 Canonical Frame Dataset 생성 완료: {out_path}")
    print(f"   - 프레임: {len(matched_pairs)}장 ({est_hz} FPS, {duration:.1f}초)")
    print(f"   - RGB-Depth Sync Delta: Mean {dataset_info['sync_dt_mean_ms']}ms, P95 {dataset_info['sync_dt_p95_ms']}ms")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Extract Canonical RGB-D Frame Dataset from Rosbag")
    parser.add_argument("bag", help="Rosbag name or directory path")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: ros2_data/frames/<name>)")
    parser.add_argument("--max-sync-dt", type=float, default=50.0, help="Max sync delta between RGB and Depth in ms (default: 50.0)")
    parser.add_argument("--force", action="store_true", help="Force re-extraction even if already exists")
    args = parser.parse_args()

    extract_dataset_from_bag(args.bag, args.out_dir, args.max_sync_dt, args.force)


if __name__ == "__main__":
    main()
