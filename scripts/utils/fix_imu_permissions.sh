#!/bin/bash
# D435i IMU(IIO) sysfs 권한 복구 — usbipd 재연결/재부팅 후 udev RUN이 누락될 때
#
# 증상: realsense2_camera_node가
#   "Failed to open scan_element .../scan_elements/in_anglvel_x_en ... Permission denied"
# 를 출력 (IIO 파일이 0644로 남아 root만 쓰기 가능)
#
# 사용법: sudo ./scripts/utils/fix_imu_permissions.sh
#   (usbipd로 카메라를 새로 attach 한 뒤 1회 실행, 그 후 camera.launch.py 재시작)

set -e

# 8086(Intel) VID의 HID-IIO 장치 검색 (HID 장치명 형식: "0003:8086:0B3A.xxxx")
FOUND=0
for hid in /sys/bus/hid/devices/*; do
    [ -d "$hid" ] || continue
    vid=$(basename "$hid" | cut -d: -f2)
    [ "$vid" = "8086" ] || continue
    FOUND=1

    echo "▶ $hid"
    for iio in "$hid"/*/iio:device* "$hid"/iio:device*; do
        [ -d "$iio" ] || continue
        echo "  chmod $iio"
        # scan_elements (in_*_en 등) — librealsense가 여기에 쓰기 접근
        if [ -d "$iio/scan_elements" ]; then
            chmod 0666 "$iio"/scan_elements/in_* 2>/dev/null || true
        fi
        # buffer / trigger
        chmod 0666 "$iio"/buffer*/enable "$iio"/buffer*/length 2>/dev/null || true
        chmod 0666 "$iio"/buffer/enable "$iio"/buffer/length 2>/dev/null || true
        chmod 0666 "$iio"/trigger/current_trigger 2>/dev/null || true
        chmod 0666 "$iio"/in_* 2>/dev/null || true
    done

    # enable_sensor (hid_sensor_custom)
    chmod 0666 "$hid"/*/enable_sensor 2>/dev/null || true
done

if [ "$FOUND" = 0 ]; then
    echo "⚠️  Intel(8086) HID 장치를 찾지 못했습니다. usbipd로 카메라가 attach 되어 있는지 확인하세요."
    exit 1
fi

echo "✅ IMU 권한 복구 완료. 이제 camera.launch.py 를 다시 실행하세요."
