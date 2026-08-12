#!/usr/bin/env python3
"""
카메라 파라미터 유효성 검증 CLI (realsense2_camera 4.58.3 실측 기준)

사용법:
  python3 scripts/utils/check_camera_params.py
    → src/auto_mobility/config.py 의 CAMERA_PARAMS 검증
  python3 scripts/utils/check_camera_params.py <yaml 파일>
    → YAML로 정의된 카메라 파라미터 검증 (사용자 커스텀 설정 확인용)

종료 코드: 문제 없음 0 / 문제 있음 1
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from auto_mobility.config import CAMERA_PARAMS, validate_camera_params


def main():
    if len(sys.argv) > 1:
        import yaml
        with open(sys.argv[1], encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        label = sys.argv[1]
    else:
        params = dict(CAMERA_PARAMS)
        label = "config.CAMERA_PARAMS"

    issues = validate_camera_params(params)
    print(f"[검증 대상] {label} ({len(params)}개 키)")
    if issues:
        print("[실패] 문제 키 발견:")
        for i in issues:
            print("  -", i)
        sys.exit(1)
    print("[통과] 문제 없음")
    sys.exit(0)


if __name__ == "__main__":
    main()
