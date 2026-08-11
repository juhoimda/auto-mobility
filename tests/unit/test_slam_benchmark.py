import unittest
from auto_mobility.slam.benchmark_slam import (
    SLAM_BASE_PARAMS,
    build_camera_cmd,
    _coerce_param,
)
from auto_mobility.launch.launch_common import RTABMAP_PARAMS


class TestSlamBenchmarkConsistency(unittest.TestCase):
    def test_slam_base_params_derived_from_production(self):
        # 벤치마크 기준 파라미터가 production(RTABMAP_PARAMS)과 일치해야 drift 가 없다.
        for key, prod_val in RTABMAP_PARAMS.items():
            self.assertIn(key, SLAM_BASE_PARAMS,
                          f"production param '{key}' 가 벤치마크 기준에 없음")
            self.assertEqual(SLAM_BASE_PARAMS[key], _coerce_param(prod_val),
                             f"{key}: production={prod_val} vs benchmark={SLAM_BASE_PARAMS[key]}")

        # 벤치마크 전용 launch 키
        self.assertEqual(SLAM_BASE_PARAMS["approx_sync"], "true")
        self.assertEqual(SLAM_BASE_PARAMS["approx_sync_max_interval"], 0.15)
        self.assertEqual(SLAM_BASE_PARAMS["topic_queue_size"], 30)

    def test_build_camera_cmd_uses_camera_params(self):
        cmd = build_camera_cmd("640x480x30", "SENSOR_DATA")
        self.assertIn("depth_module.depth_profile:=640x480x30", cmd)
        self.assertIn("rgb_camera.color_profile:=640x480x30", cmd)
        self.assertIn("color_qos:=SENSOR_DATA", cmd)
        self.assertIn("align_depth.enable:=false", cmd)
        # production 단일 소스의 필터/에미터 설정이 벤치마크에도 반영
        self.assertIn("filters:=spatial,temporal,hole_filling", cmd)
        self.assertIn("depth_module.emitter_enabled:=1", cmd)
        # bool 은 rtabmap/camera CLI 파싱과 일치하도록 소문자로 렌더링
        self.assertNotIn("align_depth.enable:=False", cmd)
        self.assertNotIn("enable_sync:=True", cmd)

    def test_coerce_param(self):
        self.assertIs(_coerce_param("true"), True)
        self.assertIs(_coerce_param("false"), False)
        self.assertEqual(_coerce_param("10"), 10)
        self.assertEqual(_coerce_param("4.0"), 4.0)
        self.assertEqual(_coerce_param("abc"), "abc")


if __name__ == "__main__":
    unittest.main()
