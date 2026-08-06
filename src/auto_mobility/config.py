import os
import yaml
from pathlib import Path

def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 경로 반환"""
    return Path(__file__).resolve().parent.parent.parent

def load_topics_config() -> dict:
    """config/topics.yaml 파일 로드"""
    config_path = get_project_root() / "config" / "topics.yaml"
    if not config_path.exists():
        share_path = Path("/opt/ros") / os.getenv("ROS_DISTRO", "humble") / "share" / "auto_mobility" / "config" / "topics.yaml"
        if share_path.exists():
            config_path = share_path

    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

TOPICS_CONFIG = load_topics_config()
