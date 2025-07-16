"""
pytest 설정 파일

이 파일은 pytest가 테스트를 실행할 때 사용하는 설정을 포함합니다.
"""

import sys
import os

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 테스트 설정
def pytest_configure(config):
    """pytest 설정을 구성합니다."""
    # 테스트 마커 등록
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")


def pytest_collection_modifyitems(config, items):
    """테스트 아이템을 수정합니다."""
    for item in items:
        # 기본적으로 모든 테스트를 unit으로 마킹
        if not any(item.iter_markers()):
            item.add_marker("unit")
