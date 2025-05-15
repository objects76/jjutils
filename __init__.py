"""
jjutils - 유틸리티 함수 모음 패키지
"""

__version__ = "0.1.0"

# struct.py가 jj_struct.py로 이름이 변경되었으므로 호환성을 위해 import
from .jj_struct import Struct

# logging.py가 jj_logging.py로 이름이 변경되었으므로 호환성을 위해 import
from .jj_logging import get_logger, val_name, out

# 자주 사용되는 모듈 미리 로드
from . import clrs
