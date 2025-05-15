from setuptools import setup, find_packages
import os
import re
import sys

# 버전 정보를 읽어오는 함수
def get_version():
    init_py = os.path.join(os.path.dirname(__file__), "__init__.py")
    if os.path.exists(init_py):
        with open(init_py, "r", encoding="utf-8") as f:
            version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
            if version_match:
                return version_match.group(1)
    return "0.1.0"  # 기본 버전

# README 정보 읽기
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="jjutils",
    version=get_version(),
    author="jjkim",
    description="유틸리티 함수 모음",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/objects76/jjutils",
    packages=find_packages(),
    python_requires=">=3.10",  # Python 3.9 이상으로 변경
    install_requires=[
        # 필요한 의존성 패키지들을 여기에 추가하세요
        # 예: "numpy>=1.20.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)