#!/usr/bin/env python
# encoding: utf-8

import os
import inspect
import logging
import subprocess
import sys
import time
from typing import Any

from .clrs import text_color

# Define text colors
red = text_color(91)
green = text_color(92)
yellow = text_color(93)
blue = text_color(94)

def fname(inc_line: bool = False) -> str:
    """현재 실행 중인 함수의 이름을 반환합니다.

    Args:
        inc_line (bool): 라인 번호를 포함할지 여부.

    Returns:
        str: 함수 이름 (및 라인 번호).
    """
    caller = inspect.currentframe().f_back  # type: ignore
    if inc_line:
        return f"{caller.f_code.co_name}:{caller.f_lineno}"  # type: ignore
    return caller.f_code.co_name  # type: ignore

def typeinfo(obj: Any, short: bool = False) -> str:
    """객체의 타입 정보를 반환합니다.

    Args:
        obj (Any): 타입을 확인할 객체.
        short (bool): 짧은 형식으로 반환할지 여부.

    Returns:
        str: 객체의 타입 정보.
    """
    s = str(type(obj))
    if s.startswith("<class '"):
        s = s[8:-2]
    if short:
        return s.split(".")[-1]
    return s

def static_vars(**kwargs):
    """함수에 정적 변수를 추가합니다."""
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def devlog_null(*argv, **kwargv): pass
def devlog(*argv, **kwargv):
    """개발 로그를 출력합니다."""
    frame = inspect.currentframe().f_back  # type: ignore
    info = inspect.getframeinfo(frame)  # type: ignore
    caller = info.function
    line_no = info.lineno
    src_path = info.filename
    if src_path.startswith("/tmp/ipykernel"):
        src_path = './notebook'
    else:
        src_path = src_path.replace(os.getcwd(), '.')

    kwargv['end'] = kwargv.get('end', '') + f" at {caller}() {src_path}:{line_no}\n"
    print(*argv, **kwargv)

def set_default_logger(source=True):
    """기본 로거를 설정합니다."""
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        filename=None, filemode='a',
                        datefmt="%H:%M:%S")

    class CustomFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\33[107m",    # White
            logging.INFO: "\33[102m",     # Green
            logging.WARNING: "\33[103m",  # Yellow foreground
            logging.ERROR: "\33[101m",    # Red foreground
            logging.CRITICAL: "\33[101m", # Red background
        }

        def format(self, record):
            record.levelname = record.levelname[:4]
            if record.pathname.startswith("/tmp/ipykernel_"):
                record.pathname = "notebook"
            else:
                record.pathname = record.pathname.replace(os.getcwd(), '.')
                record.pathname = record.pathname.replace('/home/jjkim/anaconda3/envs/pyan320-weaviate/lib/python3.10/site-', './')

            record.color_code = self.COLORS.get(record.levelno, "")
            record.reset_code = "\33[0m"
            record.name = record.name[-15:]
            format = super().format(record)

            # full line coloring
            if record.levelno >= logging.WARNING:
                format = self.COLORS.get(record.levelno, "") + format.replace('\33[0m', '') + "\33[0m"
            return format

    for handler in logging.getLogger().handlers:
        handler.setFormatter(CustomFormatter((
            '%(name)s - %(color_code)s%(levelname)s%(reset_code)s: %(message)s'
            ' at %(funcName)s() %(pathname)s:%(lineno)d' if source else ''
        ), datefmt="%H:%M:%S"))

def open_file(filename: str) -> None:
    """파일을 엽니다.

    Args:
        filename (str): 열 파일의 이름.
    """
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

class Struct:
    """딕셔너리를 속성으로 변환하는 클래스."""
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                val = Struct(**val)
            setattr(self, key, val)

    def __repr__(self) -> str:
        lines = []
        for k, v in self.__dict__.items():
            details = f'{k}: {type(v)}, {v.shape if hasattr(v, "shape") else v}'
            if hasattr(v, 'dtype'):
                details += f', dtype={v.dtype}'
            lines.append(details)
        return '\n'.join(lines)

    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, key):
        return dict(self.__dict__)[key]

    def apply(self, func, for_only_attr=None):
        for k, v in self.__dict__.items():
            if for_only_attr is None or hasattr(v, for_only_attr):
                setattr(self, k, func(v))

def fmt(*args: Any, sep: str = ', ') -> str:
    """인자를 포맷하여 문자열로 반환합니다.

    Args:
        *args (Any): 포맷할 인자들.
        sep (str): 구분자.

    Returns:
        str: 포맷된 문자열.
    """
    return sep.join([repr(a) for a in args])

def print_obj(obj: Any) -> None:
    """객체의 속성을 출력합니다.

    Args:
        obj (Any): 출력할 객체.
    """
    try:
        print('\n'.join([f"{k|green}: {v}" for k, v in vars(obj).items()]))
    except Exception as e:
        logging.error(f"Error printing object: {e}")
        print(str(obj))

class StopExecution(Exception):
    """실행 중지를 위한 예외 클래스."""
    def _render_traceback_(self):
        print('stop execution')

class ExecutionTime:
    """실행 시간을 측정하는 클래스."""
    def __init__(self) -> None:
        self.start()

    def start(self) -> None:
        """타이머를 시작합니다."""
        self.tick = time.time()

    def check(self, name: str) -> None:
        """경과 시간을 출력합니다.

        Args:
            name (str): 측정할 이름.
        """
        elapsed = time.time() - self.tick
        print(f'time of {name}: {elapsed:.1f}sec'|yellow)
        self.tick = time.time()

if __name__ == '__main__':
    print(fmt(1, 'abc', list(), f'100+100={100+100}'))
    print(fmt(2, 'abc', list(), f'100+100={100+100}', sep='/'))
    print(fmt(3, 'abc', list(), f'100+100={100+100}', sep='/'))
    print(fmt())