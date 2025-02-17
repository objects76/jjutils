#!/usr/bin/env python
# encoding: utf-8

import os
import inspect
import glob
import inspect

# black,red,green,yellow,blue,magenta,cyan,white = [text_color(clr) for clr in range(90,98)]
from .clrs import text_color
red = text_color(91)
green = text_color(92)
yellow = text_color(93)
blue = text_color(94)

# import sys, os
# if os.path.abspath('.') not in sys.path: sys.path.append(os.path.abspath('.'))
def fname(inc_line=False) -> str:
    """현재 실행 중인 함수의 이름을 반환"""
    caller = inspect.currentframe().f_back  # type: ignore
    if inc_line:
        return f"{caller.f_code.co_name}:{caller.f_lineno}" # type: ignore
    return caller.f_code.co_name # type: ignore

def typeinfo(obj, short=False):
    s = str(type(obj))
    if s.startswith("<class '"): s = s[8:-2]
    if short:
        return s.split(".")[-1]
    return s

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def set_default_logger():
    import logging
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(name)s - %(levelname)s: %(message)s',
                        filename= None, filemode='a',
                        datefmt="%H:%M:%S", # "%Y-%m-%d %H:%M:%S",
    )

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

            return super().format(record)

    for handler in logging.getLogger().handlers:
        handler.setFormatter(CustomFormatter(
            # '%(asctime)s - '
            '%(name)s - %(color_code)s%(levelname)s%(reset_code)s: '
            '%(message)s at %(funcName)s() %(pathname)s:%(lineno)d',
            datefmt="%H:%M:%S"
        ))



# def static_var(varname, value):
#     def decorate(func):
#         setattr(func, varname, value)
#         return func
#     return decorate
#
#
# def dump_packed(rawdata_path, printout=False):
#     import numpy as np
#     import os
#     npy_data = None
#     if isinstance(rawdata_path, str):
#         if rawdata_path.endswith('.npy'):
#             assert os.path.exists(rawdata_path)
#             npy_data = np.load(rawdata_path, allow_pickle=True, encoding='latin1')
#             if npy_data.ndim == 0:
#                 npy_data = npy_data.item()
#         elif rawdata_path.endswith('.npz'):
#             assert os.path.exists(rawdata_path)
#             npy_data = np.load(rawdata_path, allow_pickle=True, encoding='latin1')
#             npy_data =  { k: npy_data[k] for k in npy_data.files }
#         elif rawdata_path.endswith('.pkl'):
#             npy_data = np.load(rawdata_path, allow_pickle=True, encoding='latin1')
#         else:
#             print("\33[31m Not supported file format. \33[0m")
#             print('\t', rawdata_path)
#             return
#     elif type(rawdata_path) == dict:
#         npy_data = rawdata_path
#     elif ".npyio." in str(type(rawdata_path)):
#         npy_data = rawdata_path.item()
#
#     if printout:
#         def print_item(k,v):
#             if hasattr(v,'shape'):
#                 typestr = str(type(v))
#                 if 'scipy.sparse' in typestr: v = np.array(v.todense()) # SciPy sparse matrix => numpy matrix.
#
#                 print(f"\33[33m{k}: shape={v.shape}, dtype={v.dtype}, {typestr}\33[0m")
#                 head = str(v)[:100]
#                 head = head.replace('\n', '\n      ')
#                 print( '     ', head )
#             elif len(v) > 10:
#                 print(f"\33[33m{k}: len={len(v)}, {v[:10]}, {type(v)}\33[0m")
#             else:
#                 print(f"\33[33m{k}: {v}, {type(v)}\33[0m")

#         if isinstance( npy_data, dict):
#             for k,v in npy_data.items():
#                 print_item(k,v)
#         else:
#             filename = os.path.basename(rawdata_path)
#             print_item(filename, npy_data)
#     return npy_data

#
#
#

from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers.web import JsonLexer
from pygments.style import Style
from pygments.token import Token
import json

# https://pygments.org/docs/styles/
class MyStyle(Style):
    styles = {
        Token.String: "ansiyellow",
        Token.Name: "ansibrightgreen",
    }

class cout:
    black,red,green,yellow,blue,magenta,cyan,white,gray = *range(30,38), 90

    def __ror__(self, txt):
        if self.clr != 0:
            print(f"\33[{self.clr}m{txt}\33[0m")
            self.clr = 0
        else:
            print(txt)

    def __call__(self, clr):
        self.clr = clr
        return self
out = cout()

#
#
#
import os, sys, subprocess

def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])

#
# dict to attrs
#
class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if isinstance(val, dict):
                val = Struct(**val)
            setattr(self, key, val)

    def __repr__(self):
        lines = []
        for k,v in self.__dict__.items():
            details = f'{k}: {type(v)}, {v.shape if hasattr(v,"shape") else v}'
            if hasattr(v, 'dtype'): details += f', dtype={v.dtype}'
            lines.append(details)
        return '\n'.join(lines)

    # for dict(obj) cast
    def keys(self): return self.__dict__.keys()
    def __getitem__(self, key): return dict(self.__dict__)[key]

    def apply(self, func, for_only_attr = None):
        for k,v in self.__dict__.items():
            if for_only_attr is None or hasattr(v, for_only_attr):
                setattr(self, k, func(v))

#
#
#
def fmt(*args, sep=', '):
    return sep.join([ repr(a) for a in args])


def print_obj(obj)->str:
    try:
        print( '\n'.join([f"{k|green}: {v}" for k,v in vars(obj).items()]) )
    except:
        print( str(obj) )
        pass


class StopExecution(Exception):
    def _render_traceback_(self): print('stop execution')


# class _srcpos:
#     def __get__(self, inst, owner):
#         return f"{inspect.stack()[1].function} :{inspect.stack()[1].lineno}"


# class Debug:
#     class raise_StopCell:
#         class StopCell(BaseException):
#             def _render_traceback_(self):
#                 print('\033[30;100m', '[ STOP CELL EXECUTION ]', '\033[0m')

#         def __get__(self, instance, owner):
#             raise self.StopCell()

#     stop_cell = raise_StopCell()
#     srcpos = _srcpos()


import time
class ExecutionTime:
    def __init__(self) -> None:
        self.start()

    def start(self):
        self.tick = time.time()

    def check(self, name):
        elapsed = time.time() - self.tick
        print(f'time of {name}: {elapsed:.1f}sec'|yellow)
        self.tick = time.time()



if __name__ == '__main__':

    # test_cxtmgr()
    # text_color_table()

    print( fmt(1, 'abc', list(), f'100+100={100+100}') )
    print( fmt(2, 'abc', list(), f'100+100={100+100}', sep='/') )


    print( fmt(3, 'abc', list(), f'100+100={100+100}', sep='/') )
    print( fmt() )
