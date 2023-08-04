# 2022.08: runtime code analysis
# - replace builtin print
#

import os
import io
import inspect
import glob
import builtins
import re
import typing
import enum
import time
from pathlib import Path
import threading
from dataclasses import dataclass
from types import FrameType
from typing import Dict, Union, Optional, Set, Tuple
#
# function call history
#
GRAY, RED, GREEN, YELLOW, RESET = '\33[30m', '\33[31m', '\33[32m', '\33[33m', '\33[0m' # logging colors

class scope:
    # static operation flags
    CONTROL_PRINT = True # enable/disable builtins.print: disable all following print() if suppress is not None

    _logger_initialized = False
    _blacklist_files:typing.Set[str]=set()
    _builtin_print = builtins.print

    _workdir = os.getcwd()+'/'
    @staticmethod
    def _trim_workingdir(path):
        if scope._workdir in path:
            return path[len(scope._workdir):]
        return path

    @enum.unique
    class Suppress(enum.Enum):
        Following = enum.auto() # suppress all scope message in the function.
        SelfAndFollowing = enum.auto() # suppress all scope message including me in the function.

    #
    # exclude this file from scope logging.
    @staticmethod
    def blacklist_files(items:Union[str,list,set], all_subdir:bool=False):
        if not isinstance(items, (list,set)):
            items = [items]

        for i in items:
            if Path(i).is_dir():
                exts = '**/*.py' if all_subdir else '*.py'
                folder = os.path.abspath(i)
                srcs = glob.glob( os.path.join(folder, exts), recursive=all_subdir)
                scope._blacklist_files.update(srcs)
                print(f'+ {len(srcs)} for {i}')

            elif Path(i).is_file():
                file = os.path.abspath(i)
                scope._blacklist_files.add(file)

    @staticmethod
    def args():
        tls = scope._tls()
        if tls.suppress == scope.Suppress.SelfAndFollowing:
            return

        caller_fi = inspect.getouterframes(inspect.currentframe())[1]
        args2,_,_,values = inspect.getargvalues(caller_fi.frame)

        # print args
        for i,name in enumerate(args2):
            if name == 'self': continue
            value = values[name]

            if ".npyio." in str(type(value)): value = value.item()

            if type(value) == dict:
                scope.log(f"- args{i}. {name}=")
                for k,v in value.items():
                    scope.log(f"  - {k}: {v.shape if hasattr(v, 'shape') else v}")

            elif hasattr(value, 'shape'): # tensor or np.array
                scope.log(f"- args{i}. {name}= {value.shape}")

            #
            # TODO - add some codes to support another type.
            #
            else:
                scope.log(f"- args{i}. {name}= {value}")

    # _suppress=None # No thread concept.
    # _indent='' # No thread concept.
    @staticmethod
    def _tls():
        @dataclass
        class TlsData:
            name:str= threading.current_thread().name
            suppress:scope.Suppress=None
            indent:str=''

        this_thread = threading.current_thread()
        tlsdata = getattr(this_thread, 'tlsdata', None)
        if tlsdata is None:
            setattr(this_thread, 'tlsdata', TlsData())
            tlsdata = getattr(this_thread, 'tlsdata', None)
        return tlsdata

    @staticmethod
    def _getcaller(frame):
        if frame is None: return ('?', '?', -1)

        func_name = frame.f_code.co_name
        if 'self' in frame.f_locals:
            class_name = frame.f_locals['self'].__class__.__name__
            func_name = class_name + '.' + func_name
        src_path = frame.f_code.co_filename
        return (func_name, src_path, frame.f_lineno)

    @staticmethod
    def fn(func:typing.Union[typing.Callable, None]=None,
        suppress:typing.Union[Suppress, None] = None,
        max_count:typing.Union[int,None]=None) -> typing.Callable:

        # when using loguru
        # if not scope._logger_initialized is None:
        #     scope._logger_initialized = True
        #     # init logger for scope_fn, There is no way to check SCOPE log level exists.
        #     # logger.level("SCOPE", no=4, color='<green><d>') # SCOPE level.
        #     # logger.add(lambda msg: print(msg, end=''), level='SCOPE', format='<level>{message}</level>', filter=lambda r: r['level'].no == 4, colorize=True)

        def deco(func):
            logcnt=0
            # @functools.wraps(func)
            def wrapper(*args, **kwargs):
                tls = scope._tls()
                if (tls.suppress != None) or (func.__code__.co_filename in scope._blacklist_files):
                    return func(*args, **kwargs)

                nonlocal suppress, logcnt

                if max_count is not None:
                    logcnt += 1
                    if logcnt > max_count and suppress != scope.Suppress.SelfAndFollowing:
                        scope.log(f'limit funclog: {str(func)}, {logcnt=}, {max_count=}')
                        suppress = scope.Suppress.SelfAndFollowing

                #
                tls.suppress = suppress

                # set print function for following func
                #  disable all following print() if suppress is not None
                if scope.CONTROL_PRINT:
                    oprint = builtins.print
                    builtins.print = scope.indented_print if suppress is None else (lambda *args, **kwargs: None)

                if suppress == scope.Suppress.SelfAndFollowing: # print caller func scope.
                    result = func(*args, **kwargs)
                else:
                    assert suppress == scope.Suppress.Following or suppress == None
                    func_name = str(func).split()[1]

                    caller_fn, caller_file, caller_line = scope._getcaller(inspect.currentframe().f_back)
                    caller_file = scope._trim_workingdir(caller_file)

                    callee_file = scope._trim_workingdir(func.__code__.co_filename)
                    callee_line = func.__code__.co_firstlineno + 1

                    # ENTER....
                    scope.log(f'{{ {func_name} {callee_file}:{callee_line} from {caller_fn}() {caller_file}:{caller_line} - {tls.name}')

                    tls.indent += '\t'

                    result = func(*args, **kwargs)

                    tls.indent = tls.indent[:-1]

                    # EXIT....
                    scope.log(f'}} {func_name} - {tls.name}')

                if scope.CONTROL_PRINT:
                    builtins.print = oprint
                tls.suppress = None # restore suppress flag
                #
                return result
            return wrapper

        if callable(func): return deco(func)
        return deco

    # print(..., end='') handling?
    @staticmethod
    def indented_print(*args, **kwargs):
        tls = scope._tls()
        scope._builtin_print(tls.indent, *args, **kwargs)

    @staticmethod
    def log(*args, **kwargs):
        tls = scope._tls()
        buf = io.StringIO()
        scope._builtin_print(*args, **kwargs, end='', file=buf)
        message = buf.getvalue().replace('\n', '\n' + tls.indent)
        buf.close()

        if scope.logfp is not None:
            scope._builtin_print(tls.indent, message, file=scope.logfp)
        scope._builtin_print(tls.indent,GREEN + message + RESET)

        # if scope.logfp is not None:
        #   scope.builtin_print(tls.indent, *args, **kwargs, file=logfp)
        # scope.builtin_print(tls.indent,GREEN, *args, RESET, **kwargs)


    logfp:io.FileIO = None
    @staticmethod
    def set_logfile(logpath):
        scope.logfp = open(logpath, 'w', buffering=1) # line buffer



#
# apply scop_fn to py file.
#

def remove_scope_from_py(src_path, dst_path=None):

    src_path:Path = Path(src_path)
    if os.path.isdir():
        files = glob.glob( os.path.join(src_path, '**/*.py'), recursive=True)
        for pyfile in files:
            apply_scope_to_py(pyfile, dst_path)
        return

    assert src_path.endswith('.py'), src_path + ' is not py file.'
    with open(src_path, 'r', encoding='utf-8') as fp:
        txt = fp.read()

    if 'from scope_fn' not in txt and 'import scope_fn' not in txt:
        print(src_path, ': Already de-patched!!!')
        return # already patched.

    # remove import
    # remove # @scope.fn
    txt = re.sub(r'^[ \t]*from\s+scope_fn\s+import\s+scope[^\n]*\n', '', txt, count=1, flags=re.MULTILINE)
    txt = re.sub(r'^[ \t]*import\s+scope_fn[^\n]*\n', '', txt, count=1, flags=re.MULTILINE)
    txt = re.sub(r'^[ \t]*@scope\.fn[^\n]*\n', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'(^[ \t]+)scope.args\(\)([^\n]*\n)', r'\g<1># # scope.args()\g<2>', txt, flags=re.MULTILINE)


    dst_path = src_path if dst_path is None else Path(dst_path)
    with open(dst_path, 'w', encoding='utf-8') as fp:
        fp.write(txt)
        print(dst_path, 'is patched with no scope')
    return


#
#
#
def apply_scope_to_py(src_path, dst_path=None):

    if os.path.isdir(src_path):
        files = glob.glob( os.path.join(src_path, '**/*.py'), recursive=True)
        for pyfile in files:
            apply_scope_to_py(pyfile, dst_path)
        return

    if os.path.basename(src_path) in 'scope_fn.py, dbg.py, __init__.py':
        print(src_path, 'is skipped')
        return

    assert src_path.endswith('.py'), src_path + ' is not py file.'
    with open(src_path, 'r', encoding='utf-8') as fp:
        txt = fp.read()
    if 'from scope_fn import scope' in txt:
        print(src_path, ': Already patched!!!')
        return # already patched.

    txt = re.sub(r'\n(^[ \t]*)def\s+\w+', r'\n\g<1># @scope.fn\g<0>', txt, flags=re.MULTILINE)
    # txt = re.sub(r'\n\s*(from \w+\s+)?import\s+', r'\nfrom scope_fn import scope, scope.args\n\g<0>', txt, count=1, re.MULTILINE)

    scope_fn_dir = os.path.dirname(__file__)
    if not os.path.isabs(scope_fn_dir): scope_fn_dir = os.path.abspath(scope_fn_dir)

    pathinsert = '\n'.join([
        "import sys",
        f"if '{scope_fn_dir}' not in sys.path: sys.path.append('{scope_fn_dir}')\n",
    ])
    txt = re.sub(r'^\s*(from \w+\s+)?import\s+', pathinsert + r'from scope_fn import scope\n\n\g<0>', txt, count=1, flags=re.MULTILINE)

    if dst_path is None: dst_path = src_path
    with open(dst_path, 'w', encoding='utf-8') as fp:
        fp.write(txt)
        print(dst_path, 'is patched with scope')
    return


# add following to launch.json
# https://github.com/microsoft/debugpy/blob/fad8ae6577fcb14d762acac837000d5b758c00cd/src/debugpy/_vendored/pydevd/tests_python/test_pydevd_filtering.py
# "rules" : [{"module":"scope_fn", "include":false}], // NOTE: undocumented features, work on vscode 1.76.0

def disable_stepinto_scope_fn_when_debugging():
    launch_path = './.vscode/launch.json'
    if not os.path.exists(launch_path):
        print('no file:', launch_path)
        return

    with open(launch_path) as fp:
        txt = fp.read()
        # "rules": [{ "module": "scope_fn", "include": false }]
        if '"rules"' not in txt:
            justMyCode = txt.find('"justMyCode"')
            txt = txt[:justMyCode] + '"rules": [{ "module": "scope_fn", "include": false }],\n\t\t\t' + txt[justMyCode:]
            print(justMyCode)
            print(txt)
            with open(launch_path, 'w') as fp:
                fp.write(txt)
                print('scope_fn is excluded from step into!!!')
        else:
            print('scope_fn is already excluded from step into!!!')


#
# examples
#

# scope.blacklist_files(__file__)

class A:
    @scope.fn
    def __init__(self, a=1, b=0) -> None:
        scope.args()
        pass

    @scope.fn
    def memberfunc(self, a,b):
        scope.args()
        print('memberfunc', a,b)

@scope.fn
def scope_test():

    print('-------------- max_count=2 test ---------------')
    @scope.fn(max_count=2)
    def max_count_test(a,b,c):
        scope.args()
        print(f'It will shown only max_count(2) times', a,b,c)

        obj = A(100,101)
        obj.memberfunc(4,5)
        return a+b+c

    for _ in range(5):
        max_count_test(1,2,3)

    print('-------------- suppress test ---------------')
    @scope.fn
    def in_suppress_1():
        print('Not seen: in_suppress_1')

    @scope.fn(suppress=scope.Suppress.SelfAndFollowing)
    def suppress_following():
        print('Not seen: suppress_following()')
        in_suppress_1()
        print('Not seen: suppress_following: after in_suppress_1')
        return

    suppress_following()
    return

# (^\s+)def\s\S+  => $1@scope.fn\n$0


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', help='remove scope.fn from folder or python source')
    parser.add_argument('--apply', help='add scope.fn from folder or python source')
    parser.add_argument('--patch-launch-json', action='store_true', help='disable_stepinto_scope_fn_when_debugging')

    args = parser.parse_args()

    if args.apply is not None:
        print('try apply scope:', args.apply)
        apply_scope_to_py(args.apply)

    elif args.remove is not None:
        remove_scope_from_py(args.remove)

    elif args.patch_launch_json:
        disable_stepinto_scope_fn_when_debugging()

    else:
        # print('--------- scope.suppress_all() test ---------'); scope.suppress_all()
        scope_test()

