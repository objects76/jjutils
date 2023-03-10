#!/usr/bin/env python
# encoding: utf-8

import os
import inspect
import glob
import builtins


import functools
import re
import typing
import enum


# def static_vars(**kwargs):
#     def decorate(func):
#         for k in kwargs:
#             setattr(func, k, kwargs[k])
#         return func
#     return decorate


#
# logging
#
GRAY, RED, GREEN, YELLOW, RESET = '\33[30m', '\33[31m', '\33[32m', '\33[33m', '\33[0m'

#
# function call history
#
from loguru import logger
import time

class scope:
    _suppress=None # No thread concept.
    _indent=None # No thread concept.
    # _basepath_len=len(os.getcwd()) + 1
    _basepath_len = len(os.path.dirname(__file__)) + 1
    _skipfiles:typing.Set[str]=set()
    builtins_print = builtins.print

    @enum.unique
    class Suppress(enum.Enum):
        Following = enum.auto()
        SelfAndFollowing = enum.auto()
        SuppressAll = 99

    @staticmethod
    def suppress_all():
        scope._suppress = scope.Suppress.SuppressAll()

    @staticmethod
    def skip_this_file():
        fi = inspect.getouterframes(inspect.currentframe())[1]
        scope._skipfiles.add(fi.filename)
        logger.log('SCOPE', f'{YELLOW}scope.fn disabled in {fi.filename}{RESET}')

    @staticmethod
    def args():
        caller_fi = inspect.getouterframes(inspect.currentframe())[1]
        args2,_,_,values = inspect.getargvalues(caller_fi.frame)

        for i,name in enumerate(args2):
            if name == 'self': continue
            value = values[name]

            if ".npyio." in str(type(value)): value = value.item()

            if type(value) == dict:
                print(f"- args{i}. {name}=")
                for k,v in value.items():
                    print('-',k,':', v.shape if hasattr(v, 'shape') else v)
            elif hasattr(value, 'shape'):
                print(f"- args{i}. {name}= {value.shape}")
            else:
                print(f"- args{i}. {name}= {value}")
    @staticmethod
    def fn(func:typing.Union[typing.Callable, None]=None,
        suppress:typing.Union[Suppress, None] = None,
        max_count:typing.Union[int,None]=None) -> typing.Callable:

        if scope._indent is None:
            scope._indent = ''
            # init logger for scope_fn, There is no way to check SCOPE log level exists.
            logger.level("SCOPE", no=4, color='<green><d>')
            logger.add(lambda msg: print(msg, end=''), level='SCOPE', format='<level>{message}</level>', filter=lambda r: r['level'].no == 4, colorize=True)

        def deco(func):
            outcnt=0
            # @functools.wraps(func)
            def wrapper(*args, **kwargs):
                '''
                scope_fn wrapper func
                '''
                if scope._suppress != None:
                    return func(*args, **kwargs)

                nonlocal suppress, outcnt
                assert type(func).__name__ == 'function'
                callee_file = func.__code__.co_filename

                if callee_file in scope._skipfiles:
                    suppress = scope.Suppress.SelfAndFollowing

                elif max_count is not None:
                    outcnt += 1
                    if outcnt > max_count and suppress != scope.Suppress.SelfAndFollowing:
                        logger.log('SCOPE', f'limit funclog: {str(func)}, {outcnt=}, {max_count=}')
                        suppress = scope.Suppress.SelfAndFollowing

                #
                assert None == scope._suppress # save suppress
                scope._suppress = suppress

                # set print function for following func
                oprint = builtins.print
                builtins.print = scope.nullprint

                if suppress == scope.Suppress.SelfAndFollowing: # print caller func scope.
                    result = func(*args, **kwargs)
                else:
                    assert suppress == scope.Suppress.Following or suppress == None
                    func_name = str(func).split()[1]

                    caller_fi = inspect.getouterframes(inspect.currentframe())[1]
                    caller_file = caller_fi.filename[scope._basepath_len:] \
                        if caller_fi.filename[0] == '/' else caller_fi.filename
                    # scope.builtins_print(f'{caller_fi.filename=}, {scope._basepath_len=}') # debug

                    callee_file = func.__code__.co_filename[scope._basepath_len:]
                    callee_line = func.__code__.co_firstlineno + 1
                    callee_pos = f'(./{callee_file}:{callee_line})'

                    # ENTER....
                    if suppress is None:
                        builtins.print = scope.myprint

                    logger.log('SCOPE', f'> {func_name} {callee_pos} from {caller_fi.function}() ./{caller_file}:{caller_fi.lineno}')

                    scope._indent += ' '*3
                    start_time = time.perf_counter()

                    result = func(*args, **kwargs)

                    scope._indent = scope._indent[:-3]

                    # EXIT....
                    logger.log('SCOPE', f'< {func_name}, exectime= {time.perf_counter()-start_time:.3f} sec')

                builtins.print = oprint
                scope._suppress = None # restore suppress flag
                #
                return result
            return wrapper

        if callable(func): return deco(func)
        return deco

    # print(..., end='') handling?
    @staticmethod
    def myprint(*args, **kwargs):
        scope.builtins_print(scope._indent, *args, **kwargs)

    @staticmethod
    def nullprint(*args, **kwargs): pass





def remove_scope_from_py(src_path, dst_path=None):

    if os.path.isdir(src_path):
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
    # remove @scope.fn
    txt = re.sub(r'^[ \t]*from\s+scope_fn\s+import\s+scope[^\n]*\n', '', txt, count=1, flags=re.MULTILINE)
    txt = re.sub(r'^[ \t]*import\s+scope_fn[^\n]*\n', '', txt, count=1, flags=re.MULTILINE)
    txt = re.sub(r'^[ \t]*@scope\.fn[^\n]*\n', '', txt, flags=re.MULTILINE)
    txt = re.sub(r'(^[ \t]+)scope.args\(\)([^\n]*\n)', r'\g<1># scope.args()\g<2>', txt, flags=re.MULTILINE)



    if dst_path is None: dst_path = src_path
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

    txt = re.sub(r'\n(^[ \t]*)def\s+\w+', r'\n\g<1>@scope.fn\g<0>', txt, flags=re.MULTILINE)
    # txt = re.sub(r'\n\s*(from \w+\s+)?import\s+', r'\nfrom scope_fn import scope, scope.args\n\g<0>', txt, count=1, re.MULTILINE)
    txt = re.sub(r'^\s*(from \w+\s+)?import\s+', r'from scope_fn import scope\n\g<0>', txt, count=1, flags=re.MULTILINE)

    if dst_path is None: dst_path = src_path
    with open(dst_path, 'w', encoding='utf-8') as fp:
        fp.write(txt)
        print(dst_path, 'is patched with scope')
    return


class A:
    @scope.fn
    def __init__(self, a=1, b=0) -> None:
        scope.args()
        pass

    @scope.fn
    def memberfunc(self, a,b):
        scope.args()
        print('memberfunc', a,b)


def scope_test():

    @scope.fn(max_count=2)
    def max2_test(a,b,c):
        scope.args()
        print('max2_test', a,b,c)

        obj = A(100,101)
        obj.memberfunc(4,5)
        return a+b+c

    for _ in range(5):
        max2_test(1,2,3)

    print('-------------- suppress test ---------------')
    @scope.fn
    def inner1():
        print('inner1')

    @scope.fn
    def inner2():
        print('inner2')

    @scope.fn(suppress=scope.Suppress.SelfAndFollowing)
    def suppress_following():
        print('suppress_following')
        inner1()
        print('suppress_following: after inner1')
        inner2()
        print('suppress_following: after inner2')
        return

    suppress_following(); return


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



def test():
    print('-- test --')
    exit(0)




if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--remove', help='remove scope.fn from folder or python source')
    parser.add_argument('--apply', help='add scope.fn from folder or python source')
    parser.add_argument('--patch-launch-json', action='store_true', help='disable_stepinto_scope_fn_when_debugging')

    args = parser.parse_args()
    print(args)
    if args.apply is not None:
        print('try apply scope:', args.apply)
        apply_scope_to_py(args.apply)
    elif args.remove is not None:
        remove_scope_from_py(args.remove)

    elif args.patch_launch_json:
        disable_stepinto_scope_fn_when_debugging()

    else:
        test()

