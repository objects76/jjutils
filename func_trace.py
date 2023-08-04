
from pathlib import Path
import glob
import sys
import io, os
import inspect
from dataclasses import dataclass
import re
from types import FrameType
from typing import Dict, Union, Optional, Set
import builtins
import threading
import enum

GRAY, RED, GREEN, YELLOW, RESET = '\33[30m', '\33[31m', '\33[32m', '\33[33m', '\33[0m' # logging colors
MAX_FUNC_LOGGING = 20


@enum.unique
class Suppress(enum.Enum):
    Following = enum.auto()
    SelfAndFollowing = enum.auto()


class Profile:
    _workdir = None
    _builtin_print = builtins.print

    @dataclass
    class FuncContext:
        # id:str
        show_args:bool = False
        logging_cnt:int=MAX_FUNC_LOGGING
        suppress:Optional[Suppress] = None

    _FuncCxt:Dict[str, FuncContext] = dict()

    @staticmethod
    def start(indent_print:bool = False):
        print('start profile...')
        assert len(Profile._whitelist_files) > 0, "No whitelist files"
        Profile.rmv_sources(__file__) # remove self.

        @dataclass
        class TlsData:
            name:str= threading.current_thread().name
            suppress_from:int = 0
            suppress:Suppress=None
            indent:str=''

        this_thread = threading.current_thread()
        assert getattr(this_thread, 'tlsdata', None) is None
        setattr(this_thread, 'tlsdata', TlsData())

        sys.setprofile(Profile._trace)
        if indent_print:
            builtins.print = Profile.log


    @staticmethod
    def set_func_context(func_name:str, srcpath:str=None, *,
                         show_args:bool=False,
                         logging_cnt:int=MAX_FUNC_LOGGING,
                         suppress:Suppress=None,
                         block:bool=False):
        assert Path(Profile._workdir).is_dir()

        if srcpath is None and block:
            # set global blacklist func
            Profile._global_blackfunc.add(func_name)
            return

        srcpath = Profile._trim_workingdir( os.path.abspath(srcpath) )
        id = f"{func_name}@{srcpath}"

        if block:
            Profile._blacklist_funcs.add(id)
            return

        Profile._FuncCxt[id] = Profile.FuncContext(show_args=show_args, logging_cnt=logging_cnt, suppress=suppress)


    @staticmethod
    def set_func_context2(funcstr:str, *,
                          show_args:bool=False,
                          logging_cnt:int=MAX_FUNC_LOGGING,
                          suppress:Suppress=None,
                          block:bool=False):
        '''
        funcstr: you can removed the line with python comment.
            # ex) get func_name and srcpath from ' IBasicBlock.forward() models/arcface.py:61, from '
        '''
        funcstr = re.sub(r'^\s*#.*$', '', funcstr, flags=re.MULTILINE) # remove comment(#) line
        matches = re.findall(r'([\w<](?!\(\)).+)\(\)\s+([^:]+):\d+', funcstr)
        if matches:
            for match in matches:
                func_name, srcpath = match
                func_name = func_name.strip()
                srcpath = srcpath.strip()
                Profile.set_func_context(func_name, srcpath,
                                         show_args=show_args, logging_cnt=logging_cnt, suppress=suppress, block=block)

    _whitelist_files:Set[str] = set()      # Look for these words in the file path.
    _blacklist_funcs:Set[str] = set()      # Ignore func, etc. in the function name.
    _global_blackfunc = set(['<listcomp>', '<genexpr>']) # + ['<module>']


    @staticmethod
    def add_sources(items:Union[str,list,set], all_subdir:bool=False):
        if Profile._workdir is None:
            Profile._workdir = os.path.abspath(os.getcwd())
            if Profile._workdir[-1] != '/': Profile._workdir += '/'
            print('working dir:' + Profile._workdir)

        if not isinstance(items, (list,set)):
            items = [items]

        for i in items:
            if Path(i).is_dir():
                exts = '**/*.py' if all_subdir else '*.py'
                folder = os.path.abspath(i)
                srcs = glob.glob( os.path.join(folder, exts), recursive=all_subdir)
                # srcs = [Profile._trim_workingdir(i) for i in srcs if '__init__.py' not in i] # remove __init__.py
                srcs = [i for i in srcs if '__init__.py' not in i] # remove __init__.py
                Profile._whitelist_files.update(srcs)
                print(f'+ {len(srcs)} for {i}')
                # print(f'+ <{srcs[0]}>') # debug

            elif Path(i).is_file():
                # file = Profile._trim_workingdir( os.path.abspath(i) )
                file = os.path.abspath(i)
                Profile._whitelist_files.add(file)
                # print('+', i)


    @staticmethod
    def rmv_sources(items:Union[str,list,set]):
        if not isinstance(items, (list,set)):
            items = [items]

        for i in items:
            if Path(i).is_dir():
                all_subdir = False
                folder = os.path.abspath(i)
                exts = '**/*.py' if all_subdir else '*.py'
                srcs = glob.glob( os.path.join(folder, exts), recursive=all_subdir)
                # srcs = [Profile._trim_workingdir(i) for i in srcs]
                Profile._whitelist_files.discard(srcs)
                print(f'+ {len(srcs)} for {i}')
                # print(f'+ <{srcs[0]}>') # debug

            elif Path(i).is_file():
                # file = Profile._trim_workingdir( os.path.abspath(i) )
                file = os.path.abspath(i)
                Profile._whitelist_files.discard(file)


    @staticmethod
    def rmv_sources2(filestrs:str):
        filestrs = re.sub(r'^\s*#.*$', '', filestrs, flags=re.MULTILINE) # remove comment(#) line
        py_files = re.findall(r'(\S+\.py)', filestrs)
        Profile.rmv_sources(py_files)

    #
    # function arguments
    #
    @staticmethod
    def _args(frame:FrameType):
        args,_,_,values = inspect.getargvalues(frame)
        for i,name in enumerate(args):
            if name == 'self': continue
            value = values[name]

            if ".npyio." in str(type(value)): value = value.item()

            if type(value) == dict:
                Profile.log(f"- args{i}. {name}=", clr=YELLOW)
                for k,v in value.items():
                    Profile.log(f"  - {k}: {v.shape if hasattr(v, 'shape') else v}", clr=YELLOW)

            elif hasattr(value, 'shape'): # tensor or np.array
                Profile.log(f"- args{i}. {name}= {value.shape}", clr=YELLOW)

            #
            # TODO - add some codes to support another type.
            #
            else:
                Profile.log(f"- args{i}. {name}= {value}", clr=YELLOW)


    @staticmethod
    def args():
        frame = inspect.currentframe().f_back
        if not any(x in frame.f_code.co_filename for x in Profile._whitelist_files):
            return

        callee_func, callee_file, callee_line = Profile._getfi(frame)
        id = f"{callee_func}@{callee_file}"

        if id in Profile._blacklist_funcs:
            return

        funcxt = Profile._FuncCxt.get(id, Profile.FuncContext())
        if funcxt.logging_cnt < 0:
            return
        Profile._args(frame)


    @staticmethod
    def _trim_workingdir(path):
        if Profile._workdir in path:
            return path[len(Profile._workdir):]
        return path


    @staticmethod
    def _getfi(frame):
        if frame is None: return ('?', '?', -1)
        src_path = frame.f_code.co_filename
        # src_path = Profile._trim_workingdir(frame.f_code.co_filename)

        func_name = frame.f_code.co_name
        if 'self' in frame.f_locals:
            for cls in inspect.getmro(frame.f_locals['self'].__class__):
                if hasattr(cls, func_name):
                    member = getattr(cls, func_name)
                    if inspect.isfunction(member) and member.__code__ == frame.f_code:
                        return (member.__qualname__, src_path, frame.f_lineno)

            func_name = frame.f_locals['self'].__class__.__name__ + '.' + func_name
        return (func_name, src_path, frame.f_lineno)


    @staticmethod
    def _trace(frame, event, arg):
        if event != "call" and event != 'return': return

        # filtering non ROI
        if frame.f_code.co_filename not in Profile._whitelist_files:
            return
        if frame.f_code.co_name in Profile._global_blackfunc:
            return

        callee_func, callee_file, callee_line = Profile._getfi(frame)
        fnid = f"{callee_func}@{callee_file}"

        if fnid in Profile._blacklist_funcs:
            return

        tls = threading.current_thread().tlsdata
        if event == "call":
            if tls.suppress is not None:
                return

            funcxt = Profile._FuncCxt.get(fnid, Profile.FuncContext())
            if funcxt.suppress is not None:
                tls.suppress_from = id(frame)
                tls.suppress = funcxt.suppress
                if funcxt.suppress == Suppress.SelfAndFollowing:
                    return

            funcxt.logging_cnt -= 1
            if funcxt.logging_cnt < 0:
                return

            Profile._FuncCxt[fnid] = funcxt

            caller_func, caller_file, caller_line = Profile._getfi(frame.f_back)
            Profile.log(f'{{ {callee_func}() { Profile._trim_workingdir(callee_file)}:{callee_line}, '
                        f'from {caller_func} ({Profile._trim_workingdir(caller_file)}:{caller_line}) - {tls.name}',
                        clr=GREEN)
            tls.indent += '\t'
            # print('debug: enter', id(frame))

            if funcxt.show_args:
                Profile._args(frame)

        elif event == "return":

            if tls.suppress:
                if tls.suppress_from != id(frame):
                    return
                suppress = tls.suppress
                tls.suppress = None
                if suppress == Suppress.SelfAndFollowing:
                    return

            funcxt = Profile._FuncCxt.get(fnid, Profile.FuncContext())
            if funcxt.logging_cnt >= 0:
                # print('debug: exit', id(frame))
                tls.indent = tls.indent[:-1]
                Profile.log(f'}} {callee_func}() - {tls.name}', clr=GREEN)



    #
    # logging funcs
    #
    # @staticmethod
    # def fmt(*args, **kwargs):
    #     buf = io.StringIO()
    #     print(*args, **kwargs, end='', file=buf)
    #     message = buf.getvalue().replace('\n', '\n' + Profile.indent)
    #     buf.close()
    #     return Profile.indent + message


    logfp:io.FileIO = None
    @staticmethod
    def set_logfile(logpath:Union[str,Path]):
        Profile.logfp = open(str(logpath), 'w', buffering=1) # line buffer

    @staticmethod
    def log(*args, clr:Optional[str]=None, **kwargs):

        tls = threading.current_thread().tlsdata

        end = kwargs.pop('end', '\n')
        buf = io.StringIO()
        Profile._builtin_print(*args, **kwargs, end='', file=buf)
        message = buf.getvalue()
        if message[-1] == '\n':
            message = tls.indent + message[:-1].replace('\n', '\n' + tls.indent) + '\n' # do not indent last '\n'
        else:
            message = tls.indent + message.replace('\n', '\n' + tls.indent)
        buf.close()

        if clr is None:
            Profile._builtin_print(message, end=end)
        else:
            Profile._builtin_print(clr, message, RESET, end=end)

        if Profile.logfp is not None:
            Profile._builtin_print(message, file = Profile.logfp, end=end)




# #
# # examples
# #
# class Base:
#     def __init__(self, b) -> None:
#         print('b', b, ' in Base')
#         pass

# class A(Base):

#     def __init__(self, a=1, b=0) -> None:
#         super().__init__(b)
#         pass


#     def memberfunc(self, a,b):
#         print('memberfunc', a,b)


# def scope_test():

#     print('-------------- max_count=2 test ---------------')
#     def max_count_test(a,b,c):
#         print(f'It will shown only max_count(2) times', a,b,c)

#         obj = A(100,101)
#         obj.memberfunc(4,5)
#         return a+b+c

#     for _ in range(5):
#         max_count_test(1,2,3)

#     print('-------------- suppress test ---------------')

#     def in_suppress_1():
#         print('Not seen: in_suppress_1')

#     def suppress_following():
#         print('enter: suppress_following()')
#         in_suppress_1()
#         print('exit: suppress_following: after in_suppress_1')
#         return

#     suppress_following()
#     return




# if __name__ == '__main__':

#     Profile.add_sources('./')
#     Profile.set_func_context('suppress_following', __file__, suppress=Suppress.SelfAndFollowing)
#     Profile.set_func_context('suppress_following', __file__, suppress=Suppress.Following)

#     Profile.start(indent_print=True)
#     Profile.add_sources(__file__)
#     scope_test()

