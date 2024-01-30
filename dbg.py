#!/usr/bin/env python
# encoding: utf-8

import os
import inspect
import inspect
import glob


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


# def static_var(varname, value):
#     def decorate(func):
#         setattr(func, varname, value)
#         return func
#     return decorate

def srcpos(abspath=False):
    fi = inspect.getouterframes(inspect.currentframe())[1]

    filename = os.path.abspath(fi.filename) if abspath else os.path.basename(fi.filename)
    return f'\33[36mat {fi.function}() ./{filename}:{fi.lineno}\33[0m'




#
#
#
def dump_packed(rawdata_path, printout=False):
    import numpy as np
    import os
    npy_data = None
    if isinstance(rawdata_path, str):
        if rawdata_path.endswith('.npy'):
            assert os.path.exists(rawdata_path)
            npy_data = np.load(rawdata_path, allow_pickle=True, encoding='latin1')
            if npy_data.ndim == 0:
                npy_data = npy_data.item()
        elif rawdata_path.endswith('.npz'):
            assert os.path.exists(rawdata_path)
            npy_data = np.load(rawdata_path, allow_pickle=True, encoding='latin1')
            npy_data =  { k: npy_data[k] for k in npy_data.files }
        elif rawdata_path.endswith('.pkl'):
            npy_data = np.load(rawdata_path, allow_pickle=True, encoding='latin1')
        else:
            print("\33[31m Not supported file format. \33[0m")
            print('\t', rawdata_path)
            return
    elif type(rawdata_path) == dict:
        npy_data = rawdata_path
    elif ".npyio." in str(type(rawdata_path)):
        npy_data = rawdata_path.item()

    if printout:
        def print_item(k,v):
            if hasattr(v,'shape'):
                typestr = str(type(v))
                if 'scipy.sparse' in typestr: v = np.array(v.todense()) # SciPy sparse matrix => numpy matrix.

                print(f"\33[33m{k}: shape={v.shape}, dtype={v.dtype}, {typestr}\33[0m")
                head = str(v)[:100]
                head = head.replace('\n', '\n      ')
                print( '     ', head )
            elif len(v) > 10:
                print(f"\33[33m{k}: len={len(v)}, {v[:10]}, {type(v)}\33[0m")
            else:
                print(f"\33[33m{k}: {v}, {type(v)}\33[0m")

        if isinstance( npy_data, dict):
            for k,v in npy_data.items():
                print_item(k,v)
        else:
            filename = os.path.basename(rawdata_path)
            print_item(filename, npy_data)

    return npy_data

#
#
#
import inspect
class out:
    def __init__(self, suppress=False):
        self.suppress = suppress

    markers = []
    def __ror__(self, txt):

        if self.suppress:
            return

        frame = inspect.currentframe()
        caller_locals = frame.f_back.f_locals

        # print( caller_locals.items() )

        arg_name = [name for name, value in caller_locals.items() if value is txt]
        label = arg_name[0] if arg_name else None
        # print( arg_name )
        # arg_names = [name for name, value in caller_locals.items() if value in argv]

        try:
            for k in txt.keys():
                v = str(txt[k])
                if label: print(f'{label}: '|blue, end='')
                for m in out.markers:
                    v = v.replace(m, m|yellow)
                print(f"{k|green}= {v}")
        except:
            for m in out.markers:
                txt = str(txt).replace(m, m|yellow)
            if label: print(f'{label}= '|blue, end='')
            print(txt)

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

from .clrs import text_color
green = text_color(text_color.green)

def print_obj(obj)->str:
    try:
        print( '\n'.join([f"{k|green}: {v}" for k,v in vars(obj).items()]) )
    except:
        print( str(obj) )
        pass


class StopExecution(Exception):
    def _render_traceback_(self): print('stop execution')


if __name__ == '__main__':
    # test_cxtmgr()
    text_color_table()
    print( fmt(1, 'abc', list(), f'100+100={100+100}') )
    print( fmt(2, 'abc', list(), f'100+100={100+100}', sep='/') )
    print( fmt(3, 'abc', list(), f'100+100={100+100}', sep='/') )
    print( fmt() )
