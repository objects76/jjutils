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



# console color:
#   https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
#logx = lambda *args: print(caller(), "\33[32m", *args, '\033[0m')

def logx(*args, caller=False, **kwargs):
    if caller:
        print(_caller())

    if 'label' in kwargs.keys():
        print("\33[0;36;40m", kwargs['label']+':', '\33[0m')
        del kwargs['label']

    # if 'end' not in kwargs.keys():
    #     kwargs['end'] = '\n'
    if len(args)>0:
        print("\33[32m", *args, '\33[0m', **kwargs)


def text_color_table():
    """
    prints table of formatted text format options
    """
    for fg in range(30,38):
        s1 = ''
        format = ';'.join([str(fg)])
        s1 += f'\33[{format}m  \\33[{format}m \33[0m'
        print(s1)

    # for style in range(8):
    #     for fg in range(30,38):
    #         s1 = ''
    #         format = ';'.join([str(style), str(fg)])
    #         s1 += f'\33[{format}m  \\33[{format}m \33[0m'
    #         print(s1)
    #         # for bg in range(40,48):
    #         #     format = ';'.join([str(style), str(fg), str(bg)])
    #         #     s1 += f'\33[{format}m  \\33[{format}m \33[0m'
    #         # print(s1)
    #     print('\n')


def text_lightcolor_table():
    """
    prints table of formatted text format options
    """
    for fg in range(30+60,38+60):
        s1 = ''
        format = ';'.join([str(fg)])
        s1 += f'\33[{format}m  \\33[{format}m \33[0m'
        print(s1)

    # for style in range(8):
    #     for fg in range(30+60,38+60):
    #         s1 = ''
    #         for bg in range(40+60,48+60):
    #             format = ';'.join([str(style), str(fg), str(bg)])
    #             s1 += f'\33[{format}m  \\33[{format}m \33[0m'
    #         print(s1)
    #     print('\n')


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
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import torch

    def show_image(img, title=''):
        '''
        input imgs can be single or multiple tensor(s), this function uses matplotlib to visualize.
        Single input example:
        show(x) gives the visualization of x, where x should be a torch.Tensor
            if x is a 4D tensor (like image batch with the size of b(atch)*c(hannel)*h(eight)*w(eight), this function splits x in batch dimension, showing b subplots in total, where each subplot displays first 3 channels (3*h*w) at most.
            if x is a 3D tensor, this function shows first 3 channels at most (in RGB format)
            if x is a 2D tensor, it will be shown as grayscale map

        Multiple input example:
        show(x,y,z) produces three windows, displaying x, y, z respectively, where x,y,z can be in any form described above.
        '''
        if not isinstance(img, torch.Tensor):
            raise Exception("unsupported type:  "+str(type(img)))

        # plt.figure(img_idx)
        img = img.detach().cpu()

        if img.dim()==4: # 4D tensor
            bz = img.shape[0]
            c = img.shape[1]
            if bz==1 and c==1:  # single grayscale image
                img=img.squeeze()
            elif bz==1 and c==3: # single RGB image
                img=img.squeeze()
                img=img.permute(1,2,0)
            elif bz==1 and c > 3: # multiple feature maps
                img = img[:,0:3,:,:]
                img = img.permute(0, 2, 3, 1)[:]
                print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
            elif bz > 1 and c == 1:  # multiple grayscale images
                img=img.squeeze()
            elif bz > 1 and c == 3:  # multiple RGB images
                img = img.permute(0, 2, 3, 1)
            elif bz > 1 and c > 3:  # multiple feature maps
                img = img[:,0:3,:,:]
                img = img.permute(0, 2, 3, 1)[:]
                print('warning: more than 3 channels! only channels 0,1,2 are preserved!')
            else:
                raise Exception("unsupported type!  " + str(img.size()))
        elif img.dim()==3: # 3D tensor
            bz = 1
            c = img.shape[0]
            if c == 1:  # grayscale
                img=img.squeeze()
            elif c == 3:  # RGB
                img = img.permute(1, 2, 0)
            else:
                raise Exception("unsupported type!  " + str(img.size()))
        elif img.dim()==2:
            pass
        else:
            raise Exception("unsupported type!  "+str(img.size()))

        if len(title)>0:
            plt.title(title)

        img = img.numpy()  # convert to numpy
        img = img.squeeze()
        if bz ==1:
            plt.imshow(img, cmap='gray')
            # plt.colorbar()
            # plt.show()
        else:
            for idx in range(0,bz):
                plt.subplot(int(bz**0.5),int(np.ceil(bz/int(bz**0.5))),int(idx+1))
                plt.imshow(img[idx], cmap='gray')

        plt.show()
except:
    pass


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

if __name__ == '__main__':
    # test_cxtmgr()
    text_color_table()
    print( fmt(1, 'abc', list(), f'100+100={100+100}') )
    print( fmt(2, 'abc', list(), f'100+100={100+100}', sep='/') )
    print( fmt(3, 'abc', list(), f'100+100={100+100}', sep='/') )
    print( fmt() )
