# llmutils.py
import re
import io
import builtins
import inspect

# type("Object", (), a_dict )()
class Struct(object):
    def __init__(self, *args, **kwargs):
        merged_dict = {}
        argid = 0
        for d in args:
            if isinstance(d, dict):
                merged_dict.update(d) # added for new or update the key.
            else:
                merged_dict[f'arg{argid}'] = d
                argid += 1

        merged_dict.update(kwargs)

        for key, val in merged_dict.items():
            if isinstance(val, dict): val = Struct(**val)
            setattr(self, key, val)

    def __repr__(self):
        lines = []
        for k,v in self.__dict__.items():
            # details = f'{k}= {type(v)}, {v.shape if hasattr(v,"shape") else v}'
            details = f'{k}= {v.shape if hasattr(v,"shape") else v}'
            if hasattr(v, 'dtype'): details += f', dtype={v.dtype}'
            lines.append(details)
        return '\n'.join(lines)

    # for dict(obj) cast
    def keys(self): return self.__dict__.keys()
    def __getitem__(self, key): return dict(self.__dict__)[key]

#
# gpu
#
import os, subprocess, re, gc
def free_gpumem():
    import torch
    smi_process = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    smi_output = smi_process.stdout
    smi_processes_section = smi_output[smi_output.find('Processes:'):]

    path_pattern = r'(\d+)\s+\w+\s+(\.+\w+(?:\/(?:[\w.-]+\/?)*)?)\s+([^\s]+)'
    match = re.findall(path_pattern, smi_processes_section)
    for m in match:
        pid = int(m[0])
        if pid == os.getpid():
            print(f'Current process({pid=}) use', m[2])
        else:
            print('kill', m)
            try:
                subprocess.run(["kill", "-9", m[0]], check=True)
            except subprocess.CalledProcessError:
                pass
    torch.cuda.empty_cache()  # Clear GPU cache
    gc.collect()  # Run garbage collection

def get_gpu_memory_usage():
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        used_memory_str = result.stdout.strip()
        used_memory = int(used_memory_str)

        return used_memory
    except Exception as e:
        print(f"Error: {e}")
        return None

def free_vram():
    import torch
    torch.cuda.empty_cache()
    gc.collect()




#
# console text colors
#
class WhiteCar:
    def __ror__(self, x): return x.replace('\n', '\\n\n').replace('\t', '\\t\t').replace(' ', '⎵')
wc = WhiteCar()

class text_color:
    black,red,green,yellow,blue,magenta,cyan,white, gray = [*range(30,38), 90] # fgclr,  [*range(90,98), ''] # light-fgclr
    bold, italic, underline, strike = 1, 3, 4, 9  # attrs supported on vscode notebook.
    def __init__(self, fg,bg=0,attr=0):
        attr = f"{attr};" if attr > 0 else ''
        bg = f"{bg+10};" if bg > 0 else ''
        self.clr = f'\33[{attr}{bg}{fg}m'

    def __ror__(self, obj): return self.clr + str(obj) + '\33[0m'

black,red,green,yellow,blue,magenta,cyan,white,gray = (
    text_color(text_color.black),
    text_color(text_color.red),
    text_color(text_color.green),
    text_color(text_color.yellow),
    text_color(text_color.blue),
    text_color(text_color.magenta),
    text_color(text_color.cyan),
    text_color(text_color.white),
    text_color(text_color.gray),)

class cout:
    def __ror__(self, obj): print(f"[{inspect.stack()[1].lineno}] {str(obj)}")
    def __call__(self, *args, **kwds): print(f"[{inspect.stack()[1].lineno+1}]", *args, **kwds)
out = cout()


def print_tokens(tokens, tokenizer):
    token_clrs = [red,green,yellow,blue,magenta,cyan,white,gray]
    decoded = []
    for i, token in enumerate(tokens):
        clr = token_clrs[i%len(token_clrs)]
        decoded.append(tokenizer.decode(token)|wc|clr)
    print(f"tokens#: {len(tokens)}")
    print(''.join(decoded))

#
#
#
def format(*argv, **kwargv):
    with io.StringIO() as buffer:
        kwargv['file']=buffer
        builtins.print(*argv, **kwargv)
        return buffer.getvalue()

#
#
#
import random
def rand_index(lst):
    return random.randint(0, len(lst)-1)

import pandas as pd
def rand_item(lst, get_index = True):
    idx = rand_index(lst)
    if isinstance(lst, pd.DataFrame):
        return lst.iloc[idx], idx

    if get_index:
        return lst[idx], idx
    return lst[idx]


def static_vars(**kwargs):
    '''
    @static_vars(sep=' ')
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

