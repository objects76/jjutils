# llmutils.py
import re

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
class text_color:
    black,red,green,yellow,blue,magenta,cyan,white, gray = [*range(30,38), 90] # fgclr,  [*range(90,98), ''] # light-fgclr
    light = 60
    bold, underline, strike  = 1,4,9 # attrs supported on vscode notebook.
    def __init__(self, fg:int=0,bg:int=0,attr:int=0):
        self.clr = f'\33[{attr}'
        # assert fg != bg, f"invalid {fg=}, {bg=}"
        if fg: self.clr += f';{fg}'
        if bg: self.clr += f';{bg+10}'
        self.clr += 'm'

    def __ror__(self, obj): return self.clr + str(obj) + '\33[0m'

    @staticmethod
    def decolorize(text):
        return re.sub(r"\x1b\[[\d;]+m", "", text)
bg_gray = text_color(0, text_color.gray)
bg_red = text_color(0, text_color.red)
green = text_color(text_color.green)
blue = text_color(text_color.blue)
yellow = text_color(text_color.yellow)

#
#
#
import inspect
class out:
    def __init__(self, nl_after = False, suppress=False):
        self.suppress = suppress
        self.nl_after = nl_after

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

            if self.nl_after:
                print()
        except:
            for m in out.markers:
                txt = str(txt).replace(m, m|yellow)
            if label: print(f'{label}= '|blue, end='')
            print(txt)
            if self.nl_after:
                print()

#
#
#
import io
def format(*argv, **kwargv):
    with io.StringIO() as buffer:
        kwargv['file']=buffer
        builtins.print(*argv, **kwargv)
        return buffer.getvalue()

def colorize_llm(
        text:str,
        clr:text_color= text_color(text_color.magenta+text_color.light),
        tokens:str=None):
    ''' colorize_llm '''
    if tokens is None:
        tokens = colorize_llm.tokens

    for token in tokens.split():
        colored_token = token|clr
        text = text.replace(token, colored_token)
    return text
colorize_llm.tokens = '<s> </s> [INST] [/INST] <<SYS>> <</SYS>> <FUNCTIONS> </FUNCTIONS> <|system|> <|user|> <|assistant|> <eom> <|end_of_turn|>'
#
#
#
_special_tokens = '<s> </s> <unk> <pad> '.split() + [0, 1, 2, 32000]
def readable_tokens(lst, special_tokens=None) -> list[str]:
    import torch
    if isinstance(lst, torch.Tensor):
        lst = lst.tolist()

    count = 1
    strs = []
    if special_tokens is None:
        special_tokens = _special_tokens

    i_new = 0
    token = lst[i_new]
    count = 1

    for i in range(1, len(lst)):
        if token == lst[i]:
            count += 1

        else:
            tokenclr = bg_red if (token in special_tokens) else bg_gray
            if count >= 3:
                strs.append(f"{token}*{count}@{i_new}"|tokenclr)
            else:
                if tokenclr == bg_red:
                    strs.append(f"{token}@{i_new}"|tokenclr)
                    for _ in range(count-1): strs.append(token|tokenclr )
                else:
                    for _ in range(count): strs.append(token|tokenclr )
            i_new = i
            token = lst[i_new]
            count = 1


    tokenclr = bg_red if token in special_tokens else bg_gray
    if count >= 3:
        strs.append(f"{token}*{count}@{i_new}"|tokenclr)
    else:
        if tokenclr == bg_red:
            strs.append(f"{token}@{i_new}"|tokenclr)
            for _ in range(count-1): strs.append(token|tokenclr )
        else:
            for _ in range(count): strs.append(token|tokenclr )


    # for i in range(1, len(lst)):
    #     if lst[i] == lst[i - 1]:
    #         count += 1
    #     else:
    #         token = lst[i-1]
    #         tokenclr = bg_red if token in special_tokens else bg_gray

    #         if count >= 3:
    #             strs.append(f"{token}*{count}@{i - count}"|tokenclr)
    #         else:
    #             if tokenclr == bg_red:
    #                 strs.append(f"{token}@{i - count}"|tokenclr)
    #                 for _ in range(count-1): strs.append(token|tokenclr )
    #             else:
    #                 for _ in range(count): strs.append(token|tokenclr )
    #         count = 1

    # # last item.
    # token = lst[-1]
    # tokenclr = bg_red if token in special_tokens else bg_gray
    # if count >= 3:
    #     strs.append(f"{token}*{count}@{len(lst) - count}"|tokenclr)
    # else:
    #     if tokenclr == bg_red:
    #         strs.append(f"{token}@{i - count}"|tokenclr)
    #         for _ in range(count-1): strs.append(token|tokenclr )
    #     else:
    #         for _ in range(count): strs.append(token|tokenclr )

    return strs

def readable_tokens_str(lst, special_tokens=None) -> str:
    return  ' '.join(readable_tokens(lst, special_tokens))
#
#
#
import random
def rand_index(lst):
    return random.randint(0, len(lst)-1)

import pandas as pd
def rand_item(lst):
    idx = rand_index(lst)
    if isinstance(lst, pd.DataFrame):
        return lst.iloc[idx], idx

    return lst[idx], idx

import textwrap
import datetime
from pathlib import Path
import builtins

class LLMLog:
    def __init__(self, *, log_path:Path=None) -> None:
        # log_path =  Path(log_path)
        self.log_path = log_path.with_stem(log_path.stem + datetime.date.today().strftime('_%Y-%m-%d'))
        self.label_clr = text_color(text_color.yellow)
        self.to_file_ = True
        pass

    def to_file(self, enable):
        self.to_file_ = enable

    def tokens(self, tokens, label=None):
        import torch
        if label:
            label = f"[{label}]: "
            self.print(label|self.label_clr, end='')

        if isinstance(tokens, torch.Tensor): tokens = tokens.tolist()
        tokens = readable_tokens( tokens )
        tokens_str = ' '.join(tokens)
        wrapped = textwrap.fill(tokens_str, width=int(100*3.5),
                    replace_whitespace=False,
                    initial_indent='', subsequent_indent='    ',
                    break_on_hyphens=False,
                    # break_long_words=True,
                    )
        self.print(wrapped)

        # return tokens

    def print(self, *argv, **kwargv):
        fileonly = 'fileonly' in kwargv
        if fileonly:
            fileonly = kwargv['fileonly']
            del kwargv['fileonly']

        # format
        with io.StringIO() as buffer:
            kwargv['file']=buffer
            builtins.print(*argv, **kwargv)
            text = buffer.getvalue()

        if not fileonly: builtins.print(text, end='')
        if self.to_file_ and self.log_path:
            with open(self.log_path, 'a') as fp:
                builtins.print(text_color.decolorize(text), file=fp, end='')

    def line(self): self.print('_'*80)

    def message(self, message=None, label=None):
        if label:
            self.print(f"[{label}]: "|self.label_clr)

        if message:
            message = textwrap.indent(message+ '<eom>', prefix=' '*4)
            message = colorize_llm(message)
            self.print(message)

    def openai_api_messages(self, messages, label=None):
        if label:
            self.print(f"[{label}]: "|self.label_clr)

        prompt = '\n'.join([f"{m['role']|green}: {m['content']}" for m in messages])
        prompt = textwrap.indent(prompt, prefix=' '*4)
        prompt = colorize_llm(prompt)
        self.print(prompt)


def static_vars(**kwargs):
    '''
    @static_vars(sep=' ')
    '''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


#
#
#
class Messages(list):
    def __init__(self, iterable = []):
        super().__init__(item for item in iterable)

    def extend(self, other):
        for item in other:
            self.add(item['role'], item['content'])
        return self

    def __getitem__(self, index):
        result = super().__getitem__(index)
        if isinstance(index, slice):
            return self.__class__(result)
        return result

    def __repr__(self) -> str:
        return '\n'.join([f'''{item['role']|yellow}: {item['content']}''' for item in self])
    def rewind(self, n):
        self[:] = self[:-n] if n < len(self) else []

    # def append(self, v0):
    #     v = vars(v0)
    #     if 'function_call' in v and v['content']:
    #         content = v['function_call']
    #     elif 'tool_calls' in v and v['tool_calls']:
    #         content = v['tool_calls']
    #     elif 'content' in v and v['content']:
    #         content = v['content']
    #     else:
    #         content = str(v)
    #     self.add(v['role'], content)

    def export(self) -> str:
        return '\n\n'.join([f'''__{item['role']}: {item['content']}''' for item in self])


    def head(self, n):
        self[:] = self[:n]

    # message
    def remove_role(self, role):
        i = 0
        while i < len(self):
            if self[i]['role'] == role:
                self.pop(i)
            else:
                i += 1
    # message
    def keep_role_only(self, role):
        i = 0
        while i < len(self):
            if self[i]['role'] != role:
                self.pop(i)
            else:
                i += 1

    def add(self, role, content):
        if len(content) == 0:
            raise RuntimeError('No content...')
        super().append( {"role":role, 'content': content} )
        return self

    def system(self, x):
        return self.add('system', x)
    def user(self, x):
        return self.add('user', x)
    def assistant(self, x):
        return self.add('assistant', x)

    # helpr
    def filter(self, cond:callable):
        return [ msg for msg in self if cond(msg)]

    def get_last_message(self):
        return self[-1]['content']

    def count(self, role):
        return sum([ 1 for msg in self if msg['role'] == role])
#
#
#
import openai # pip install openai
@static_vars(client = openai.OpenAI()) # api_key=os.getenv('OPENAI_API_KEY')
def chatgpt_conversation(messages, model_name = 'gpt-3.5-turbo-1106'):
    # model_name = 'gpt-3.5-turbo-0613'
    # model_name = 'gpt-3.5-turbo-1106'
    # model_name = 'gpt-4-0613'
    # model_name = 'gpt-4-1106-preview'

    try:
        response = chatgpt_conversation.client.chat.completions.create(
            model=model_name,
            messages=messages,
            # temperature=0.0,
            top_p=0.0,
            max_tokens=256,
            seed=42,
        )

        response_message = response.choices[0].message.content.strip()
        return response_message

    except openai.RateLimitError as ex:
        raise RuntimeError(str(ex))
