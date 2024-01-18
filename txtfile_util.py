

# text file util
import random

class Txtfile:
    @staticmethod
    def read(x, idx=-1, condition=None):
        if isinstance(x,list):
            if idx>=0:
                x = x[idx]
                if not condition or condition(txt):
                    return txt, x
            else:
                for _ in range(len(x)):
                    selected = x[random.randint(0,len(x)-1)]
                    with open(selected.strip()) as fp:
                        txt = fp.read()
                    if not condition or condition(txt):
                        return txt, selected
            return None, None

        with open(x.strip()) as fp:
            txt = fp.read()
        if not condition or condition(txt):
            return txt
        else:
            return None

    @staticmethod
    def write(x, txt):
        with open(x.strip(), 'w') as fp: fp.write(txt)

    @staticmethod
    def append(x, txt):
        with open(x.strip(), 'a') as fp: fp.write(txt)

