
import inspect
from .clrs import yellow, green, blue

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

