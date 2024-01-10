
import inspect
from .clrs import yellow, green, blue


import logging
import logging.handlers
from pathlib import Path

def get_logger(logfile, name=None):

    logger = logging.getLogger( name if name else Path(logfile).stem)
    formatter = logging.Formatter('%(asctime)s %(levelname)8s - %(message)s', "%m/%d %H:%M:%S")

    streamHandler = logging.StreamHandler()
    fileHandler = logging.FileHandler( Path(logfile).with_suffix('.log') )
    streamHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    try:
        while logger.hasHandlers():
            logger.removeHandler(logger.handlers[0])
    except:pass

    logger.addHandler(streamHandler)
    logger.addHandler(fileHandler)

    # logger.setLevel(level=logging.DEBUG)
    return logger

#
#
#
def val_name(variable, local_vars = None):
    if local_vars is None:
        frame = inspect.currentframe()
        local_vars = frame.f_back.f_locals
    for name, value in local_vars.items():
        if value is variable:
            return name
    return None


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

