


#
# dict to attrs
#
class Struct(object):
    def __init__(self, *args, **kwargs):
        for arg in args:
            assert isinstance(arg, dict)
            kwargs.update(dict(arg))

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
