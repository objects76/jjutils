### import notebook
# - https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Importing%20Notebooks.html


import io, os, sys, types, re, typing
from IPython import get_ipython
from nbformat import read
from IPython.core.interactiveshell import InteractiveShell
from pathlib import Path
from datetime import datetime

def find_notebook(fullname, path=None):
    """find a notebook, given its fully qualified name and an optional path

    This turns "foo.bar" into "foo/bar.ipynb"
    and tries turning "Foo_Bar" into "Foo Bar" if Foo_Bar
    does not exist.
    """
    name = fullname.rsplit('.', 1)[-1]
    if not path:
        path = ['']
    for d in path:
        nb_path = os.path.join(d, name + ".ipynb")
        if os.path.isfile(nb_path):
            return nb_path
        # let import Notebook_Name find "Notebook Name.ipynb"
        nb_path = nb_path.replace("_", " ")
        if os.path.isfile(nb_path):
            return nb_path


class NotebookLoader(object):
    """Module Loader for Jupyter Notebooks"""
    def __init__(self, path=None):
        self.shell = InteractiveShell.instance()
        self.path = path

    def load_module(self, fullname):
        """import a notebook as a module"""
        path = find_notebook(fullname, self.path)

        print (f'importing notebook from "{path}"')

        # load the notebook object
        with io.open(path, 'r', encoding='utf-8') as f:
            nb = read(f, 4)

        # create the module and add it to sys.modules
        # if name in sys.modules:
        #    return sys.modules[name]
        mod = types.ModuleType(fullname)
        mod.__file__ = path
        mod.__loader__ = self
        mod.__dict__['get_ipython'] = get_ipython
        sys.modules[fullname] = mod

        # extra work to ensure that magics that would affect the user_ns
        # actually affect the notebook module's ns
        save_user_ns = self.shell.user_ns
        self.shell.user_ns = mod.__dict__

        try:
          for cell in nb.cells:
            if cell.cell_type == 'code':
                # transform the input to executable Python
                code = self.shell.input_transformer_manager.transform_cell(cell.source)
                # run the code in themodule
                exec(code, mod.__dict__)
        finally:
            self.shell.user_ns = save_user_ns
        return mod

class NotebookFinder(object):
    """Module finder that locates Jupyter Notebooks"""
    def __init__(self):
        self.loaders = {}

    def find_module(self, fullname, path=None):
        nb_path = find_notebook(fullname, path)
        if not nb_path:
            return

        key = path
        if path:
            # lists aren't hashable
            key = os.path.sep.join(path)

        if key not in self.loaders:
            self.loaders[key] = NotebookLoader(path)
        return self.loaders[key]

# register hook

# if sum(1 for obj in sys.meta_path if isinstance(obj, NotebookFinder)) == 0:
#     sys.meta_path.append(NotebookFinder())


# from 종목정보.stock_price2 import get_codes_by_marcap



#
# get all tags
#
def get_tags(nbfile:Path):
    import nbformat, io
    from datetime import datetime

    with io.open(nbfile, 'r', encoding='utf-8') as fp:
        nb = nbformat.read(fp, as_version=4)

    tags = [f"{nbfile}, {datetime.now():%Y-%m-%d, %H:%M}"]
    for cell in nb.cells:
        if cell.cell_type == 'code':
            if len(cell.source.strip()) == 0: continue

            if ('tags' in cell.metadata):
                tags.append(','.join(cell.metadata.tags))

    return '\n'.join([i for i in tags if i.strip()]) + '\n\n'

#
#
#
import nbformat, io, re

# export_notebook( globals() )
def export_notebook(globals_ = None, *, nbfile:str = None):
    ''' pip install nbformat '''
    if nbfile is None:
        nbfile:str = globals_['__vsc_ipynb_file__']

    with io.open(nbfile, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    markdowns = []
    exported_path = nbfile.replace('.ipynb', '.py')
    with open(exported_path, 'w') as fp:
        fp.write(f"# from {nbfile}\n# {datetime.now()}\n\n")
        for cell in nb.cells:
            if cell.cell_type == 'code':
                if len(cell.source.strip()) == 0:
                    continue

                is_x = ('tags' in cell.metadata) and ('x' in cell.metadata.tags)

                if not is_x and 'export_notebook' not in cell.source:
                    source = cell.source

                    # export markdowns upper code cell.
                    for markdown in markdowns:
                        fp.write('# '+ markdown.replace('\n', '\n# ') + '\n')

                    # remove notebook %reload_ext autoreload like statements
                    source = re.sub(r'^%\w+[^\n]+\n', '', source, flags=re.MULTILINE)

                    # comment out ! statement
                    source = re.sub(r'(\n[ \t]+![^\n]+)', r'\n#\1', source, flags=re.MULTILINE|re.DOTALL)

                    # remove __nbook__
                    # source = re.sub(r"if __name__ == '__nbook__':\s+(\n[ \t]+[^\n]+|\n)+", r'', cell.source, flags=re.MULTILINE)

                    # fp.write('\n# %%  ------------------------------------\n')
                    fp.write(source)
                    fp.write('\n\n')
                markdowns = []
            else:
                markdowns.append(cell.source)

        print( exported_path , 'done!')

# import notebook_utils
# notebook_utils.export_notebook(globals())
