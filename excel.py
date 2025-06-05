
import pandas as pd
from pathlib import Path

# pip install openpyxl
class Excel:
    def __init__(self, file_path:str|Path):
        self.file_path = Path(file_path)
        mode = 'a' if self.file_path.exists() else 'w'
        self.writer = pd.ExcelWriter(self.file_path, engine='openpyxl', mode=mode)

    def add_page(self, sheet_name:str, data:pd.DataFrame):
        data.to_excel(self.writer, sheet_name=sheet_name, index=True)

# excel = Excel('trained/rsup-eval.xlsx')
# excel.add_page('new_page', pd.DataFrame())


