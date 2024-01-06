

import pandas as pd



def apply_comma(df:pd.DataFrame, inplace=False):
    numeric_columns = df.select_dtypes(include=['number']).columns
    tmp = df if inplace else df.copy()
    tmp[numeric_columns] = df[numeric_columns].applymap('{:,}'.format)
    return tmp
