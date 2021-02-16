import pandas as pd

'''
    FUNCIONES GENERALES
'''
def read_file_xls(path, sheet_name=None):
    """
        Lee el archivo y devuelve un dataframe
    """
    if sheet_name:
        df = pd.read_excel(path, sheet_name=sheet_name)
        df = df.dropna()
        return df
    df = pd.read_excel(path)
    df = df.dropna()
    return df

def read_file_csv(path, sheet_name=None):
    """
        Lee el archivo y devuelve un dataframe
    """
    if sheet_name:
        df = pd.read_csv(path, sheet_name=sheet_name)
        df = df.dropna()
        return df
    df = pd.read_csv(path)
    df = df.dropna()
    return df


def appendDataFrames(dforiginal, dfappend, cols):
    """
        Concatena dataframes, al dforiginal agrega el dfappend
    """
    return dforiginal.append(dfappend.iloc[:, cols], ignore_index=True)

'''
    FUNCIONES ESPECIFICAS
'''

def wrong(dataframe):
    del_row = []
    for i, row in enumerate(dataframe['descripcion']):
        if type(row) != str or row.lower() == 'comprometida' or row == '' or row == ' ':
            del_row.append(i)
    return del_row

def separador(ds):
    """
        devuelve los 4 dataset predefinidos auto,moto,bici,peaton
    """
    ds['cod_accidente'] = ds['cod_accidente'].str.strip()
    for idx, value in enumerate(ds['cod_accidente']):
        if value.startswith('p'):
            ds.loc[idx, 'cod_accidente'] = 'PEATON'
    ds_a = ds[ds['cod_accidente'] == 'AA']
    ds_m = ds[ds['cod_accidente'] == 'AM']
    ds_b = ds[ds['cod_accidente'] == 'ACI']
    ds_p = ds[ds['cod_accidente'] == 'PEATON']
    return ds_a, ds_m, ds_b, ds_p