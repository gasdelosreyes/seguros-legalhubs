import re
import nltk
import numpy as np
import pandas as pd
from fuzzywuzzy import fuzz
from nltk.probability import FreqDist
from nltk.text import Text, ConcordanceIndex
from nltk.tokenize import word_tokenize


def read_file(path, sheet_name=None):
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


def appendDataFrames(dforiginal, dfappend, cols):
    """
        Concatena dataframes, al dforiginal agrega el dfappend
    """
    return dforiginal.append(dfappend.iloc[:, cols], ignore_index=True)


def cleaner(w):
    """
        elimina caracteres no deseados
        w = texto tipo string
    """
    vector = np.array([[r'\'e1', r'á'],
                       [r'\'e9', r'é'],
                       [r'\'ed', r'í'],
                       [r'\'cd', r'í'],
                       [r'\'fa', r'ú'],
                       [r'\'f3', r'ó'],
                       [r'\'e0', r'à'],
                       [r'\'e8', r'è'],
                       [r'\'ec', r'ì'],
                       [r'\'f2', r'ò'],
                       [r'\'f9', r'ù'],
                       [r'\'c1', r'Á'],
                       [r'\'d3', r'Ó'],
                       [r'\tab ', ''],
                       [r'\n', ''],
                       [r'\'d1', r'ñ'],
                       [r'\par_x000D_', ''],
                       [r'_x000D_', ''],
                       ['redgreenblue', ''],
                       ['fcharset', ''],
                       ['agaranond', ''],
                       ['redgreenlue', ''],
                       ['agaranond', ''],
                       ['d ', ''],
                       ['dfs', ''],
                       ['agaramond', ''],
                       ['ff','']])
    w = w.lower()
    w = w.replace('descripcion hecho', '')
    w = re.sub(r'[^\w\s]', '', w)
    w = re.sub(r'(\r\n?|\n)+', '', w)
    w = re.sub(r'\d+', '', w)
    for row in vector:
        w = w.replace(row[0], row[1])
    return w


def clean(serie):
    """
    Limpia la columna donde estan las descripciones 

    :function: estructura todas las descripciones
    :returns: devuelve la misma descripcion pero con las palabras de to_rep a for_rep

    """
    # TODO: hacer con pares ordenados
    to_rep = [
        r'izq.*? ', 'derecho', 'delantero', 'trasero', 'frontal', r'lat.*? ', 'lado',
        'frente', 'de atras', 'puerta', 'parte lateral', 'choque',
        r'tercero .* impacta', r'desde izq.*? ', r'vh.*? ', r'colis.*? ', r'choc.*? ',
        r'impac.*? ', r'su delat.*? ', ' p ', r'aseg.*? ', r'emb.*? ', r'redg.*? ',
        'parte parte', 'roza', r'golp.*? ', r'contra\s', 'por detras', 'detras',
        ' ro '
    ]  # tentativo ,'precede']

    for_rep = [
        'izquierda ', 'derecha', 'delantera', 'trasera', 'parte delantera', 'parte ',
        'parte', 'parte delantera', 'en parte trasera', 'parte', 'parte',
        'mi parte delantera', 'tercero impacta', 'en parte izquierda ',
        'vehiculo ', 'colisiona ', 'colisiona ', 'colisiona ',
        'su parte delantera ', ' parte ', 'asegurado ', 'colisiona ', '', 'parte',
        'colisiona', 'colisiona ', 'con ', 'en parte trasera', 'trasera', 'tercero'
    ]  # ,'con parte delantera']

    for i, w in enumerate(serie):
        try:
            for j in range(len(to_rep) - 1):
                w = re.sub(to_rep[j], for_rep[j], w)
            # esto se puso muy extraño
            w = re.sub(to_rep[len(to_rep) - 1], for_rep[len(for_rep) - 1], w)
        except:
            pass
        serie.iloc[i] = w
    return serie

# algunas stopwords que no aportan valor


def nonStop(w):
    return ' '.join(i for i in w.split() if i not in ['el', 'la', 'los', 'las', 'ellos', 'nosotros', 'lo', 'le',
                                                      'que', 'un', 'se', 'de', 'a', 'y', 'sobre', 'cuando', 'do', 'una'])


def separador(ds):
    """
        devuelve los 4 dataset predefinidos auto,moto,bici,peaton
    """
    ds['cod_accidente'] = ds['cod_accidente'].apply(cleaner)
    ds['cod_accidente'] = ds['cod_accidente'].str.strip()
    for idx, value in enumerate(ds['cod_accidente']):
        if value.startswith('p'):
            ds.loc[idx, 'cod_accidente'] = 'peaton'
    ds_a = ds[ds['cod_accidente'] == 'aa']
    ds_m = ds[ds['cod_accidente'] == 'am']
    ds_b = ds[ds['cod_accidente'] == 'aci']
    ds_p = ds[ds['cod_accidente'] == 'peaton']
    return ds_a, ds_m, ds_b, ds_p


if __name__ == "__main__":
    df1 = read_file('../dataset/casos_universidad.xlsx')
    df2 = read_file('../dataset/casos_zurich_20201228.xlsx', 'Dataset')

    '''
        Getting the columns that needs to be processed
    '''

    dataframe = pd.DataFrame()
    dataframe = appendDataFrames(dataframe, df1, [5, 11, 7])
    dataframe = appendDataFrames(dataframe, df2, [5, 11, 7])
    dataframe = dataframe.rename(
        columns={"descripcion_del_hecho - Final": "descripcion"})

    del_row = []
    for i, row in enumerate(dataframe['descripcion']):
        if type(row) != str or row.lower() == 'comprometida' or row == '' or row == ' ':
            del_row.append(i)

    dataframe = dataframe.drop(del_row)
    print(dataframe['descripcion'][1000])
    dataframe['descripcion'] = dataframe['descripcion'].apply(cleaner)
    dataframe['descripcion'] = dataframe['descripcion'].apply(nonStop)
    dataframe['descripcion'] = clean(dataframe['descripcion'])
    print(dataframe['descripcion'][1000])
    '''
        Divide el DataFrame en 4 DataFrames, cada uno por categoria.
    '''
    auto, moto, bici, peaton = separador(dataframe)
    print(len(auto), len(moto), len(bici), len(peaton))

    auto.to_csv('../dataset/casos/auto.csv', index=False, header=True)
    moto.to_csv('../dataset/casos/moto.csv', index=False, header=True)
    bici.to_csv('../dataset/casos/bici.csv', index=False, header=True)
    peaton.to_csv('../dataset/casos/peaton.csv', index=False, header=True)
