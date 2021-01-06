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


def changeStrings(row):
    vector = [
        (r'vw','vehiculo'),(r'veh','vehiculo'),(r'vena','venia'),
        (r'derecho', 'derecha'), (r'delantero', 'delantera'), (r'trasero', 'trasera'), (r'frontal', 'parte delantera'), (r'lado', 'parte'),
        (r'frente delantero', 'parte delantera'), (r'de atras', 'en parte trasera'), (r'pb','parte'), (r'puerta', 'parte'), (r'parte lateral', 'parte'),
        (r'conmi','con mi'), (r'choque', 'mi parte delantera'),(r'circ','circulaba'), (r'stro','siniestro'), (r'ero','tercero'),(r'gge','garage'),
        (r' p ', ' parte '), (r'contra\s', 'con '), (r'por detras', 'en parte trasera'), (r'detras','trasera'),(r'parte parte', 'parte'), (r'roza', 'colisiona'),
        (r' ro ', 'tercero'),(r'gral','general'),(r'paragolpes delantero','parte delantera'), (r'trompa','delantera')
    ]
    words = []
    for w in row.split():
        for value in vector:
            if(w == value[0]):
                w = value[1]
        words.append(w)
    row = ' '.join(words)
    return row

def changeRegex(row):
    vector = [ 
        (r'izq.*? ', 'izquierda '), (r'lat.*? ', 'parte '), (r'av.*? ', 'avenida '), (r'amb.*? ', 'ambulancia '),
        (r'tercero .* impacta', 'tercero impacta'), (r'desde izq.*? ', 'en parte izquierda '), (r'vh.*? ', 'vehiculo '), (r'colis.*? ', 'colisiona '), (r'\scho.*? ', 'colisiona '),
        (r'\simpac.*? ', 'colisiona '), (r'su delat.*? ', 'su parte delantera '), (r'aseg.*? ', 'asegurado '), (r'emb.*? ', 'colisiona '), (r'redg.*? ',''),(r'\sgolp.*? ','colisiona')
    ]
    for value in vector:
        row = re.sub(value[0],value[1], row)
    return row

def clean(serie):
    """
    Limpia la columna donde estan las descripciones 

    :function: estructura todas las descripciones
    :returns: devuelve la misma descripcion pero con las palabras de to_rep a for_rep

    """
    for index, row in enumerate(serie):
        serie.iloc[index] = changeStrings(row)

    for index, row in enumerate(serie):
        serie.iloc[index] = changeRegex(row)
    
    return serie 

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
    # print(dataframe['descripcion'][59:61])
    dataframe['descripcion'] = dataframe['descripcion'].apply(cleaner)
    dataframe['descripcion'] = dataframe['descripcion'].apply(nonStop)
    dataframe['descripcion'] = clean(dataframe['descripcion'])
    '''
        Divide el DataFrame en 4 DataFrames, cada uno por categoria.
    '''
    auto, moto, bici, peaton = separador(dataframe)
    print(len(auto), len(moto), len(bici), len(peaton))
    print(auto['descripcion'][49])

    auto.to_csv('../dataset/casos/auto.csv', index=False, header=True)
    moto.to_csv('../dataset/casos/moto.csv', index=False, header=True)
    bici.to_csv('../dataset/casos/bici.csv', index=False, header=True)
    peaton.to_csv('../dataset/casos/peaton.csv', index=False, header=True)
