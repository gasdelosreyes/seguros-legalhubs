import nltk
import numpy as np
import pandas as pd
import re
import time

from fuzzywuzzy import fuzz
from nltk.probability import FreqDist
from nltk.text import ConcordanceIndex
from nltk.text import Text
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
#matriz de correlacion
def changeStrings(row):
    vector = [
        (r'vhl','vehiculo'),(r'vw','vehiculo'),(r'veh','vehiculo'),(r'vena','venia'),(r'guardabarro','parte'),(r'guardabarros','parte'),(r'auto','vehiculo'),(r'automovil','vehiculo'),(r'costado','parte'),
        (r'derecho', 'derecha'), (r'delantero', 'delantera'), (r'trasero', 'trasera'), (r'frontal', 'parte delantera'), (r'lado', 'parte'),
        (r'pb','parte'), (r'puerta', 'parte'),
        (r'conmi','con mi'), (r'choque', 'mi parte delantera'),(r'circ','circulaba'), (r'stro','siniestro'), (r'ero','tercero'),(r'gge','garage'),
        (r' p ', ' parte '), (r'contra\s', 'con '), (r'detras','trasera'),(r'atras','trasera'), (r'roza', 'colisiona'),
        (r' ro ', ' tercero '),(r'gral','general'),(r'paragolpe','delantera'), (r'trompa','delantera'),(r'izq', 'izquierda'),
        (r'toco','colisiona'),(r'toca','colisiona'),(r'roza','colisiona'),(r'adelante','delantera'),(r'izquierdo','izquierda'),(r'posterior','delantera'),(r'vehiculo',''),
        (r'raspon','colisiona'),(r'zona','parte'),(r'roce','colisiona'),(r'ro','tercero'),(r'raye','colisiona'),(r'aprt','parte')
    ]
    words = []
    for w in row.split():
        for value in vector:
            if w == value[0]:
                w = value[1]
        words.append(w)
    row = ' '.join(words)
    return row

def changeRegex(row):
    vector = [ 
        (r'*vh.*?', 'vehiculo'),(r'*izq.*?','izquierda'),(r'*der.*?','derecha'),(r'*trase.*?','trasera'),
        (r'*envest.*?','colisiona'),(r'*envist.*?','colisiona'),(r'*embest.*?','colisiona'),
        (r'amb.*?', 'ambulancia'),(r'tercero .* impacta', 'tercero impacta'), (r'desde izq.*?', 'en parte izquierda'),
        (r'colis.*?', 'colisiona'), (r'*cho.*?', 'colisiona'),(r'impac.*?', ' colisiona')
    ]

    for value in vector:
        oldString = ' ' + value[0] + ' '
        newString = ' ' + value[1] + ' '
        row = re.sub(oldString,newString, row)

    vector2 = [
        (r'su delat.*?', 'su parte delantera'), (r'aseg.*?', 'asegurado'), (r'emb.*?', 'colisiona'), (r'redg.*?',''),
        (r'paragolpe delantera','parte delantera'), (r'posterior','delantera'), (r'av.*?', 'avenida'), (r'der.*?','derecha') , (r'lat.*?', 'parte'),
        (r'frente delantero', 'parte delantera'), (r'de atras', 'en parte trasera'), (r'por detras', 'en parte trasera'),(r'parte conductor','parte izquierda'),
        (r'golp.*?',' colisiona'),(r'delant.*?','delantera'),(r'contacto','colisiona'),(r'parte acompanante','parte derecha'),(r'parte medio','parte'), 
        (r'parte lateral', 'parte'),('sector','parte'),('zona','parte')
    ]

    for value in vector2:
        oldString = value[0] + ' '
        newString = value[1] + ' '
        row = re.sub(oldString,newString, row)
    row = re.sub(r'embes.*$','colisiona',row)
    row = re.sub(r'embes.$','colisiona',row)
    row = re.sub(r'embes*$','colisiona',row)
    row = re.sub(r'embesti$','colisiona',row)
    return row


vector = ['su','mi','tercero','asegurado','vehiculo','parte','colisiona','lateral','delantera',
    'derecha','izquierda','trasera','delantero','derecho','izquierdo','trasero','paragolpe','acompanante',
    'acompadante','guardabarro','guardabarros','auto','del','agaramond','agaramon','en','estaba','el']

unitedStrings = []
separatedStrings = []
for a in vector:
    for b in vector:
        unitedStrings.append(a+b)
        separatedStrings.append((a,b))
unitedStrings = np.array(unitedStrings)
separatedStrings = np.array(separatedStrings)

def divideParts(row):
    for i in range(len(unitedStrings)):
        newString = separatedStrings[i][0] + ' ' + separatedStrings[i][1]
        row = re.sub(unitedStrings[i],newString, row)   
        # if re.search(r' '+unitedStrings[i],row) or re.search(' '+unitedStrings[i]+' ',row):
            
    return row

def ratios(w):
    try:
        dic = [
            'trasera','delantera','izquierda','derecha','acompanante','atras','vehiculo','embesti'
        ]
        aux = 0
        word = ''
        for i in dic:
            if (aux <= fuzz.ratio(w, i) and 80 <= fuzz.ratio(w, i)):
                aux = fuzz.ratio(w, i)
                word = i
        if word != '':
            return word
        return w
    except TypeError:
        return w

def cleanRatios(w):
    try:
        if(len(w) <= 5):
            return w
        dic = ['izquierda','derecha','izquierdo','derecho', 'paragolpe','vehiculo']
        aux = 0
        word = ''
        for i in dic:
            if (aux <= fuzz.partial_ratio(w, i) and 100 <= fuzz.partial_ratio(w, i)):
                aux = fuzz.partial_ratio(w, i)
                word = i
        if word != '':
            return word
        return w
    except TypeError:
        return w

def deleteRepeated(row):
    row = row.split()
    i = 0
    while i < len(row) - 1:
        if row[i] == row[i + 1]:
            del row[i]
        i += 1
    return ' '.join(row)


def changePersons(row):
    words = [('mi','asegurado')]
    vector = ['delantera','derecha','trasera','izquierda']
    for word in words:
        for value in vector:
            oldString = word[0] +  ' ' + value
            oldStringv2 = word[0] +  ' parte'
            newString = word[1] + ' parte ' + value
            newStringv2 = word[1] +  ' parte'
            if(re.search(oldString,row)):
                row = re.sub(oldString,newString,row)
            elif(re.search(oldStringv2,row)):
                row = re.sub(oldStringv2,newStringv2,row)
    return row

def clean(serie):
    """
    Limpia la columna donde estan las descripciones 
    :function: estructura todas las descripciones
    :returns: devuelve la misma descripcion pero con las palabras de to_rep a for_rep
    tarda al rededor de 1 min 
    """

    serie = pd.Series(list(map(changeRegex, serie)))
    serie = pd.Series(list(map(changeStrings, serie)))
    serie = pd.Series(list(map(divideParts, serie)))
    for index, row in enumerate(serie):
        serie.iloc[index] = ' '.join(list(map(cleanRatios,row.split())))
        serie.iloc[index] = ' '.join(list(map(ratios,row.split())))
    serie = pd.Series(list(map(changeStrings, serie)))
    serie = pd.Series(list(map(changeRegex, serie)))
    serie = pd.Series(list(map(deleteRepeated, serie)))        
    return serie 

def nonStop(w):
    #me
    return ' '.join(i for i in w.split() if i not in ['el', 'la', 'los', 'las', 'ellos', 'nosotros', 'lo', 'le',
                                                      'que', 'un', 'se', 'de', 'a', 'y', 'sobre', 'cuando', 'do', 'una',
                                                      'en', 'del', 'al','ella','del','por','con','no','si','ni','en'])

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
    start = time.time()
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
    # from matplotlib import pyplot as plt
    auto.to_csv('../dataset/casos/auto.csv', index=False, header=True)
    moto.to_csv('../dataset/casos/moto.csv', index=False, header=True)
    bici.to_csv('../dataset/casos/bici.csv', index=False, header=True)
    peaton.to_csv('../dataset/casos/peaton.csv', index=False, header=True)
    print('Tiempo de ejecución: ',round(time.time()-start,2)//60,'min',round(time.time()-start,2)%60,'s')

    # auto_df1 = df1[df1['cod_accidente'] == 'aa']
    # # Data to plot
    # labels = ['COMPROMETIDA', 'DISCUTIDA', 'SIN RESPONSABILIDAD', 'CONCURRENTE']
    # compro = len(df1['responsabilidad'][df1['responsabilidad'] == 'COMPROMETIDA'])
    # concu = len(df1['responsabilidad'][df1['responsabilidad'] == 'CONCURRENTE'])
    # sin = len(df1['responsabilidad'][df1['responsabilidad'] == 'SIN RESPONSABILIDAD'])
    # disc = len(df1['responsabilidad'][df1['responsabilidad'] == 'DISCUTIDA'])
    # sizes = [compro, disc, sin, concu]
    # colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    # explode = (0.1, 0, 0, 0)  # explode 1st slice

    # # Plot
    # plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    #         autopct='%1.1f%%', shadow=True, startangle=140)
    # plt.savefig('resposabilidad_auto_datos_laCaja.png')
