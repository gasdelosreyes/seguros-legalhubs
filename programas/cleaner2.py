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
import matplotlib.pyplot as plt

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
                       ['redgreenlue', ''],
                       ['agaranond', ''],
                       ['d ', ''],
                       ['dfs', ''],
                       ['agaramond', ''],
                       ['agaramon', ''],
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

vector = ['su','mi','tercero','asegurado','vehiculo','parte','colisiona','lateral','delantera','puerta','ampliacion',
    'derecha','izquierda','trasera','delantero','derecho','izquierdo','trasero','paragolpe','acompanante','conductor'
    'acompadante','guardabarro','guardabarros','auto','del','agaramond','agaramon','en','estaba','el','lado','sector','trazera','lados']

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
    # que hacemos con mio
    words = [('mi','asegurado'),('yo','asegurado')]
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

        #(r'choque', 'mi parte delantera'),
        #(r'contra\s', 'con '), (r'detras','trasera'),
        #venia detras mio
        #venia delante mio
        # Que hacemos con el contra
notcommon = [
        (r'ero','tercero'), (r'ro', 'tercero'), (r'veh','vehiculo'), (r'lat', 'parte'), (r'av','avenida'), (r'stro','siniestro'), (r'circ','circulaba'),
        (r'conmi','con mi'), (r'vena','venia'),(r'p', 'parte'), (r'frontal', 'parte delantera'), (r'gral','general'),
        (r'su delant\w*', 'su parte delantera'), (r'aseg\w*', 'asegurado'), (r'de atras', 'parte trasera'), (r'por detras', 'parte trasera'),
        (r'parte conductor','parte izquierda'), (r'parte acompanante','parte derecha'), (r'gol','vehiculo'), (r'coche','vehiculo'), (r'auto','vehiculo'),(r'ka','vehiculo')
]

#(r'roce','colisiona'),(r'raye','colisiona'), ,(r'toco','colisiona'),(r'toca','colision'a),(r'me colisiona', 'tercero colision')
vehicle = [
        (r'vh','vehiculo'),(r'vhl','vehiculo'),(r'vw','vehiculo'),(r'automovil','vehiculo'),(r'taxi','vehiculo'),
        (r'rodado','vehiculo'), (r'autos','vehiculos'), (r'camioneta','vehiculo'), (r'camion','vehiculo'),
        (r'peugeot','vehiculo'), (r'toyota','vehiculo'), (r'corolla','vehiculo'),(r'chevrolet','vehiculo'), (r'ford','vehiculo'),
        (r'focus','vehiculo'),(r'corsa','vehiculo'),(r'fiat','vehiculo'),(r'clio','vehiculo'),(r'renault','vehiculo'), (r'sandero','vehiculo'), (r'kangoo','vehiculo')
        ]

third = [
        (r'taxista', 'tercero'), (r'vecino','tercero')
]

locations = [
        (r'garage','garaje'),(r'gge','garaje'),(r'cochera','garaje')
]

parts = [
        (r'sector','parte'),(r'zona','parte'), (r'parte medio','parte'), (r'puerta', 'parte'),(r'lado', 'parte'),
        (r'aprt','parte'), (r'costado','parte'), (r'pb','parte'), (r'guardabarro','parte'), (r'lateral', 'parte')
]

# choco choca por colisiona
crash = [
        (r'envest','colisiona'),(r'envist','colisiona'),(r'embest','colisiona'),(r'embist','colisiona'), (r'roza', 'colisiona'),
        (r'raspon','colisiona'), (r'impac', 'colisiona'), (r'colis', 'colisiona'),
        (r'golp',' colisiona'), (r'contacto','colisiona') 
]

front = [
        (r'paragolpe','delantera'), (r'trompa','delantera'), (r'posterior','delantera'), (r'delant', 'delantera'), (r'frente delantero', 'parte delantera')
]

left = [
        (r'izq', 'izquierda')
]

right = [
        (r'dere', 'derecha')
]

back = [
        (r'atras','trasera'), (r'trase', 'trasera')
]
def swapStrings(row, vector):
    changes = 0
    for value in vector:
        oldString = r' ?' + value[0] + r'\w* '
        newString = r' ' + value[1] + r' '
        match = re.search(oldString, row)
        row, number = re.subn(oldString,newString, row)
        # if(match and (number != 0) and match.group() != newString):
        #     f.write('('+ str(match.group()) +',' + newString + ') \n' )
        changes += number
        firstOldString = r'^' + value[0] + r' '
        firstNewString = value[1] + r' '
        match = re.search(firstOldString, row)
        row, number = re.subn(firstOldString, firstNewString, row)
        # if(match and (number != 0) and match.group() != firstNewString):
        #     f.write('('+ str(match.group()) +',' + firstNewString + ') \n')
        changes += number
        lastOldString = r' ' + value[0] + r'\w*$'
        lastNewString = r' ' + value[1]
        match = re.search(lastOldString, row)
        row, number = re.subn(lastOldString, lastNewString, row)
        # if(match and (number != 0) and match.group() != lastNewString):
        #     f.write('('+ str(match.group()) +',' + lastNewString + ') \n')
        changes += number
    return row, changes
        
def changeCommonRegex(row, vector):
    for value in vector:
        oldString = ' ' + value[0] + ' '
        newString = ' ' + value[1] + ' '
        match = re.search(oldString, row)
        row, number = re.subn(oldString,newString, row)
        # if(match and (number != 0) and match.group() != newString):
        #     f.write('('+ str(match.group()) +',' + newString + ') \n')
    return row, number

def clean(serie):
    print('LLEGUE AL CLEANER')
    """
    Limpia la columna donde estan las descripciones 
    :function: estructura todas las descripciones
    :returns: devuelve la misma descripcion pero con las palabras de to_rep a for_rep
    tarda al rededor de 1 min 
    """
    serie = pd.Series(list(map(divideParts, serie)))
    nocomun = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = changeCommonRegex(row, notcommon)
        nocomun += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *NO COMUNES*: ' + str(nocomun))

    vehiculo = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,vehicle)
        vehiculo += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *VEHICULO*: ' + str(vehiculo))

    tercero = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,third)
        tercero += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *TERCERO*: ' + str(tercero))

    ubicacion = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,locations)
        ubicacion += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *UBICACIONES*: ' + str(ubicacion))

    parte = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,parts)
        parte += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *PARTE*: ' + str(parte))

    colition = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,crash)
        colition += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *COLISIONA*: ' + str(colition))

    delantera = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,front)
        delantera += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *DELANTERA*: ' + str(delantera))

    izquierda = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,left)
        izquierda += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *IZQUIERDA*: ' + str(izquierda))

    derecha = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,right)
        derecha += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *DERECHA*: ' + str(derecha))

    trasera = 0
    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,back)
        trasera += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *TRASERA*: ' + str(trasera))

    serie = pd.Series(list(map(divideParts, serie)))

    for index, row in enumerate(serie):
        serie.iloc[index] = ' '.join(list(map(cleanRatios,row.split())))
        serie.iloc[index] = ' '.join(list(map(ratios,row.split())))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = changeCommonRegex(row, notcommon)
        nocomun += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *NO COMUNES*: ' + str(nocomun))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,vehicle)
        vehiculo += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *VEHICULO*: ' + str(vehiculo))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,third)
        tercero += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *TERCERO*: ' + str(tercero))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,locations)
        ubicacion += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *UBICACIONES*: ' + str(ubicacion))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,parts)
        parte += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *PARTE*: ' + str(parte))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,crash)
        colition += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *COLISIONA*: ' + str(colition))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,front)
        delantera += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *DELANTERA*: ' + str(delantera))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,left)
        izquierda += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *IZQUIERDA*: ' + str(izquierda))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,right)
        derecha += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *DERECHA*: ' + str(derecha))

    for index, row in enumerate(serie):
        serie.iloc[index], cantidad = swapStrings(row,back)
        trasera += cantidad
    print('NUMERO DE CAMBIOS EN DATASET *TRASERA*: ' + str(trasera))

    serie = pd.Series(list(map(deleteRepeated, serie)))
    for index, row in enumerate(serie):
        serie.iloc[index] = changePersons(row)

    labels = 'nocomun', 'vehiculo', 'tercero', 'ubicacion', 'parte','delantera','izquierda','derecha','trasera'
    sizes = [nocomun, vehiculo, tercero, ubicacion, parte, delantera, izquierda, derecha, trasera]

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig('pieChanges.png')
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