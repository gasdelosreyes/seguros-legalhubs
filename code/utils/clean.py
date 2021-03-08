import re
from utils import dictionary, ratio

def general(txt):
    """
        elimina caracteres no deseados
        w = texto tipo string
    """
    txt = txt.translate(str.maketrans('áéíóúýàèìòùÁÉÍÓÚÀÈÌÒÙÝ','aeiouyaeiouAEIOUAEIOUY'))
    txt = txt.lower()
    txt = txt.replace('\r', ' ').replace('\n', ' ').replace("\v", ' ').replace("\t", ' ').replace("\f", ' ').replace("\a", ' ').replace("\b", ' ')
    txt = re.sub(r'3ro','tercero', txt)
    txt = txt.replace('envest','embest').replace('envist','embist')
    txt = txt.replace('coali','coli')
    txt = txt.replace('dana','dania').replace('dano','danio').replace('dane','danie')
    txt = txt.replace('roso','rozo').replace('rose','roce').replace('roze','roce')
    txt = txt.replace('cruze','cruce').replace('cruse','cruce').replace('crus','cruz')
    txt = txt.replace('aboll','aboy')
    txt = txt.replace('rall','ray')
    txt = txt.replace(' una ',' un ').replace('asegurada','asegurado').replace(' do ',' del ')
    txt = txt.replace(' la ',' ').replace(' el ',' ').replace(' las ',' ').replace(' los ',' ')
    txt = txt.replace('tracer','traser')
    txt = re.sub(r'[^\w\s]', ' ', txt)
    txt = re.sub(r'\d+',' ',txt)
    txt = re.sub(' +', ' ',txt)
    txt = txt.strip()
    return txt

def changeWords(dataframe, vector):
    fstring = ''
    for row in dataframe:
        fstring += row + ' | '
    for value in vector:
        for word in value[0]:
            fstring,cantidad = re.subn(r'\ '+ word + r'\ ',r' '+ value[1] + r' ',fstring)
            # print(word + ' SE CAMBIO POR ' + value[1] + ' ' + str(cantidad) + ' VECES')
    return fstring.split(' | ')[:-1]

def changeRatios(dataframe, vector):
    fstring = ''
    for row in dataframe:
        fstring += row + ' | '
    for w in set(fstring.split()):
        if(w != ratio.ratios(w,vector)):
            word = ratio.ratios(w,vector)
            fstring, cantidad = re.subn(r'\ ' + w + r'\ ',r' ' + word + r' ', fstring)
            print(w + ' SE CAMBIO POR ' + word + ' ' + str(cantidad) + ' VECES')
    return fstring.split(' | ')[:-1]

def deleteRepeated(row):
    row = row.split()
    i = 0
    while i < len(row) - 1:
        if row[i] == row[i + 1]:
            del row[i]
        i += 1
    return ' '.join(row)

def detectOther(row):
    if(' su parte ' in row):
        if(' tercero parte ' in row):
            row = re.sub(' su parte ',' asegurado parte ',row)
        elif(' asegurado parte ' in row):
            row = re.sub(' su parte ',' tercero parte ',row)
    if(' con parte ' in row):
        if(' tercero parte ' in row):
            row = re.sub(' con parte ',' con asegurado parte ',row)
        elif(' asegurado parte ' in row):
            row = re.sub(' con parte ',' con tercero parte ',row)
    if(' en parte ' in row):
        if(' tercero parte ' in row):
            row = re.sub(' en parte ',' en asegurado parte ',row)
        elif(' asegurado parte ' in row):
            row = re.sub(' en parte ',' en tercero parte ',row)
    return row

def changeRepeated(dataframe):
    fstring = ''
    for row in dataframe:
        fstring += detectOther(deleteRepeated(row)) + ' | '
    return fstring.split(' | ')[:-1]
