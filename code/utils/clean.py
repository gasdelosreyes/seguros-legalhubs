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
            print(word + ' SE CAMBIO POR ' + value[1] + ' ' + str(cantidad) + ' VECES')
    return fstring.split('|')[:-1]

def changeRatios(dataframe, vector):
    fstring = ''
    for row in dataframe:
        fstring += row + ' | '
    for w in set(fstring.split()):
        if(w != ratio.ratios(w,vector)):
            word = ratio.ratios(w,vector)
            fstring, cantidad = re.subn(r'\ ' + w + r'\ ',r' ' + word + r' ', fstring)
            print(w + ' SE CAMBIO POR ' + word + ' ' + str(cantidad) + ' VECES')
    return fstring.split('|')[:-1]
