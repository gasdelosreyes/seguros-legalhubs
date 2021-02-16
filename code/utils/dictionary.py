import re

def changeDic(text, **kwargs):
    """
        Cambia todos los valores en el diccionario especificado
    """
    for value in kwargs.get('vector'):
        text = re.sub(value[0],value[1],text)
    return text

'''
    UNICODE DICTIONARYS
'''

def codexDic():
    return [
            (r'\\\'b1', r'ni'),
            (r'\\\'d0', r'ni'),
            (r'\\\'e1', r'á'),
            (r'\\\'df', r'á'),
            (r'\\\'e9', r'é'),
            (r'\\\'ed', r'í'),
            (r'\\\'cd', r'í'),
            (r'\\\'y' , r'í'),
            (r'\\\'f3', r'ó'),
            (r'\\\'cb', r'ó'),
            (r'\\\'be', r'ó'),
            (r'\\\'fa', r'ú'),
            (r'\\\'dc', r'ú'),
            (r'\\\'fd', r'ý'),
            (r'\\\'e0', r'à'),
            (r'\\\'e8', r'è'),
            (r'\\\'ec', r'ì'),
            (r'\\\'f2', r'ò'),
            (r'\\\'f9', r'ù'),
            (r'\\\'b7', r'ù'),
            (r'\\\'c0', r'À'),
            (r'\\\'c8', r'È'),
            (r'\\\'cc', r'Ì'),
            (r'\\\'d2', r'Ò'),
            (r'\\\'d9', r'Ù'),
            (r'\\\'c1', r'Á'),
            (r'\\\'dd', r'Ý'),
            (r'\\\'c9', r'É'),
            (r'\\\'cd', r'Í'),
            (r'\\\'d3', r'Ó'),
            (r'\\\'da', r'Ú'),
    ]

def unicodexDic():
    return [
            (r'\\u9524\?',r'Á'),
            (r'\\u9492\?',r'Á'),
            (r'\\u9552\?',r'Í'),
            (r'\\u9568\?',r'Í'),
            (r'\\u8215\?',r'Ó'),
            (r'\\u9556\?',r'É'),
            (r'\\u9484\?',r'Ú'),
            (r'\\nu9553\?',r'numero'),
            (r'\\u9617\?',r' grados'),
            (r'\\u9508\?',r' '),
    ]

def formatDic():
    return [
            (r'\\uc1', r''),
            (r'\\red\d+\\green\d+lue\d+\;', r''),
            (r'\\green\d+\;', r''),
            (r'ff(?:\:|\,|\.|\-)', r''),
            (r'cc(?:\:|\,|\.|\-)', r''),
            (r'denuncia asegurado:', r''),
            (r'\\fcharset\d+\ \w+\;', r''),
    ]

def postformatDic():
    return [
            (r'ff\ ', r''),
            (r'fs\ ', r''),
            (r'cc\ ', r''),
            (r'tx\ ', r''),
    ]

def verbsDic():
    return [
        r'colisionandome',r'colisionandole',r'colisionandolo',r'colisionando',r'colisionado',r'colisionar',r'colisiono',r'colisione',r'colisiona',r'colision',
        r'impactandome',r'impactandole',r'impactandolo',r'impactando',r'impactado',r'impactarme',r'impactarlo',r'impactarle',r'impactar',r'impactado',r'impacta',r'impacto',r'impacte',
        r'embestido',r'embestirlo',r'embestirme',r'embestirle',r'embestir',r'embesti',
        r'embistiendome',r'embistiendole',r'embistiendolo',r'embistiendo',r'embistio',r'embisto',r'embiste',
        r'chocandome',r'chocandole',r'chocandolo',r'chocarlo',r'chocarle',r'chocarme',r'chocando'r'chocado',r'chocar',r'choque',r'choco',r'choca',
        r'tocandome',r'tocandole',r'tocandolo',r'tocando',r'tocarle',r'tocarme',r'tocado',r'tocarlo',r'tocar',r'toco',r'toca',r'toque',
    ]

def carDic():
    return [
            ((r'vhlo',r'vh',r'vhc',r'vehculo',r'vhl',r'camioneta',r'auto',r'automovil',r'rodado',r'autos',r'veh',r'coche',r'vehyculo'), r'vehiculo'),
            ((r'garage', r'gge', r'cochera'), r'garaje'),
            ((r'av',r'avda'), r'avenida'),
            ((r'lateral', r'costado', r'sector'), r'parte'),
            ((r'ero', r'taxi', r'taxista',r'terc'), r'tercero'),
            ((r'asegurada', r'aseg'), r'asegurado'),
            ((r'izquierdo', r'izq'),r'izquierda'),
            ((r'derecho', r'der', r'dere',r'dcha'),r'derecha'),
            ((r'trasero', ),r'trasera'),
            ((r'delantero', r'trompa',r'posterior'),r'delantera'),
    ]

def crashDic():
    return [
            ((r'y colisiono',r'lo colisiono', r'le colisiono', 
                r'colisione',r'embesti',r'embisto',r'choco',r'choque'), r'asegurado colisiona')
    ]